from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

NormType = Literal["group", "instance", "batch", "none"]
ActType = Literal["relu", "lrelu"]


def _act(act: ActType) -> nn.Module:
    if act == "relu":
        return nn.ReLU(inplace=True)
    return nn.LeakyReLU(0.1, inplace=True)


def _norm3d(c: int, norm: NormType, gn_groups: int = 8) -> nn.Module:
    norm = (norm or "group").lower()  # type: ignore[assignment]
    if norm == "instance":
        return nn.InstanceNorm3d(c, affine=True, eps=1e-5)
    if norm == "batch":
        return nn.BatchNorm3d(c, eps=1e-5, momentum=0.1)
    if norm == "group":
        g = min(int(gn_groups), c)
        while g > 1 and (c % g) != 0:
            g -= 1
        return nn.GroupNorm(g, c, eps=1e-5)
    if norm == "none":
        return nn.Identity()
    raise ValueError(f"Unsupported norm: {norm}")


def _blur3d(vol: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return vol
    if k % 2 == 0:
        k += 1
    return F.avg_pool3d(vol, kernel_size=k, stride=1, padding=k // 2)


def _normalize_map(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dims = tuple(range(2, x.dim()))
    xmin = x.amin(dim=dims, keepdim=True)
    xmax = x.amax(dim=dims, keepdim=True)
    return (x - xmin) / (xmax - xmin + eps)



def _crop_or_resize_to(x: torch.Tensor, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """Match [D,H,W] safely after patch expanding.

    PatchMerging3D pads odd spatial sizes on the positive side before merging.
    PatchExpanding3D can therefore return one voxel larger than the skip feature.
    Cropping from the positive side preserves the original voxel origin. If a
    tensor is unexpectedly smaller, fall back to trilinear interpolation.
    """
    td, th, tw = [int(v) for v in target_shape]
    d, h, w = x.shape[2:]
    if d >= td and h >= th and w >= tw:
        return x[:, :, :td, :th, :tw]
    return F.interpolate(x, size=(td, th, tw), mode="trilinear", align_corners=False)


class ConvBlock3D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        n_convs: int = 1,
        norm: NormType = "group",
        act: ActType = "lrelu",
        gn_groups: int = 8,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        for i in range(max(1, int(n_convs))):
            cin = in_ch if i == 0 else out_ch
            layers.append(nn.Conv3d(cin, out_ch, kernel_size=3, stride=1, padding=1, bias=bias))
            layers.append(_norm3d(out_ch, norm, gn_groups))
            layers.append(_act(act))
            if dropout > 0:
                layers.append(nn.Dropout3d(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearEmbed3D(nn.Module):
    """Initial 3D linear embedding for the Swin-UNet refiner.

    This replaces the previous Stem/Initial ConvBlock.  It is intentionally a
    per-voxel linear projection, implemented as 1x1x1 Conv3d:

        [B, C_in, D, H, W] -> [B, C_embed, D, H, W]

    No spatial downsampling is performed here.  Down/up scaling is controlled
    only by the two supported refiner versions:
      v1: Conv3D stride downsampling + ConvTranspose/interpolate upsampling
      v2: PatchMerging3D + PatchExpanding3D
    """
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True, bias: bool = True):
        super().__init__()
        self.proj = nn.Conv3d(int(in_ch), int(out_ch), kernel_size=1, stride=1, padding=0, bias=bool(bias))
        self.norm = nn.LayerNorm(int(out_ch)) if bool(norm) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        # LayerNorm is applied over the embedding/channel dimension.
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()
        x_cl = self.norm(x_cl)
        return x_cl.permute(0, 4, 1, 2, 3).contiguous()


def _window_partition(x: torch.Tensor, ws: int) -> torch.Tensor:
    # x: [B,D,H,W,C] -> [B*nW, ws^3, C]
    B, D, H, W, C = x.shape
    x = x.view(B, D // ws, ws, H // ws, ws, W // ws, ws, C)
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, ws * ws * ws, C)
    return windows


def _window_reverse(windows: torch.Tensor, ws: int, B: int, D: int, H: int, W: int) -> torch.Tensor:
    C = windows.shape[-1]
    x = windows.view(B, D // ws, H // ws, W // ws, ws, ws, ws, C)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, D, H, W, C)
    return x


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.dim = int(dim)
        self.num_heads = max(1, int(num_heads))
        while self.dim % self.num_heads != 0 and self.num_heads > 1:
            self.num_heads -= 1
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.dim, self.dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask[None, :, None, :, :]
            attn = attn.view(-1, self.num_heads, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class SwinBlock3D(nn.Module):
    """3D shifted-window transformer block used at every U-Net resolution."""
    def __init__(
        self,
        dim: int,
        num_heads: int = 2,
        window_size: int = 4,
        shift: bool = False,
        mlp_ratio: float = 2.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.window_size = max(2, int(window_size))
        self.shift_size = self.window_size // 2 if bool(shift) else 0
        self.norm1 = nn.LayerNorm(self.dim)
        self.attn = WindowAttention3D(self.dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(self.dim)
        self.mlp = Mlp(self.dim, int(self.dim * float(mlp_ratio)), drop=drop)
        # Cache shifted-window masks by padded spatial size. Rebuilding the 3D
        # SW-MSA mask on full/near-full resolution is extremely expensive.
        self._attn_mask_cache: Dict[tuple, torch.Tensor] = {}

    @staticmethod
    def _attn_mask(Dp: int, Hp: int, Wp: int, ws: int, shift: int, device: torch.device) -> Optional[torch.Tensor]:
        if shift <= 0:
            return None
        img_mask = torch.zeros((1, Dp, Hp, Wp, 1), device=device)
        cnt = 0
        d_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        for d in d_slices:
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, d, h, w, :] = cnt
                    cnt += 1
        mask_windows = _window_partition(img_mask, ws).squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)
        return attn_mask

    def _cached_attn_mask(self, Dp: int, Hp: int, Wp: int, ws: int, shift: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if shift <= 0:
            return None
        key = (int(Dp), int(Hp), int(Wp), int(ws), int(shift), device.type, device.index, str(dtype))
        mask = self._attn_mask_cache.get(key)
        if mask is None or mask.device != device or mask.dtype != dtype:
            mask = self._attn_mask(Dp, Hp, Wp, ws, shift, device)
            if mask is not None:
                mask = mask.to(dtype=dtype)
            self._attn_mask_cache[key] = mask
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        ws = min(self.window_size, D, H, W)
        if ws < 2:
            return x
        shift = 0 if min(D, H, W) <= ws else min(self.shift_size, ws // 2)

        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()
        shortcut = x_cl
        x_norm = self.norm1(x_cl)

        pad_d = (ws - D % ws) % ws
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_d or pad_h or pad_w:
            x_norm = F.pad(x_norm, (0, 0, 0, pad_w, 0, pad_h, 0, pad_d))
        Dp, Hp, Wp = x_norm.shape[1:4]

        if shift > 0:
            x_shifted = torch.roll(x_norm, shifts=(-shift, -shift, -shift), dims=(1, 2, 3))
            attn_mask = self._cached_attn_mask(Dp, Hp, Wp, ws, shift, x.device, x_norm.dtype)
        else:
            x_shifted = x_norm
            attn_mask = None

        x_windows = _window_partition(x_shifted, ws)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        x_shifted = _window_reverse(attn_windows, ws, B, Dp, Hp, Wp)
        if shift > 0:
            x_attn = torch.roll(x_shifted, shifts=(shift, shift, shift), dims=(1, 2, 3))
        else:
            x_attn = x_shifted

        x_cl = shortcut + x_attn[:, :D, :H, :W, :]
        x_cl = x_cl + self.mlp(self.norm2(x_cl))
        return x_cl.permute(0, 4, 1, 2, 3).contiguous()


def _swin_enabled_for_stage(cfg: "SwinUNet3DRefinerConfig", role: str, stage_index: int) -> bool:
    if not bool(getattr(cfg, "swin_each_layer", True)):
        return False
    placement = str(getattr(cfg, "swin_placement", "all")).lower()
    if placement in ("all", "true", "full", "layerwise", "layer-wise"):
        return True
    if placement in ("encoder", "enc"):
        return role in ("stem", "down", "mid")
    if placement in ("decoder", "dec"):
        return role in ("mid", "up")
    if placement in ("bottleneck", "mid", "middle"):
        return role == "mid"
    if placement in ("none", "conv", "off"):
        return False
    raise ValueError(f"Unsupported swin_placement={placement!r}; use all/encoder/decoder/bottleneck/none")


def _stage_swin_block_count(cfg: "SwinUNet3DRefinerConfig", role: str, stage_index: int) -> int:
    """Return block count for a stage.

    Speed-safe default:
      - stage 0 / full-resolution stages use only W-MSA by default.
        Full-resolution SW-MSA creates a huge attention mask and was the main
        reason for the 20~40 s/iter slowdown.
      - lower-resolution stages keep swin_blocks=2, i.e. W-MSA + SW-MSA.
    """
    if stage_index == 0:
        if role == "up":
            return max(0, int(getattr(cfg, "swin_first_decoder_blocks", getattr(cfg, "swin_first_blocks", 1))))
        return max(0, int(getattr(cfg, "swin_first_blocks", 1)))
    return max(0, int(getattr(cfg, "swin_blocks", 2)))


class SwinStage3D(nn.Module):
    def __init__(self, dim: int, cfg: "SwinUNet3DRefinerConfig", stage_index: int, role: str = "mid"):
        super().__init__()
        if not _swin_enabled_for_stage(cfg, role=role, stage_index=stage_index):
            self.blocks = nn.Identity()
            return
        if stage_index == 0:
            ws = int(getattr(cfg, "swin_first_window_size", 2))
        else:
            ws = int(getattr(cfg, "swin_window_size", 4))
        n_blocks = _stage_swin_block_count(cfg, role=role, stage_index=stage_index)
        if n_blocks <= 0:
            self.blocks = nn.Identity()
            return
        blocks: List[nn.Module] = []
        for i in range(n_blocks):
            blocks.append(
                SwinBlock3D(
                    dim=dim,
                    num_heads=int(cfg.swin_num_heads),
                    window_size=ws,
                    shift=(i % 2 == 1),
                    mlp_ratio=float(cfg.swin_mlp_ratio),
                    drop=float(cfg.swin_drop),
                    attn_drop=float(cfg.swin_attn_drop),
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)



class PatchMerging3D(nn.Module):
    """3D patch merging by 2x2x2 token concatenation + linear projection.

    This is the 3D analogue of Swin patch merging. It replaces stride-2
    Conv3D downsampling in the v2 refiner. Input/output use PyTorch's
    [B,C,D,H,W] layout, while the token projection is applied in channels-last
    form. Odd spatial sizes are padded on the positive side and later cropped
    by the decoder, so the final refiner output keeps the original size.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.norm = nn.LayerNorm(8 * self.in_ch)
        self.reduction = nn.Linear(8 * self.in_ch, self.out_ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("PatchMerging3D expects [B,C,D,H,W]")
        B, C, D, H, W = x.shape
        pad_d = D % 2
        pad_h = H % 2
        pad_w = W % 2
        if pad_d or pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        # [B,C,D,H,W] -> [B,D,H,W,C]
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 1::2, 0::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], dim=-1)
        x = self.reduction(self.norm(x))
        return x.permute(0, 4, 1, 2, 3).contiguous()


class PatchExpanding3D(nn.Module):
    """3D patch expanding by linear channel expansion + 2x2x2 rearrange.

    It replaces ConvTranspose3D/interpolate upsampling in the v2 refiner.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.expand = nn.Linear(self.in_ch, 8 * self.out_ch, bias=False)
        self.norm = nn.LayerNorm(self.out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError("PatchExpanding3D expects [B,C,D,H,W]")
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B,D,H,W,C]
        x = self.expand(x)  # [B,D,H,W,8*out_ch]
        x = x.view(B, D, H, W, 2, 2, 2, self.out_ch)
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
        x = x.view(B, D * 2, H * 2, W * 2, self.out_ch)
        x = self.norm(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cfg: "SwinUNet3DRefinerConfig", stage_index: int):
        super().__init__()
        self.down = nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=cfg.bias)
        self.norm = _norm3d(out_ch, cfg.norm, cfg.gn_groups)
        self.act = _act(cfg.act)
        self.conv = ConvBlock3D(out_ch, out_ch, 1, cfg.norm, cfg.act, cfg.gn_groups, cfg.dropout, cfg.bias)
        self.swin = SwinStage3D(out_ch, cfg, stage_index, role="down")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.norm(self.down(x)))
        x = self.conv(x)
        return self.swin(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cfg: "SwinUNet3DRefinerConfig", stage_index: int):
        super().__init__()
        self.use_transpose = cfg.use_transpose
        if cfg.use_transpose:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=cfg.bias)
            self.proj = None
        else:
            self.up = None
            self.proj = nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=cfg.bias)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch, 1, cfg.norm, cfg.act, cfg.gn_groups, cfg.dropout, cfg.bias)
        self.swin = SwinStage3D(out_ch, cfg, stage_index, role="up")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if self.use_transpose:
            x = self.up(x)  # type: ignore[misc]
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="trilinear", align_corners=False)
            x = self.proj(x)  # type: ignore[operator]
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.swin(x)


class DownBlockV2(nn.Module):
    """PatchMerging3D down block: concat 2x2x2 tokens + projection."""
    def __init__(self, in_ch: int, out_ch: int, cfg: "SwinUNet3DRefinerConfig", stage_index: int):
        super().__init__()
        self.merge = PatchMerging3D(in_ch, out_ch)
        self.conv = ConvBlock3D(out_ch, out_ch, 1, cfg.norm, cfg.act, cfg.gn_groups, cfg.dropout, cfg.bias)
        self.swin = SwinStage3D(out_ch, cfg, stage_index, role="down")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.merge(x)
        x = self.conv(x)
        return self.swin(x)


class UpBlockV2(nn.Module):
    """PatchExpanding3D up block: linear expansion + 2x2x2 token rearrange."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cfg: "SwinUNet3DRefinerConfig", stage_index: int):
        super().__init__()
        self.expand = PatchExpanding3D(in_ch, out_ch)
        self.conv = ConvBlock3D(out_ch + skip_ch, out_ch, 1, cfg.norm, cfg.act, cfg.gn_groups, cfg.dropout, cfg.bias)
        self.swin = SwinStage3D(out_ch, cfg, stage_index, role="up")

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.expand(x)
        if x.shape[2:] != skip.shape[2:]:
            x = _crop_or_resize_to(x, skip.shape[2:])
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return self.swin(x)


@dataclass
class SwinUNet3DRefinerConfig:
    in_channels: int = 2
    base_channels: int = 8
    depth: int = 3
    norm: NormType = "group"
    act: ActType = "lrelu"
    gn_groups: int = 8
    dropout: float = 0.0
    bias: bool = False
    use_transpose: bool = True

    # Initial layer.  The previous Stem/Initial ConvBlock is removed; both
    # Swin-UNet versions now start with LinearEmbed3D.
    linear_embed_norm: bool = True
    linear_embed_bias: bool = True

    # Only two Swin-UNet refiner versions are supported:
    #   v1 = Conv3D stride downsampling + ConvTranspose/interpolate upsampling.
    #   v2 = PatchMerging3D + PatchExpanding3D.
    # Both versions share the same LinearEmbed3D input layer, gated residual API,
    # and final output size.
    refiner_version: str = "v2"

    # Layer-wise Swin settings.
    swin_placement: str = "all"
    swin_each_layer: bool = True
    swin_first_window_size: int = 2
    swin_window_size: int = 4
    swin_num_heads: int = 2
    # Speed-safe layer-wise Swin settings.
    #   full-resolution stages: swin_first_blocks=1 -> W-MSA only
    #   lower-resolution stages: swin_blocks=2 -> W-MSA + SW-MSA
    # This keeps the standard W/SW pair where it is affordable, while avoiding
    # the huge full-resolution SW-MSA mask that caused the 20~40 s/iter slowdown.
    swin_first_blocks: int = 1
    swin_first_decoder_blocks: int = 1
    swin_blocks: int = 2
    swin_mlp_ratio: float = 2.0
    swin_drop: float = 0.0
    swin_attn_drop: float = 0.0

    # Refiner residual output is deliberately simple:
    #   out[:, 0:1] = delta_raw
    #   out[:, 1:2] = gate_logit
    #   chi_ref = chi_gs + residual_scale * gate * clip(delta_raw)
    # The old low/high-frequency split was removed to keep the refiner easier
    # to interpret and avoid mixing it with CFR high-frequency terms.
    refine_mode: str = "fixed_ratio"
    fixed_ref_ratio: float = 1.00
    residual_scale: float = 0.90
    tanh_tau: float = 8.0
    delta_clip_abs: float = 0.080
    zero_init_last: bool = True
    gate_bias_init: float = -1.4
    only_update_in_mask: bool = True

    gate_use_input_prior: bool = False
    gate_prior_scale: float = 0.0
    gate_smooth_kernel: int = 3
    gate_cap: float = 0.75
    gate_confidence_floor: float = 0.0
    gate_confidence_channel: int = 1


class SwinUNet3DRefiner(nn.Module):
    """Layer-wise 3D Swin-UNet gated residual refiner.

    API-compatible with the previous Swin-UNet-lite / UNet3DRefiner interface:
    forward(x, base_chi, mask, return_aux) returns chi_ref or (chi_ref, aux).

    Version switch:
      - refiner_version="v1": LinearEmbed3D + Conv3D down/up sampling + SwinStage3D.
      - refiner_version="v2": LinearEmbed3D + PatchMerging3D/PatchExpanding3D + SwinStage3D.

    Both versions output the same two residual-control channels and keep
    chi_ref = chi_gs + gated residual delta.
    """
    def __init__(self, cfg: Optional[SwinUNet3DRefinerConfig] = None, **kwargs):
        super().__init__()
        if cfg is None:
            cfg = SwinUNet3DRefinerConfig(**kwargs)
        else:
            for k, v in kwargs.items():
                if hasattr(cfg, k):
                    setattr(cfg, k, v)
        if cfg.depth < 2:
            raise ValueError("depth must be >= 2")
        self.cfg = cfg
        raw_version = str(getattr(cfg, "refiner_version", "v2")).lower().replace("-", "_")
        version_alias = {
            "v1": "v1",
            "updown": "v1",
            "up_down": "v1",
            "conv_updown": "v1",
            "conv_down_up": "v1",
            "v2": "v2",
            "merging_expanding": "v2",
            "merge_expand": "v2",
        }
        if raw_version not in version_alias:
            raise ValueError(
                f"Unsupported refiner_version={raw_version!r}; only two versions are supported: "
                "'v1' for down/up sampling, or 'v2' for PatchMerging/PatchExpanding."
            )
        self.refiner_version = version_alias[raw_version]

        chs = [int(cfg.base_channels) * (2 ** i) for i in range(int(cfg.depth))]
        self.linear_embed = LinearEmbed3D(
            cfg.in_channels,
            chs[0],
            norm=bool(getattr(cfg, "linear_embed_norm", True)),
            bias=bool(getattr(cfg, "linear_embed_bias", True)),
        )
        self.embed_swin = SwinStage3D(chs[0], cfg, stage_index=0, role="stem")

        self.downs = nn.ModuleList()
        down_cls = DownBlockV2 if self.refiner_version == "v2" else DownBlock
        for i in range(1, cfg.depth):
            self.downs.append(down_cls(chs[i - 1], chs[i], cfg, stage_index=i))

        self.mid_conv = ConvBlock3D(chs[-1], chs[-1], 1, cfg.norm, cfg.act, cfg.gn_groups, cfg.dropout, cfg.bias)
        self.mid_swin = SwinStage3D(chs[-1], cfg, stage_index=cfg.depth - 1, role="mid")

        self.ups = nn.ModuleList()
        up_cls = UpBlockV2 if self.refiner_version == "v2" else UpBlock
        for i in range(cfg.depth - 1, 0, -1):
            self.ups.append(up_cls(chs[i], chs[i - 1], chs[i - 1], cfg, stage_index=i - 1))
        self.out_conv = nn.Conv3d(chs[0], 2, kernel_size=1, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                if m is self.out_conv and self.cfg.zero_init_last:
                    continue
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.cfg.zero_init_last:
            nn.init.zeros_(self.out_conv.weight)
            nn.init.zeros_(self.out_conv.bias)
            with torch.no_grad():
                self.out_conv.bias[1].fill_(float(self.cfg.gate_bias_init))

    def _gate_prior_from_input(self, x: torch.Tensor) -> torch.Tensor:
        # Minimal-input version. The refiner no longer receives hand-crafted HP,
        # grad, field-residual, or reliability channels. Keep this as a safe
        # fallback only; by default gate_use_input_prior=False.
        if x.shape[1] >= 2:
            return x[:, 1:2].clamp(0.0, 1.0)
        return x.new_zeros((x.shape[0], 1, *x.shape[2:]))

    def forward(
        self,
        x: torch.Tensor,
        base_chi: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_aux: bool = False,
    ):
        if x.dim() != 5:
            raise ValueError("x must be 5D [B,C,D,H,W]")
        if base_chi is None:
            base_chi = x[:, :1]
        if base_chi.dim() != 5 or base_chi.shape[1] != 1:
            raise ValueError("base_chi must be [B,1,D,H,W]")
        if self.cfg.only_update_in_mask and mask is None:
            raise ValueError("mask required when only_update_in_mask=True")
        if mask is not None and mask.shape != base_chi.shape:
            raise ValueError("mask shape must match base_chi")

        y = self.embed_swin(self.linear_embed(x))
        skips: List[torch.Tensor] = [y]
        for d in self.downs:
            y = d(y)
            skips.append(y)

        y = self.mid_swin(self.mid_conv(y))

        for i, u in enumerate(self.ups):
            y = u(y, skips[-2 - i])

        out = self.out_conv(y)
        delta_raw_unclipped = out[:, 0:1]
        gate_logit = out[:, 1:2]

        # Bound the predicted residual. This remains trainable because tanh has
        # derivative 1 around zero, while preventing extreme chi jumps.
        delta_raw = delta_raw_unclipped
        tau = float(self.cfg.tanh_tau)
        if tau > 0:
            delta_raw = torch.tanh(delta_raw / tau) * tau

        delta_clip = float(getattr(self.cfg, "delta_clip_abs", 0.0))
        if delta_clip > 0:
            delta_raw = torch.tanh(delta_raw / delta_clip) * delta_clip

        refine_mode = str(getattr(self.cfg, "refine_mode", "fixed_ratio")).lower()
        use_fixed_ratio = refine_mode in ("fixed", "fixed_ratio", "fixed-ratio", "ratio", "no_gate", "nogate")

        if use_fixed_ratio:
            # Fixed-ratio residual refiner:
            #   chi_ref = chi_gs + fixed_ref_ratio * clip(delta_raw)
            # This bypasses the learned gate, so refiner strength is explicitly controlled.
            ratio = float(getattr(self.cfg, "fixed_ref_ratio", 1.0))
            gate = torch.ones_like(delta_raw) * ratio
            delta = ratio * delta_raw
        else:
            # Backward-compatible learned-gate mode.
            if self.cfg.gate_use_input_prior:
                gate_logit = gate_logit + float(self.cfg.gate_prior_scale) * self._gate_prior_from_input(x)
            gate = torch.sigmoid(gate_logit)
            if int(self.cfg.gate_smooth_kernel) > 1:
                gate = _blur3d(gate, int(self.cfg.gate_smooth_kernel))
            gate = gate * float(self.cfg.gate_cap)
            floor = float(getattr(self.cfg, "gate_confidence_floor", 0.0))
            ch = int(getattr(self.cfg, "gate_confidence_channel", 4))
            if floor > 0.0 and x.shape[1] > ch:
                struct_prior = x[:, ch:ch + 1].clamp(0.0, 1.0)
                gate = torch.maximum(gate, floor * struct_prior)
            delta = float(self.cfg.residual_scale) * gate * delta_raw

        chi_ref = base_chi + delta

        if mask is not None:
            chi_ref = chi_ref * mask + base_chi * (1.0 - mask)
            delta = delta * mask
            delta_raw = delta_raw * mask
            gate = gate * mask

        if return_aux:
            aux: Dict[str, torch.Tensor] = {
                "delta": delta,                 # effective residual added to chi_gs
                "delta_raw": delta_raw,         # clipped residual before fixed ratio / gate
                "delta_raw_unclipped": delta_raw_unclipped,
                "gate": gate,                   # fixed ratio map in fixed_ratio mode
                "gate_logit": gate_logit,
            }
            return chi_ref, aux
        return chi_ref

    def predict_delta(self, x: torch.Tensor, base_chi: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, aux = self.forward(x, base_chi=base_chi, mask=mask, return_aux=True)
        return aux["delta"]
