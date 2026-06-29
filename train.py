from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, load_config, validate_config
from model.forward import ForwardOp, build_forward_from_cfg
from model.initialize import build_init_params_from_cfg
from utils.loss import tv_loss_3d, param_regularization
from model.gs_model import GaussianChiModel, GSStats
from utils.tool import (
    ensure_dir,
    load_data,
    load_nifti,
    save_nifti,
    set_seed,
    repair_mask_np,
    estimate_lfs_noise_score_np,
    bbox_from_mask,
    crop3d,
    paste_crop3d,
    force_nii_path,
    voxel_size_from_affine_mm_zyx,
)
from model.cfr import cfr_stage2_chi_denoise






def _maybe_load_default_cfg() -> Config:
    cfg = Config()
    validate_config(cfg)
    print("[INFO] config: using defaults from config.py")
    return cfg


def _weighted_data_fidelity(
    phi_pred: torch.Tensor,
    phi_gt: torch.Tensor,
    weight: Optional[torch.Tensor],
    mode: str = "l1",
    eps: float = 1e-6,
) -> torch.Tensor:
    r = phi_pred - phi_gt
    mode = str(mode).lower()
    if mode == "l1":
        err = r.abs()
    elif mode == "l2":
        err = r.pow(2)
    elif mode == "charbonnier":
        err = torch.sqrt(r.pow(2) + eps * eps)
    else:
        raise ValueError(f"Unsupported data mode: {mode}")
    if weight is None:
        return err.mean()
    w = weight.to(device=err.device, dtype=err.dtype).clamp_min(0.0)
    return (err * w).sum() / w.sum().clamp_min(1.0e-6)




def _cfr_feedback_weight(cfg: Config, it: int) -> float:
    """Piecewise weight for the staged CFR feedback loss."""
    if not bool(getattr(cfg.cfr, "feedback_enable", True)):
        return 0.0
    if int(it) < int(getattr(cfg.cfr, "from_iter", 50)):
        return 0.0
    if int(it) >= int(getattr(cfg.cfr, "feedback_late_iter", 250)):
        return float(getattr(cfg.cfr, "feedback_w_late", 0.12))
    if int(it) >= int(getattr(cfg.cfr, "feedback_mid_iter", 150)):
        return float(getattr(cfg.cfr, "feedback_w_mid", 0.08))
    return float(getattr(cfg.cfr, "feedback_w_early", 0.03))


def _weighted_chi_consistency(
    chi_current: torch.Tensor,
    chi_ref: torch.Tensor,
    weight: Optional[torch.Tensor],
    mask: torch.Tensor,
    mode: str = "charbonnier",
    eps: float = 1.0e-3,
    weight_floor: float = 0.0,
) -> torch.Tensor:
    """Reliability/mask-weighted chi-domain consistency to a detached CFR result."""
    m = mask.to(device=chi_current.device, dtype=chi_current.dtype).clamp(0.0, 1.0)
    if weight is None:
        w = m
    else:
        w = weight.to(device=chi_current.device, dtype=chi_current.dtype).clamp(0.0, 1.0) * m
        if float(weight_floor) > 0.0:
            w = torch.maximum(w, m * float(weight_floor))
    r = (chi_current - chi_ref.detach()) * m
    mode = str(mode).lower()
    if mode == "l1":
        err = r.abs()
    elif mode == "l2":
        err = 0.5 * r.pow(2)
    elif mode == "charbonnier":
        err = torch.sqrt(r.pow(2) + float(eps) * float(eps))
    else:
        raise ValueError("CFR feedback loss must be one of: l1, l2, charbonnier")
    return (err * w).sum() / w.sum().clamp_min(1.0e-6)


def _should_refresh_cfr_reference(cfg: Config, it: int) -> bool:
    return (
        bool(cfg.cfr.enable)
        and bool(getattr(cfg.cfr, "feedback_enable", True))
        and int(it) >= int(getattr(cfg.cfr, "from_iter", 50))
        and int(it) % int(getattr(cfg.cfr, "update_every", 50)) == 0
    )

def _blur3d(vol: torch.Tensor, k: int) -> torch.Tensor:
    if int(k) <= 1:
        return vol
    kk = int(k)
    if kk % 2 == 0:
        kk += 1
    return F.avg_pool3d(vol[None, None], kernel_size=kk, stride=1, padding=kk // 2)[0, 0]

def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    return (x * mask).sum() / mask.sum().clamp_min(1.0)


def _build_refiner_features(
    chi_base: torch.Tensor,
    mask: torch.Tensor,
    residual_field_n: Optional[torch.Tensor],
    cfg: Config,
) -> torch.Tensor:
    mode = str(getattr(cfg.unet, "input_mode", "residual")).lower()
    m = mask.detach().to(device=chi_base.device, dtype=chi_base.dtype).clamp(0.0, 1.0)
    if mode == "mask":
        aux = m
    elif mode == "residual":
        if residual_field_n is None:
            aux = m
        else:
            aux = residual_field_n
            if bool(getattr(cfg.unet, "detach_residual_input", True)):
                aux = aux.detach()
            clipv = float(getattr(cfg.unet, "residual_input_clip_abs", 3.0))
            if clipv > 0:
                aux = aux.clamp(-clipv, clipv) / max(clipv, 1.0e-6)
            aux = aux.to(device=chi_base.device, dtype=chi_base.dtype) * m
    else:
        raise ValueError("cfg.unet.input_mode must be 'residual' or 'mask'")
    return torch.stack([chi_base, aux], dim=0)[None]


def _build_refiner(cfg: Config, device: torch.device):
    if not bool(cfg.unet.enable):
        print("[INFO] Swin-UNet: disabled by cfg.unet.enable=False")
        return None, None

    from model.swin_unet3d import SwinUNet3DRefiner, SwinUNet3DRefinerConfig

    keys = (
        "in_channels", "base_channels", "depth", "norm", "act", "gn_groups", "dropout", "bias", "use_transpose",
        "refiner_version", "linear_embed_norm", "linear_embed_bias", "refine_mode", "fixed_ref_ratio",
        "residual_scale", "tanh_tau", "delta_clip_abs", "zero_init_last", "gate_bias_init", "only_update_in_mask",
        "gate_use_input_prior", "gate_prior_scale", "gate_smooth_kernel", "gate_cap", "gate_confidence_floor",
        "gate_confidence_channel", "swin_placement", "swin_each_layer", "swin_first_window_size", "swin_window_size",
        "swin_num_heads", "swin_first_blocks", "swin_first_decoder_blocks", "swin_blocks", "swin_mlp_ratio",
        "swin_drop", "swin_attn_drop",
    )
    allowed = set(SwinUNet3DRefinerConfig.__dataclass_fields__.keys())
    ucfg_dict = {k: getattr(cfg.unet, k) for k in keys if hasattr(cfg.unet, k) and k in allowed}
    unet = SwinUNet3DRefiner(SwinUNet3DRefinerConfig(**ucfg_dict)).to(device)
    optim = torch.optim.Adam(unet.parameters(), lr=float(cfg.unet.lr), weight_decay=float(cfg.unet.weight_decay))
    print(
        f"[INFO] Swin-UNet: enabled version={cfg.unet.refiner_version} in=2 input={cfg.unet.input_mode} "
        f"base={cfg.unet.base_channels} depth={cfg.unet.depth} placement={cfg.unet.swin_placement} "
        f"mode={cfg.unet.refine_mode} ratio={float(cfg.unet.fixed_ref_ratio):.2f} "
        f"dclip={float(cfg.unet.delta_clip_abs):.3f} from={int(cfg.unet.from_iter)}"
    )
    return unet, optim


def _forward_refiner(
    unet,
    chi_base: torch.Tensor,
    mask: torch.Tensor,
    cfg: Config,
    residual_field_n: Optional[torch.Tensor] = None,
    return_aux: bool = False,
):
    feat = _build_refiner_features(chi_base, mask, residual_field_n, cfg)
    out = unet(
        feat,
        base_chi=chi_base[None, None],
        mask=mask[None, None] if bool(cfg.unet.only_update_in_mask) else None,
        return_aux=return_aux,
    )
    if return_aux:
        chi_ref, aux = out
        chi_ref = chi_ref[0, 0]
        if cfg.unet.chi_clip is not None:
            chi_ref = torch.clamp(chi_ref, -float(cfg.unet.chi_clip), float(cfg.unet.chi_clip))
        return chi_ref * mask, aux
    chi_ref = out[0, 0]
    if cfg.unet.chi_clip is not None:
        chi_ref = torch.clamp(chi_ref, -float(cfg.unet.chi_clip), float(cfg.unet.chi_clip))
    return chi_ref * mask


def _candidate_magnitude_paths(phi_path: str, cfg: Config) -> list[str]:
    paths: list[str] = []
    if cfg.io.magnitude_path:
        paths.append(str(cfg.io.magnitude_path))
    try:
        d = os.path.dirname(os.path.abspath(str(phi_path)))
        for name in ("Magnitude.nii.gz", "Magnitude.nii", "magnitude.nii.gz", "magnitude.nii", "mag.nii.gz", "mag.nii", "Mag.nii.gz", "Mag.nii"):
            paths.append(os.path.join(d, name))
    except Exception:
        pass
    out: list[str] = []
    seen = set()
    for path in paths:
        if path and path not in seen:
            out.append(path)
            seen.add(path)
    return out


def _load_magnitude_base_weight_np(cfg: Config, phi_np: np.ndarray, mask_np: np.ndarray) -> Tuple[Optional[np.ndarray], str]:
    if not bool(cfg.cfr.enable):
        return None, "cfr-disabled"
    for path in _candidate_magnitude_paths(str(cfg.io.phi_path), cfg):
        if not os.path.exists(path):
            continue
        try:
            mag, _, _ = load_nifti(path)
            mag = np.asarray(mag, dtype=np.float32)
            mag = np.squeeze(mag)
            if mag.ndim == 4:
                mag = mag[..., 0]
            if mag.shape != phi_np.shape:
                print(f"[WARN] magnitude shape mismatch, skipped: {path} shape={mag.shape} expected={phi_np.shape}")
                continue
            valid = (mask_np > 0) & np.isfinite(mag)
            if int(valid.sum()) < 16:
                print(f"[WARN] magnitude has too few valid voxels in mask, skipped: {path}")
                continue
            mag = np.nan_to_num(np.abs(mag), nan=0.0, posinf=0.0, neginf=0.0)
            vmax = float(np.max(mag[valid]))
            if not np.isfinite(vmax) or vmax <= 1.0e-8:
                print(f"[WARN] magnitude max inside mask invalid, skipped: {path}")
                continue
            bw = np.clip(mag / vmax, 0.0, 1.0).astype(np.float32) * (mask_np > 0).astype(np.float32)
            bw_valid = bw[mask_np > 0]
            print(
                "[INFO] CFR base_weight stats before crop | "
                f"min={bw_valid.min():.4e} "
                f"p5={np.percentile(bw_valid, 5):.4e} "
                f"p25={np.percentile(bw_valid, 25):.4e} "
                f"p50={np.percentile(bw_valid, 50):.4e} "
                f"p75={np.percentile(bw_valid, 75):.4e} "
                f"p95={np.percentile(bw_valid, 95):.4e} "
                f"max={bw_valid.max():.4e} "
                f"mean={bw_valid.mean():.4e} "
                f"std={bw_valid.std():.4e}"
            )
            return bw, path
        except Exception as e:
            print(f"[WARN] failed to load magnitude candidate {path}: {e}")
    return None, "mask-fallback"



def _build_optimizer(model: GaussianChiModel, cfg: Config) -> torch.optim.Optimizer:
    groups = [
        {"name": "xyz", "params": [model.xyz], "lr": float(cfg.train.lr_xyz)},
        {"name": "sigma", "params": [model.log_sigma], "lr": float(cfg.train.lr_sigma)},
        {"name": "a", "params": [model.a], "lr": float(cfg.train.lr_a)},
    ]
    if bool(getattr(cfg.train, "rotation_enable", True)):
        groups.append({"name": "rot", "params": [model.rot_q], "lr": float(getattr(cfg.train, "lr_rot", 8.0e-4))})
    return torch.optim.Adam(groups, weight_decay=float(cfg.train.weight_decay))


def _lr_factor(kind: str, t: float, cfg: Config) -> float:
    t = float(max(0.0, min(1.0, t)))
    final_ratio = {
        "xyz": float(cfg.train.lr_xyz_final_ratio),
        "sigma": float(cfg.train.lr_sigma_final_ratio),
        "a": float(cfg.train.lr_a_final_ratio),
        "rot": float(getattr(cfg.train, "lr_rot_final_ratio", 0.35)),
    }[kind]
    final_ratio = max(final_ratio, 1.0e-4)
    mode = str(cfg.train.lr_schedule).lower()
    if mode == "cosine":
        return final_ratio + (1.0 - final_ratio) * 0.5 * (1.0 + float(np.cos(np.pi * t)))
    # Exponential decay is close to original 3DGS position schedule behavior.
    return float(np.exp(np.log(final_ratio) * t))


def _rotation_start_iter(cfg: Config) -> int:
    frac = float(getattr(cfg.train, "rotation_from_frac", 0.0))
    if frac > 0.0:
        return max(1, int(round(frac * max(1, int(cfg.train.iters)))))
    return int(getattr(cfg.train, "rotation_from_iter", 100))


def _update_learning_rates(optim: torch.optim.Optimizer, cfg: Config, it: int) -> None:
    if not bool(cfg.train.lr_schedule_enable):
        return
    total = max(1, int(cfg.train.iters))
    t = max(0, int(it) - 1) / max(1, total - 1)
    a_only_from_frac = float(getattr(cfg.train, "a_only_from_frac", 0.0))
    a_only = a_only_from_frac > 0 and float(it) >= a_only_from_frac * float(total)
    base = {
        "xyz": float(cfg.train.lr_xyz),
        "sigma": float(cfg.train.lr_sigma),
        "a": float(cfg.train.lr_a),
        "rot": float(getattr(cfg.train, "lr_rot", 8.0e-4)),
    }
    for group in optim.param_groups:
        name = str(group.get("name", ""))
        if name in base:
            if a_only and name in ("xyz", "sigma", "rot"):
                group["lr"] = 0.0
            elif name == "rot" and (not bool(getattr(cfg.train, "rotation_enable", True)) or int(it) < _rotation_start_iter(cfg)):
                group["lr"] = 0.0
            else:
                group["lr"] = base[name] * _lr_factor(name, t, cfg)


def _capture_optimizer_state(model: GaussianChiModel, optim: torch.optim.Optimizer) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, param in (("xyz", model.xyz), ("sigma", model.log_sigma), ("a", model.a), ("rot", model.rot_q)):
        st = optim.state.get(param, {})
        copied: Dict[str, torch.Tensor] = {}
        for k, v in st.items():
            if torch.is_tensor(v):
                copied[k] = v.detach().clone()
            else:
                copied[k] = torch.tensor(float(v), device=param.device)
        out[name] = copied
    return out


def _rebuild_optimizer_with_state(
    model: GaussianChiModel,
    cfg: Config,
    *,
    old_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
    source_index: Optional[torch.Tensor] = None,
    it: int = 1,
) -> torch.optim.Optimizer:
    optim = _build_optimizer(model, cfg)
    _update_learning_rates(optim, cfg, it)
    if old_state is None or source_index is None:
        return optim
    src = source_index.to(device=model.device, dtype=torch.long).view(-1)
    for name, param in (("xyz", model.xyz), ("sigma", model.log_sigma), ("a", model.a), ("rot", model.rot_q)):
        st_old = old_state.get(name, {})
        if not st_old:
            continue
        st_new = optim.state[param]
        for k, v in st_old.items():
            if torch.is_tensor(v) and v.ndim >= 1 and v.shape[0] >= int(src.max().item()) + 1:
                st_new[k] = v.to(device=param.device, dtype=param.dtype)[src].clone()
            elif torch.is_tensor(v):
                st_new[k] = v.to(device=param.device).clone()
            else:
                st_new[k] = v
    return optim


def _effective_rmax(model: GaussianChiModel, cfg: Config) -> int:
    base = int(cfg.render.rmax)
    if not bool(getattr(cfg.render, "auto_rmax", False)):
        return base
    with torch.no_grad():
        sig = model.sigma().detach().max(dim=1).values
        if sig.numel() == 0:
            return base
        q = min(max(float(cfg.render.auto_rmax_quantile), 0.0), 1.0)
        s = float(torch.quantile(sig, q).detach().cpu())
    auto = int(np.ceil(float(cfg.render.radius_factor) * max(s, 1.0e-6)))
    return int(max(int(cfg.render.min_rmax), min(int(cfg.render.max_rmax), max(base, auto))))


def _build_direction_map_from_chi0(chi0: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Low-cost local guide for split placement: normalized chi0 gradient direction.
    z = F.pad(chi0[None, None], (0, 0, 0, 0, 1, 1), mode="replicate")[0, 0]
    y = F.pad(chi0[None, None], (0, 0, 1, 1, 0, 0), mode="replicate")[0, 0]
    x = F.pad(chi0[None, None], (1, 1, 0, 0, 0, 0), mode="replicate")[0, 0]
    dz = 0.5 * (z[2:] - z[:-2])
    dy = 0.5 * (y[:, 2:] - y[:, :-2])
    dx = 0.5 * (x[:, :, 2:] - x[:, :, :-2])
    g = torch.stack([dz, dy, dx], dim=0) * mask[None]
    n = torch.sqrt((g * g).sum(dim=0, keepdim=True)).clamp_min(1.0e-6)
    return g / n


def _build_direction_guide_from_chi0(chi0: torch.Tensor, mask: torch.Tensor, cfg: Config) -> Optional[torch.Tensor]:
    if not bool(cfg.densify.directional_split):
        return None
    mode = str(getattr(cfg.densify, "directional_mode", "grad_normal")).lower()
    if mode in ("grad", "grad_normal", "normal"):
        return _build_direction_map_from_chi0(chi0, mask)
    if mode in ("hessian", "hessian_tangent", "tangent", "structure"):
        # Return scalar guide. model.py computes local Hessian directions only for selected
        # densify parents, so this is cheap and avoids a full-volume eigen-decomposition.
        return chi0.detach()
    raise ValueError(f"Unknown cfg.densify.directional_mode={mode}; valid: hessian_tangent, grad_normal")


def _density_stage_params(cfg: Config, it: int) -> Dict[str, object]:
    total = max(1, int(cfg.train.iters))
    if bool(cfg.densify.schedule_by_train_iters):
        start = max(1, int(round(float(cfg.densify.start_frac) * total)))
        split_until = max(start, int(round(float(cfg.densify.split_until_frac) * total)))
        clone_until = max(split_until, int(round(float(cfg.densify.clone_until_frac) * total)))
    else:
        start = int(cfg.densify.from_iter)
        split_until = int(cfg.densify.until_iter)
        clone_until = int(cfg.densify.until_iter)

    if it < start:
        return {"active": False, "stage": "off", "prune_only": False}

    if it <= split_until:
        return {
            "active": True,
            "stage": "split_clone",
            "prune_only": False,
            "clone_budget_frac": float(cfg.densify.stage1_clone_budget_frac),
            "clone_first": False,
            "allow_split": True,
            "allow_clone": True,
        }

    if it <= clone_until:
        return {
            "active": True,
            "stage": "clone_refine",
            "prune_only": False,
            "clone_budget_frac": float(cfg.densify.stage2_clone_budget_frac),
            "clone_first": True,
            "allow_split": True,
            "allow_clone": True,
        }

    if bool(getattr(cfg.densify, "prune_only_after_density", True)):
        return {"active": True, "stage": "prune_only", "prune_only": True}

    return {"active": False, "stage": "off", "prune_only": False}


def _build_chi0_from_cfg(*, phi_raw: torch.Tensor, mask: torch.Tensor, fwd: ForwardOp, cfg: Config) -> torch.Tensor:
    mode = str(cfg.init.chi0_mode).lower()
    if mode in ("zero", "none"):
        chi0 = torch.zeros_like(phi_raw)
    elif mode in ("mask", "const", "constant"):
        chi0 = torch.ones_like(phi_raw) * float(cfg.init.mask_init_value)
    elif mode in ("phi", "lfs", "field"):
        chi0 = phi_raw.clone()
        if bool(cfg.init.phi_init_normalize):
            vals = chi0[mask > 0]
            if vals.numel() > 16:
                p01 = torch.quantile(vals, 0.01)
                p99 = torch.quantile(vals, 0.99)
                chi0 = chi0 / ((p99 - p01).abs() / 4.0).clamp_min(1.0e-6)
    elif mode in ("tkd", "tkd_inverse"):
        chi0 = fwd.tkd(phi_raw, padded=True, thresh=float(cfg.phys.tkd_thresh))
    else:
        raise ValueError(f"Unknown cfg.init.chi0_mode={mode}; valid: zero, mask, phi, tkd")

    chi0 = chi0 * mask
    if int(cfg.init.chi0_blur_kernel) > 1:
        chi0 = _blur3d(chi0, int(cfg.init.chi0_blur_kernel)) * mask
    if cfg.init.chi0_clip is not None and float(cfg.init.chi0_clip) > 0:
        chi0 = chi0.clamp(-float(cfg.init.chi0_clip), float(cfg.init.chi0_clip))
    return chi0


class VolumeSaver:
    def __init__(self, run_dir: str, affine: np.ndarray, full_shape: Tuple[int, int, int], bbox: Optional[Tuple[int, int, int, int, int, int]]):
        self.run_dir = run_dir
        self.affine = affine
        self.full_shape = tuple(int(v) for v in full_shape)
        self.bbox = bbox

    def to_full_np(self, vol) -> np.ndarray:
        if isinstance(vol, torch.Tensor):
            arr = vol.detach().cpu().numpy().astype(np.float32)
        else:
            arr = np.asarray(vol, dtype=np.float32)
        if self.bbox is None:
            return arr.astype(np.float32, copy=False)
        if tuple(arr.shape) == self.full_shape:
            return arr.astype(np.float32, copy=False)
        full = np.zeros(self.full_shape, dtype=np.float32)
        paste_crop3d(full, arr.astype(np.float32, copy=False), self.bbox)
        return full

    def save(self, rel_path: str, vol) -> None:
        path = rel_path if os.path.isabs(rel_path) else os.path.join(self.run_dir, rel_path)
        save_nifti(self.to_full_np(vol), self.affine, force_nii_path(path))


def _maybe_apply_bbox_crop(
    *,
    phi_np: np.ndarray,
    mask_np: np.ndarray,
    mag_base_weight_np: Optional[np.ndarray],
    cfg: Config,
    run_dir: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[Tuple[int, int, int, int, int, int]], Tuple[int, int, int]]:
    full_shape = tuple(int(v) for v in phi_np.shape)
    nvox = int(np.prod(full_shape))
    use_crop = bool(cfg.bbox.enable) and (bool(cfg.bbox.force) or nvox >= int(cfg.bbox.voxel_threshold))
    if not use_crop:
        print(f"[INFO] bbox crop: disabled/not needed shape={full_shape} voxels={nvox} threshold={int(cfg.bbox.voxel_threshold)}")
        return phi_np, mask_np, mag_base_weight_np, None, full_shape

    bbox = bbox_from_mask(mask_np, margin=int(cfg.bbox.margin))
    z0, z1, y0, y1, x0, x1 = bbox
    crop_shape = (z1 - z0, y1 - y0, x1 - x0)
    if crop_shape == full_shape:
        print(f"[INFO] bbox crop: skipped because crop equals full volume shape={full_shape}")
        return phi_np, mask_np, mag_base_weight_np, None, full_shape

    if bool(cfg.bbox.save_debug):
        with open(os.path.join(run_dir, "bbox_info.json"), "w", encoding="utf-8") as f:
            json.dump({
                "enabled": True,
                "full_shape": list(full_shape),
                "crop_shape": list(crop_shape),
                "bbox_zyx": [z0, z1, y0, y1, x0, x1],
                "margin": int(cfg.bbox.margin),
                "voxels_full": nvox,
                "voxels_crop": int(np.prod(crop_shape)),
            }, f, indent=2)

    print(f"[INFO] bbox crop: full={full_shape} -> crop={crop_shape} bbox={bbox}")
    phi_c = crop3d(phi_np, bbox).astype(np.float32, copy=False)
    mask_c = crop3d(mask_np, bbox).astype(np.float32, copy=False)
    mag_c = crop3d(mag_base_weight_np, bbox).astype(np.float32, copy=False) if mag_base_weight_np is not None else None
    return phi_c, mask_c, mag_c, bbox, full_shape

def run(cfg: Config) -> None:
    validate_config(cfg)
    set_seed(int(cfg.train.seed))

    run_dir = os.path.join(cfg.io.out_dir, cfg.io.run_name)
    ensure_dir(run_dir)

    device = torch.device("cpu") if str(cfg.io.device).lower() == "cuda" and not torch.cuda.is_available() else torch.device(str(cfg.io.device))
    print("[INFO] GSQSM v5 staged-CFR pipeline: rotation-covariance GS + Swin-UNet + CFR feedback; final-mild-denoise removed")
    print(
        f"[INFO] switches: GS=on Swin-UNet={bool(cfg.unet.enable)} "
        f"CFR={bool(cfg.cfr.enable)} CFR_feedback={bool(getattr(cfg.cfr, 'feedback_enable', True))} "
        f"bbox={bool(cfg.bbox.enable)}"
    )
    print(f"[INFO] device={device} dtype={cfg.io.dtype}")

    phi_full_np, mask_full_np, affine, meta = load_data(str(cfg.io.phi_path), str(cfg.io.mask_path or ""))
    mask_full_np, mask_info = repair_mask_np(phi_full_np, mask_full_np, cfg.mask, meta)
    full_shape0 = tuple(int(v) for v in phi_full_np.shape)

    mag_base_weight_full_np, mag_source = _load_magnitude_base_weight_np(cfg, phi_full_np, mask_full_np)
    if bool(cfg.cfr.enable):
        if mag_base_weight_full_np is not None:
            print(f"[INFO] CFR base_weight: magnitude loaded from {mag_source}")
        else:
            print("[INFO] CFR base_weight: magnitude not found/invalid; fallback to no-mag mag^2=0.1")

    noise_info = estimate_lfs_noise_score_np(phi_full_np, mask_full_np, kernel=int(cfg.loss.noise_kernel))
    print(
        "[INFO] data: "
        f"phi={full_shape0} mask_ratio={mask_info.get('output_ratio', -1):.4f} source={meta.get('source','')} "
        f"mask_fixed={mask_info.get('fixed', False)} reason={mask_info.get('reason', '')}"
    )
    print(
        "[INFO] lfs_noise_score: "
        f"score={noise_info['score']:.5f} hp_mad={noise_info['hp_mad']:.4e} "
        f"flat_hp_mad={noise_info['flat_hp_mad']:.4e} phi_std={noise_info['phi_std']:.4e}"
    )

    if float(noise_info["score"]) >= 0.045:
        cfg.init.n_init = min(int(cfg.init.n_init), 80_000)
        cfg.densify.from_iter = max(int(cfg.densify.from_iter), 80)
        cfg.densify.max_points = min(int(cfg.densify.max_points), 115_000)
        # Keep v5 structure-aware initialization; do not revert to mostly uniform.
        cfg.init.edge_uniform_frac = min(float(cfg.init.edge_uniform_frac), 0.70)
        cfg.init.edge_grad_frac = max(float(cfg.init.edge_grad_frac), 0.20)
        cfg.init.edge_log_frac = max(float(cfg.init.edge_log_frac), 0.10)
        cfg.init.chi0_clip = 0.025 if cfg.init.chi0_clip is None else min(float(cfg.init.chi0_clip), 0.025)
        cfg.init.chi0_blur_kernel = max(int(cfg.init.chi0_blur_kernel), 5)
        print("[INFO] strong-noise safe init/grow applied")

    phi_np, mask_np, mag_base_weight_np, bbox, full_shape = _maybe_apply_bbox_crop(
        phi_np=phi_full_np,
        mask_np=mask_full_np,
        mag_base_weight_np=mag_base_weight_full_np,
        cfg=cfg,
        run_dir=run_dir,
    )
    saver = VolumeSaver(run_dir, affine, full_shape, bbox)

    vz_aff = voxel_size_from_affine_mm_zyx(affine)
    voxel_cfg = tuple(float(v) for v in cfg.phys.voxel_size_mm)
    if str(meta.get("source", "")).lower() == ".mat" or (np.allclose(vz_aff, (1.0, 1.0, 1.0)) and voxel_cfg != (1.0, 1.0, 1.0)):
        voxel_size_zyx = voxel_cfg
    else:
        voxel_size_zyx = vz_aff
    print(f"[INFO] voxel_size_mm_zyx={voxel_size_zyx} (affine={vz_aff}, cfg={voxel_cfg})")

    fwd: ForwardOp = build_forward_from_cfg(cfg.phys, device=str(device), dtype_str=str(cfg.io.dtype))
    fwd.voxel_size_mm_zyx = voxel_size_zyx

    Z, Y, X = map(int, phi_np.shape)
    phi_raw = torch.tensor(phi_np, device=device, dtype=torch.float32)
    mask = torch.tensor(mask_np, device=device, dtype=torch.float32).clamp(0.0, 1.0)
    cfr_base_weight = torch.tensor(mag_base_weight_np, device=device, dtype=torch.float32).clamp(0.0, 1.0) if mag_base_weight_np is not None else None

    with torch.no_grad():
        phi_std = phi_raw[mask > 0].std().clamp_min(1.0e-6)
        phi_raw_n = phi_raw / phi_std
        chi0 = _build_chi0_from_cfg(phi_raw=phi_raw, mask=mask, fwd=fwd, cfg=cfg)
        direction_map = _build_direction_guide_from_chi0(chi0, mask, cfg)

    print(
        f"[INFO] render formula: normalized={bool(cfg.render.normalize_kernel)} "
        f"mask_aware={bool(getattr(cfg.render, 'mask_aware_normalization', True))} "
        f"voxel_size_aware={bool(getattr(cfg.render, 'voxel_size_aware', True))} "
        f"voxel_size_zyx={tuple(float(v) for v in voxel_size_zyx)}"
    )
    print(
        f"[INFO] chi0 init: mode={cfg.init.chi0_mode} sample_mode={cfg.init.sample_mode} "
        f"n_init={int(cfg.init.n_init)} edge_split=({float(cfg.init.edge_uniform_frac):.2f},"
        f"{float(cfg.init.edge_grad_frac):.2f},{float(cfg.init.edge_log_frac):.2f}) chi0_clip={cfg.init.chi0_clip}"
    )
    print(
        "[INFO] density control: gradient + residual threshold; "
        f"sigma_mode={cfg.densify.sigma_threshold_mode} clone_q={float(cfg.densify.clone_sigma_quantile):.2f} "
        f"split_q={float(cfg.densify.split_sigma_quantile):.2f} add_max={int(cfg.densify.add_max)} "
        f"max_points={int(cfg.densify.max_points)} schedule_frac=({float(cfg.densify.start_frac):.2f},"
        f"{float(cfg.densify.split_until_frac):.2f},{float(cfg.densify.clone_until_frac):.2f}) "
        f"direction={bool(cfg.densify.directional_split)}:{getattr(cfg.densify, 'directional_mode', 'none')} "
        f"rotation={bool(getattr(cfg.train, 'rotation_enable', True))}@{_rotation_start_iter(cfg)}"
    )
    print(
        f"[INFO] CFR: enable={bool(cfg.cfr.enable)} iters={int(cfg.cfr.iters)} loss={cfg.cfr.loss} "
        f"data_w={float(cfg.cfr.data_w):.2e} tv={float(cfg.cfr.tv_w):.2e} fd={float(cfg.cfr.fd_w):.2e} "
        f"weight_mode={cfg.cfr.weight_mode} feedback={bool(getattr(cfg.cfr, 'feedback_enable', True))} "
        f"from={int(getattr(cfg.cfr, 'from_iter', 50))} every={int(getattr(cfg.cfr, 'update_every', 50))} "
        f"w=({float(getattr(cfg.cfr, 'feedback_w_early', 0.03)):.2e},"
        f"{float(getattr(cfg.cfr, 'feedback_w_mid', 0.08)):.2e},"
        f"{float(getattr(cfg.cfr, 'feedback_w_late', 0.12)):.2e})"
    )

    initp = build_init_params_from_cfg(
        chi0=chi0.detach().cpu().numpy().astype(np.float32),
        mask=mask_np,
        cfg=cfg,
        voxel_size_mm_zyx=voxel_size_zyx,
        device=str(device),
        dtype=torch.float32,
        score_volume=phi_np,
    )
    model = GaussianChiModel(
        initp.xyz,
        initp.sigma,
        initp.a,
        sigma_min=float(cfg.densify.prune_sigma_min),
        sigma_max=float(cfg.densify.prune_sigma_max),
        use_rotation=bool(getattr(cfg.train, "rotation_enable", True)),
    ).to(device)
    optim = _build_optimizer(model, cfg)
    _update_learning_rates(optim, cfg, 1)
    stats = GSStats.init(model.num_points(), device=device, dtype=torch.float32, decay=float(cfg.densify.ema_decay))
    unet, unet_optim = _build_refiner(cfg, device)

    wall_t0 = time.time()
    last_density_info = {"added": 0, "clone_parents": 0, "split_parents": 0, "grad_thr": 0.0, "res_thr": 0.0, "stage": "off", "clone_sigma_thr": 0.0, "split_sigma_thr": 0.0}

    # Periodically refreshed CFR result used by the staged feedback loss.
    # It is detached from the graph; gradients flow through the current
    # chi_ref/chi_gs branch only, not through the internal CFR optimizer.
    cfr_ref_chi: Optional[torch.Tensor] = None
    cfr_ref_weight: Optional[torch.Tensor] = None
    cfr_ref_iter: int = 0
    cfr_ref_hist: Dict[str, float] = {"loss": 0.0, "data": 0.0, "tv": 0.0, "fd": 0.0}

    for it in range(1, int(cfg.train.iters) + 1):
        iter_t0 = time.perf_counter()
        pruned = 0
        model.train()
        if unet is not None:
            unet.train()
        _update_learning_rates(optim, cfg, it)
        rmax_cur = _effective_rmax(model, cfg)
        optim.zero_grad(set_to_none=True)
        if unet_optim is not None:
            unet_optim.zero_grad(set_to_none=True)

        chi_gs, phi_gs = model.forward_qsm(
            (Z, Y, X),
            fwd,
            mask=mask,
            padded=True,
            render="gaussian_local",
            rmax=int(rmax_cur),
            radius_factor=float(cfg.render.radius_factor),
            normalize_kernel=bool(cfg.render.normalize_kernel),
            mask_aware_normalization=bool(getattr(cfg.render, "mask_aware_normalization", True)),
            voxel_size_mm_zyx=voxel_size_zyx,
            voxel_size_aware=bool(getattr(cfg.render, "voxel_size_aware", True)),
            chunk=int(cfg.render.chunk),
        )
        phi_gs_n = phi_gs / phi_std
        weight = mask if bool(cfg.loss.use_mask) else None
        l_gs_data = _weighted_data_fidelity(phi_gs_n, phi_raw_n.detach(), weight, str(cfg.loss.data))
        l_tv = tv_loss_3d(chi_gs[None, None], mask=mask[None, None] if bool(cfg.loss.use_mask) else None, mode="anisotropic") if float(cfg.loss.tv_w) > 0 else chi_gs.new_zeros(())
        l_reg = param_regularization(model, sigma_w=float(cfg.loss.sigma_w), a_l2_w=float(cfg.loss.a_l2_w))
        loss_total = float(cfg.loss.data_w) * l_gs_data + float(cfg.loss.tv_w) * l_tv + l_reg

        residual_field_n = (phi_raw_n.detach() - phi_gs_n.detach()) * mask
        residual_abs_n = residual_field_n.abs()
        if int(cfg.densify.residual_smooth_kernel) > 1:
            residual_abs_n_for_density = _blur3d(residual_abs_n, int(cfg.densify.residual_smooth_kernel)) * mask
        else:
            residual_abs_n_for_density = residual_abs_n

        l_ref_raw = chi_gs.new_zeros(())
        l_cfr_feedback = chi_gs.new_zeros(())
        cfr_feedback_w_cur = 0.0
        cfr_ref_age = -1
        gate_mean = chi_gs.new_zeros(())
        gate_max = chi_gs.new_zeros(())
        chi_ref_joint = None
        do_refiner = unet is not None and unet_optim is not None and it >= int(cfg.unet.from_iter) and (it % int(cfg.unet.every) == 0)
        if do_refiner:
            chi_ref_joint, aux = _forward_refiner(
                unet,
                chi_gs,
                mask,
                cfg,
                residual_field_n=residual_field_n,
                return_aux=True,
            )
            phi_ref_joint_n = fwd.apply(chi_ref_joint, padded=True) / phi_std
            l_ref_raw = _weighted_data_fidelity(phi_ref_joint_n, phi_raw_n.detach(), weight, str(cfg.unet.data))
            loss_total = loss_total + float(cfg.unet.ref_raw_w) * l_ref_raw
            if isinstance(aux, dict) and aux.get("gate", None) is not None:
                gate = aux["gate"]
                gate_mean = _masked_mean(gate[0, 0], mask)
                gate_max = gate.max()

        if bool(cfg.cfr.enable) and bool(getattr(cfg.cfr, "feedback_enable", True)) and cfr_ref_chi is not None:
            target_mode = str(getattr(cfg.cfr, "feedback_target", "ref")).lower()
            if target_mode == "ref" and chi_ref_joint is not None:
                cfr_current_chi = chi_ref_joint
            else:
                cfr_current_chi = chi_gs
            cfr_feedback_w_cur = _cfr_feedback_weight(cfg, it)
            if cfr_feedback_w_cur > 0.0:
                l_cfr_feedback = _weighted_chi_consistency(
                    cfr_current_chi,
                    cfr_ref_chi,
                    cfr_ref_weight,
                    mask,
                    mode=str(getattr(cfg.cfr, "feedback_loss", "charbonnier")),
                    eps=float(getattr(cfg.cfr, "feedback_eps", 1.0e-3)),
                    weight_floor=float(getattr(cfg.cfr, "feedback_weight_floor", 0.0)),
                )
                loss_total = loss_total + cfr_feedback_w_cur * l_cfr_feedback
                cfr_ref_age = int(it) - int(cfr_ref_iter)

        loss_total.backward()
        if cfg.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        if unet is not None and cfg.unet.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(unet.parameters(), float(cfg.unet.grad_clip))
        model.update_stats_from_grads(stats)
        optim.step()
        if unet_optim is not None and do_refiner:
            unet_optim.step()

        density_changed = False
        source_index_for_state = None
        old_optim_state = None
        last_density_info = {"added": 0, "clone_parents": 0, "split_parents": 0, "grad_thr": 0.0, "res_thr": 0.0, "stage": "off", "clone_sigma_thr": 0.0, "split_sigma_thr": 0.0}
        stage_info = _density_stage_params(cfg, it)
        if bool(cfg.densify.enable) and bool(stage_info.get("active", False)):
            prune_only_stage = bool(stage_info.get("prune_only", False))
            if (not prune_only_stage) and it % int(cfg.densify.every) == 0 and model.num_points() < int(cfg.densify.max_points):
                old_optim_state = _capture_optimizer_state(model, optim)
                old_stats = stats
                last_density_info = model.densify_clone_split(
                    stats,
                    residual_abs_map=residual_abs_n_for_density,
                    mask=mask,
                    metric=str(cfg.densify.metric),
                    grad_threshold_mode=str(cfg.densify.grad_threshold_mode),
                    grad_abs_min=float(cfg.densify.grad_abs_min),
                    grad_mad_k=float(cfg.densify.grad_mad_k),
                    grad_percentile=float(cfg.densify.grad_percentile),
                    residual_sample_mode=str(cfg.densify.residual_sample_mode),
                    residual_rmax=int(rmax_cur),
                    residual_radius_factor=float(cfg.render.radius_factor),
                    voxel_size_mm_zyx=voxel_size_zyx,
                    voxel_size_aware=bool(getattr(cfg.render, "voxel_size_aware", True)),
                    residual_threshold_mode=str(cfg.densify.residual_threshold_mode),
                    residual_abs_min=float(cfg.densify.residual_abs_min),
                    residual_mad_k=float(cfg.densify.residual_mad_k),
                    residual_percentile=float(cfg.densify.residual_percentile),
                    sigma_threshold_mode=str(cfg.densify.sigma_threshold_mode),
                    clone_sigma_max=float(cfg.densify.clone_sigma_max),
                    split_sigma_min=float(cfg.densify.split_sigma_min),
                    clone_sigma_quantile=float(cfg.densify.clone_sigma_quantile),
                    split_sigma_quantile=float(cfg.densify.split_sigma_quantile),
                    clone_k=int(cfg.densify.clone_k),
                    split_k=int(cfg.densify.split_k),
                    add_max=int(cfg.densify.add_max),
                    max_points=int(cfg.densify.max_points),
                    clone_budget_frac=float(stage_info.get("clone_budget_frac", cfg.densify.stage1_clone_budget_frac)),
                    clone_first=bool(stage_info.get("clone_first", False)),
                    allow_split=bool(stage_info.get("allow_split", True)),
                    allow_clone=bool(stage_info.get("allow_clone", True)),
                    clone_jitter_scale=float(cfg.densify.clone_jitter_scale),
                    split_jitter_scale=float(cfg.densify.split_jitter_scale),
                    split_sigma_scale_children=float(cfg.densify.split_sigma_scale_children),
                    direction_map=direction_map,
                    directional_split=bool(cfg.densify.directional_split),
                    directional_mode=str(getattr(cfg.densify, "directional_mode", "grad_normal")),
                    directional_clone=bool(getattr(cfg.densify, "directional_clone", False)),
                    directional_strength=float(cfg.densify.directional_strength),
                    directional_clone_strength=float(getattr(cfg.densify, "directional_clone_strength", cfg.densify.directional_strength)),
                    directional_random_scale=float(cfg.densify.directional_random_scale),
                )
                last_density_info["stage"] = str(stage_info.get("stage", "active"))
                source_index_for_state = last_density_info.get("source_index", None)
                density_changed = int(last_density_info.get("added", 0)) != 0
                if density_changed and isinstance(source_index_for_state, torch.Tensor):
                    stats = old_stats.remap_from_old(source_index_for_state)

            if prune_only_stage:
                last_density_info["stage"] = "prune_only"

            if it >= int(cfg.densify.prune_warmup) and it % int(cfg.densify.prune_every) == 0:
                if old_optim_state is None:
                    old_optim_state = _capture_optimizer_state(model, optim)
                    old_stats = stats
                    source_index_for_state = torch.arange(model.num_points(), device=device, dtype=torch.long)
                prune_info = model.prune(
                    mask=mask if bool(cfg.densify.prune_outside_mask) else None,
                    a_min=float(cfg.densify.prune_a_min),
                    contribution_min=float(cfg.densify.prune_contribution_min),
                    sigma_min=float(cfg.densify.prune_sigma_min),
                    sigma_max=float(cfg.densify.prune_sigma_max),
                    keep_at_least=int(cfg.densify.keep_at_least),
                    normalize_kernel=bool(cfg.render.normalize_kernel),
                    voxel_size_mm_zyx=voxel_size_zyx,
                )
                pruned = int(prune_info.get("pruned", 0))
                prune_src = prune_info.get("source_index", None)
                if pruned > 0 and isinstance(prune_src, torch.Tensor):
                    if isinstance(source_index_for_state, torch.Tensor):
                        source_index_for_state = source_index_for_state[prune_src]
                    else:
                        source_index_for_state = prune_src
                    stats = old_stats.remap_from_old(source_index_for_state)
                density_changed = density_changed or pruned > 0

            if density_changed:
                optim = _rebuild_optimizer_with_state(
                    model,
                    cfg,
                    old_state=old_optim_state,
                    source_index=source_index_for_state if isinstance(source_index_for_state, torch.Tensor) else None,
                    it=it,
                )

        if _should_refresh_cfr_reference(cfg, it):
            model.eval()
            if unet is not None:
                unet.eval()
            with torch.no_grad():
                chi_gs_for_cfr = model.splat_gaussian_local(
                    (Z, Y, X),
                    mask=mask,
                    rmax=int(_effective_rmax(model, cfg)),
                    radius_factor=float(cfg.render.radius_factor),
                    normalize_kernel=bool(cfg.render.normalize_kernel),
                    mask_aware_normalization=bool(getattr(cfg.render, "mask_aware_normalization", True)),
                    voxel_size_mm_zyx=voxel_size_zyx,
                    voxel_size_aware=bool(getattr(cfg.render, "voxel_size_aware", True)),
                    chunk=int(cfg.render.chunk),
                ).detach()
                cfr_source_chi = chi_gs_for_cfr
                if unet is not None and it >= int(cfg.unet.from_iter):
                    phi_gs_for_cfr_n = fwd.apply(chi_gs_for_cfr, padded=True) / phi_std
                    residual_for_cfr_n = (phi_raw_n.detach() - phi_gs_for_cfr_n.detach()) * mask
                    chi_ref_for_cfr = _forward_refiner(
                        unet,
                        chi_gs_for_cfr,
                        mask,
                        cfg,
                        residual_field_n=residual_for_cfr_n,
                        return_aux=False,
                    ).detach()
                    if str(getattr(cfg.cfr, "feedback_target", "ref")).lower() == "ref":
                        cfr_source_chi = chi_ref_for_cfr

            cfr_ref_chi, cfr_ref_weight, _, _, cfr_ref_hist = cfr_stage2_chi_denoise(
                chi_init=cfr_source_chi.detach(),
                phi_raw_n=phi_raw_n,
                fwd=fwd,
                phi_scale=phi_std,
                mask=mask,
                base_weight=cfr_base_weight,
                cfg=cfg,
                verbose=bool(it == int(cfg.train.iters)),
            )
            cfr_ref_chi = cfr_ref_chi.detach()
            cfr_ref_weight = cfr_ref_weight.detach() if cfr_ref_weight is not None else None
            cfr_ref_iter = int(it)
            print(
                f"[CFR-REF] it={it:06d} refreshed staged CFR result | "
                f"loss={cfr_ref_hist['loss']:.6e} data={cfr_ref_hist['data']:.6e} "
                f"tv={cfr_ref_hist['tv']:.6e} fd={cfr_ref_hist.get('fd', 0.0):.6e}"
            )
            model.train()
            if unet is not None:
                unet.train()

        if it % int(cfg.train.log_every) == 0 or it == 1:
            with torch.no_grad():
                wall_dt = time.time() - wall_t0
                wall_t0 = time.time()
                iter_dt = time.perf_counter() - iter_t0

                ref_val = float(l_ref_raw.detach()) if unet is not None else 0.0
                cfr_val = float(l_cfr_feedback.detach()) if bool(cfg.cfr.enable) else 0.0
                stage = str(last_density_info.get("stage", "off"))
                added = int(last_density_info.get("added", 0))
                clone_p = int(last_density_info.get("clone_parents", 0))
                split_p = int(last_density_info.get("split_parents", 0))

            adc_msg = ""
            if stage != "off" or added != 0 or pruned != 0:
                adc_msg = f" adc={stage} +{added}/-{pruned} cp/sp={clone_p}/{split_p}"

            print(
                f"[{it:06d}/{int(cfg.train.iters)}] "
                f"loss={float(loss_total.detach()):.3e} "
                f"gs={float(l_gs_data.detach()):.3e} "
                f"ref={ref_val:.3e} "
                f"cfr={cfr_val:.3e} "
                f"N={model.num_points()}"
                f"{adc_msg} "
                f"iter={iter_dt:.3f}s "
                f"wall={wall_dt:.1f}s"
            )

        if it == int(cfg.train.iters):
            model.eval()
            if unet is not None:
                unet.eval()

            with torch.no_grad():
                chi_gs_save = model.splat_gaussian_local(
                    (Z, Y, X),
                    mask=mask,
                    rmax=int(_effective_rmax(model, cfg)),
                    radius_factor=float(cfg.render.radius_factor),
                    normalize_kernel=bool(cfg.render.normalize_kernel),
                    mask_aware_normalization=bool(getattr(cfg.render, "mask_aware_normalization", True)),
                    voxel_size_mm_zyx=voxel_size_zyx,
                    voxel_size_aware=bool(getattr(cfg.render, "voxel_size_aware", True)),
                    chunk=int(cfg.render.chunk),
                ).detach()

                chi_final = chi_gs_save
                if unet is not None:
                    phi_gs_save_n = fwd.apply(chi_gs_save, padded=True) / phi_std
                    residual_save_n = (phi_raw_n.detach() - phi_gs_save_n.detach()) * mask
                    chi_final = _forward_refiner(
                        unet,
                        chi_gs_save,
                        mask,
                        cfg,
                        residual_field_n=residual_save_n,
                        return_aux=False,
                    ).detach()

            if bool(cfg.cfr.enable):
                if cfr_ref_chi is not None and int(cfr_ref_iter) == int(it):
                    chi_final = cfr_ref_chi.detach()
                    print(f"[SAVE] final reused staged CFR result at it={it:06d}")
                else:
                    chi_final, _, _, _, _ = cfr_stage2_chi_denoise(
                        chi_init=chi_final.detach(),
                        phi_raw_n=phi_raw_n,
                        fwd=fwd,
                        phi_scale=phi_std,
                        mask=mask,
                        base_weight=cfr_base_weight,
                        cfg=cfg,
                        verbose=True,
                    )
                    chi_final = chi_final.detach()

            saver.save("gsqsm.nii", chi_final)
            print(f"[SAVE] final result saved: {os.path.join(run_dir, 'gsqsm.nii')}")

    print("Done:", run_dir)
    return run_dir


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GSQSM training/reconstruction.")
    p.add_argument("--config", default=None, help="Path to config json/yaml.")
    return p.parse_args()


def _load_cfg(config_path: Optional[str]) -> Config:
    if config_path:
        cfg = load_config(config_path)
        validate_config(cfg)
        print(f"[INFO] config: loaded {config_path}")
        return cfg
    return _maybe_load_default_cfg()


if __name__ == "__main__":
    args = _parse_args()
    run(_load_cfg(args.config))

