# initialize.py
# GSQSM - initialization + sanity test driven by config
#
# Pipeline:
#   load lfs/phi -> TKD chi0 -> init gaussian params -> splat back to chi_voxel -> save .nii only
#
# Usage:
#   python -m model.initialize --config ./config.json
# or:
#   python -m model.initialize --config ./config.yaml

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

try:
    from scipy import ndimage as ndi
except Exception:  # scipy is expected in this project, but keep a safe fallback
    ndi = None

from config import Config, load_config, validate_config
from utils.tool import load_data, save_nifti, ensure_dir, force_nii_path, voxel_size_from_affine_mm_zyx, _as_float_np
from model.forward import build_forward_from_cfg, ForwardOp



@dataclass
class InitParams:
    """
    Voxel coordinate convention:
      xyz: continuous voxel coords (z,y,x)
      sigma: in voxel units (z,y,x)
      a: chi amplitude
    """
    xyz: torch.Tensor     # (N,3)
    sigma: torch.Tensor   # (N,3)
    a: torch.Tensor       # (N,1)


def _sample_points_from_mask(mask: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask empty.")
    if n >= len(coords):
        idx = rng.integers(0, len(coords), size=n)
        return coords[idx].astype(np.int64)
    idx = rng.choice(len(coords), size=n, replace=False)
    return coords[idx].astype(np.int64)


def _sample_points_topk_abs_chi0(mask: np.ndarray, chi0: np.ndarray, n: int, topk_ratio: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask empty.")
    vals = np.abs(chi0[coords[:, 0], coords[:, 1], coords[:, 2]])
    k = max(1, int(len(vals) * float(topk_ratio)))
    top_idx = np.argpartition(vals, -k)[-k:]
    top_coords = coords[top_idx]
    if n >= len(top_coords):
        idx = rng.integers(0, len(top_coords), size=n)
        return top_coords[idx].astype(np.int64)
    idx = rng.choice(len(top_coords), size=n, replace=False)
    return top_coords[idx].astype(np.int64)


def _robust_normalize_inside_mask(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Robustly scale a 3D score source to roughly [-1, 1] inside mask."""
    x = _as_float_np(x)
    m = mask > 0
    vals = x[m]
    vals = vals[np.isfinite(vals)]
    if vals.size < 32:
        return x
    p01, p99 = np.percentile(vals, [1.0, 99.0])
    center = np.median(vals)
    scale = max(float((p99 - p01) / 4.0), 1e-6)
    y = (x - float(center)) / scale
    return np.clip(y, -5.0, 5.0).astype(np.float32)


def _sample_points_topk_score(mask: np.ndarray, score: np.ndarray, n: int, topk_ratio: float, seed: int) -> np.ndarray:
    """Sample points from a top-score candidate pool inside mask."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.int64)
    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask empty.")
    s = _as_float_np(score)[coords[:, 0], coords[:, 1], coords[:, 2]]
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    if float(np.max(s)) <= 0:
        return _sample_points_from_mask(mask, n, seed)
    k = max(1, int(len(s) * float(topk_ratio)))
    k = min(k, len(s))
    top_idx = np.argpartition(s, -k)[-k:]
    top_coords = coords[top_idx]
    if n >= len(top_coords):
        idx = rng.integers(0, len(top_coords), size=n)
    else:
        idx = rng.choice(len(top_coords), size=n, replace=False)
    return top_coords[idx].astype(np.int64)


def _edge_and_log_scores(score_volume: np.ndarray, mask: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build lesion/detail allocation scores from phi/LFS-like volume.

    grad_score: high at lesion boundaries and strong field transitions.
    log_score : high at blob/ridge-like local changes, useful for lesion interiors.

    These scores are used only for Gaussian center allocation, not as direct chi.
    """
    src = _robust_normalize_inside_mask(score_volume, mask)
    src = src * (mask > 0).astype(np.float32)

    if ndi is not None:
        gs = float(getattr(cfg.init, "edge_score_smooth_sigma", 1.0))
        ls = float(getattr(cfg.init, "log_score_smooth_sigma", 1.0))
        src_g = ndi.gaussian_filter(src, sigma=max(0.0, gs)) if gs > 0 else src
        src_l = ndi.gaussian_filter(src, sigma=max(0.0, ls)) if ls > 0 else src
    else:
        src_g = src_l = src

    dz, dy, dx = np.gradient(src_g.astype(np.float32))
    grad = np.sqrt(dz * dz + dy * dy + dx * dx).astype(np.float32)

    if ndi is not None:
        # Negative LoG highlights bright blobs; abs handles sign variations.
        log_resp = np.abs(ndi.gaussian_laplace(src_l.astype(np.float32), sigma=max(0.5, float(getattr(cfg.init, "log_score_smooth_sigma", 1.0))))).astype(np.float32)
    else:
        lap = (
            -6.0 * src_l
            + np.roll(src_l, 1, axis=0) + np.roll(src_l, -1, axis=0)
            + np.roll(src_l, 1, axis=1) + np.roll(src_l, -1, axis=1)
            + np.roll(src_l, 1, axis=2) + np.roll(src_l, -1, axis=2)
        )
        log_resp = np.abs(lap).astype(np.float32)

    m = mask > 0
    grad *= m
    log_resp *= m

    # Robustly scale scores to [0, 1] inside mask so top-k is stable.
    for arr in (grad, log_resp):
        vals = arr[m]
        vals = vals[np.isfinite(vals)]
        if vals.size > 32:
            hi = np.percentile(vals, 99.5)
            if hi > 0:
                arr[:] = np.clip(arr / float(hi), 0.0, 1.0)
        arr[~m] = 0.0

    return grad.astype(np.float32), log_resp.astype(np.float32)


def _split_counts(n: int, fractions: Tuple[float, float, float]) -> Tuple[int, int, int]:
    f = np.asarray(fractions, dtype=np.float64)
    f = np.maximum(f, 0.0)
    if f.sum() <= 0:
        f[:] = (1.0, 0.0, 0.0)
    f = f / f.sum()
    counts = np.floor(f * int(n)).astype(np.int64)
    # assign remainder to largest fractional parts
    rem = int(n) - int(counts.sum())
    frac = f * int(n) - counts
    for j in np.argsort(-frac)[:rem]:
        counts[j] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def build_init_params_from_cfg(
    chi0: np.ndarray,
    mask: np.ndarray,
    cfg: Config,
    voxel_size_mm_zyx: Tuple[float, float, float],
    device: str,
    dtype: torch.dtype,
    score_volume: Optional[np.ndarray] = None,
) -> InitParams:
    """
    Init params from chi0 + mask using cfg.init settings.

    New lesion-aware mode:
      sample_mode="mask_edge_hybrid"
        - Gaussian centers are sampled from mask-uniform + phi-gradient + LoG score.
        - TKD is not used for center support unless you explicitly set score_volume/chi0 to TKD.
        - Edge/LoG points receive smaller sigma for sharper lesion details.
    """
    n_init = int(cfg.init.n_init)
    seed = int(cfg.train.seed)
    sample_mode = str(cfg.init.sample_mode).lower()

    sigma_scale_per_point: Optional[np.ndarray] = None

    if sample_mode in ("mask_uniform", "mask", "uniform"):
        idx = _sample_points_from_mask(mask, n_init, seed)

    elif sample_mode in ("topk_chi0", "topk", "chi0_topk"):
        idx = _sample_points_topk_abs_chi0(mask, chi0, n_init, cfg.init.topk_ratio, seed)

    elif sample_mode == "hybrid":
        hybrid_top_frac = float(getattr(cfg.init, "hybrid_top_frac", 0.10))
        hybrid_top_frac = max(0.0, min(1.0, hybrid_top_frac))

        n_top = int(round(n_init * hybrid_top_frac))
        n_uni = n_init - n_top

        parts = []
        if n_top > 0:
            parts.append(_sample_points_topk_abs_chi0(mask, chi0, n_top, cfg.init.topk_ratio, seed))
        if n_uni > 0:
            parts.append(_sample_points_from_mask(mask, n_uni, seed + 1))
        if not parts:
            raise ValueError("No initialization points selected; check n_init and hybrid_top_frac.")
        idx = np.concatenate(parts, axis=0)

    elif sample_mode in ("mask_edge_hybrid", "edge_hybrid", "phi_edge_hybrid", "lfs_edge_hybrid"):
        src = score_volume if score_volume is not None else chi0
        grad_score, log_score = _edge_and_log_scores(src, mask, cfg)

        n_uni, n_grad, n_log = _split_counts(
            n_init,
            (
                float(getattr(cfg.init, "edge_uniform_frac", 0.70)),
                float(getattr(cfg.init, "edge_grad_frac", 0.20)),
                float(getattr(cfg.init, "edge_log_frac", 0.10)),
            ),
        )

        parts = []
        scales = []
        if n_uni > 0:
            p0 = _sample_points_from_mask(mask, n_uni, seed + 1)
            parts.append(p0)
            scales.append(np.ones((p0.shape[0],), dtype=np.float32))
        if n_grad > 0:
            p1 = _sample_points_topk_score(
                mask,
                grad_score,
                n_grad,
                float(getattr(cfg.init, "edge_topk_ratio", 0.16)),
                seed + 2,
            )
            parts.append(p1)
            scales.append(np.full((p1.shape[0],), float(getattr(cfg.init, "edge_sigma_scale", 0.55)), dtype=np.float32))
        if n_log > 0:
            p2 = _sample_points_topk_score(
                mask,
                log_score,
                n_log,
                float(getattr(cfg.init, "log_topk_ratio", 0.12)),
                seed + 3,
            )
            parts.append(p2)
            scales.append(np.full((p2.shape[0],), float(getattr(cfg.init, "log_sigma_scale", 0.65)), dtype=np.float32))

        if not parts:
            raise ValueError("No initialization points selected in mask_edge_hybrid.")
        idx = np.concatenate(parts, axis=0)
        sigma_scale_per_point = np.concatenate(scales, axis=0)

        # Shuffle so optimizer chunks do not contain only one point type.
        rng_shuffle = np.random.default_rng(seed + 33)
        perm = rng_shuffle.permutation(idx.shape[0])
        idx = idx[perm]
        sigma_scale_per_point = sigma_scale_per_point[perm]

        print(
            "[INIT] mask_edge_hybrid: "
            f"N={n_init} uniform={n_uni} grad={n_grad} log={n_log} "
            f"edge_sigma_scale={float(getattr(cfg.init, 'edge_sigma_scale', 0.55)):.3f} "
            f"log_sigma_scale={float(getattr(cfg.init, 'log_sigma_scale', 0.65)):.3f}"
        )

    else:
        raise ValueError(
            f"Unknown cfg.init.sample_mode: {cfg.init.sample_mode}. "
            "Valid choices: mask_uniform, topk_chi0, hybrid, mask_edge_hybrid."
        )

    # xyz with jitter
    rng = np.random.default_rng(seed + 7)
    jitter = rng.uniform(low=-0.30, high=0.30, size=idx.shape).astype(np.float32)
    xyz = idx.astype(np.float32) + jitter

    # a from chi0 at sampled voxels. For phi/zero modes this is intentionally weak.
    a0 = chi0[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32) * float(cfg.init.init_a_scale)
    if cfg.init.chi0_clip is not None:
        clipv = float(cfg.init.chi0_clip)
        a0 = np.clip(a0, -clipv, clipv)
    a0 = a0.reshape(-1, 1)

    # sigma: mm -> voxel units
    init_sigma_mm = float(cfg.init.init_sigma_mm)
    vz, vy, vx = voxel_size_mm_zyx
    sigma_base = np.array([init_sigma_mm / vz, init_sigma_mm / vy, init_sigma_mm / vx], dtype=np.float32)

    j = rng.normal(loc=0.0, scale=float(cfg.init.init_sigma_jitter), size=(xyz.shape[0], 3)).astype(np.float32)
    sigma = sigma_base[None, :] * (1.0 + j)
    if sigma_scale_per_point is not None:
        sigma = sigma * sigma_scale_per_point[:, None]
    sigma = np.clip(sigma, 1e-3, 1e6)

    return InitParams(
        xyz=torch.tensor(xyz, device=device, dtype=dtype),
        sigma=torch.tensor(sigma, device=device, dtype=dtype),
        a=torch.tensor(a0, device=device, dtype=dtype),
    )


# -------------------------
# Splat (fast sanity-check only)
# -------------------------

@torch.no_grad()
def splat_fixed_sigma_gaussians_to_volume(
    xyz_zyx: torch.Tensor,   # (N,3)
    a: torch.Tensor,         # (N,1)
    sigma_zyx: torch.Tensor, # (N,3) or (3,)
    vol_shape_zyx: Tuple[int, int, int],
    radius_factor: float = 3.0,
    chunk: int = 2048,
) -> torch.Tensor:
    """
    Fast init sanity-check splat (NOT training splat).
    Uses median sigma and a separable Gaussian footprint.
    """
    device = xyz_zyx.device
    dtype = xyz_zyx.dtype
    Z, Y, X = vol_shape_zyx

    sig = torch.median(sigma_zyx, dim=0).values if sigma_zyx.ndim == 2 else sigma_zyx
    sig = torch.clamp(sig, min=1e-3)

    rz = int(torch.ceil(radius_factor * sig[0]).item())
    ry = int(torch.ceil(radius_factor * sig[1]).item())
    rx = int(torch.ceil(radius_factor * sig[2]).item())

    dz = torch.arange(-rz, rz + 1, device=device, dtype=dtype)
    dy = torch.arange(-ry, ry + 1, device=device, dtype=dtype)
    dx = torch.arange(-rx, rx + 1, device=device, dtype=dtype)
    DZ, DY, DX = torch.meshgrid(dz, dy, dx, indexing="ij")
    offs = torch.stack([DZ, DY, DX], dim=-1).reshape(-1, 3)  # (K,3)

    w = torch.exp(-0.5 * ((offs[:, 0] / sig[0]) ** 2 + (offs[:, 1] / sig[1]) ** 2 + (offs[:, 2] / sig[2]) ** 2))
    w = w / torch.clamp(w.sum(), min=1e-12)

    vol = torch.zeros((Z, Y, X), device=device, dtype=dtype)
    vol_flat = vol.view(-1)

    ctr = torch.round(xyz_zyx).to(torch.long)
    offs_long = offs.to(torch.long)
    K = offs_long.shape[0]

    for s in range(0, ctr.shape[0], chunk):
        e = min(s + chunk, ctr.shape[0])
        c = ctr[s:e]
        M = c.shape[0]

        pos = c[:, None, :] + offs_long[None, :, :]
        pz, py, px = pos[..., 0], pos[..., 1], pos[..., 2]
        inb = (pz >= 0) & (pz < Z) & (py >= 0) & (py < Y) & (px >= 0) & (px < X)

        idx_flat = (pz * (Y * X) + py * X + px)
        amp = a[s:e].view(M, 1)
        val = amp * w.view(1, K)

        vol_flat.scatter_add_(0, idx_flat[inb], val[inb])

    return vol


# -------------------------
# Main init-test runner
# -------------------------

def run_init_test(cfg: Config) -> None:
    validate_config(cfg)

    ensure_dir(cfg.io.out_dir)
    out_dir = os.path.join(cfg.io.out_dir, "init_test")
    ensure_dir(out_dir)

    # load lfs/phi + mask (mat or nii)
    phi_np, mask_np, affine, meta = load_data(
        data_path=cfg.io.phi_path,
        mask_path=cfg.io.mask_path,
        prefer_phi_key="lfs",     # common default; load_data will also search other keys
        prefer_mask_key="mask",
        squeeze=True,
    )

    # voxel size: affine wins if it looks valid, else use cfg.phys.voxel_size_mm
    vz_aff = voxel_size_from_affine_mm_zyx(affine)
    if np.allclose(vz_aff, (1.0, 1.0, 1.0)) and tuple(cfg.phys.voxel_size_mm) != (1.0, 1.0, 1.0):
        voxel_size_zyx = tuple(float(x) for x in cfg.phys.voxel_size_mm)
    else:
        voxel_size_zyx = vz_aff

    # build forward op from cfg.phys (but override voxel_size with inferred if needed)
    # (We keep the same naming as config.py: voxel_size_mm_zyx in PhysConfig)
    cfg_phys = cfg.phys
    # If you used PhysConfig.voxel_size_mm, Forward builder can pick it up.
    fwd: ForwardOp = build_forward_from_cfg(cfg_phys, device=cfg.io.device, dtype_str=cfg.io.dtype)
    # ensure forward uses correct voxel size (override)
    fwd.voxel_size_mm_zyx = tuple(float(x) for x in voxel_size_zyx)

    device = cfg.io.device if (cfg.io.device == "cpu" or torch.cuda.is_available()) else "cpu"
    dtype = torch.float32 if cfg.io.dtype == "float32" else torch.float16

    phi = torch.tensor(phi_np, device=device, dtype=torch.float32)  # TKD should stay float32
    mask = torch.tensor(mask_np, device=device, dtype=torch.float32)

    # TKD -> chi0
    chi0 = fwd.tkd(phi, padded=True, thresh=cfg.phys.tkd_thresh)
    chi0 = chi0 * mask

    chi0_np = chi0.detach().cpu().numpy().astype(np.float32)

    # init gaussians from chi0
    init_params = build_init_params_from_cfg(
        chi0=chi0_np,
        mask=mask_np,
        cfg=cfg,
        voxel_size_mm_zyx=voxel_size_zyx,
        device=device,
        dtype=torch.float32,  # keep init params float32 for sanity check
        score_volume=phi_np,
    )

    # splat back to voxel for sanity check
    chi_gs = splat_fixed_sigma_gaussians_to_volume(
        xyz_zyx=init_params.xyz,
        a=init_params.a,
        sigma_zyx=init_params.sigma,
        vol_shape_zyx=phi_np.shape,
        radius_factor=3.0,
        chunk=2048,
    )
    chi_gs = chi_gs * mask
    chi_gs_np = chi_gs.detach().cpu().numpy().astype(np.float32)

    # Save ONLY .nii
    save_nifti(phi_np.astype(np.float32), affine, force_nii_path(os.path.join(out_dir, "lfs_input.nii")))
    save_nifti(mask_np.astype(np.float32), affine, force_nii_path(os.path.join(out_dir, "mask.nii")))
    save_nifti(chi0_np, affine, force_nii_path(os.path.join(out_dir, "chi_tkd.nii")))
    save_nifti(chi_gs_np, affine, force_nii_path(os.path.join(out_dir, "chi_gs_init.nii")))
    save_nifti((chi_gs_np - chi0_np).astype(np.float32), affine, force_nii_path(os.path.join(out_dir, "chi_diff_gs_minus_tkd.nii")))

    # Console stats
    def _stats(x: np.ndarray) -> str:
        return f"min={x.min():.4g} max={x.max():.4g} mean={x.mean():.4g} std={x.std():.4g}"

    print("=== Init Test Done ===")
    print("meta:", meta)
    print("voxel_size_zyx(mm):", voxel_size_zyx)
    print("phi:", _stats(phi_np))
    print("chi_tkd:", _stats(chi0_np))
    print("chi_gs_init:", _stats(chi_gs_np))
    diff = chi_gs_np - chi0_np
    print("diff(gs-tkd):", _stats(diff))
    print("saved to:", out_dir)


def _try_load_default_cfg(config_path: Optional[str] = None) -> Config:
    if config_path:
        return load_config(config_path)
    for p in ("config.json", "config.yaml", "config.yml"):
        if os.path.exists(p):
            return load_config(p)
    cfg = Config()
    validate_config(cfg)
    print("[WARN] No config file found. Using Config() defaults.")
    return cfg


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run GSQSM initialization test.")
    parser.add_argument("--config", default=None, help="Optional json/yaml config.")
    args = parser.parse_args()
    run_init_test(_try_load_default_cfg(args.config))

