# tool.py
from __future__ import annotations

import os
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy import ndimage as ndi


# -------------------------
# basic
# -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def force_nii_path(path: str) -> str:
    # Normalize NIfTI suffix.
    p = str(path)
    if p.lower().endswith(".nii.gz"):
        p = p[:-3]
    if not p.lower().endswith(".nii"):
        p += ".nii"
    return p


def voxel_size_from_affine_mm_zyx(affine: np.ndarray) -> Tuple[float, float, float]:
    # Return voxel size in z-y-x order.
    A = np.asarray(affine, dtype=np.float32)[:3, :3]
    vx = float(np.linalg.norm(A[:, 0]))
    vy = float(np.linalg.norm(A[:, 1]))
    vz = float(np.linalg.norm(A[:, 2]))
    return (
        vz if np.isfinite(vz) and vz > 0 else 1.0,
        vy if np.isfinite(vy) and vy > 0 else 1.0,
        vx if np.isfinite(vx) and vx > 0 else 1.0,
    )


# -------------------------
# nifti io
# -------------------------

def load_nifti(path: str) -> Tuple[np.ndarray, np.ndarray, Any]:
    import nibabel as nib
    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)
    return vol, affine, img.header


def save_nifti(vol: np.ndarray, affine: np.ndarray, path: str) -> None:
    import nibabel as nib
    img = nib.Nifti1Image(np.asarray(vol), np.asarray(affine, dtype=np.float32))
    nib.save(img, path)


# -------------------------
# mask / roi
# -------------------------

def make_mask_from_phi(phi: np.ndarray, mode: str = "phi!=0") -> np.ndarray:
    if mode == "phi!=0":
        return (phi != 0).astype(np.uint8)
    if mode == "abs>0":
        return (np.abs(phi) > 0).astype(np.uint8)
    raise ValueError(f"Unknown mask mode: {mode}")


def bbox_from_mask(mask: np.ndarray, margin: int = 0) -> Tuple[int, int, int, int, int, int]:
    if mask.ndim != 3:
        raise ValueError(f"mask must be 3D, got {mask.ndim}D")

    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        z, y, x = mask.shape
        return 0, z, 0, y, 0, x

    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0) + 1

    z0 = max(0, int(z0) - margin)
    y0 = max(0, int(y0) - margin)
    x0 = max(0, int(x0) - margin)

    z1 = min(mask.shape[0], int(z1) + margin)
    y1 = min(mask.shape[1], int(y1) + margin)
    x1 = min(mask.shape[2], int(x1) + margin)

    return z0, z1, y0, y1, x0, x1


def crop3d(vol: np.ndarray, bbox: Tuple[int, int, int, int, int, int]) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    return vol[z0:z1, y0:y1, x0:x1]


def crop3d_torch(vol: torch.Tensor, bbox: Tuple[int, int, int, int, int, int]) -> torch.Tensor:
    z0, z1, y0, y1, x0, x1 = bbox
    return vol[..., z0:z1, y0:y1, x0:x1]


def paste_crop3d(dst: np.ndarray, crop: np.ndarray, bbox: Tuple[int, int, int, int, int, int]) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    dst[z0:z1, y0:y1, x0:x1] = crop
    return dst


# -------------------------
# padding
# -------------------------

def pad3d_np(vol: np.ndarray, pad: int, mode: str = "edge", constant_values: float = 0.0) -> np.ndarray:
    if pad <= 0:
        return vol
    pads = ((pad, pad), (pad, pad), (pad, pad))
    if mode == "edge":
        return np.pad(vol, pads, mode="edge")
    if mode == "constant":
        return np.pad(vol, pads, mode="constant", constant_values=constant_values)
    if mode == "reflect":
        return np.pad(vol, pads, mode="reflect")
    raise ValueError(f"Unknown pad mode: {mode}")


def unpad3d_np(vol: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return vol
    return vol[pad:-pad, pad:-pad, pad:-pad]


def pad3d_torch(vol: torch.Tensor, pad: int, mode: str = "replicate", value: float = 0.0) -> torch.Tensor:
    if pad <= 0:
        return vol

    import torch.nn.functional as F

    if mode not in ("replicate", "reflect", "constant"):
        raise ValueError(f"Unknown torch pad mode: {mode}")

    pads = (pad, pad, pad, pad, pad, pad)  # x1,x2,y1,y2,z1,z2

    if mode == "constant":
        return F.pad(vol, pads, mode="constant", value=value)

    if vol.ndim < 3:
        raise ValueError(f"pad3d_torch expects at least 3 dims, got {vol.ndim}")

    orig_shape = vol.shape
    Z, Y, X = orig_shape[-3], orig_shape[-2], orig_shape[-1]
    leading = orig_shape[:-3]

    v = vol.reshape(-1, 1, Z, Y, X)
    v = F.pad(v, pads, mode=mode)
    return v.reshape(*leading, v.shape[-3], v.shape[-2], v.shape[-1])


def unpad3d_torch(vol: torch.Tensor, pad: int) -> torch.Tensor:
    if pad <= 0:
        return vol
    if vol.shape[-3] <= 2 * pad or vol.shape[-2] <= 2 * pad or vol.shape[-1] <= 2 * pad:
        raise ValueError(f"unpad3d_torch: pad={pad} too large for shape {tuple(vol.shape)}")
    return vol[..., pad:-pad, pad:-pad, pad:-pad]


# -------------------------
# torch / numpy
# -------------------------

def to_torch(x: Any, device: str = "cuda", dtype: str = "float32") -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        t = x
    else:
        t = torch.from_numpy(np.asarray(x))

    if dtype == "float16":
        t = t.to(torch.float16)
    else:
        t = t.to(torch.float32)

    return t.to(device)


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# -------------------------
# fft
# -------------------------

def fft3(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftn(x, dim=(-3, -2, -1))


def ifft3(X: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftn(X, dim=(-3, -2, -1))


# -------------------------
# small math
# -------------------------

def safe_norm(x: torch.Tensor, dim: Optional[int] = None, keepdim: bool = False, eps: float = 1e-12) -> torch.Tensor:
    if dim is None:
        return torch.sqrt(torch.clamp((x * x).sum(), min=eps))
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))


def print_tensor_stats(name: str, x: torch.Tensor, max_items: int = 8) -> None:
    xd = x.detach()
    flat = xd.flatten()
    if flat.numel() == 0:
        print(f"[{name}] empty")
        return
    sel = flat[: min(flat.numel(), max_items)]
    print(
        f"[{name}] shape={tuple(xd.shape)} dtype={xd.dtype} device={xd.device} "
        f"min={float(flat.min()):.4g} max={float(flat.max()):.4g} mean={float(flat.mean()):.4g} "
        f"std={float(flat.std()):.4g} sample={sel.cpu().numpy()}"
    )



# -------------------------
# sampling
# -------------------------

def sample_voxels_from_mask(mask: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        raise ValueError("Mask is empty; cannot sample voxels.")
    if n >= len(coords):
        idx = rng.integers(0, len(coords), size=n)
        return coords[idx].astype(np.int64)
    idx = rng.choice(len(coords), size=n, replace=False)
    return coords[idx].astype(np.int64)


def voxel_idx_to_world_mm(
    idx_zyx: np.ndarray,
    voxel_size_mm: Tuple[float, float, float],
    origin_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    idx = idx_zyx.astype(np.float32)
    vz, vy, vx = voxel_size_mm
    oz, oy, ox = origin_mm
    out = np.empty_like(idx, dtype=np.float32)
    out[:, 0] = oz + idx[:, 0] * vz
    out[:, 1] = oy + idx[:, 1] * vy
    out[:, 2] = ox + idx[:, 2] * vx
    return out


# -------------------------
# data loading (nii / mat)
# -------------------------

_DEFAULT_PHI_KEYS = (
    "phi", "lfs", "tfs", "field", "local_field", "localField", "totalField",
    "phase", "lfs_ppm", "lfs_rad", "b0map", "b0_map",
)

_DEFAULT_MASK_KEYS = (
    "mask", "msk", "brain_mask", "brainMask", "roi", "ROI", "mask_eroded",
)


def load_data(
    data_path: str,
    mask_path: str = "",
    phi_keys: Sequence[str] = _DEFAULT_PHI_KEYS,
    mask_keys: Sequence[str] = _DEFAULT_MASK_KEYS,
    prefer_phi_key: str = "",
    prefer_mask_key: str = "",
    squeeze: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"data_path not found: {data_path}")

    ext = _lower_ext(data_path)
    meta: Dict[str, Any] = {"data_path": data_path, "mask_path": mask_path, "source": ext}

    # phi
    if ext in (".nii", ".nii.gz"):
        phi, affine, _ = load_nifti(data_path)
        meta.update({"phi_key": None, "phi_shape_raw": tuple(phi.shape), "phi_dtype_raw": str(phi.dtype)})
        mat = None
    elif ext == ".mat":
        mat = _load_mat(data_path)
        phi_key, phi = _extract_from_mat(mat, prefer_phi_key, phi_keys, what="phi/lfs")
        affine = np.eye(4, dtype=np.float32)
        meta.update({"phi_key": phi_key, "phi_shape_raw": tuple(np.asarray(phi).shape), "phi_dtype_raw": str(np.asarray(phi).dtype)})
    else:
        raise ValueError(f"Unsupported data file extension: {ext}")

    phi = _to_3d_float32(phi, squeeze=squeeze, name="phi")

    # mask
    mask: Optional[np.ndarray] = None
    if mask_path:
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"mask_path not found: {mask_path}")

        mext = _lower_ext(mask_path)
        if mext in (".nii", ".nii.gz"):
            mask_np, _, _ = load_nifti(mask_path)
            meta["mask_key"] = None
        elif mext == ".mat":
            mmat = _load_mat(mask_path)
            mk, mask_np = _extract_from_mat(mmat, prefer_mask_key, mask_keys, what="mask")
            meta["mask_key"] = mk
        else:
            raise ValueError(f"Unsupported mask file extension: {mext}")

        mask = _to_3d_mask(mask_np, squeeze=squeeze, name="mask")

    else:
        if ext == ".mat" and mat is not None:
            mk, mask_np = _extract_from_mat(mat, prefer_mask_key, mask_keys, what="mask", allow_missing=True)
            if mask_np is not None:
                mask = _to_3d_mask(mask_np, squeeze=squeeze, name="mask")
                meta["mask_key"] = mk

    if mask is None:
        mask = (phi != 0).astype(np.uint8)
        meta["mask_key"] = "derived(phi!=0)"

    if mask.shape != phi.shape:
        raise ValueError(f"Shape mismatch: phi {phi.shape} vs mask {mask.shape}")

    meta["phi_shape"] = tuple(phi.shape)
    meta["mask_shape"] = tuple(mask.shape)
    return phi, mask, affine.astype(np.float32), meta


# -------------------------
# internals (mat helpers)
# -------------------------

def _lower_ext(path: str) -> str:
    p = path.lower()
    if p.endswith(".nii.gz"):
        return ".nii.gz"
    return os.path.splitext(p)[1]


def _load_mat(path: str) -> Dict[str, Any]:
    import scipy.io as sio
    mat = sio.loadmat(path, squeeze_me=False, struct_as_record=False)
    return {k: v for k, v in mat.items() if not k.startswith("__")}


def _extract_from_mat(
    mat: Dict[str, Any],
    prefer_key: str,
    candidate_keys: Sequence[str],
    what: str,
    allow_missing: bool = False,
) -> Tuple[Optional[str], Optional[np.ndarray]]:
    if prefer_key:
        v = _get_mat_key(mat, prefer_key)
        if v is not None:
            return prefer_key, v

    for k in candidate_keys:
        v = _get_mat_key(mat, k)
        if v is not None:
            return k, v

    for root_k, root_v in mat.items():
        fields = _struct_fields(root_v)
        if not fields:
            continue

        if prefer_key and prefer_key in fields:
            return f"{root_k}.{prefer_key}", getattr(root_v, prefer_key)

        for k in candidate_keys:
            if k in fields:
                return f"{root_k}.{k}", getattr(root_v, k)

    if allow_missing:
        return None, None

    keys = ", ".join(list(mat.keys())[:30])
    raise KeyError(
        f"Cannot find {what} in .mat. prefer_key='{prefer_key}', candidates={list(candidate_keys)}. "
        f"Top-level keys: {keys}"
    )


def _get_mat_key(mat: Dict[str, Any], key: str) -> Optional[Any]:
    if key in mat:
        return mat[key]
    lk = key.lower()
    for k, v in mat.items():
        if k.lower() == lk:
            return v
    return None


def _struct_fields(x: Any) -> Sequence[str]:
    if hasattr(x, "_fieldnames") and isinstance(getattr(x, "_fieldnames"), (list, tuple)):
        return list(x._fieldnames)
    return []


def _to_3d_float32(arr: Any, squeeze: bool, name: str) -> np.ndarray:
    a = np.asarray(arr)
    if squeeze:
        a = np.squeeze(a)
    if np.iscomplexobj(a):
        a = np.real(a)

    if a.ndim == 4:
        if a.shape[-1] <= 4:
            a = a[..., 0]
        elif a.shape[0] <= 4:
            a = a[0, ...]
        else:
            raise ValueError(f"{name} is 4D with ambiguous layout: {a.shape}")

    if a.ndim != 3:
        raise ValueError(f"{name} must be 3D after squeeze; got shape {a.shape}")

    return a.astype(np.float32, copy=False)


def _to_3d_mask(arr: Any, squeeze: bool, name: str) -> np.ndarray:
    m = np.asarray(arr)
    if squeeze:
        m = np.squeeze(m)
    if np.iscomplexobj(m):
        m = np.real(m)

    if m.ndim == 4:
        if m.shape[-1] <= 4:
            m = m[..., 0]
        elif m.shape[0] <= 4:
            m = m[0, ...]
        else:
            raise ValueError(f"{name} is 4D with ambiguous layout: {m.shape}")

    if m.ndim != 3:
        raise ValueError(f"{name} must be 3D after squeeze; got shape {m.shape}")

    return (m != 0).astype(np.uint8)


# ============================================================
# Mask repair / LFS noise estimation utilities
# Merged here to avoid any extra utility module after removing denoise modules.
# ============================================================

# ============================================================
# Config helpers
# ============================================================

def _cfg_get(cfg: Any, name: str, default: Any = None) -> Any:
    """
    Read config value from dataclass-like object or dict.

    Supports:
      cfg.name
      cfg.cfr.name
      cfg.mask.name
      cfg["name"]
      cfg["cfr"]["name"]
      cfg["mask"]["name"]
    """
    if cfg is None:
        return default

    if isinstance(cfg, dict):
        if name in cfg:
            return cfg[name]
        for sec in ("cfr", "mask", "train", "loss", "unet", "io"):
            obj = cfg.get(sec, None)
            if isinstance(obj, dict) and name in obj:
                return obj[name]
        return default

    if hasattr(cfg, name):
        return getattr(cfg, name)

    for sec in ("cfr", "mask", "train", "loss", "unet", "io"):
        if hasattr(cfg, sec):
            obj = getattr(cfg, sec)
            if isinstance(obj, dict):
                if name in obj:
                    return obj[name]
            elif hasattr(obj, name):
                return getattr(obj, name)

    return default


def _as_float_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    m = np.asarray(mask)
    m = np.nan_to_num(m, nan=0.0, posinf=0.0, neginf=0.0)
    return m > 0.5


# ============================================================
# Robust mask repair / building
# ============================================================

def _largest_connected_component(mask: np.ndarray, connectivity: int = 2) -> np.ndarray:
    m = mask.astype(bool)
    if m.sum() == 0:
        return m

    structure = ndi.generate_binary_structure(3, int(connectivity))
    lab, n = ndi.label(m, structure=structure)
    if n <= 1:
        return m

    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = int(np.argmax(counts))
    return lab == keep


def _remove_small_components(mask: np.ndarray, min_voxels: int = 128) -> np.ndarray:
    m = mask.astype(bool)
    if m.sum() == 0:
        return m

    structure = ndi.generate_binary_structure(3, 2)
    lab, n = ndi.label(m, structure=structure)
    if n <= 1:
        return m

    counts = np.bincount(lab.ravel())
    keep_ids = np.where(counts >= int(min_voxels))[0]
    keep_ids = keep_ids[keep_ids != 0]
    if keep_ids.size == 0:
        return _largest_connected_component(m, connectivity=2)

    return np.isin(lab, keep_ids)


def _fill_holes_slice_wise(mask: np.ndarray) -> np.ndarray:
    """
    Fill holes in 3D and then along all three slice directions.

    This is stronger than single 3D fill and is useful for masks with
    many intra-slice black holes.
    """
    m = mask.astype(bool)

    # 3D fill first
    out = ndi.binary_fill_holes(m)

    # Slice-wise fill along z/y/x
    for axis in range(3):
        moved = np.moveaxis(out, axis, -1).copy()
        for i in range(moved.shape[-1]):
            moved[..., i] = ndi.binary_fill_holes(moved[..., i])
        out = np.moveaxis(moved, -1, axis)

    return out.astype(bool)


def _hole_fraction(mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.sum() == 0:
        return 1.0
    filled = _fill_holes_slice_wise(m)
    holes = filled & (~m)
    return float(holes.sum() / (filled.sum() + 1e-8))


def _boundary_fragment_score(mask: np.ndarray) -> float:
    m = mask.astype(bool)
    if m.sum() == 0:
        return 1.0
    structure = ndi.generate_binary_structure(3, 1)
    eroded = ndi.binary_erosion(m, structure=structure, iterations=1)
    boundary = m & (~eroded)
    return float(boundary.sum() / (m.sum() + 1e-8))


def _postprocess_brain_mask(
    mask: np.ndarray,
    close_iter: int = 4,
    open_iter: int = 1,
    dilate_iter: int = 0,
    erode_iter: int = 0,
    min_component_voxels: int = 128,
    keep_largest_only: bool = True,
) -> np.ndarray:
    """
    Strong brain support post-processing.

    Designed to repair:
      - black holes inside brain
      - fragmented edges
      - small isolated islands
      - phi!=0 fallback masks that are too noisy
    """
    m = mask.astype(bool)
    if m.sum() == 0:
        return m.astype(np.uint8)

    structure1 = ndi.generate_binary_structure(3, 1)
    structure2 = ndi.generate_binary_structure(3, 2)

    # Close cracks and small gaps first.
    if close_iter > 0:
        m = ndi.binary_closing(m, structure=structure2, iterations=int(close_iter))

    m = _remove_small_components(m, min_voxels=int(min_component_voxels))
    m = _fill_holes_slice_wise(m)

    if keep_largest_only:
        m = _largest_connected_component(m, connectivity=2)

    # Remove burrs.
    if open_iter > 0:
        m = ndi.binary_opening(m, structure=structure1, iterations=int(open_iter))

    # Close again after opening.
    if close_iter > 0:
        m = ndi.binary_closing(m, structure=structure2, iterations=max(1, int(close_iter) // 2))

    m = _fill_holes_slice_wise(m)

    if keep_largest_only:
        m = _largest_connected_component(m, connectivity=2)

    if dilate_iter > 0:
        m = ndi.binary_dilation(m, structure=structure1, iterations=int(dilate_iter))

    if erode_iter > 0:
        m = ndi.binary_erosion(m, structure=structure1, iterations=int(erode_iter))

    m = _fill_holes_slice_wise(m)

    if keep_largest_only:
        m = _largest_connected_component(m, connectivity=2)

    return m.astype(np.uint8)


def _build_mask_from_lfs(
    phi_np: np.ndarray,
    target_ratio: float = 0.20,
    min_ratio: float = 0.04,
    max_ratio: float = 0.55,
    smooth_sigma: float = 2.0,
    close_iter: int = 4,
    open_iter: int = 1,
) -> np.ndarray:
    """
    Fallback support mask from single LFS.

    This is not a precise brain extraction method. It is a stable support
    estimator for GS-QSM loss/statistics when no reliable mask is available.
    """
    phi = _as_float_np(phi_np)
    score = np.abs(phi)

    finite = np.isfinite(score)
    vals = score[finite]
    vals = vals[vals > 0]

    if vals.size < 128:
        raw = score > 0
        return _postprocess_brain_mask(raw, close_iter=close_iter, open_iter=open_iter)

    # Clip outliers so that thresholding is not dominated by a few values.
    p995 = np.percentile(vals, 99.5)
    if p995 > 0:
        score = np.clip(score, 0.0, p995)

    if smooth_sigma > 0:
        score = ndi.gaussian_filter(score, sigma=float(smooth_sigma))

    vals = score[np.isfinite(score)]
    vals = vals[vals > 0]
    if vals.size < 128:
        raw = score > 0
        return _postprocess_brain_mask(raw, close_iter=close_iter, open_iter=open_iter)

    # Try several quantiles and choose the most reasonable support.
    # Lower q -> larger mask; higher q -> smaller mask.
    q_list = [5, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

    candidates = []
    for q in q_list:
        thr = np.percentile(vals, q)
        raw = score > thr

        m = _postprocess_brain_mask(
            raw,
            close_iter=close_iter,
            open_iter=open_iter,
            min_component_voxels=128,
            keep_largest_only=True,
        )

        ratio = float(m.mean())
        holes = _hole_fraction(m)
        frag = _boundary_fragment_score(m)

        ratio_penalty = abs(ratio - target_ratio)
        if ratio < min_ratio:
            ratio_penalty += 10.0 * (min_ratio - ratio)
        if ratio > max_ratio:
            ratio_penalty += 10.0 * (ratio - max_ratio)

        score_val = ratio_penalty + 0.75 * holes + 0.10 * frag
        candidates.append((score_val, q, ratio, holes, frag, m))

    candidates.sort(key=lambda x: x[0])
    return candidates[0][-1].astype(np.uint8)


def fix_or_build_brain_mask(
    phi_np: np.ndarray,
    mask_np: Optional[np.ndarray] = None,
    cfg: Any = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Main mask function.

    Returns:
        mask_fixed: uint8, same shape as phi_np
        info: dict
    """
    phi = _as_float_np(phi_np)
    shape = phi.shape

    min_ratio = float(_cfg_get(cfg, "mask_min_ratio", 0.04))
    max_ratio = float(_cfg_get(cfg, "mask_max_ratio", 0.55))
    full_ratio = float(_cfg_get(cfg, "mask_full_ratio", 0.92))
    target_ratio = float(_cfg_get(cfg, "mask_target_ratio", 0.20))

    hole_thresh = float(_cfg_get(cfg, "mask_hole_thresh", 0.025))
    smooth_sigma = float(_cfg_get(cfg, "mask_smooth_sigma", 2.0))

    close_iter = int(_cfg_get(cfg, "mask_close_iter", 4))
    open_iter = int(_cfg_get(cfg, "mask_open_iter", 1))
    dilate_iter = int(_cfg_get(cfg, "mask_dilate_iter", 0))
    erode_iter = int(_cfg_get(cfg, "mask_erode_iter", 0))
    force_rebuild = bool(_cfg_get(cfg, "mask_force_rebuild", False))
    keep_largest_only = bool(_cfg_get(cfg, "mask_keep_largest_only", True))
    min_component_voxels = int(_cfg_get(cfg, "mask_min_component_voxels", 128))

    info: Dict[str, Any] = {
        "input_ratio": -1.0,
        "output_ratio": -1.0,
        "input_hole_fraction": -1.0,
        "output_hole_fraction": -1.0,
        "input_boundary_fragment": -1.0,
        "output_boundary_fragment": -1.0,
        "fixed": False,
        "rebuilt": False,
        "reason": "",
    }

    use_input = False
    m0 = None

    if mask_np is not None:
        arr = np.asarray(mask_np)
        if arr.shape != shape:
            info["reason"] += f"shape_mismatch:{arr.shape}->{shape}; "
        else:
            m0 = _as_bool_mask(arr)
            input_ratio = float(m0.mean())
            input_holes = _hole_fraction(m0)
            input_frag = _boundary_fragment_score(m0)

            info["input_ratio"] = input_ratio
            info["input_hole_fraction"] = input_holes
            info["input_boundary_fragment"] = input_frag

            if force_rebuild:
                info["reason"] += "force_rebuild; "
            elif input_ratio <= 1e-8:
                info["reason"] += "input_empty; "
            elif input_ratio >= full_ratio:
                info["reason"] += "input_almost_full; "
            elif input_ratio < min_ratio:
                info["reason"] += "input_too_small; "
            elif input_ratio > max_ratio:
                info["reason"] += "input_too_large; "
            else:
                use_input = True
    else:
        info["reason"] += "no_input_mask; "

    if use_input and m0 is not None:
        m = _postprocess_brain_mask(
            m0,
            close_iter=close_iter,
            open_iter=open_iter,
            dilate_iter=dilate_iter,
            erode_iter=erode_iter,
            min_component_voxels=min_component_voxels,
            keep_largest_only=keep_largest_only,
        )

        out_ratio = float(m.mean())
        out_holes = _hole_fraction(m)

        if out_ratio < min_ratio:
            info["reason"] += f"post_too_small:{out_ratio:.4f}; rebuild_from_lfs; "
            rebuild = True
        elif out_ratio > max_ratio:
            info["reason"] += f"post_too_large:{out_ratio:.4f}; rebuild_from_lfs; "
            rebuild = True
        elif out_holes > hole_thresh:
            info["reason"] += f"post_holes_too_many:{out_holes:.4f}; rebuild_from_lfs; "
            rebuild = True
        else:
            rebuild = False

        if rebuild:
            m = _build_mask_from_lfs(
                phi,
                target_ratio=target_ratio,
                min_ratio=min_ratio,
                max_ratio=max_ratio,
                smooth_sigma=smooth_sigma,
                close_iter=close_iter,
                open_iter=open_iter,
            )
            info["rebuilt"] = True
        else:
            info["reason"] += "postprocess_input_mask; "

        info["fixed"] = True
    else:
        m = _build_mask_from_lfs(
            phi,
            target_ratio=target_ratio,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            smooth_sigma=smooth_sigma,
            close_iter=close_iter,
            open_iter=open_iter,
        )
        info["fixed"] = True
        info["rebuilt"] = True
        info["reason"] += "build_from_lfs; "

    # Final safety: never return an almost full mask.
    final_ratio = float(m.mean())
    if final_ratio >= full_ratio:
        info["reason"] += "final_almost_full_strict_rebuild; "
        m = _build_mask_from_lfs(
            phi,
            target_ratio=min(target_ratio, 0.18),
            min_ratio=min_ratio,
            max_ratio=min(max_ratio, 0.40),
            smooth_sigma=smooth_sigma,
            close_iter=max(close_iter, 4),
            open_iter=open_iter,
        )
        info["rebuilt"] = True
        info["fixed"] = True

    # Final fill and largest component again.
    m = _fill_holes_slice_wise(m)
    m = _largest_connected_component(m, connectivity=2)

    info["output_ratio"] = float(m.mean())
    info["output_hole_fraction"] = _hole_fraction(m)
    info["output_boundary_fragment"] = _boundary_fragment_score(m)

    return m.astype(np.uint8), info


# ============================================================
# Old V3 train.py compatibility: repair_mask_np
# ============================================================

def repair_mask_np(phi_np, mask_np=None, mask_cfg=None, meta=None):
    """
    Compatibility wrapper for train.py.

    train.py calls:
        mask_np, mask_info = repair_mask_np(phi_np, mask_np, cfg.mask, meta)
    """
    mask_fixed, info = fix_or_build_brain_mask(phi_np=phi_np, mask_np=mask_np, cfg=mask_cfg)

    if meta is not None and isinstance(meta, dict):
        if "mask_key" in meta:
            info["mask_key"] = str(meta.get("mask_key", ""))
        elif "mask_path" in meta:
            info["mask_key"] = str(meta.get("mask_path", ""))
        else:
            info["mask_key"] = ""
    else:
        info["mask_key"] = ""

    return mask_fixed.astype(np.float32), info


# ============================================================
# Single LFS noise estimation
# ============================================================

def estimate_lfs_noise_score(
    phi_np: np.ndarray,
    mask_np: Optional[np.ndarray] = None,
    smooth_sigma: float = 2.0,
) -> Dict[str, float]:
    """
    Heuristic noise score from single LFS.

    score = MAD(high-pass(phi)) / robust_std(phi)
    """
    phi = _as_float_np(phi_np)
    if mask_np is None:
        mask = np.isfinite(phi)
    else:
        mask = _as_bool_mask(mask_np)

    if mask.sum() < 32:
        return {"score": 1.0, "hp_mad": 0.0, "robust_std": 0.0, "mask_ratio": float(mask.mean())}

    phi_s = ndi.gaussian_filter(phi, sigma=float(smooth_sigma))
    hp = phi - phi_s

    vals_hp = hp[mask]
    vals_phi = phi[mask]

    med_hp = np.median(vals_hp)
    hp_mad = 1.4826 * np.median(np.abs(vals_hp - med_hp))

    p1 = np.percentile(vals_phi, 1)
    p99 = np.percentile(vals_phi, 99)
    robust_std = max((p99 - p1) / 4.0, 1e-8)

    score = float(hp_mad / robust_std)
    return {
        "score": score,
        "hp_mad": float(hp_mad),
        "robust_std": float(robust_std),
        "mask_ratio": float(mask.mean()),
    }


def estimate_lfs_noise_score_np(phi_np, mask_np=None, kernel: int = 5):
    """
    Compatibility wrapper used by train.py.

    train.py expects keys:
        score, hp_mad, flat_hp_mad, phi_std
    """
    smooth_sigma = max(1.0, float(kernel) / 3.0)
    info = estimate_lfs_noise_score(phi_np, mask_np, smooth_sigma=smooth_sigma)

    hp_mad = float(info.get("hp_mad", 0.0))
    robust_std = float(info.get("robust_std", 0.0))

    return {
        "score": float(info.get("score", 1.0)),
        "hp_mad": hp_mad,
        "flat_hp_mad": hp_mad,
        "phi_std": robust_std,
        "mask_ratio": float(info.get("mask_ratio", 0.0)),
    }


