from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor


def _safe_norm(x: Tensor, dim: int = -1, keepdim: bool = False, eps: float = 1e-12) -> Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim, keepdim=keepdim), min=eps))


def _resize_like(x: Tensor, n: int) -> Tensor:
    old_n = x.shape[0]
    if n == old_n:
        return x
    if n < old_n:
        return x[:n].contiguous()
    pad = torch.zeros((n - old_n, *x.shape[1:]), device=x.device, dtype=x.dtype)
    return torch.cat([x, pad], dim=0).contiguous()


def _adaptive_threshold(
    values: Tensor,
    *,
    mode: str,
    abs_min: float = 0.0,
    mad_k: float = 2.5,
    percentile: float = 75.0,
) -> Tensor:
    """Return a scalar threshold from finite non-negative values."""
    vals = values.detach().reshape(-1)
    vals = vals[torch.isfinite(vals)]
    vals = vals[vals >= 0]
    if vals.numel() == 0:
        return torch.tensor(float("inf"), device=values.device, dtype=values.dtype)

    mode = str(mode).lower()
    if mode == "abs":
        thr = torch.tensor(float(abs_min), device=values.device, dtype=values.dtype)
    elif mode == "percentile":
        q = min(max(float(percentile) / 100.0, 0.0), 1.0)
        thr = torch.quantile(vals, q)
        if float(abs_min) > 0:
            thr = torch.maximum(thr, torch.tensor(float(abs_min), device=values.device, dtype=values.dtype))
    elif mode == "mad":
        med = torch.median(vals)
        mad = torch.median((vals - med).abs()).clamp_min(1.0e-12)
        thr = med + float(mad_k) * mad
        if float(abs_min) > 0:
            thr = torch.maximum(thr, torch.tensor(float(abs_min), device=values.device, dtype=values.dtype))
    else:
        raise ValueError(f"Unknown threshold mode: {mode}")
    return thr


def _identity_quat(n: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    q = torch.zeros((int(n), 4), device=device, dtype=dtype)
    q[:, 0] = 1.0
    return q


def _normalize_quat(q: Tensor, eps: float = 1.0e-8) -> Tensor:
    # q = [w, x, y, z]
    return q / torch.clamp(_safe_norm(q, dim=1, keepdim=True), min=eps)


def _quat_to_rotmat(q: Tensor) -> Tensor:
    """Quaternion [w,x,y,z] -> rotation matrix R with columns as local axes in world coords."""
    q = _normalize_quat(q)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    ww, xx, yy, zz = w*w, x*x, y*y, z*z
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z
    R = torch.empty((q.shape[0], 3, 3), device=q.device, dtype=q.dtype)
    R[:, 0, 0] = ww + xx - yy - zz
    R[:, 0, 1] = 2.0 * (xy - wz)
    R[:, 0, 2] = 2.0 * (xz + wy)
    R[:, 1, 0] = 2.0 * (xy + wz)
    R[:, 1, 1] = ww - xx + yy - zz
    R[:, 1, 2] = 2.0 * (yz - wx)
    R[:, 2, 0] = 2.0 * (xz - wy)
    R[:, 2, 1] = 2.0 * (yz + wx)
    R[:, 2, 2] = ww - xx - yy + zz
    return R


def _local_to_world(vec_local: Tensor, rot_q: Tensor) -> Tensor:
    """Row-vector local coordinates -> world coordinates using R columns as local axes."""
    R = _quat_to_rotmat(rot_q)
    if vec_local.ndim == 2:
        return torch.einsum('bi,bij->bj', vec_local, R.transpose(1, 2))
    if vec_local.ndim == 3:
        return torch.einsum('bki,bij->bkj', vec_local, R.transpose(1, 2))
    raise ValueError(f"vec_local must be 2D/3D, got {tuple(vec_local.shape)}")


def _world_to_local(vec_world: Tensor, rot_q: Tensor) -> Tensor:
    """Row-vector world coordinates -> local coordinates using R^T."""
    R = _quat_to_rotmat(rot_q)
    if vec_world.ndim == 2:
        return torch.einsum('bi,bij->bj', vec_world, R)
    if vec_world.ndim == 3:
        return torch.einsum('bki,bij->bkj', vec_world, R)
    raise ValueError(f"vec_world must be 2D/3D, got {tuple(vec_world.shape)}")


def _precision_elements_from_sigma_rot(sig: Tensor, rot_q: Tensor, use_rotation: bool = True) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Return symmetric precision matrix elements for q=d^T A d.

    This is mathematically equivalent to rotating every offset into local
    coordinates and dividing by sigma, but it avoids a per-offset matrix
    multiply in the splatting inner loop. A = R diag(1/sigma^2) R^T.
    Element order follows z/y/x coordinates: Azz, Ayy, Axx, Azy, Azx, Ayx.
    """
    inv = 1.0 / (sig.clamp_min(1.0e-6) * sig.clamp_min(1.0e-6))
    if not bool(use_rotation):
        zero = torch.zeros_like(inv[:, 0])
        return inv[:, 0], inv[:, 1], inv[:, 2], zero, zero, zero
    R = _quat_to_rotmat(rot_q)  # rows: world z/y/x, cols: local axes
    A = (R * inv[:, None, :]) @ R.transpose(1, 2)
    return A[:, 0, 0], A[:, 1, 1], A[:, 2, 2], A[:, 0, 1], A[:, 0, 2], A[:, 1, 2]


def _quadratic_form_from_precision(d: Tensor, elems: Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    """Compute d^T A d for batched offsets d=(B,K,3)."""
    Azz, Ayy, Axx, Azy, Azx, Ayx = elems
    dz, dy, dx = d[..., 0], d[..., 1], d[..., 2]
    return (
        Azz[:, None] * dz * dz
        + Ayy[:, None] * dy * dy
        + Axx[:, None] * dx * dx
        + 2.0 * Azy[:, None] * dz * dy
        + 2.0 * Azx[:, None] * dz * dx
        + 2.0 * Ayx[:, None] * dy * dx
    )


@dataclass
class GSStats:
    ema_grad_xyz: Tensor  # (N,1)
    ema_grad_a: Tensor    # (N,1)
    decay: float = 0.95

    @staticmethod
    def init(n: int, device: torch.device, dtype: torch.dtype, decay: float = 0.95) -> "GSStats":
        return GSStats(
            ema_grad_xyz=torch.zeros((n, 1), device=device, dtype=dtype),
            ema_grad_a=torch.zeros((n, 1), device=device, dtype=dtype),
            decay=float(decay),
        )

    def ensure_size(self, n: int) -> None:
        if self.ema_grad_xyz.shape[0] != n:
            self.ema_grad_xyz = _resize_like(self.ema_grad_xyz, n)
        if self.ema_grad_a.shape[0] != n:
            self.ema_grad_a = _resize_like(self.ema_grad_a, n)

    def remap_from_old(self, source_index: Tensor) -> "GSStats":
        """Copy EMA statistics through densify/prune using new->old source indices."""
        idx = source_index.to(device=self.ema_grad_xyz.device, dtype=torch.long).view(-1)
        idx = idx.clamp(0, max(self.ema_grad_xyz.shape[0] - 1, 0))
        return GSStats(
            ema_grad_xyz=self.ema_grad_xyz[idx].clone(),
            ema_grad_a=self.ema_grad_a[idx].clone(),
            decay=float(self.decay),
        )


class GaussianChiModel(nn.Module):
    """Additive QSM Gaussian field.

    Parameters:
      xyz       (N,3): centers in voxel coords (z,y,x)
      log_sigma (N,3): axis-aligned Gaussian scale, sigma=exp(log_sigma)
      a         (N,1): signed susceptibility amplitude

    There is intentionally no alpha/opacity parameter in this QSM version.
    For an additive chi field, a signed amplitude already controls positive,
    negative, and near-zero contribution.
    """

    def __init__(
        self,
        xyz: Tensor,
        sigma: Tensor,
        a: Tensor,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e2,
        rot_q: Optional[Tensor] = None,
        use_rotation: bool = True,
    ):
        super().__init__()
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError(f"xyz must be (N,3), got {tuple(xyz.shape)}")
        if sigma.ndim != 2 or sigma.shape[1] != 3:
            raise ValueError(f"sigma must be (N,3), got {tuple(sigma.shape)}")
        if a.ndim != 2 or a.shape[1] != 1:
            raise ValueError(f"a must be (N,1), got {tuple(a.shape)}")
        if not (xyz.shape[0] == sigma.shape[0] == a.shape[0]):
            raise ValueError("xyz/sigma/a must have the same N")
        if rot_q is not None and (rot_q.ndim != 2 or rot_q.shape[1] != 4 or rot_q.shape[0] != xyz.shape[0]):
            raise ValueError(f"rot_q must be (N,4), got {tuple(rot_q.shape)}")

        # sigma limits are not hard-clamped in forward; they are enforced by prune.
        # Keep only a tiny positive lower bound for numerical log/division safety.
        self.sigma_min = float(max(float(sigma_min), 1.0e-6))
        self.sigma_max = float(sigma_max)
        self.xyz = nn.Parameter(xyz)
        sigma = torch.clamp(sigma, min=1.0e-6)
        self.log_sigma = nn.Parameter(torch.log(sigma))
        self.a = nn.Parameter(a)
        self.use_rotation = bool(use_rotation)
        if rot_q is None:
            rot_q = _identity_quat(xyz.shape[0], xyz.device, xyz.dtype)
        self.rot_q = nn.Parameter(_normalize_quat(rot_q.to(device=xyz.device, dtype=xyz.dtype)))
        self._offset_cache: Dict[Tuple[int, str, int, torch.dtype], Tuple[Tensor, Tensor]] = {}
        self._check_n()

    @property
    def device(self) -> torch.device:
        return self.xyz.device

    @property
    def dtype(self) -> torch.dtype:
        return self.xyz.dtype

    def num_points(self) -> int:
        return int(self.xyz.shape[0])

    def sigma(self) -> Tensor:
        # No upper clamp: over-large or over-small Gaussians are removed by prune.
        return torch.exp(self.log_sigma).clamp_min(1.0e-6)

    def rotation(self) -> Tensor:
        return _normalize_quat(self.rot_q)

    def rotmat(self) -> Tensor:
        return _quat_to_rotmat(self.rotation())

    def amplitude(self) -> Tensor:
        return self.a

    def _voxel_size_vec(self, voxel_size_mm_zyx: Optional[Tuple[float, float, float]] = None) -> Tensor:
        """Return voxel size as (z,y,x) tensor on the model device.

        A value of None means isotropic unit voxels. The model keeps xyz and
        sigma in voxel-coordinate units; render-time formulas can convert both
        offsets and scales into physical mm units for anisotropic data.
        """
        if voxel_size_mm_zyx is None:
            vals = (1.0, 1.0, 1.0)
        else:
            vals = tuple(float(v) for v in voxel_size_mm_zyx)
            if len(vals) != 3:
                raise ValueError(f"voxel_size_mm_zyx must have 3 values, got {vals}")
            vals = tuple(max(float(v), 1.0e-6) for v in vals)
        return torch.tensor(vals, device=self.device, dtype=self.dtype)

    def contribution_metric(
        self,
        *,
        normalize_kernel: bool = True,
        voxel_size_mm_zyx: Optional[Tuple[float, float, float]] = None,
    ) -> Tensor:
        """Contribution score used for logging/pruning.

        With normalized splatting, a Gaussian is a discrete basis whose weights
        sum to one inside its valid support, so |a| is the correct first-order
        total contribution. With unnormalized splatting, total mass scales with
        the physical Gaussian volume, so |a| * prod(sigma_mm) is used.
        """
        amp = self.a.abs().view(-1)
        if bool(normalize_kernel):
            return amp
        vox = self._voxel_size_vec(voxel_size_mm_zyx)
        sig_mm = self.sigma() * vox[None, :]
        return amp * sig_mm.prod(dim=1)

    def contribution_volume(self) -> Tensor:
        # Backward-compatible alias. For the current normalized basis this is |a|.
        return self.contribution_metric(normalize_kernel=True)

    def _check_n(self) -> None:
        n = self.xyz.shape[0]
        if self.log_sigma.shape[0] != n or self.a.shape[0] != n or self.rot_q.shape[0] != n:
            raise RuntimeError(
                f"Inconsistent N: xyz={self.xyz.shape[0]}, log_sigma={self.log_sigma.shape[0]}, "
                f"a={self.a.shape[0]}, rot_q={self.rot_q.shape[0]}"
            )

    def _offsets_for_rmax(self, rmax: int) -> Tuple[Tensor, Tensor]:
        """Cached local integer offsets for the fixed rmax splat window."""
        dev_index = -1 if self.device.index is None else int(self.device.index)
        key = (int(rmax), self.device.type, dev_index, self.dtype)
        cached = self._offset_cache.get(key, None)
        if cached is not None:
            return cached
        rng = torch.arange(-int(rmax), int(rmax) + 1, device=self.device, dtype=self.dtype)
        DZ, DY, DX = torch.meshgrid(rng, rng, rng, indexing="ij")
        offs = torch.stack([DZ, DY, DX], dim=-1).reshape(-1, 3).contiguous()
        offs_i = offs.to(torch.long).contiguous()
        self._offset_cache[key] = (offs, offs_i)
        return offs, offs_i

    def splat_gaussian_local(
        self,
        vol_shape_zyx: Tuple[int, int, int],
        mask: Optional[Tensor] = None,
        *,
        rmax: int = 4,
        radius_factor: float = 3.0,
        normalize_kernel: bool = False,
        mask_aware_normalization: bool = True,
        voxel_size_mm_zyx: Optional[Tuple[float, float, float]] = None,
        voxel_size_aware: bool = True,
        chunk: int = 8192,
    ) -> Tensor:
        Z, Y, X = map(int, vol_shape_zyx)
        N = self.num_points()
        chi = torch.zeros((Z, Y, X), device=self.device, dtype=self.dtype)
        chi_flat = chi.view(-1)

        xyz = self.xyz
        sig = self.sigma().clamp_min(1.0e-6)
        amp = self.amplitude().view(-1, 1)
        rot = self.rotation()
        vox = self._voxel_size_vec(voxel_size_mm_zyx)
        if bool(voxel_size_aware):
            # Precision and offsets are evaluated in physical mm space. The
            # stored sigma remains in voxel-coordinate units for compatibility
            # with existing initialization and prune thresholds.
            sig_for_precision = sig * vox[None, :]
        else:
            sig_for_precision = sig
        precision = _precision_elements_from_sigma_rot(sig_for_precision, rot, self.use_rotation)

        _, offs_i = self._offsets_for_rmax(int(rmax))
        r2_gate = float(radius_factor) ** 2
        ctr = torch.round(xyz).to(torch.long)
        mask_flat = None
        if mask is not None:
            mask_flat = mask.reshape(-1).to(device=self.device, dtype=self.dtype)

        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            c = ctr[s:e]
            p = xyz[s:e]
            a = amp[s:e]
            prec = tuple(v[s:e] for v in precision)

            pos = c[:, None, :] + offs_i[None, :, :]
            z, y, x = pos[..., 0], pos[..., 1], pos[..., 2]
            inb = (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X)
            idx = (z * (Y * X) + y * X + x).to(torch.long).clamp(0, Z * Y * X - 1)

            if bool(voxel_size_aware):
                d = (pos.to(self.dtype) - p[:, None, :]) * vox.view(1, 1, 3)
            else:
                d = pos.to(self.dtype) - p[:, None, :]
            d2 = _quadratic_form_from_precision(d, prec)
            valid = inb
            if mask_flat is not None and bool(mask_aware_normalization):
                mvals = mask_flat[idx].reshape_as(inb) > 0.5
                valid = valid & mvals

            w = torch.exp(-0.5 * d2) * (d2 <= r2_gate).to(self.dtype) * valid.to(self.dtype)
            if normalize_kernel:
                # Normalize over valid in-bounds/mask voxels. This avoids losing
                # mass at brain-mask boundaries and makes a_i the total signed
                # contribution of the discrete basis.
                w = w / torch.clamp(w.sum(dim=1, keepdim=True), min=1.0e-12)

            val = (a * w).reshape(-1)
            idx_flat = idx.reshape(-1).to(torch.long)
            valid_flat = valid.reshape(-1)
            if valid_flat.any():
                chi_flat.scatter_add_(0, idx_flat[valid_flat], val[valid_flat])

        chi = chi_flat.view(Z, Y, X)
        if mask is not None:
            chi = chi * mask
        return chi

    def forward_qsm(
        self,
        vol_shape_zyx: Tuple[int, int, int],
        fwd,
        mask: Optional[Tensor] = None,
        *,
        padded: bool = True,
        render: Literal["gaussian_local"] = "gaussian_local",
        rmax: int = 4,
        radius_factor: float = 3.0,
        normalize_kernel: bool = False,
        mask_aware_normalization: bool = True,
        voxel_size_mm_zyx: Optional[Tuple[float, float, float]] = None,
        voxel_size_aware: bool = True,
        chunk: int = 8192,
    ) -> Tuple[Tensor, Tensor]:
        if render != "gaussian_local":
            raise ValueError("Only render='gaussian_local' is supported.")
        chi = self.splat_gaussian_local(
            vol_shape_zyx=vol_shape_zyx,
            mask=mask,
            rmax=rmax,
            radius_factor=radius_factor,
            normalize_kernel=normalize_kernel,
            mask_aware_normalization=mask_aware_normalization,
            voxel_size_mm_zyx=voxel_size_mm_zyx,
            voxel_size_aware=voxel_size_aware,
            chunk=chunk,
        )
        phi = fwd.apply(chi, padded=padded)
        if mask is not None:
            phi = phi * mask
        return chi, phi

    @torch.no_grad()
    def update_stats_from_grads(self, stats: GSStats) -> None:
        n = self.num_points()
        stats.ensure_size(n)
        d = float(stats.decay)
        if self.xyz.grad is not None and self.xyz.grad.shape[0] == n:
            gxyz = _safe_norm(self.xyz.grad, dim=1, keepdim=True)
        else:
            gxyz = torch.zeros((n, 1), device=self.device, dtype=self.dtype)
        if self.a.grad is not None and self.a.grad.shape[0] == n:
            ga = self.a.grad.abs()
        else:
            ga = torch.zeros((n, 1), device=self.device, dtype=self.dtype)
        stats.ema_grad_xyz.mul_(d).add_(gxyz * (1.0 - d))
        stats.ema_grad_a.mul_(d).add_(ga * (1.0 - d))

    @torch.no_grad()
    def _sample_map_nearest(self, vol: Tensor) -> Tensor:
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {tuple(vol.shape)}")
        Z, Y, X = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
        ctr = torch.round(self.xyz).to(torch.long)
        z, y, x = ctr[:, 0], ctr[:, 1], ctr[:, 2]
        inb = (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X)
        vals = torch.zeros((self.num_points(),), device=self.device, dtype=self.dtype)
        if inb.any():
            idx = (z[inb] * (Y * X) + y[inb] * X + x[inb]).to(torch.long)
            vals[inb] = vol.reshape(-1)[idx].to(device=self.device, dtype=self.dtype)
        return vals

    @torch.no_grad()
    def _sample_map_gaussian_footprint_mean(
        self,
        vol: Tensor,
        *,
        rmax: int = 3,
        radius_factor: float = 2.8,
        normalize_kernel: bool = True,
        voxel_size_mm_zyx: Optional[Tuple[float, float, float]] = None,
        voxel_size_aware: bool = True,
        chunk: int = 4096,
    ) -> Tensor:
        """Per-Gaussian weighted mean of a 3D map over the Gaussian footprint.

        This is used only at densify steps, so it improves spatial assignment of
        residual without affecting every training iteration.
        """
        if vol.ndim != 3:
            raise ValueError(f"Expected 3D volume, got {tuple(vol.shape)}")
        Z, Y, X = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
        N = self.num_points()
        out = torch.zeros((N,), device=self.device, dtype=self.dtype)
        vol_flat = vol.reshape(-1).to(device=self.device, dtype=self.dtype)
        xyz = self.xyz
        sig = self.sigma().clamp_min(1.0e-6)
        rot = self.rotation()
        vox = self._voxel_size_vec(voxel_size_mm_zyx)
        sig_for_precision = sig * vox[None, :] if bool(voxel_size_aware) else sig
        precision = _precision_elements_from_sigma_rot(sig_for_precision, rot, self.use_rotation)
        _, offs_i = self._offsets_for_rmax(int(rmax))
        r2_gate = float(radius_factor) ** 2
        ctr = torch.round(xyz).to(torch.long)
        for s in range(0, N, int(chunk)):
            e = min(s + int(chunk), N)
            c = ctr[s:e]
            p = xyz[s:e]
            prec = tuple(v[s:e] for v in precision)
            pos = c[:, None, :] + offs_i[None, :, :]
            z, y, x = pos[..., 0], pos[..., 1], pos[..., 2]
            inb = (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X)
            d = pos.to(self.dtype) - p[:, None, :]
            d2 = _quadratic_form_from_precision(d, prec)
            w = torch.exp(-0.5 * d2) * (d2 <= r2_gate).to(self.dtype) * inb.to(self.dtype)
            idx = (z * (Y * X) + y * X + x).to(torch.long).clamp(0, Z * Y * X - 1)
            vals = vol_flat[idx] * inb.to(self.dtype)
            denom = w.sum(dim=1).clamp_min(1.0e-12)
            if normalize_kernel:
                out[s:e] = (w * vals).sum(dim=1) / denom
            else:
                out[s:e] = (w * vals).sum(dim=1) / denom
        return out

    @torch.no_grad()
    def _sample_vector_nearest(self, vec_zyx: Optional[Tensor]) -> Optional[Tensor]:
        if vec_zyx is None:
            return None
        if vec_zyx.ndim != 4 or vec_zyx.shape[0] != 3:
            raise ValueError(f"Expected vector map (3,Z,Y,X), got {tuple(vec_zyx.shape)}")
        Z, Y, X = int(vec_zyx.shape[1]), int(vec_zyx.shape[2]), int(vec_zyx.shape[3])
        ctr = torch.round(self.xyz).to(torch.long)
        z, y, x = ctr[:, 0], ctr[:, 1], ctr[:, 2]
        inb = (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X)
        out = torch.zeros((self.num_points(), 3), device=self.device, dtype=self.dtype)
        if inb.any():
            idx = (z[inb] * (Y * X) + y[inb] * X + x[inb]).to(torch.long)
            flat = vec_zyx.reshape(3, -1).to(device=self.device, dtype=self.dtype)
            out[inb] = flat[:, idx].T
            nrm = _safe_norm(out[inb], dim=1, keepdim=True)
            out[inb] = out[inb] / nrm.clamp_min(1.0e-6)
        return out

    @torch.no_grad()
    def _sample_hessian_tangent_direction(self, vol: Tensor, idx_points: Tensor) -> Optional[Tensor]:
        """Estimate a local structure tangent direction from scalar guide volume.

        For line-like or elongated structures, the Hessian eigenvector with the
        smallest absolute curvature is a cheap tangent proxy. This is evaluated
        only for selected densify parents, not for all voxels.
        """
        if vol is None or idx_points is None or idx_points.numel() == 0:
            return None
        if vol.ndim != 3:
            return None
        Z, Y, X = int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2])
        v = vol.to(device=self.device, dtype=self.dtype)
        ctr = torch.round(self.xyz[idx_points]).to(torch.long)
        z0 = ctr[:, 0].clamp(1, max(Z - 2, 1))
        y0 = ctr[:, 1].clamp(1, max(Y - 2, 1))
        x0 = ctr[:, 2].clamp(1, max(X - 2, 1))

        def smp(dz: int, dy: int, dx: int) -> Tensor:
            z = (z0 + int(dz)).clamp(0, Z - 1)
            y = (y0 + int(dy)).clamp(0, Y - 1)
            x = (x0 + int(dx)).clamp(0, X - 1)
            return v[z, y, x]

        c = smp(0, 0, 0)
        fzp, fzm = smp(1, 0, 0), smp(-1, 0, 0)
        fyp, fym = smp(0, 1, 0), smp(0, -1, 0)
        fxp, fxm = smp(0, 0, 1), smp(0, 0, -1)
        dzz = fzp - 2.0 * c + fzm
        dyy = fyp - 2.0 * c + fym
        dxx = fxp - 2.0 * c + fxm
        dzy = 0.25 * (smp(1, 1, 0) - smp(1, -1, 0) - smp(-1, 1, 0) + smp(-1, -1, 0))
        dzx = 0.25 * (smp(1, 0, 1) - smp(1, 0, -1) - smp(-1, 0, 1) + smp(-1, 0, -1))
        dyx = 0.25 * (smp(0, 1, 1) - smp(0, 1, -1) - smp(0, -1, 1) + smp(0, -1, -1))

        H = torch.zeros((idx_points.numel(), 3, 3), device=self.device, dtype=torch.float32)
        H[:, 0, 0] = dzz.float(); H[:, 1, 1] = dyy.float(); H[:, 2, 2] = dxx.float()
        H[:, 0, 1] = H[:, 1, 0] = dzy.float()
        H[:, 0, 2] = H[:, 2, 0] = dzx.float()
        H[:, 1, 2] = H[:, 2, 1] = dyx.float()

        vals, vecs = torch.linalg.eigh(H)
        choice = vals.abs().argmin(dim=1)
        dirs = torch.gather(vecs, 2, choice.view(-1, 1, 1).expand(-1, 3, 1)).squeeze(-1).to(self.dtype)

        # If the Hessian is nearly flat, fall back to local gradient normal.
        gz = 0.5 * (fzp - fzm)
        gy = 0.5 * (fyp - fym)
        gx = 0.5 * (fxp - fxm)
        grad = torch.stack([gz, gy, gx], dim=1).to(self.dtype)
        flat = vals.abs().sum(dim=1).to(self.dtype) < 1.0e-8
        if flat.any():
            dirs[flat] = grad[flat]
        nrm = _safe_norm(dirs, dim=1, keepdim=True).clamp_min(1.0e-6)
        return dirs / nrm

    @torch.no_grad()
    def _sample_direction_for_indices(self, guide: Optional[Tensor], idx_points: Tensor, mode: str = "grad_normal") -> Optional[Tensor]:
        if guide is None or idx_points is None or idx_points.numel() == 0:
            return None
        mode = str(mode).lower()
        if guide.ndim == 4:
            all_dirs = self._sample_vector_nearest(guide)
            return all_dirs[idx_points] if all_dirs is not None else None
        if guide.ndim == 3 and mode in ("hessian", "hessian_tangent", "tangent", "structure"):
            return self._sample_hessian_tangent_direction(guide, idx_points)
        return None

    @torch.no_grad()
    def densify_clone_split(
        self,
        stats: GSStats,
        *,
        residual_abs_map: Tensor,
        mask: Optional[Tensor],
        metric: Literal["xyz_grad", "a_grad", "xyz_a_grad"] = "xyz_grad",
        grad_threshold_mode: str = "mad",
        grad_abs_min: float = 0.0,
        grad_mad_k: float = 2.5,
        grad_percentile: float = 75.0,
        residual_sample_mode: str = "footprint",
        residual_rmax: int = 3,
        residual_radius_factor: float = 2.8,
        voxel_size_mm_zyx: Optional[Tuple[float, float, float]] = None,
        voxel_size_aware: bool = True,
        residual_threshold_mode: str = "mad",
        residual_abs_min: float = 0.0,
        residual_mad_k: float = 1.5,
        residual_percentile: float = 70.0,
        sigma_threshold_mode: str = "quantile",
        clone_sigma_max: float = 1.20,
        split_sigma_min: float = 1.35,
        clone_sigma_quantile: float = 0.60,
        split_sigma_quantile: float = 0.78,
        clone_k: int = 1,
        split_k: int = 2,
        add_max: int = 900,
        max_points: int = 300_000,
        clone_budget_frac: float = 0.50,
        clone_first: bool = False,
        allow_split: bool = True,
        allow_clone: bool = True,
        clone_jitter_scale: float = 0.15,
        split_jitter_scale: float = 0.45,
        split_sigma_scale_children: float = 0.65,
        direction_map: Optional[Tensor] = None,
        directional_split: bool = False,
        directional_mode: str = "grad_normal",
        directional_clone: bool = False,
        directional_strength: float = 0.75,
        directional_clone_strength: float = 0.75,
        directional_random_scale: float = 0.25,
    ) -> Dict[str, object]:
        n0 = self.num_points()
        stats.ensure_size(n0)
        empty_map = torch.arange(n0, device=self.device, dtype=torch.long)
        if n0 >= int(max_points) or int(add_max) <= 0:
            return {"added": 0, "clone_parents": 0, "split_parents": 0, "grad_thr": 0.0, "res_thr": 0.0, "source_index": empty_map}

        gx = stats.ema_grad_xyz.view(-1)
        ga = stats.ema_grad_a.view(-1)
        metric = str(metric).lower()
        if metric == "xyz_grad":
            grad_score = gx
        elif metric == "a_grad":
            grad_score = ga
        elif metric == "xyz_a_grad":
            gx_ref = torch.quantile(gx[gx >= 0], 0.95).clamp_min(1.0e-12) if (gx >= 0).any() else torch.tensor(1.0, device=self.device, dtype=self.dtype)
            ga_ref = torch.quantile(ga[ga >= 0], 0.95).clamp_min(1.0e-12) if (ga >= 0).any() else torch.tensor(1.0, device=self.device, dtype=self.dtype)
            grad_score = gx / gx_ref + 0.5 * ga / ga_ref
        else:
            raise ValueError(f"Unknown densify metric: {metric}")

        if str(residual_sample_mode).lower() == "footprint":
            residual_local = self._sample_map_gaussian_footprint_mean(
                residual_abs_map,
                rmax=int(residual_rmax),
                radius_factor=float(residual_radius_factor),
                normalize_kernel=True,
                chunk=4096,
            ).clamp_min(0.0)
        else:
            residual_local = self._sample_map_nearest(residual_abs_map).clamp_min(0.0)

        valid = torch.isfinite(grad_score) & torch.isfinite(residual_local)
        if mask is not None:
            valid &= self._sample_map_nearest(mask) > 0.5
        if not valid.any():
            return {"added": 0, "clone_parents": 0, "split_parents": 0, "grad_thr": 0.0, "res_thr": 0.0, "source_index": empty_map}

        grad_thr = _adaptive_threshold(
            grad_score[valid],
            mode=grad_threshold_mode,
            abs_min=float(grad_abs_min),
            mad_k=float(grad_mad_k),
            percentile=float(grad_percentile),
        )
        res_thr = _adaptive_threshold(
            residual_local[valid],
            mode=residual_threshold_mode,
            abs_min=float(residual_abs_min),
            mad_k=float(residual_mad_k),
            percentile=float(residual_percentile),
        )

        sig = self.sigma().clamp_min(1.0e-6)
        sigma_max = sig.max(dim=1).values
        if str(sigma_threshold_mode).lower() == "quantile":
            v_sig = sigma_max[valid]
            clone_thr = torch.quantile(v_sig, min(max(float(clone_sigma_quantile), 0.0), 1.0))
            split_thr = torch.quantile(v_sig, min(max(float(split_sigma_quantile), 0.0), 1.0))
            if split_thr < clone_thr:
                split_thr = clone_thr
        else:
            clone_thr = torch.tensor(float(clone_sigma_max), device=self.device, dtype=self.dtype)
            split_thr = torch.tensor(float(split_sigma_min), device=self.device, dtype=self.dtype)

        cand = valid & (grad_score > grad_thr) & (residual_local > res_thr)
        clone_mask = cand & (sigma_max <= clone_thr) if bool(allow_clone) else torch.zeros_like(cand)
        split_mask = cand & (sigma_max >= split_thr) if bool(allow_split) else torch.zeros_like(cand)

        # Combined score is only used to cap accepted operations when candidates exceed budget.
        g_ref = torch.quantile(grad_score[valid].clamp_min(0), 0.95).clamp_min(1.0e-12)
        r_ref = torch.quantile(residual_local[valid].clamp_min(0), 0.95).clamp_min(1.0e-12)
        combined = (grad_score / g_ref).clamp_min(0) * (residual_local / r_ref).clamp_min(0)

        net_budget = min(int(add_max), int(max_points) - n0)
        if net_budget <= 0:
            return {"added": 0, "clone_parents": 0, "split_parents": 0, "grad_thr": float(grad_thr), "res_thr": float(res_thr), "source_index": empty_map}

        split_idx = torch.nonzero(split_mask, as_tuple=False).view(-1)
        clone_idx = torch.nonzero(clone_mask, as_tuple=False).view(-1)
        if split_idx.numel() > 0:
            split_idx = split_idx[torch.argsort(combined[split_idx], descending=True)]
        if clone_idx.numel() > 0:
            clone_idx = clone_idx[torch.argsort(combined[clone_idx], descending=True)]

        split_net = max(1, int(split_k) - 1)
        clone_net = int(clone_k)
        clone_budget_frac = min(max(float(clone_budget_frac), 0.0), 1.0)
        clone_budget = int(round(net_budget * clone_budget_frac))
        split_budget = net_budget - clone_budget

        if clone_first:
            max_clone = min(clone_idx.numel(), clone_budget // clone_net if clone_net > 0 else 0)
            clone_idx = clone_idx[:max_clone]
            remaining = net_budget - int(clone_idx.numel()) * clone_net
            max_split = min(split_idx.numel(), remaining // split_net)
            split_idx = split_idx[:max_split]
            # If split was scarce, use remaining for clone.
            remaining2 = net_budget - int(clone_idx.numel()) * clone_net - int(split_idx.numel()) * split_net
            if remaining2 > 0 and clone_idx.numel() < torch.nonzero(clone_mask, as_tuple=False).numel():
                all_clone = torch.nonzero(clone_mask, as_tuple=False).view(-1)
                if all_clone.numel() > 0:
                    all_clone = all_clone[torch.argsort(combined[all_clone], descending=True)]
                    extra = all_clone[int(clone_idx.numel()): int(clone_idx.numel()) + remaining2 // max(clone_net, 1)]
                    clone_idx = torch.cat([clone_idx, extra], dim=0) if extra.numel() else clone_idx
        else:
            max_split = min(split_idx.numel(), split_budget // split_net)
            split_idx = split_idx[:max_split]
            remaining = net_budget - int(split_idx.numel()) * split_net
            max_clone = min(clone_idx.numel(), remaining // clone_net if clone_net > 0 else 0)
            clone_idx = clone_idx[:max_clone]
            remaining2 = net_budget - int(split_idx.numel()) * split_net - int(clone_idx.numel()) * clone_net
            if remaining2 > 0 and split_idx.numel() < torch.nonzero(split_mask, as_tuple=False).numel():
                all_split = torch.nonzero(split_mask, as_tuple=False).view(-1)
                if all_split.numel() > 0:
                    all_split = all_split[torch.argsort(combined[all_split], descending=True)]
                    extra = all_split[int(split_idx.numel()): int(split_idx.numel()) + remaining2 // split_net]
                    split_idx = torch.cat([split_idx, extra], dim=0) if extra.numel() else split_idx

        if split_idx.numel() == 0 and clone_idx.numel() == 0:
            return {"added": 0, "clone_parents": 0, "split_parents": 0, "grad_thr": float(grad_thr), "res_thr": float(res_thr), "source_index": empty_map}

        old_xyz = self.xyz.data.clone()
        old_sig = sig.data.clone()
        old_a = self.a.data.clone()
        old_rot = self.rotation().data.clone()
        keep = torch.ones((n0,), device=self.device, dtype=torch.bool)
        a_work = old_a.clone()

        children_xyz = []
        children_sig = []
        children_a = []
        children_rot = []
        children_src = []

        if clone_idx.numel() > 0:
            P = clone_idx.numel()
            ck = int(clone_k)
            divisor = float(ck + 1)
            parent_xyz = old_xyz[clone_idx]
            parent_sig = old_sig[clone_idx]
            parent_a = old_a[clone_idx]
            parent_rot = old_rot[clone_idx]
            a_work[clone_idx] = parent_a / divisor
            dirs = None
            if bool(directional_clone) and bool(directional_split) and direction_map is not None:
                dirs = self._sample_direction_for_indices(direction_map, clone_idx, mode=str(directional_mode))
            if dirs is not None:
                signs = torch.where(
                    torch.rand((P, ck, 1), device=self.device, dtype=self.dtype) > 0.5,
                    torch.ones((P, ck, 1), device=self.device, dtype=self.dtype),
                    -torch.ones((P, ck, 1), device=self.device, dtype=self.dtype),
                )
                sigma_proj = torch.sqrt(((dirs * parent_sig) ** 2).sum(dim=1, keepdim=True)).clamp_min(1.0e-6)
                directed = signs * dirs[:, None, :] * sigma_proj[:, None, :] * float(clone_jitter_scale) * float(directional_clone_strength)
                rnd = torch.randn((P, ck, 3), device=self.device, dtype=self.dtype) * parent_sig[:, None, :] * float(clone_jitter_scale) * float(directional_random_scale)
                offset = directed + rnd
            else:
                offset_local = torch.randn((P, ck, 3), device=self.device, dtype=self.dtype) * parent_sig[:, None, :] * float(clone_jitter_scale)
                if self.use_rotation and bool(voxel_size_aware):
                    vox = self._voxel_size_vec(voxel_size_mm_zyx)
                    offset_local_mm = offset_local * vox.view(1, 1, 3)
                    offset = _local_to_world(offset_local_mm, parent_rot) / vox.view(1, 1, 3)
                else:
                    offset = _local_to_world(offset_local, parent_rot) if self.use_rotation else offset_local
            child_xyz = parent_xyz[:, None, :].repeat(1, ck, 1) + offset
            child_sig = parent_sig[:, None, :].repeat(1, ck, 1)
            child_a = parent_a[:, None, :].repeat(1, ck, 1) / divisor
            child_rot = parent_rot[:, None, :].repeat(1, ck, 1)
            children_xyz.append(child_xyz.reshape(-1, 3))
            children_sig.append(child_sig.reshape(-1, 3))
            children_a.append(child_a.reshape(-1, 1))
            children_rot.append(child_rot.reshape(-1, 4))
            children_src.append(clone_idx[:, None].repeat(1, ck).reshape(-1))

        if split_idx.numel() > 0:
            P = split_idx.numel()
            sk = int(split_k)
            parent_xyz = old_xyz[split_idx]
            parent_sig = old_sig[split_idx]
            parent_a = old_a[split_idx]
            parent_rot = old_rot[split_idx]
            keep[split_idx] = False

            if bool(directional_split) and direction_map is not None:
                dirs = self._sample_direction_for_indices(direction_map, split_idx, mode=str(directional_mode))
            else:
                dirs = None

            if dirs is not None and sk == 2:
                signs = torch.tensor([-1.0, 1.0], device=self.device, dtype=self.dtype).view(1, 2, 1)
                sigma_proj = torch.sqrt(((dirs * parent_sig) ** 2).sum(dim=1, keepdim=True)).clamp_min(1.0e-6)
                directed = signs * dirs[:, None, :] * sigma_proj[:, None, :] * float(split_jitter_scale) * float(directional_strength)
                rnd = torch.randn((P, sk, 3), device=self.device, dtype=self.dtype) * parent_sig[:, None, :] * float(split_jitter_scale) * float(directional_random_scale)
                offset = directed + rnd
            else:
                offset_local = torch.randn((P, sk, 3), device=self.device, dtype=self.dtype) * parent_sig[:, None, :] * float(split_jitter_scale)
                if self.use_rotation and bool(voxel_size_aware):
                    vox = self._voxel_size_vec(voxel_size_mm_zyx)
                    offset_local_mm = offset_local * vox.view(1, 1, 3)
                    offset = _local_to_world(offset_local_mm, parent_rot) / vox.view(1, 1, 3)
                else:
                    offset = _local_to_world(offset_local, parent_rot) if self.use_rotation else offset_local

            child_xyz = parent_xyz[:, None, :].repeat(1, sk, 1) + offset
            child_sig = parent_sig[:, None, :].repeat(1, sk, 1) * float(split_sigma_scale_children)
            child_a = parent_a[:, None, :].repeat(1, sk, 1) / float(sk)
            child_rot = parent_rot[:, None, :].repeat(1, sk, 1)
            children_xyz.append(child_xyz.reshape(-1, 3))
            children_sig.append(child_sig.reshape(-1, 3))
            children_a.append(child_a.reshape(-1, 1))
            children_rot.append(child_rot.reshape(-1, 4))
            children_src.append(split_idx[:, None].repeat(1, sk).reshape(-1))

        source_old = torch.arange(n0, device=self.device, dtype=torch.long)[keep].contiguous()
        xyz_new = old_xyz[keep].contiguous()
        sig_new = old_sig[keep].contiguous()
        a_new = a_work[keep].contiguous()
        rot_new = old_rot[keep].contiguous()
        if children_xyz:
            xyz_new = torch.cat([xyz_new] + children_xyz, dim=0)
            sig_new = torch.cat([sig_new] + children_sig, dim=0).clamp_min(1.0e-6)
            a_new = torch.cat([a_new] + children_a, dim=0)
            rot_new = torch.cat([rot_new] + children_rot, dim=0)
            source_old = torch.cat([source_old] + children_src, dim=0)

        self.xyz = nn.Parameter(xyz_new)
        self.log_sigma = nn.Parameter(torch.log(sig_new.clamp_min(1.0e-6)))
        self.a = nn.Parameter(a_new)
        self.rot_q = nn.Parameter(_normalize_quat(rot_new))
        self._check_n()

        return {
            "added": int(self.num_points() - n0),
            "clone_parents": int(clone_idx.numel()),
            "split_parents": int(split_idx.numel()),
            "grad_thr": float(grad_thr.detach().cpu()),
            "res_thr": float(res_thr.detach().cpu()),
            "clone_sigma_thr": float(clone_thr.detach().cpu()),
            "split_sigma_thr": float(split_thr.detach().cpu()),
            "source_index": source_old,
        }

    @torch.no_grad()
    def prune(
        self,
        *,
        mask: Optional[Tensor] = None,
        a_min: float = 1e-6,
        contribution_min: float = 1e-8,
        sigma_min: float = 1e-3,
        sigma_max: float = 1e2,
        keep_at_least: int = 100,
        normalize_kernel: bool = True,
        voxel_size_mm_zyx: Optional[Tuple[float, float, float]] = None,
    ) -> Dict[str, object]:
        self._check_n()
        n0 = self.num_points()
        if n0 <= int(keep_at_least):
            return {"pruned": 0, "source_index": torch.arange(n0, device=self.device, dtype=torch.long)}

        amp = self.a.abs().view(-1)
        sig = self.sigma().clamp_min(1.0e-6)
        contrib = self.contribution_metric(normalize_kernel=normalize_kernel, voxel_size_mm_zyx=voxel_size_mm_zyx)
        keep = torch.ones((n0,), device=self.device, dtype=torch.bool)
        keep &= amp >= float(a_min)
        keep &= contrib >= float(contribution_min)
        keep &= (sig[:, 0] >= float(sigma_min)) & (sig[:, 1] >= float(sigma_min)) & (sig[:, 2] >= float(sigma_min))
        keep &= (sig[:, 0] <= float(sigma_max)) & (sig[:, 1] <= float(sigma_max)) & (sig[:, 2] <= float(sigma_max))

        if mask is not None:
            keep &= self._sample_map_nearest(mask) > 0.5

        if int(keep.sum().item()) < int(keep_at_least):
            k = min(int(keep_at_least), n0)
            top = torch.topk(contrib, k=k, largest=True).indices
            keep = torch.zeros_like(keep)
            keep[top] = True

        src = torch.arange(n0, device=self.device, dtype=torch.long)[keep].contiguous()
        self._keep_points(keep)
        self._check_n()
        return {"pruned": int(n0 - self.num_points()), "source_index": src}

    @torch.no_grad()
    def _keep_points(self, keep: Tensor) -> None:
        keep = keep.to(device=self.device, dtype=torch.bool)
        if keep.ndim != 1 or keep.shape[0] != self.num_points():
            raise ValueError(f"keep must be (N,), got {tuple(keep.shape)}")
        xyz_new = self.xyz.data[keep].contiguous()
        sig_new = self.sigma().data[keep].contiguous().clamp_min(1.0e-6)
        a_new = self.a.data[keep].contiguous()
        rot_new = self.rotation().data[keep].contiguous()
        self.xyz = nn.Parameter(xyz_new)
        self.log_sigma = nn.Parameter(torch.log(sig_new))
        self.a = nn.Parameter(a_new)
        self.rot_q = nn.Parameter(_normalize_quat(rot_new))


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Z, Y, X = 32, 32, 32
    N = 512
    xyz = torch.stack([torch.rand(N, device=dev) * (Z - 1), torch.rand(N, device=dev) * (Y - 1), torch.rand(N, device=dev) * (X - 1)], dim=1)
    sigma = torch.full((N, 3), 1.0, device=dev)
    a = torch.randn((N, 1), device=dev) * 0.01
    m = GaussianChiModel(xyz, sigma, a, sigma_min=0.1, sigma_max=3.0).to(dev)
    chi = m.splat_gaussian_local((Z, Y, X), rmax=2, chunk=256)
    loss = (chi * chi).mean()
    loss.backward()
    stats = GSStats.init(m.num_points(), dev, torch.float32)
    m.update_stats_from_grads(stats)
    info = m.densify_clone_split(stats, residual_abs_map=chi.detach().abs(), mask=None, add_max=32, max_points=1024)
    print("ok", float(loss), m.num_points(), info)
