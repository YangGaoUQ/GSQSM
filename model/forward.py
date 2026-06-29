# forward.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from utils.tool import pad3d_torch, unpad3d_torch, fft3, ifft3

Tensor = torch.Tensor


def _unit3(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    a, b, c = float(v[0]), float(v[1]), float(v[2])
    n = (a * a + b * b + c * c) ** 0.5
    if n <= 1e-12:
        raise ValueError("B0_dir_zyx has near-zero norm.")
    return (a / n, b / n, c / n)


def _to_float32(x: Tensor) -> Tensor:
    return x if x.dtype in (torch.float32, torch.float64) else x.float()


def _as_complex(x: Tensor) -> Tensor:
    x = _to_float32(x)
    return torch.complex(x, torch.zeros_like(x))


# -------------------------
# dipole kernel
# -------------------------

def build_dipole_kernel(
    shape_zyx: Tuple[int, int, int],
    voxel_size_mm_zyx: Tuple[float, float, float],
    B0_dir_zyx: Tuple[float, float, float] = (0.0, 0.0, 1.0),
    device: Union[str, torch.device] = "cuda",
    dtype: torch.dtype = torch.float32,
    eps: float = 1e-12,
) -> Tensor:
    Z, Y, X = int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2])
    vz, vy, vx = float(voxel_size_mm_zyx[0]), float(voxel_size_mm_zyx[1]), float(voxel_size_mm_zyx[2])
    bz, by, bx = _unit3(B0_dir_zyx)

    dev = torch.device(device)

    kz = torch.fft.fftfreq(Z, d=vz, device=dev, dtype=torch.float32)
    ky = torch.fft.fftfreq(Y, d=vy, device=dev, dtype=torch.float32)
    kx = torch.fft.fftfreq(X, d=vx, device=dev, dtype=torch.float32)

    KZ, KY, KX = torch.meshgrid(kz, ky, kx, indexing="ij")

    k2 = KZ * KZ + KY * KY + KX * KX
    k_dot_b = KZ * bz + KY * by + KX * bx

    D = (1.0 / 3.0) - (k_dot_b * k_dot_b) / (k2 + float(eps))
    D = torch.where(k2 > 0, D, torch.zeros_like(D))

    # kernel complex
    if dtype == torch.float64:
        Dr = D.to(torch.float64)
        return torch.complex(Dr, torch.zeros_like(Dr))
    Dr = D.to(torch.float32)
    return torch.complex(Dr, torch.zeros_like(Dr))


# -------------------------
# forward / adjoint / tkd
# -------------------------

def forward_field(chi: Tensor, dipole_k: Tensor) -> Tensor:
    chi_c = _as_complex(chi)
    Chi = fft3(chi_c)
    Phi = Chi * dipole_k
    return ifft3(Phi).real


def adjoint_field(residual_phi: Tensor, dipole_k: Tensor) -> Tensor:
    """
    g = D^T r
    FFT: ifft( fft(r) * conj(D) ).real
    dipole_k 本身实数核 -> conj 无影响，但这里写全。
    """
    r_c = _as_complex(residual_phi)
    R = fft3(r_c)
    G = R * torch.conj(dipole_k)
    return ifft3(G).real


def tkd_inverse(phi: Tensor, dipole_k: Tensor, thresh: float = 0.15) -> Tensor:
    D = dipole_k.real
    m = D.abs() > float(thresh)

    invD = torch.zeros_like(D)
    invD[m] = 1.0 / D[m]
    invD_c = torch.complex(invD, torch.zeros_like(invD))

    phi_c = _as_complex(phi)
    Phi = fft3(phi_c)
    Chi = Phi * invD_c
    return ifft3(Chi).real


# -------------------------
# cached op (padding + kernel cache)
# -------------------------

@dataclass
class ForwardMeta:
    base_shape_zyx: Tuple[int, int, int]
    padded_shape_zyx: Tuple[int, int, int]
    voxel_size_mm_zyx: Tuple[float, float, float]
    B0_dir_zyx: Tuple[float, float, float]
    pad_enable: bool
    pad_size: int
    pad_mode: str
    pad_constant_value: float
    tkd_thresh: float
    eps: float


class ForwardOp:
    def __init__(
        self,
        voxel_size_mm_zyx: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        B0_dir_zyx: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        pad_enable: bool = True,
        pad_size: int = 16,
        pad_mode: str = "replicate",
        pad_constant_value: float = 0.0,
        tkd_thresh: float = 0.15,
        eps: float = 1e-12,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        self.voxel_size_mm_zyx = (float(voxel_size_mm_zyx[0]), float(voxel_size_mm_zyx[1]), float(voxel_size_mm_zyx[2]))
        self.B0_dir_zyx = _unit3((float(B0_dir_zyx[0]), float(B0_dir_zyx[1]), float(B0_dir_zyx[2])))

        self.pad_enable = bool(pad_enable)
        self.pad_size = int(pad_size)
        self.pad_mode = str(pad_mode)
        self.pad_constant_value = float(pad_constant_value)

        self.tkd_thresh = float(tkd_thresh)
        self.eps = float(eps)

        self.device = torch.device(device)
        self.dtype = dtype

        self._kernel_cache: Dict[Tuple[int, int, int], Tensor] = {}

    def _get_kernel(self, shape_zyx: Tuple[int, int, int]) -> Tensor:
        key = (int(shape_zyx[0]), int(shape_zyx[1]), int(shape_zyx[2]))
        k = self._kernel_cache.get(key)
        if k is None:
            k = build_dipole_kernel(
                shape_zyx=key,
                voxel_size_mm_zyx=self.voxel_size_mm_zyx,
                B0_dir_zyx=self.B0_dir_zyx,
                device=self.device,
                dtype=self.dtype,
                eps=self.eps,
            )
            self._kernel_cache[key] = k
        return k

    def pad(self, vol: Tensor) -> Tensor:
        if not self.pad_enable or self.pad_size <= 0:
            return vol
        if self.pad_mode not in ("replicate", "reflect", "constant"):
            raise ValueError(f"pad_mode must be replicate/reflect/constant, got: {self.pad_mode}")
        return pad3d_torch(vol, self.pad_size, mode=self.pad_mode, value=self.pad_constant_value)

    def unpad(self, vol: Tensor) -> Tensor:
        if not self.pad_enable or self.pad_size <= 0:
            return vol
        return unpad3d_torch(vol, self.pad_size)

    def apply(self, chi: Tensor, padded: bool = True) -> Tensor:
        if padded and self.pad_enable and self.pad_size > 0:
            chi_p = self.pad(chi)
            Dp = self._get_kernel(tuple(int(x) for x in chi_p.shape[-3:]))
            phi_p = forward_field(chi_p, Dp)
            return self.unpad(phi_p)

        Db = self._get_kernel(tuple(int(x) for x in chi.shape[-3:]))
        return forward_field(chi, Db)

    def apply_adjoint(self, residual_phi: Tensor, padded: bool = True) -> Tensor:
        """
        g = D^T residual
        用于 DC：chi <- chi - eta * D^T( D chi - phi_gt )
        """
        if padded and self.pad_enable and self.pad_size > 0:
            r_p = self.pad(residual_phi)
            Dp = self._get_kernel(tuple(int(x) for x in r_p.shape[-3:]))
            g_p = adjoint_field(r_p, Dp)
            return self.unpad(g_p)

        Db = self._get_kernel(tuple(int(x) for x in residual_phi.shape[-3:]))
        return adjoint_field(residual_phi, Db)

    def tkd(self, phi: Tensor, padded: bool = True, thresh: Optional[float] = None) -> Tensor:
        t = self.tkd_thresh if thresh is None else float(thresh)

        if padded and self.pad_enable and self.pad_size > 0:
            phi_p = self.pad(phi)
            Dp = self._get_kernel(tuple(int(x) for x in phi_p.shape[-3:]))
            chi_p = tkd_inverse(phi_p, Dp, thresh=t)
            return self.unpad(chi_p)

        Db = self._get_kernel(tuple(int(x) for x in phi.shape[-3:]))
        return tkd_inverse(phi, Db, thresh=t)

    def meta(self, base_shape_zyx: Tuple[int, int, int]) -> ForwardMeta:
        Z, Y, X = int(base_shape_zyx[0]), int(base_shape_zyx[1]), int(base_shape_zyx[2])
        if self.pad_enable and self.pad_size > 0:
            p = self.pad_size
            padded_shape = (Z + 2 * p, Y + 2 * p, X + 2 * p)
        else:
            padded_shape = (Z, Y, X)

        return ForwardMeta(
            base_shape_zyx=(Z, Y, X),
            padded_shape_zyx=padded_shape,
            voxel_size_mm_zyx=self.voxel_size_mm_zyx,
            B0_dir_zyx=self.B0_dir_zyx,
            pad_enable=self.pad_enable,
            pad_size=self.pad_size,
            pad_mode=self.pad_mode,
            pad_constant_value=self.pad_constant_value,
            tkd_thresh=self.tkd_thresh,
            eps=self.eps,
        )


# -------------------------
# cfg helper
# -------------------------

def build_forward_from_cfg(cfg_phys: Any, device: str, dtype_str: str) -> ForwardOp:
    dtype = torch.float16 if str(dtype_str).lower() == "float16" else torch.float32

    B0_dir = getattr(cfg_phys, "B0_dir_zyx", None)
    if B0_dir is None:
        B0_dir = getattr(cfg_phys, "B0_dir", (0.0, 0.0, 1.0))

    voxel = getattr(cfg_phys, "voxel_size_mm_zyx", None)
    if voxel is None:
        voxel = getattr(cfg_phys, "voxel_size_mm", (1.0, 1.0, 1.0))

    eps = getattr(cfg_phys, "eps", None)
    if eps is None:
        eps = getattr(cfg_phys, "dipole_eps", 1e-12)

    return ForwardOp(
        voxel_size_mm_zyx=tuple(voxel),
        B0_dir_zyx=tuple(B0_dir),
        pad_enable=bool(getattr(cfg_phys, "pad_enable", True)),
        pad_size=int(getattr(cfg_phys, "pad_size", 16)),
        pad_mode=str(getattr(cfg_phys, "pad_mode", "replicate")),
        pad_constant_value=float(getattr(cfg_phys, "pad_constant_value", 0.0)),
        tkd_thresh=float(getattr(cfg_phys, "tkd_thresh", 0.15)),
        eps=float(eps),
        device=device,
        dtype=dtype,
    )
