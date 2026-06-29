from __future__ import annotations

from typing import Dict, Literal, Optional, Tuple

import torch

Tensor = torch.Tensor


def charbonnier(x: Tensor, eps: float = 1e-3) -> Tensor:
    return torch.sqrt(x * x + eps * eps)


def data_fidelity(
    phi_pred: Tensor,
    phi_gt: Tensor,
    mask: Optional[Tensor] = None,
    mode: Literal["l1", "l2", "charbonnier"] = "l1",
    charbonnier_eps: float = 1e-3,
) -> Tensor:
    e = phi_pred - phi_gt
    if mask is not None:
        e = e * mask
    if mode == "l1":
        return e.abs().mean()
    if mode == "l2":
        return (e * e).mean()
    if mode == "charbonnier":
        return charbonnier(e, eps=charbonnier_eps).mean()
    raise ValueError(f"Unknown data fidelity mode: {mode}")


def tv_loss_3d(
    chi: Tensor,
    mask: Optional[Tensor] = None,
    mode: Literal["anisotropic", "isotropic"] = "anisotropic",
    eps: float = 1e-6,
) -> Tensor:
    dz = chi[..., 1:, :, :] - chi[..., :-1, :, :]
    dy = chi[..., :, 1:, :] - chi[..., :, :-1, :]
    dx = chi[..., :, :, 1:] - chi[..., :, :, :-1]

    if mask is not None:
        dz = dz * (mask[..., 1:, :, :] * mask[..., :-1, :, :])
        dy = dy * (mask[..., :, 1:, :] * mask[..., :, :-1, :])
        dx = dx * (mask[..., :, :, 1:] * mask[..., :, :, :-1])

    if mode == "anisotropic":
        return dz.abs().mean() + dy.abs().mean() + dx.abs().mean()
    if mode == "isotropic":
        dz_c = dz[..., :, :-1, :-1]
        dy_c = dy[..., :-1, :, :-1]
        dx_c = dx[..., :-1, :-1, :]
        g = torch.sqrt(dz_c * dz_c + dy_c * dy_c + dx_c * dx_c + eps)
        return g.mean()
    raise ValueError(f"Unknown TV mode: {mode}")


def param_regularization(model, sigma_w: float = 0.0, a_l2_w: float = 0.0) -> Tensor:
    reg = torch.zeros((), device=model.xyz.device, dtype=model.xyz.dtype)
    if sigma_w > 0.0:
        reg = reg + float(sigma_w) * model.sigma().mean()
    if a_l2_w > 0.0:
        reg = reg + float(a_l2_w) * model.a.pow(2).mean()
    return reg


def total_loss(
    phi_pred: Tensor,
    phi_gt: Tensor,
    chi: Tensor,
    mask: Optional[Tensor],
    model,
    data_mode: Literal["l1", "l2", "charbonnier"] = "l1",
    data_w: float = 1.0,
    tv_w: float = 0.0,
    tv_mode: Literal["anisotropic", "isotropic"] = "anisotropic",
    sigma_w: float = 0.0,
    a_l2_w: float = 0.0,
    charbonnier_eps: float = 1e-3,
) -> Tuple[Tensor, Dict[str, float]]:
    l_data = data_fidelity(phi_pred, phi_gt, mask=mask, mode=data_mode, charbonnier_eps=charbonnier_eps)
    l_tv = tv_loss_3d(chi, mask=mask, mode=tv_mode) if tv_w > 0 else chi.new_zeros(())
    l_reg = param_regularization(model, sigma_w=sigma_w, a_l2_w=a_l2_w) if (sigma_w > 0 or a_l2_w > 0) else chi.new_zeros(())
    loss = float(data_w) * l_data + float(tv_w) * l_tv + l_reg
    return loss, {
        "loss": float(loss.detach().item()),
        "data": float(l_data.detach().item()),
        "tv": float(l_tv.detach().item()) if tv_w > 0 else 0.0,
        "reg": float(l_reg.detach().item()) if (sigma_w > 0 or a_l2_w > 0) else 0.0,
    }


def grad_norm_report(model) -> Dict[str, float]:
    def _mean_abs(g: Optional[Tensor]) -> float:
        if g is None:
            return 0.0
        return float(g.detach().abs().mean().item())
    return {
        "grad_xyz": _mean_abs(model.xyz.grad),
        "grad_a": _mean_abs(model.a.grad),
        "grad_log_sigma": _mean_abs(model.log_sigma.grad),
    }
