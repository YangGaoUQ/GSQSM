from __future__ import annotations

from typing import Optional, Tuple

import torch

from config import Config



def _finite_difference_l2_3d(chi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Quadratic first-order finite-difference smoothness inside mask.

  This is the FD part used together with anisotropic TV. TV handles sparse
  jumps/edges; FD gives a stronger smooth pull in flat noisy regions.
  """
  m = mask.to(device=chi.device, dtype=chi.dtype).clamp(0.0, 1.0)
  dz = chi[1:, :, :] - chi[:-1, :, :]
  dy = chi[:, 1:, :] - chi[:, :-1, :]
  dx = chi[:, :, 1:] - chi[:, :, :-1]
  mz = m[1:, :, :] * m[:-1, :, :]
  my = m[:, 1:, :] * m[:, :-1, :]
  mx = m[:, :, 1:] * m[:, :, :-1]
  nz = mz.sum().clamp_min(1.0)
  ny = my.sum().clamp_min(1.0)
  nx = mx.sum().clamp_min(1.0)
  return 0.3333333333 * (
    (dz.pow(2) * mz).sum() / nz +
    (dy.pow(2) * my).sum() / ny +
    (dx.pow(2) * mx).sum() / nx
  )




def _masked_quantile_torch(x: torch.Tensor, mask: torch.Tensor, q: float, default: float = 1.0) -> torch.Tensor:
  vals = x[(mask > 0) & torch.isfinite(x)]
  if vals.numel() < 16:
    return torch.tensor(float(default), device=x.device, dtype=x.dtype)
  qq = max(0.0, min(1.0, float(q) / 100.0))
  return torch.quantile(vals.float(), qq).to(device=x.device, dtype=x.dtype)


def _masked_tv_l1_3d(chi: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  """Anisotropic TV normalized only by valid in-mask finite-difference edges.

  The generic tv_loss_3d averages over the full dense volume after masking out
  invalid edges. For QSM masks with a small brain ratio this dilutes the TV term.
  CFR's shrinkage acts on gradient variables directly, so this normalized TV
  is a closer Adam-loss approximation of the same regularization pressure.
  """
  m = mask.to(device=chi.device, dtype=chi.dtype).clamp(0.0, 1.0)
  dz = chi[1:, :, :] - chi[:-1, :, :]
  dy = chi[:, 1:, :] - chi[:, :-1, :]
  dx = chi[:, :, 1:] - chi[:, :, :-1]
  mz = m[1:, :, :] * m[:-1, :, :]
  my = m[:, 1:, :] * m[:, :-1, :]
  mx = m[:, :, 1:] * m[:, :, :-1]
  return (
    (dz.abs() * mz).sum() / mz.sum().clamp_min(1.0) +
    (dy.abs() * my).sum() / my.sum().clamp_min(1.0) +
    (dx.abs() * mx).sum() / mx.sum().clamp_min(1.0)
  )


def _cfr_stage2_weight_from_discrepancy(
  *,
  phi_raw_n: torch.Tensor,
  phi_init_n: torch.Tensor,
  mask: torch.Tensor,
  cfg: Config,
  base_weight: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
  """stage2 discrepancy weight with an optional robust max.

  Public MATLAB code:

      params.weight = mask .* (mag_use / max(mag_use(:)))
      dphi = abs(input - D(x_stage1)) * mask
      dphi = dphi / max(dphi(:))
      weight = weight .* weight .* (1-dphi) .* (1-dphi)

  This implementation keeps exactly the same form.  The only optional
  improvement is replacing max(dphi) by a high percentile denominator, because
  a single boundary/outlier voxel can otherwise make almost every dphi too
  small and leave reliability too bright.  Set
  cfg.cfr.dphi_norm_mode = "max" for the exact MATLAB normalization.
  """
  with torch.no_grad():
    m = mask.to(device=phi_raw_n.device, dtype=phi_raw_n.dtype).clamp(0.0, 1.0)
    dphi = (phi_raw_n.detach() - phi_init_n.detach()).abs() * m
    valid = (m > 0) & torch.isfinite(dphi)

    if bool(valid.any()):
      exact_max = dphi[valid].max().clamp_min(1.0e-6)
    else:
      exact_max = torch.tensor(1.0, device=phi_raw_n.device, dtype=phi_raw_n.dtype)

    norm_mode = str(getattr(cfg.cfr, "dphi_norm_mode", "robust_max")).lower()
    if norm_mode in ("max", "original"):
      denom = exact_max
    elif norm_mode in ("robust_max", "quantile", "percentile"):
      q = float(getattr(cfg.cfr, "dphi_norm_percentile", 99.5))
      denom = _masked_quantile_torch(dphi, m, q=q, default=float(exact_max.detach().cpu())).clamp_min(1.0e-6)
      # Never use a denominator larger than the exact max. This keeps the
      # normalized dphi at least as strong as the original max normalization.
      denom = torch.minimum(denom, exact_max).clamp_min(1.0e-6)
    else:
      raise ValueError("cfg.cfr.dphi_norm_mode must be 'max' or 'robust_max'")

    # Match max-normalization behavior by forcing dphi into [0, 1].
    # This is critical when robust_max/gain is used; otherwise (1-dphi)^2 would
    # incorrectly turn dphi>1 outliers back into high reliability.
    dphi01 = (dphi / denom).clamp(0.0, 1.0)

    # reliability can be made stricter without changing its form by
    # amplifying the normalized discrepancy before applying (1-dphi)^2.  The
    # default >1 is useful when no magnitude weight is available and mask is the
    # only base weight.  Set to 1.0 for exact discrepancy strength.
    gain = float(getattr(cfg.cfr, "dphi_gain", 1.5))
    if gain != 1.0:
      dphi01 = (dphi01 * gain).clamp(0.0, 1.0)

    if base_weight is None:
      # No magnitude fallback:
      # Treat the missing magnitude-derived term as a global constant
      # normalized_mag^2 = 0.15 inside the brain mask.
      # Because reliability below is computed as bw^2 * (1-dphi)^2,
      # bw itself must be sqrt(0.15), not 0.15.
      no_mag_mag2 = 0.4
      bw = torch.sqrt(torch.as_tensor(no_mag_mag2, device=m.device, dtype=m.dtype)) * m
    else:
      bw = base_weight.to(device=phi_raw_n.device, dtype=phi_raw_n.dtype)
      if bw.shape != m.shape:
        raise ValueError(f"base_weight shape {tuple(bw.shape)} does not match mask shape {tuple(m.shape)}")
      bw = torch.nan_to_num(bw, nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0) * m
      # If the magnitude-derived weight is empty/invalid, use the same
      # no-magnitude fallback instead of reverting to mask=1.
      if float(bw[m > 0].max().detach().cpu()) <= 1.0e-6:
        no_mag_mag2 = 0.4
        bw = torch.sqrt(torch.as_tensor(no_mag_mag2, device=m.device, dtype=m.dtype)) * m
      

    reliability = bw.pow(2) * (1.0 - dphi01).pow(2)
    # Reliability must remain a true gate in [0, 1].
    reliability = reliability.clamp(0.0, 1.0)

    return reliability.detach(), dphi01.detach()


def cfr_stage2_chi_denoise(
  *,
  chi_init: torch.Tensor,
  phi_raw_n: torch.Tensor,
  fwd,
  mask: torch.Tensor,
  phi_scale: Optional[torch.Tensor] = None,
  base_weight: Optional[torch.Tensor] = None,
  cfg: Config,
  verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
  """Full-chi CFR stage2 with raw-LFS target and strong TV/FD.

  The optimization is an Adam approximation of stage2:

      min_chi data_w * loss(R * (D(chi) - raw_LFS))
            + tv_w   * TV(chi)
            + fd_w   * FD_L2(chi)

  where R = base_weight^2 * (1 - dphi)^2 is the stage2 reliability.
  base_weight is magnitude-derived when available; when magnitude is missing, normalized_mag^2 is fixed to 0.2 inside the mask.
  chi_init is the robust stage1 result, i.e. chi_ref in this pipeline. CFR
  directly optimizes chi. There is no residual/delta mode and no ref-target
  blending. The output chi_cfr can be used directly as the final output or as
  the staged CFR reference for the training feedback loss in train.py.
  """
  steps = int(getattr(cfg.cfr, "iters", 30))
  m = mask.to(device=chi_init.device, dtype=chi_init.dtype).clamp(0.0, 1.0)
  chi0 = chi_init.detach().to(dtype=phi_raw_n.dtype) * m
  scale = (
    torch.as_tensor(1.0, device=chi_init.device, dtype=phi_raw_n.dtype)
    if phi_scale is None
    else torch.as_tensor(phi_scale, device=chi_init.device, dtype=phi_raw_n.dtype).clamp_min(1.0e-6)
  )

  with torch.no_grad():
    phi_init_n = fwd.apply(chi0, padded=True) / scale
    reliability, dphi_init = _cfr_stage2_weight_from_discrepancy(
      phi_raw_n=phi_raw_n.detach(),
      phi_init_n=phi_init_n.detach(),
      mask=m,
      cfg=cfg,
      base_weight=base_weight,
    )
    mode = str(getattr(cfg.cfr, "weight_mode", "reliability")).lower()
    if mode in ("mask", "none", "uniform"):
      w = m.clone()
    elif mode in ("reliability", "discrepancy"):
      w = reliability.clone()
    else:
      raise ValueError("cfg.cfr.weight_mode must be 'reliability' or 'mask'")

  hist = {"loss": 0.0, "data": 0.0, "tv": 0.0, "fd": 0.0, "accepted": 1.0}
  if steps <= 0:
    return chi0.detach(), w.detach(), dphi_init.detach(), reliability.detach(), hist

  loss_mode = str(getattr(cfg.cfr, "loss", "l2")).lower()
  if loss_mode not in ("l1", "l2", "charbonnier"):
    raise ValueError("cfg.cfr.loss must be one of: l1, l2, charbonnier")

  data_w = float(getattr(cfg.cfr, "data_w", 0.06))
  tv_w = float(getattr(cfg.cfr, "tv_w", 2.0e-2))
  fd_w = float(getattr(cfg.cfr, "fd_w", 1.0e-2))
  grad_clip = getattr(cfg.cfr, "grad_clip", 0.75)
  clipv = getattr(cfg.cfr, "clip", None)

  # Do not square the reliability again. In the original code the
  # squared terms are already inside: weight = base_weight^2 * (1-dphi)^2.
  w_data = w.clamp(0.0, 1.0)

  var = torch.nn.Parameter(chi0.clone())
  opt = torch.optim.Adam([var], lr=float(getattr(cfg.cfr, "lr", 8.0e-4)))

  def _data_loss(phi_pred: torch.Tensor) -> torch.Tensor:
    r = phi_pred - phi_raw_n.detach()
    norm_mode = str(getattr(cfg.cfr, "data_norm_mode", "mask")).lower()
    if norm_mode in ("weight", "weighted", "old"):
      denom = w_data.sum().clamp_min(1.0e-6)
    elif norm_mode in ("mask", "absolute"):
      denom = m.sum().clamp_min(1.0e-6)
    else:
      raise ValueError("cfg.cfr.data_norm_mode must be 'mask' or 'weight'")
    if loss_mode == "l1":
      return (r.abs() * w_data).sum() / denom
    if loss_mode == "charbonnier":
      eps = float(getattr(cfg.cfr, "charbonnier_eps", 1.0e-3))
      return (torch.sqrt(r.pow(2) + eps * eps) * w_data).sum() / denom
    return 0.5 * (r.pow(2) * w_data).sum() / denom

  for s in range(1, steps + 1):
    opt.zero_grad(set_to_none=True)
    chi_m = var * m
    phi = fwd.apply(chi_m, padded=True) / scale
    data = _data_loss(phi)
    tv = _masked_tv_l1_3d(chi_m, m) if tv_w > 0.0 else chi_m.new_zeros(())
    fd = _finite_difference_l2_3d(chi_m, m) if fd_w > 0.0 else chi_m.new_zeros(())
    loss = data_w * data + tv_w * tv + fd_w * fd
    loss.backward()
    if grad_clip is not None and float(grad_clip) > 0.0:
      torch.nn.utils.clip_grad_norm_([var], float(grad_clip))
    opt.step()
    with torch.no_grad():
      var.data.mul_(m)
      if clipv is not None and float(clipv) > 0.0:
        var.data.clamp_(-float(clipv), float(clipv))

    hist = {
      "loss": float(loss.detach()),
      "data": float(data.detach()),
      "tv": float(tv.detach()),
      "fd": float(fd.detach()),
      "accepted": 1.0,
    }
    if verbose and (s == 1 or s == steps or s % 10 == 0):
      print(
        f"[CFR] step={s:03d}/{steps} "
        f"loss={hist['loss']:.6e} data={hist['data']:.6e} "
        f"tv={hist['tv']:.6e} fd={hist['fd']:.6e}"
      )

  chi_out = (var.detach() * m)
  return chi_out.detach(), w.detach(), dphi_init.detach(), reliability.detach(), hist
