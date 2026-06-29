from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple
import os
import json

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


@dataclass
class IOCfg:
    phi_path: str = "./dataset/vivo/Sim1_1/lfs.nii"
    mask_path: Optional[str] = None
    magnitude_path: Optional[str] = ""
    out_dir: str = "./deepMRI/gs30/Sim1_1"
    run_name: str = "gsqsm"
    device: str = "cuda"
    dtype: str = "float32"


@dataclass
class MaskCfg:
    mask_min_ratio: float = 0.03
    mask_max_ratio: float = 0.70
    mask_full_ratio: float = 0.92
    mask_target_ratio: float = 0.20
    mask_hole_thresh: float = 0.025
    mask_smooth_sigma: float = 2.0
    mask_open_iter: int = 1
    mask_close_iter: int = 2
    mask_dilate_iter: int = 0
    mask_erode_iter: int = 0
    mask_force_rebuild: bool = False
    mask_keep_largest_only: bool = True
    mask_min_component_voxels: int = 128


@dataclass
class BBoxCfg:
    # Large-volume mask bounding-box crop. Training is done inside the crop;
    # saved chi_gs is pasted back to the original full shape.
    enable: bool = True
    force: bool = True
    voxel_threshold: int = 12_000_000
    margin: int = 6
    save_debug: bool = True


@dataclass
class PhysCfg:
    voxel_size_mm: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    B0_dir_zyx: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    pad_enable: bool = True
    pad_size: int = 8
    pad_mode: str = "replicate"
    pad_constant_value: float = 0.0
    tkd_thresh: float = 0.15
    eps: float = 1e-12


@dataclass
class InitCfg:
    n_init: int = 60_000
    sample_mode: str = "mask_edge_hybrid"
    chi0_mode: str = "tkd"

    topk_ratio: float = 0.006
    hybrid_top_frac: float = 0.10

    edge_uniform_frac: float = 0.62
    edge_grad_frac: float = 0.26
    edge_log_frac: float = 0.12
    edge_topk_ratio: float = 0.08
    log_topk_ratio: float = 0.05
    edge_score_smooth_sigma: float = 1.80
    log_score_smooth_sigma: float = 2.00
    edge_sigma_scale: float = 0.75
    log_sigma_scale: float = 0.85

    mask_init_value: float = 0.005
    phi_init_normalize: bool = True
    chi0_blur_kernel: int = 5
    init_sigma_mm: float = 1.00
    init_sigma_jitter: float = 0.10
    init_a_scale: float = 0.075
    chi0_clip: Optional[float] = 0.025


@dataclass
class RenderCfg:
    # Local splat window. If auto_rmax=True, rmax is updated from a high
    # sigma quantile and clipped by max_rmax, avoiding both sigma truncation
    # and large speed regressions from rare huge Gaussians.
    rmax: int = 4
    auto_rmax: bool = False
    auto_rmax_quantile: float = 0.90
    min_rmax: int = 4
    max_rmax: int = 4
    radius_factor: float = 2.8
    normalize_kernel: bool = True

    # Formula-correct discrete basis: normalize only over valid brain-mask
    # voxels, and evaluate rotated covariance in physical mm coordinates.
    mask_aware_normalization: bool = True
    voxel_size_aware: bool = True

    chunk: int = 2048


@dataclass
class LossCfg:
    use_mask: bool = True
    data: str = "charbonnier"
    data_w: float = 1.0
    tv_w: float = 2e-5
    sigma_w: float = 2.0e-5
    a_l2_w: float = 0.0
    noise_kernel: int = 5


@dataclass
class DensifyCfg:
    enable: bool = True

    # Stage lengths can be defined as fractions of cfg.train.iters.
    # Stage 1: split+clone mixed, mainly decomposes coarse blobs.
    # Stage 2: clone-priority, mainly adds fine-line capacity.
    schedule_by_train_iters: bool = True
    start_frac: float = 0.10
    split_until_frac: float = 0.35
    clone_until_frac: float = 0.65
    from_iter: int = 80
    until_iter: int = 520
    every: int = 30

    # Gradient + residual threshold; top-k is not used as the decision rule.
    # If candidates are too many, add_max/max_points only cap accepted operations.
    metric: str = "xyz_grad"  # xyz_grad | a_grad | xyz_a_grad
    ema_decay: float = 0.95
    grad_threshold_mode: str = "mad"  # mad | percentile | abs
    grad_abs_min: float = 0.0
    grad_mad_k: float = 2.0
    grad_percentile: float = 70.0

    residual_smooth_kernel: int = 3
    residual_sample_mode: str = "footprint"  # center | footprint
    residual_threshold_mode: str = "mad"  # mad | percentile | abs
    residual_abs_min: float = 0.0
    residual_mad_k: float = 1.2
    residual_percentile: float = 65.0

    # Clone/split separation. Quantile mode is more robust than a fixed mm/voxel cut.
    sigma_threshold_mode: str = "quantile"  # fixed | quantile
    clone_sigma_max: float = 1.20
    split_sigma_min: float = 1.35
    clone_sigma_quantile: float = 0.65
    split_sigma_quantile: float = 0.82

    clone_k: int = 1
    split_k: int = 2
    add_max: int = 900
    max_points: int = 90_000
    stage1_clone_budget_frac: float = 0.35
    stage2_clone_budget_frac: float = 0.80
    clone_jitter_scale: float = 0.15
    split_jitter_scale: float = 0.45
    split_sigma_scale_children: float = 0.65

    # v4 uses learnable rotation-covariance, so directional child placement is off by default.
    # It can still be enabled for ablation, but rotation itself already defines child offsets.
    directional_split: bool = False
    directional_mode: str = "hessian_tangent"  # hessian_tangent | grad_normal
    directional_clone: bool = False
    directional_strength: float = 1.15
    directional_clone_strength: float = 0.90
    directional_random_scale: float = 0.10

    # Prune uses a contribution metric consistent with normalized splatting:python recon.py --phi D:/workpy/matlab/QSM/dataset/vivo/CAA/lfs.nii --out D:/workpy/matlab/QSM/deepMRI/gs27/CAA
    # with normalize_kernel=True, contribution≈|a| rather than |a|*volume.
    # Sigma limits are enforced by prune, not by hard forward-time clamp.
    prune_warmup: int = 120
    prune_every: int = 40
    prune_outside_mask: bool = True
    prune_a_min: float = 5.0e-7
    prune_contribution_min: float = 5.0e-9
    prune_sigma_min: float = 0.10
    prune_sigma_max: float = 3.4
    keep_at_least: int = 42_000

    # After clone/split stages end, continue cheap prune-only checks.
    # This follows the GS idea of removing invalid/oversized Gaussians while
    # preventing late sigma drift without adding new points.
    prune_only_after_density: bool = True


@dataclass
class TrainCfg:
    iters: int = 300
    seed: int = 0
    lr_xyz: float = 1.0e-2
    lr_sigma: float = 5.0e-3
    lr_a: float = 5.0e-3
    lr_rot: float = 8.0e-4
    rotation_enable: bool = True
    rotation_from_iter: int = 100
    rotation_from_frac: float = 0.35
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None

    # Simple exponential LR schedule: xyz/sigma decay more than signed amplitude.
    lr_schedule_enable: bool = True
    lr_schedule: str = "exp"  # exp | cosine
    lr_xyz_final_ratio: float = 0.12
    lr_sigma_final_ratio: float = 0.20
    lr_a_final_ratio: float = 0.55
    lr_rot_final_ratio: float = 0.35

    # Optional late amplitude-only polishing. Disabled by default for clean ablation.
    a_only_from_frac: float = 0.0

    log_every: int = 10
    save_every: int = 300
    save_optimizer_state: bool = False


@dataclass
class UNetCfg:
    # Optional 2-channel Swin-UNet refiner. Default is enabled for the full pipeline.
    # input_mode="residual" uses [chi_gs, raw_lfs - D(chi_gs)] as the two channels.
    # input_mode="mask" keeps the older [chi_gs, mask] input for ablation.
    enable: bool = True
    from_iter: int = 40
    every: int = 10

    input_mode: str = "residual"  # residual | mask
    residual_input_clip_abs: float = 3.0
    detach_residual_input: bool = True

    in_channels: int = 2
    base_channels: int = 8
    depth: int = 4
    norm: str = "group"
    act: str = "lrelu"
    gn_groups: int = 8
    dropout: float = 0.0
    bias: bool = False
    use_transpose: bool = True
    linear_embed_norm: bool = True
    linear_embed_bias: bool = True

    refiner_version: str = "v2"
    swin_placement: str = "all"
    swin_each_layer: bool = True
    swin_first_window_size: int = 2
    swin_window_size: int = 4
    swin_num_heads: int = 2
    swin_first_blocks: int = 1
    swin_first_decoder_blocks: int = 1
    swin_blocks: int = 2
    swin_mlp_ratio: float = 2.0
    swin_drop: float = 0.0
    swin_attn_drop: float = 0.0

    refine_mode: str = "fixed_ratio"
    fixed_ref_ratio: float = 1.30
    residual_scale: float = 1.20
    tanh_tau: float = 8.0
    delta_clip_abs: float = 0.70
    zero_init_last: bool = True
    gate_bias_init: float = 0.0
    only_update_in_mask: bool = True
    gate_use_input_prior: bool = False
    gate_prior_scale: float = 0.0
    gate_smooth_kernel: int = 3
    gate_cap: float = 1.20
    gate_confidence_floor: float = 0.0
    gate_confidence_channel: int = 1
    chi_clip: Optional[float] = None

    lr: float = 2.0e-3
    weight_decay: float = 0.0
    grad_clip: Optional[float] = None
    data: str = "charbonnier"
    ref_raw_w: float = 0.90
    save_diagnostics: bool = False
    save_suffix: str = "swin_refiner"


@dataclass
class CFRCfg:
    # CFR stage2 chi-domain denoiser. During training it is refreshed
    # periodically and its detached result is used as a standard loss term.
    enable: bool = True

    # Internal CFR optimization.
    iters: int = 30
    lr: float = 8.0e-4
    data_w: float = 0.06
    tv_w: float = 1.5e-2
    fd_w: float = 1.0e-3
    loss: str = "l2"
    charbonnier_eps: float = 1.0e-3
    weight_mode: str = "reliability"  # reliability | mask
    dphi_norm_mode: str = "robust_max"  # robust_max | max
    dphi_norm_percentile: float = 99.5
    dphi_gain: float = 1.5
    data_norm_mode: str = "mask"  # mask | weight
    clip: Optional[float] = None
    grad_clip: Optional[float] = 0.75
    save_weight: bool = False
    save_residual_diag: bool = False

    # Training feedback from the periodically refreshed CFR result.
    # The CFR result is detached; gradients flow only through the current
    # chi_ref / chi_gs branch, so this avoids unrolling CFR optimization.
    from_iter: int = 50
    update_every: int = 50
    feedback_enable: bool = True
    feedback_target: str = "ref"      # ref | gs
    feedback_loss: str = "charbonnier"  # l1 | l2 | charbonnier
    feedback_eps: float = 1.0e-3
    feedback_w_early: float = 3.0e-2   # from_iter <= epoch < feedback_mid_iter
    feedback_w_mid: float = 8.0e-2     # feedback_mid_iter <= epoch < feedback_late_iter
    feedback_w_late: float = 1.2e-1    # epoch >= feedback_late_iter
    feedback_mid_iter: int = 150
    feedback_late_iter: int = 250
    feedback_weight_floor: float = 0.0


@dataclass
class Config:
    io: IOCfg = field(default_factory=IOCfg)
    mask: MaskCfg = field(default_factory=MaskCfg)
    bbox: BBoxCfg = field(default_factory=BBoxCfg)
    phys: PhysCfg = field(default_factory=PhysCfg)
    init: InitCfg = field(default_factory=InitCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    loss: LossCfg = field(default_factory=LossCfg)
    densify: DensifyCfg = field(default_factory=DensifyCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    unet: UNetCfg = field(default_factory=UNetCfg)
    cfr: CFRCfg = field(default_factory=CFRCfg)


def _deep_update_dataclass(obj: Any, d: Dict[str, Any]) -> None:
    for k, v in d.items():
        if not hasattr(obj, k):
            continue
        cur = getattr(obj, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _deep_update_dataclass(cur, v)
        else:
            setattr(obj, k, v)


def load_config(path: str) -> Config:
    ext = os.path.splitext(path)[1].lower()
    if ext not in (".json", ".yml", ".yaml"):
        raise ValueError(f"Unsupported config format: {ext}")
    with open(path, "r", encoding="utf-8") as f:
        if ext == ".json":
            d = json.load(f)
        else:
            if yaml is None:
                raise RuntimeError("pyyaml is not installed, cannot load yaml config.")
            d = yaml.safe_load(f)
    cfg = Config()
    if isinstance(d, dict):
        _deep_update_dataclass(cfg, d)
    validate_config(cfg)
    return cfg


def config_to_dict(cfg: Config) -> Dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: Config, path: str) -> None:
    ext = os.path.splitext(path)[1].lower()
    d = config_to_dict(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if ext == ".json":
            json.dump(d, f, indent=2)
        elif ext in (".yml", ".yaml"):
            if yaml is None:
                raise RuntimeError("pyyaml is not installed, cannot save yaml config.")
            yaml.safe_dump(d, f, sort_keys=False)
        else:
            raise ValueError(f"Unsupported config format: {ext}")


def validate_config(cfg: Config) -> None:
    if int(cfg.init.n_init) <= 0:
        raise ValueError("cfg.init.n_init must be > 0")
    if int(cfg.train.iters) <= 0:
        raise ValueError("cfg.train.iters must be > 0")
    if int(cfg.train.save_every) <= 0:
        raise ValueError("cfg.train.save_every must be > 0")
    if int(cfg.densify.every) <= 0:
        raise ValueError("cfg.densify.every must be > 0")

    sample_mode = str(cfg.init.sample_mode).lower()
    valid_sample = ("mask_uniform", "mask", "uniform", "topk_chi0", "topk", "chi0_topk", "hybrid", "mask_edge_hybrid", "edge_hybrid", "phi_edge_hybrid", "lfs_edge_hybrid")
    if sample_mode not in valid_sample:
        raise ValueError("cfg.init.sample_mode must be one of: mask_uniform, topk_chi0, hybrid, mask_edge_hybrid")

    chi0_mode = str(cfg.init.chi0_mode).lower()
    if chi0_mode not in ("zero", "none", "mask", "const", "constant", "phi", "lfs", "field", "tkd", "tkd_inverse"):
        raise ValueError("cfg.init.chi0_mode must be one of: zero, mask, phi, tkd")

    if int(cfg.densify.clone_k) < 1:
        raise ValueError("cfg.densify.clone_k must be >= 1")
    if int(cfg.densify.split_k) < 2:
        raise ValueError("cfg.densify.split_k must be >= 2")
    if int(cfg.densify.max_points) < int(cfg.init.n_init):
        raise ValueError("cfg.densify.max_points must be >= cfg.init.n_init")
    if str(cfg.densify.sigma_threshold_mode).lower() == "fixed" and float(cfg.densify.clone_sigma_max) > float(cfg.densify.split_sigma_min):
        raise ValueError("clone_sigma_max should be <= split_sigma_min when sigma_threshold_mode='fixed'")
    if not (0.0 < float(cfg.densify.start_frac) < float(cfg.densify.clone_until_frac) <= 1.0):
        raise ValueError("densify stage fractions should satisfy 0 < start_frac < clone_until_frac <= 1")
    if not (float(cfg.densify.start_frac) < float(cfg.densify.split_until_frac) <= float(cfg.densify.clone_until_frac)):
        raise ValueError("densify stage fractions should satisfy start < split_until <= clone_until")

    if int(cfg.unet.every) <= 0:
        raise ValueError("cfg.unet.every must be > 0")
    if cfg.unet.enable and int(cfg.unet.in_channels) != 2:
        raise ValueError("This pipeline keeps Swin-UNet as exactly 2ch. Set cfg.unet.in_channels=2.")
    if str(cfg.unet.input_mode).lower() not in ("residual", "mask"):
        raise ValueError("cfg.unet.input_mode must be 'residual' or 'mask'")
    if str(cfg.unet.refiner_version).lower() not in ("v1", "v2"):
        raise ValueError("cfg.unet.refiner_version must be 'v1' or 'v2'")
    if str(cfg.unet.refine_mode).lower() not in ("fixed_ratio", "learned_gate"):
        raise ValueError("cfg.unet.refine_mode must be 'fixed_ratio' or 'learned_gate'")

    if str(cfg.cfr.weight_mode).lower() not in ("reliability", "mask"):
        raise ValueError("cfg.cfr.weight_mode must be 'reliability' or 'mask'")
    if str(cfg.cfr.loss).lower() not in ("l1", "l2", "charbonnier"):
        raise ValueError("cfg.cfr.loss must be one of: l1, l2, charbonnier")
    if int(cfg.cfr.from_iter) <= 0:
        raise ValueError("cfg.cfr.from_iter must be > 0")
    if int(cfg.cfr.update_every) <= 0:
        raise ValueError("cfg.cfr.update_every must be > 0")
    if str(cfg.cfr.feedback_target).lower() not in ("ref", "gs"):
        raise ValueError("cfg.cfr.feedback_target must be 'ref' or 'gs'")
    if str(cfg.cfr.feedback_loss).lower() not in ("l1", "l2", "charbonnier"):
        raise ValueError("cfg.cfr.feedback_loss must be one of: l1, l2, charbonnier")
    if int(cfg.bbox.margin) < 0:
        raise ValueError("cfg.bbox.margin must be >= 0")
