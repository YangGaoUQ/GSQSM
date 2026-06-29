# GSQSM

GSQSM is an unsupervised quantitative susceptibility mapping (QSM) reconstruction pipeline based on physics-guided Gaussian splatting. The current implementation represents the susceptibility distribution with learnable 3D Gaussian primitives, enforces consistency with the QSM dipole forward model, and optionally applies a lightweight Swin-UNet refiner and CFR-based reliability-guided denoising.

This repository is intended for research use and is organized around the current code entry points without changing the existing project structure.

## Overview

Given a preprocessed local field map, GSQSM reconstructs a susceptibility map through the following pipeline:

```text
local field map
    -> Gaussian susceptibility representation
    -> dipole forward-model consistency
    -> optional Swin-UNet refiner
    -> optional CFR denoising
    -> final QSM reconstruction
```

Main components:

- **Gaussian representation**: represents the susceptibility distribution using learnable 3D Gaussian primitives.
- **Physics-guided optimization**: constrains the reconstruction by matching the measured local field through the QSM dipole forward model.
- **Adaptive density control**: adjusts Gaussian primitives during optimization.
- **Swin-UNet refiner**: performs self-supervised residual correction using the Gaussian reconstruction and local-field residual.
- **CFR denoising**: applies reliability-guided denoising to improve spatial consistency.

## Current Project Layout

```text
GSQSM/
├── config.py
├── recon.py
├── train.py
├── model/
│   ├── cfr.py
│   ├── forward.py
│   ├── gs_model.py
│   ├── initialize.py
│   └── swin_unet3d.py
└── utils/
    ├── loss.py
    └── tool.py
```

The recommended user entry point is `recon.py`.

## Installation

Create a Python environment and install the required packages:

```bash
conda create -n gsqsm python=3.10
conda activate gsqsm
pip install -r requirements.txt
```

PyTorch installation may depend on your CUDA version. If the default `pip install -r requirements.txt` does not install a CUDA-enabled PyTorch build, install PyTorch following your local CUDA environment and then install the remaining packages.

## Quick Start

Run GSQSM on one local field map:

```bash
python recon.py \
  --phi ./dataset/vivo/CAA/lfs.nii \
  --out ./deepMRI/gsqsm/CAA
```

Run with an explicit brain mask and magnitude image:

```bash
python recon.py \
  --phi ./dataset/vivo/CAA/lfs.nii \
  --mask ./dataset/vivo/CAA/mask.nii \
  --mag ./dataset/vivo/CAA/mag.nii \
  --out ./deepMRI/gsqsm/CAA
```

Run with a configuration file:

```bash
python recon.py \
  --phi ./dataset/vivo/CAA/lfs.nii \
  --mask ./dataset/vivo/CAA/mask.nii \
  --mag ./dataset/vivo/CAA/mag.nii \
  --out ./deepMRI/gsqsm/CAA \
  --config ./configs/vivo.yaml
```

The argument `--phi` denotes the input **local field map** used by GSQSM. It should be a preprocessed local field map rather than a raw phase image.

## Command-Line Arguments

| Argument | Required | Description |
|---|---:|---|
| `--phi` | Yes | Path to the input local field map. Supported formats: `.nii`, `.nii.gz`, `.mat`. |
| `--out` | Yes | Output directory. The final result is saved under `<out>/<run-name>/`. |
| `--mask` | No | Optional brain mask path. Supported formats: `.nii`, `.nii.gz`, `.mat`. |
| `--mag` | No | Optional magnitude image path for CFR reliability weighting. Supported formats: `.nii`, `.nii.gz`. |
| `--config` | No | Optional configuration file. Supported formats: `.json`, `.yaml`, `.yml`. |
| `--run-name` | No | Optional run folder name. Default: `gsqsm`. |
| `--device` | No | Optional device override, such as `cuda` or `cpu`. |

## Input Data

GSQSM expects the input local field to be preprocessed before reconstruction.

Required input:

- Local field map specified by `--phi`.

Optional inputs:

- Brain mask specified by `--mask`.
- Magnitude image specified by `--mag`.

Recommended input requirements:

- The local field should already be background-field removed.
- The local field, mask, and magnitude image should have the same spatial shape.
- For NIfTI inputs, the affine should correctly describe the voxel spacing when possible.
- If no mask is provided, the code will derive a mask from the input local field and apply mask repair logic.
- If no magnitude image is provided, CFR will fall back to a mask-based or default reliability setting.

## Output Files

The final reconstruction is saved as:

```text
<out>/<run-name>/gsqsm.nii
```

For example:

```text
./deepMRI/gsqsm/CAA/gsqsm/gsqsm.nii
```

When the Swin-UNet refiner and CFR are enabled, `gsqsm.nii` corresponds to the final CFR-refined output. If the refiner or CFR is disabled in the configuration, the final output corresponds to the last enabled reconstruction stage.

If bounding-box cropping is enabled, a debug file such as `bbox_info.json` may also be saved in the run directory.

## Configuration

The code uses `config.py` as the default configuration. You can override selected fields using a `.json`, `.yaml`, or `.yml` config file.

A minimal YAML example:

```yaml
io:
  phi_path: ./dataset/vivo/CAA/lfs.nii
  mask_path: ./dataset/vivo/CAA/mask.nii
  magnitude_path: ./dataset/vivo/CAA/mag.nii
  out_dir: ./deepMRI/gsqsm/CAA
  run_name: gsqsm
  device: cuda

phys:
  voxel_size_mm: [1.0, 1.0, 1.0]
  B0_dir_zyx: [0.0, 0.0, 1.0]

train:
  iters: 300
  seed: 0

init:
  n_init: 60000

unet:
  enable: true

cfr:
  enable: true
  iters: 30
```

Commonly adjusted parameters:

| Section | Parameter | Description |
|---|---|---|
| `io` | `phi_path` | Default local field path if not overridden by `--phi`. |
| `io` | `mask_path` | Default brain mask path. |
| `io` | `magnitude_path` | Default magnitude image path. |
| `io` | `out_dir` | Default output directory if not overridden by `--out`. |
| `phys` | `voxel_size_mm` | Voxel size in z-y-x order, especially useful for `.mat` inputs or non-informative NIfTI affine. |
| `phys` | `B0_dir_zyx` | Main magnetic field direction in z-y-x order. |
| `train` | `iters` | Number of optimization iterations. |
| `init` | `n_init` | Initial number of Gaussian primitives. |
| `densify` | `enable` | Enable or disable adaptive density control. |
| `densify` | `max_points` | Maximum number of Gaussian primitives after density control. |
| `unet` | `enable` | Enable or disable the Swin-UNet refiner. |
| `cfr` | `enable` | Enable or disable CFR denoising. |
| `cfr` | `data_w` | Field-domain data consistency weight in CFR. |
| `cfr` | `tv_w` | Total variation weight in CFR. |
| `cfr` | `fd_w` | Field-domain regularization weight in CFR. |

## Ablation Examples

### Gaussian only

```yaml
unet:
  enable: false

cfr:
  enable: false
```

### Gaussian + Swin-UNet refiner

```yaml
unet:
  enable: true

cfr:
  enable: false
```

### Full GSQSM

```yaml
unet:
  enable: true

cfr:
  enable: true
```

## Python API

The reconstruction can also be called from Python:

```python
from recon import reconstruct

run_dir = reconstruct(
    phi_path="./dataset/vivo/CAA/lfs.nii",
    mask_path="./dataset/vivo/CAA/mask.nii",
    magnitude_path="./dataset/vivo/CAA/mag.nii",
    out_dir="./deepMRI/gsqsm/CAA",
    config_path="./configs/vivo.yaml",
    run_name="gsqsm",
    device="cuda",
)

print("Result directory:", run_dir)
```

## Troubleshooting

### CUDA is not available

If `cuda` is specified but CUDA is not available, the current code falls back to CPU. CPU reconstruction may be slow. Check your PyTorch and CUDA installation if GPU acceleration is expected.

### Shape mismatch between local field and mask

Make sure the local field, mask, and magnitude image have the same 3D shape. For NIfTI inputs, also check that they are spatially aligned.

### Large dark regions appear in the reconstruction

This may be caused by the input local field, mask quality, refiner correction, or CFR regularization. Recommended checks:

1. Use a fixed display window when viewing results.
2. Check whether the local field has been properly background-field removed.
3. Check whether the brain mask contains non-brain regions.
4. Compare ablation outputs by disabling CFR and/or the refiner.
5. If the dark regions mainly appear after CFR, reduce `cfr.tv_w` and `cfr.fd_w`, and increase `cfr.data_w`.
6. Provide a magnitude image with `--mag` when available.

A conservative CFR setting for testing is:

```yaml
cfr:
  enable: true
  iters: 10
  data_w: 0.20
  tv_w: 0.003
  fd_w: 0.0002
  feedback_w_early: 0.01
  feedback_w_mid: 0.03
  feedback_w_late: 0.05
  clip: 0.20
```

### Reconstruction is too smooth

Try reducing the smoothing-related CFR weights:

```yaml
cfr:
  tv_w: 0.003
  fd_w: 0.0002
```

You can also compare against a reconstruction with CFR disabled.

### Out-of-memory error

Try reducing the number of Gaussian primitives:

```yaml
init:
  n_init: 30000

densify:
  max_points: 60000
```

You can also reduce `train.iters` for a quick test run.

## Notes

- This code is intended for research use.
- The input local field should be prepared by a proper QSM preprocessing pipeline.
- Quantitative evaluation should be performed with consistent masks, voxel spacing, and display windows.

## Citation

If you use this code, please cite the corresponding paper when available.

```bibtex
@inproceedings{GSQSM2026,
  title     = {GSQSM: Physics-Guided Gaussian Splatting for Quantitative Susceptibility Mapping},
  author    = {...},
  booktitle = {...},
  year      = {2026}
}
```

## Related Work

GSQSM is related to research on QSM reconstruction, unsupervised dipole inversion, implicit neural representations, and Gaussian splatting. Related QSM reconstruction projects include MoDIP and INR-QSM.
