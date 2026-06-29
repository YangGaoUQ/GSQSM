# GSQSM: Unsupervised Gaussian Splatting for Quantitative Susceptibility Mapping

GSQSM is an unsupervised quantitative susceptibility mapping (QSM) reconstruction framework based on physics-guided Gaussian splatting. It represents the susceptibility distribution with learnable 3D Gaussian primitives and optimizes them under the QSM dipole forward model through field-domain consistency.

The current framework combines Gaussian susceptibility representation, adaptive density control, a lightweight Swin-UNet refiner, and Confidence-Field Reliability (CFR) based denoising. GSQSM is designed for label-free QSM reconstruction from preprocessed local field maps.

---

## Overview

Quantitative susceptibility mapping reconstructs tissue magnetic susceptibility from MRI phase-derived local fields. The field-to-source inversion is ill-posed because the dipole kernel contains zero and near-zero regions in k-space, making the reconstruction sensitive to noise and streaking artifacts.

GSQSM formulates QSM reconstruction as an explicit primitive optimization problem. Instead of directly optimizing a dense voxel-wise susceptibility map, GSQSM uses learnable 3D Gaussian primitives to represent the susceptibility distribution. The voxelized susceptibility map is generated through differentiable splatting and constrained by the QSM dipole forward model.

Main characteristics of GSQSM:

- **Unsupervised reconstruction** without paired local field-susceptibility labels.
- **Physics-guided optimization** using QSM dipole forward-model consistency.
- **Gaussian susceptibility representation** with learnable 3D primitives.
- **Adaptive density control** for updating Gaussian primitives during optimization.
- **Lightweight Swin-UNet refiner** for residual susceptibility correction.
- **CFR denoising** for reliability-guided refinement of the reconstructed susceptibility map.

---

## Pipeline

<p align="center">
  <img src="figures/pipeline.png" width="950">
</p>

<p align="center">
  <em>Figure 1. Overall pipeline of GSQSM. The local field map is used to initialize Gaussian primitives, which are voxelized through differentiable splatting and optimized with the QSM dipole forward model. Adaptive density control updates the Gaussian representation, the lightweight Swin-UNet refiner predicts residual susceptibility correction, and the CFR module performs reliability-guided denoising.</em>
</p>

The GSQSM reconstruction pipeline contains the following steps:

1. **Input local field map**  
   GSQSM takes a preprocessed local field map as input. The local field is expected to have been processed by phase unwrapping and background field removal before reconstruction.

2. **Gaussian primitive initialization**  
   A set of 3D Gaussian primitives is initialized inside the brain region. Each primitive contains a position, scale, orientation-related covariance, and signed susceptibility amplitude.

3. **Differentiable voxel splatting**  
   Gaussian primitives are converted into a voxelized susceptibility map, denoted as `chi_gs`, through differentiable splatting.

4. **Physics-guided optimization**  
   The predicted local field is generated from `chi_gs` using the QSM dipole forward model. The Gaussian parameters are optimized by minimizing the discrepancy between the predicted and measured local fields within the brain mask.

5. **Adaptive density control**  
   Gaussian primitives are cloned, split, or pruned according to gradient statistics, local-field residuals, and pruning criteria.

6. **Swin-UNet refiner**  
   A lightweight Swin-UNet refiner takes the Gaussian reconstruction and normalized local-field residual as input and predicts a bounded residual correction to obtain `chi_refiner`.

7. **CFR denoising**  
   The CFR module estimates a reliability map from the field discrepancy and performs reliability-guided denoising to generate the final susceptibility map.

---

## Results

On the 2019 QSM Challenge simulation data, GSQSM achieved average NRMSE, HFEN, and XSIM values of **47.76%**, **38.46%**, and **75.05%**, respectively, across the four simulation settings reported in Table 1.

<p align="center">
  <img src="figures/combine%20on%20challenge.png" width="900">
</p>

<p align="center">
  <em>Figure 3. Comparison of different QSM reconstruction methods on representative 2019 QSM Challenge simulation settings. The rows show reconstructed susceptibility maps and corresponding error maps.</em>
</p>

### Quantitative Results on the 2019 QSM Challenge

Values are reported in the order of **Sim1SNR1 / Sim1SNR2 / Sim2SNR1 / Sim2SNR2**.

| Method | NRMSE (%) | HFEN (%) | XSIM (%) |
|---|---:|---:|---:|
| iLSQR | 77.88 / 64.51 / 57.57 / 51.70 | 49.07 / 43.82 / 47.07 / 44.19 | 50.51 / 62.32 / 60.53 / 66.98 |
| MEDI | 57.14 / 49.01 / 51.66 / 48.89 | 37.58 / 34.20 / 46.04 / 45.42 | 72.99 / 76.59 / 70.62 / 72.57 |
| LPCNN | 55.57 / 51.75 / 47.61 / 46.91 | 47.97 / 46.52 / 47.63 / 46.69 | 61.20 / 66.37 / 66.99 / 68.28 |
| INRQSM | 63.69 / 57.78 / 56.78 / 55.32 | 55.17 / 51.38 / 60.94 / 60.95 | 56.29 / 60.30 / 57.54 / 59.04 |
| MODIP | 65.50 / 55.14 / 55.73 / 51.39 | 47.31 / 45.33 / 50.47 / 49.99 | 57.51 / 67.25 / 62.67 / 66.60 |
| **GSQSM** | **53.55 / 46.74 / 46.37 / 44.38** | **38.01 / 35.56 / 37.39 / 42.88** | **73.12 / 78.62 / 73.66 / 74.80** |

These results indicate that GSQSM provides stable reconstruction performance on the 2019 QSM Challenge simulation data, particularly in reducing reconstruction error and preserving susceptibility-related structures.

---

## Quick Start

### Installation

```bash
git clone https://github.com/Itachi711/GSQSM.git
cd GSQSM

conda create -n gsqsm python=3.10
conda activate gsqsm
pip install -r requirements.txt
```

### Run a Single Case

```bash
python recon.py \
  --phi ./dataset/vivo/CAA/lfs.nii \
  --out ./deepMRI/gsqsm/CAA
```

If a brain mask is available:

```bash
python recon.py \
  --phi ./dataset/vivo/CAA/lfs.nii \
  --mask ./dataset/vivo/CAA/mask.nii \
  --out ./deepMRI/gsqsm/CAA
```

The final reconstruction will be saved under the output directory, typically as:

```text
<out>/<run_name>/gsqsm.nii
```

For example:

```text
./deepMRI/gsqsm/CAA/gsqsm/gsqsm.nii
```

---

## Command-Line Arguments

| Argument | Required | Description |
|---|---:|---|
| `--phi` | Yes | Path to the input local field map in NIfTI format. This should be a preprocessed local field map rather than raw phase. |
| `--mask` | No | Path to the brain mask in NIfTI format. If not provided, the code may attempt to use an internal mask handling strategy depending on the current configuration. |
| `--out` | Yes | Output directory for reconstructed results, logs, and intermediate outputs. |
| `--config` | No | Path to a configuration file if supported by the current code version. |
| `--device` | No | Computation device, such as `cuda` or `cpu`, depending on the current implementation. |

Example:

```bash
python recon.py \
  --phi ./dataset/vivo/CAA/lfs.nii \
  --mask ./dataset/vivo/CAA/mask.nii \
  --out ./deepMRI/gsqsm/CAA \
  --device cuda
```

> Note: In the current command-line interface, `--phi` refers to the local field map used for QSM reconstruction.

---

## Input and Output

### Input

GSQSM expects a preprocessed local field map as input.

Recommended input preparation:

- phase unwrapping has been performed;
- background field removal has been performed;
- the local field map and brain mask are spatially aligned;
- the local field map and mask have the same matrix size and affine information.

### Output

The main output is:

```text
gsqsm.nii
```

This file corresponds to the final reconstructed susceptibility map. When the refiner and CFR modules are enabled, `gsqsm.nii` represents the final output after the full GSQSM pipeline. When some modules are disabled for ablation, the output corresponds to the last enabled reconstruction stage.

---

## Related Work

GSQSM is related to several lines of research in QSM reconstruction and Gaussian-based representation learning.

### Conventional QSM Reconstruction

Classical QSM methods stabilize the ill-posed dipole inversion problem through thresholding, iterative optimization, morphological priors, or artifact suppression. Representative methods include COSMOS, TKD, iLSQR, MEDI, and STAR-QSM.

### Deep Learning for QSM

Learning-based QSM methods use neural networks to learn the mapping from local fields or phase measurements to susceptibility maps. Representative methods include QSMnet, xQSM, autoQSM, LPCNN, MoDL-QSM, iQSM, and iQSM+.

### Unsupervised and Subject-Specific QSM

Unsupervised QSM methods reduce the dependence on paired training labels by using subject-specific optimization or model-based constraints. Related methods include AdaIN-based resolution-agnostic QSM, MoDIP, and INR-QSM.

### Gaussian Splatting and Medical Imaging

3D Gaussian splatting provides an explicit and differentiable representation based on Gaussian primitives. Recent Gaussian-based medical imaging studies, such as X-Gaussian and R2-Gaussian, suggest that Gaussian representations can be combined with imaging forward models for volumetric reconstruction tasks.

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@inproceedings{GSQSM2026,
  title={GSQSM: Unsupervised Gaussian Splatting for Quantitative Susceptibility Mapping},
  author={...},
  booktitle={...},
  year={2026}
}
```

---

## Notes

This repository is currently under active development. The code and default parameters may be updated as the paper and experiments are finalized.
