from __future__ import annotations

import argparse
import time
from typing import Optional

from config import Config, load_config, validate_config
from train import run


def reconstruct(
    phi_path: str,
    out_dir: str,
    mask_path: Optional[str] = None,
    magnitude_path: Optional[str] = None,
    config_path: Optional[str] = None,
    run_name: Optional[str] = None,
    device: Optional[str] = None,
):
    """Minimal external API for GSQSM reconstruction."""
    cfg = load_config(config_path) if config_path else Config()

    # Required IO.
    cfg.io.phi_path = phi_path
    cfg.io.out_dir = out_dir

    if mask_path is not None:
        cfg.io.mask_path = mask_path
    if magnitude_path is not None:
        cfg.io.magnitude_path = magnitude_path
    if run_name is not None:
        cfg.io.run_name = run_name
    if device is not None:
        cfg.io.device = device

    validate_config(cfg)

    # Time one full reconstruction.
    t0 = time.perf_counter()
    try:
        return run(cfg)
    finally:
        elapsed = time.perf_counter() - t0
        print(f"[TIME] reconstruction elapsed: {elapsed:.2f} s ({elapsed / 60.0:.2f} min)")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GSQSM reconstruction.")
    p.add_argument("--phi", required=True, help="Local-field NIfTI path.")
    p.add_argument("--out", required=True, help="Output directory.")
    p.add_argument("--mask", default=None, help="Optional brain mask path.")
    p.add_argument("--mag", default=None, help="Optional magnitude path.")
    p.add_argument("--config", default=None, help="Optional json/yaml config.")
    p.add_argument("--run-name", default=None, help="Optional run folder name.")
    p.add_argument("--device", default=None, help="Optional device, e.g. cuda or cpu.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    reconstruct(
        phi_path=args.phi,
        out_dir=args.out,
        mask_path=args.mask,
        magnitude_path=args.mag,
        config_path=args.config,
        run_name=args.run_name,
        device=args.device,
    )
