"""
Run all 8 PINN experiments sequentially, then generate all comparison graphs.
Usage: python src/run_experiments.py
"""

import json
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

torch.set_num_interop_threads(1)

from train import train
from plot import plot_loss_curves_to_path
from eval import plot_evals

EXPERIMENTS = [
    {"pde_type": "heat",    "use_lbfgs": False, "use_resampling": False, "run_name": "heat_adam"},
    {"pde_type": "heat",    "use_lbfgs": False, "use_resampling": True,  "run_name": "heat_adam_resample"},
    {"pde_type": "heat",    "use_lbfgs": True,  "use_resampling": False, "run_name": "heat_adam_lbfgs"},
    {"pde_type": "heat",    "use_lbfgs": True,  "use_resampling": True,  "run_name": "heat_adam_resample_lbfgs"},
    {"pde_type": "burgers", "use_lbfgs": False, "use_resampling": False, "run_name": "burgers_adam"},
    {"pde_type": "burgers", "use_lbfgs": False, "use_resampling": True,  "run_name": "burgers_adam_resample"},
    {"pde_type": "burgers", "use_lbfgs": True,  "use_resampling": False, "run_name": "burgers_adam_lbfgs"},
    {"pde_type": "burgers", "use_lbfgs": True,  "use_resampling": True,  "run_name": "burgers_adam_resample_lbfgs"},
]

HEAT_NAMES    = [e["run_name"] for e in EXPERIMENTS if e["pde_type"] == "heat"]
BURGERS_NAMES = [e["run_name"] for e in EXPERIMENTS if e["pde_type"] == "burgers"]

PLOTS_DIR = os.path.join("plots", "experiments")


def main():
    manifest = {}

    # --- training ---
    for cfg in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"starting: {cfg['run_name']}")
        print(f"{'='*60}")
        run_id, run_path = train(**cfg, use_checkpoint=False)
        manifest[cfg["run_name"]] = {"run_id": run_id, "run_path": run_path}

    os.makedirs("logs", exist_ok=True)
    manifest_path = os.path.join("logs", "experiment_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nmanifest saved: {manifest_path}")

    # --- loss curve plots (one per experiment) ---
    print("\ngenerating loss curve plots...")
    for cfg in EXPERIMENTS:
        name = cfg["run_name"]
        run_path = manifest[name]["run_path"]
        bundle = torch.load(run_path, map_location="cpu", weights_only=False)
        out_path = os.path.join(PLOTS_DIR, f"{name}_loss_curves.png")
        plot_loss_curves_to_path(bundle, name, out_path)

    # --- error bar charts ---
    print("\ngenerating error bar charts...")
    plot_evals(
        names=HEAT_NAMES,
        title="Heat Equation 1D — error comparison",
        out_path=os.path.join(PLOTS_DIR, "heat_errors.png"),
    )
    plot_evals(
        names=BURGERS_NAMES,
        title="Viscous Burgers 1D — error comparison",
        out_path=os.path.join(PLOTS_DIR, "burgers_errors.png"),
    )

    print("\nall done.")
    print(f"plots in: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
