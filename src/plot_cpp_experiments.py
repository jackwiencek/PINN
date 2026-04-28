"""
Generate all C++ experiment graphs into plots/cpp_experiments/.

Produces:
  8 loss curve PNGs        (<name>_loss_curves.png)
  2 error bar chart PNGs   (heat_errors.png, burgers_errors.png)
  8 per-epoch timing PNGs  (<name>_timing.png)
  1 Python vs C++ comparison (timing_comparison_py_vs_cpp.png)

Usage: python src/plot_cpp_experiments.py
"""

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch
import matplotlib.pyplot as plt
import numpy as np

from plot import plot_loss_curves_to_path, load_perf_csv

EXPERIMENTS = [
    "heat_adam",
    "heat_adam_resample",
    "heat_adam_lbfgs",
    "heat_adam_resample_lbfgs",
    "burgers_adam",
    "burgers_adam_resample",
    "burgers_adam_lbfgs",
    "burgers_adam_resample_lbfgs",
]
HEAT_NAMES    = EXPERIMENTS[:4]
BURGERS_NAMES = EXPERIMENTS[4:]
PLOTS_DIR     = os.path.join("plots", "cpp_experiments")


def load_manifest(path):
    with open(path) as f:
        return json.load(f)


def plot_cpp_evals(names, title, out_path, evals_csv="c_logs/evals.csv"):
    if not os.path.exists(evals_csv):
        sys.exit(f"evals CSV not found: {evals_csv}")

    all_rows = []
    with open(evals_csv, newline="") as f:
        for row in csv.DictReader(f):
            all_rows.append({
                "name":          row["name"],
                "max_abs_error": float(row["max_abs_error"]),
                "l2_error":      float(row["l2_error"]),
            })

    name_set = set(names)
    by_name  = {}
    for r in all_rows:
        if r["name"] in name_set:
            by_name[r["name"]] = r
    rows = [by_name[n] for n in names if n in by_name]
    if not rows:
        sys.exit(f"no matching rows in {evals_csv} for {names}")

    row_names = [r["name"] for r in rows]
    max_errs  = [r["max_abs_error"] for r in rows]
    l2_errs   = [r["l2_error"]      for r in rows]
    x         = np.arange(len(rows))
    width     = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.9 + 2), 5))
    ax.bar(x - width / 2, max_errs, width, label="max abs error", color="tab:red",  alpha=0.8)
    ax.bar(x + width / 2, l2_errs,  width, label="L2 error",      color="tab:blue", alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(row_names, rotation=30, ha="right")
    ax.set_ylabel("error (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_epoch_timing(rows, name, out_path):
    adam_rows  = [r for r in rows if r["phase"] == "adam"]
    lbfgs_rows = [r for r in rows if r["phase"] == "lbfgs"]
    adam_n     = len(adam_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    if adam_rows:
        ax.plot(
            [r["epoch"] for r in adam_rows],
            [r["epoch_time_s"] * 1000 for r in adam_rows],
            color="tab:blue", linewidth=0.8, alpha=0.7, label="Adam (ms/epoch)",
        )
    if lbfgs_rows:
        lx = [adam_n + r["epoch"] for r in lbfgs_rows]
        ax.plot(
            lx, [r["epoch_time_s"] * 1000 for r in lbfgs_rows],
            color="tab:green", linewidth=0.8, alpha=0.9, label="L-BFGS (ms/step)",
        )
        if adam_rows:
            ax.axvline(adam_n, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvspan(adam_n, adam_n + len(lbfgs_rows), alpha=0.08, color="tab:green", zorder=0)
    ax.set_xlabel("step")
    ax.set_ylabel("time per step (ms)")
    ax.set_title(f"Per-epoch timing — {name} (C++)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_timing_comparison(py_manifest, cpp_manifest, out_path):
    py_times, cpp_times = [], []
    labels = []
    for name in EXPERIMENTS:
        py_run_id  = py_manifest[name]["run_id"]
        cpp_run_id = cpp_manifest[name]["run_id"]
        py_rows    = load_perf_csv(f"logs/perf_{py_run_id}.csv")
        cpp_rows   = load_perf_csv(f"c_logs/perf_{cpp_run_id}.csv")
        py_times.append(py_rows[-1]["wall_time_s"])
        cpp_times.append(cpp_rows[-1]["wall_time_s"])
        labels.append(name)

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.1 + 2), 5))
    ax.bar(x - width / 2, py_times,  width, label="Python",  color="tab:orange", alpha=0.85)
    ax.bar(x + width / 2, cpp_times, width, label="C++",     color="tab:blue",   alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("total training time (s)")
    ax.set_title("Training time comparison — Python vs C++")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    cpp_manifest = load_manifest("c_logs/experiment_manifest.json")
    py_manifest  = load_manifest("logs/experiment_manifest.json")

    # --- 8 loss curves ---
    print("\ngenerating loss curves...")
    for name in EXPERIMENTS:
        run_path = cpp_manifest[name]["run_path"]
        bundle   = torch.load(run_path, map_location="cpu", weights_only=False)
        out      = os.path.join(PLOTS_DIR, f"{name}_loss_curves.png")
        plot_loss_curves_to_path(bundle, f"{name} (C++)", out)

    # --- 2 error bar charts ---
    print("\ngenerating error bar charts...")
    plot_cpp_evals(
        names=HEAT_NAMES,
        title="Heat Equation 1D — error comparison (C++)",
        out_path=os.path.join(PLOTS_DIR, "heat_errors.png"),
    )
    plot_cpp_evals(
        names=BURGERS_NAMES,
        title="Viscous Burgers 1D — error comparison (C++)",
        out_path=os.path.join(PLOTS_DIR, "burgers_errors.png"),
    )

    # --- 8 per-epoch timing graphs ---
    print("\ngenerating per-epoch timing graphs...")
    for name in EXPERIMENTS:
        run_id   = cpp_manifest[name]["run_id"]
        csv_path = f"c_logs/perf_{run_id}.csv"
        rows     = load_perf_csv(csv_path)
        out      = os.path.join(PLOTS_DIR, f"{name}_timing.png")
        plot_epoch_timing(rows, name, out)

    # --- 1 Python vs C++ comparison ---
    print("\ngenerating Python vs C++ timing comparison...")
    plot_timing_comparison(
        py_manifest,
        cpp_manifest,
        out_path=os.path.join(PLOTS_DIR, "timing_comparison_py_vs_cpp.png"),
    )

    print(f"\nall done. plots in: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
