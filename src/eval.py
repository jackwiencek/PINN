import argparse
import csv
import datetime
import glob
import os
import sys
import torch

from model import PINN
from physics import HeatEquation1D, ViscousBurgers1D

EVALS_CSV = os.path.join("logs", "evals.csv")
EVALS_FIELDNAMES = ["name", "timestamp", "run_id", "max_abs_error", "l2_error"]


def evaluate(model, pde, device="cpu", grid=100):
    model.eval()
    with torch.no_grad():
        xs = torch.linspace(pde.x_min, pde.x_max, grid, device=device)
        ts = torch.linspace(0.0, pde.t_max, grid, device=device)
        X, Tm = torch.meshgrid(xs, ts, indexing="ij")
        x_flat = X.reshape(-1, 1)
        t_flat = Tm.reshape(-1, 1)

        u_pred = model(x_flat, t_flat)

        result = {
            "u_pred": u_pred,
            "X": X,
            "Tm": Tm,
        }

        u_exact = pde.analytical_solution(x_flat, t_flat)
        if u_exact is not None:
            max_err = (u_pred - u_exact).abs().max().item()
            l2_err = ((u_pred - u_exact) ** 2).mean().sqrt().item()
            result["u_exact"] = u_exact
            result["max_abs_error"] = max_err
            result["l2_error"] = l2_err

    return result


def resolve_run_path(arg):
    if arg is not None:
        return arg
    candidates = sorted(glob.glob(os.path.join("runs", "run_*.pt")))
    if not candidates:
        sys.exit("no runs found in runs/ — pass a path or run train.py first")
    return candidates[-1]


def load_run(run_path):
    if not os.path.exists(run_path):
        sys.exit(f"run file not found: {run_path}")
    print(f"loading {run_path}")
    return torch.load(run_path, map_location="cpu", weights_only=False)


def build_pde_from_bundle(bundle):
    """Reconstruct PDE from saved run bundle."""
    pde_class = bundle.get("pde_class", "HeatEquation1D")
    pde_params = bundle.get("pde_params")
    if pde_params:
        if pde_class == "HeatEquation1D":
            return HeatEquation1D(**pde_params)
        elif pde_class == "ViscousBurgers1D":
            return ViscousBurgers1D(**pde_params)
    # backward compat: old bundles without pde_class
    return HeatEquation1D(
        L=bundle.get("L", 1.0),
        T=bundle.get("T", 1.0),
        alpha=bundle.get("alpha", 1.0),
    )


def append_eval_result(name, run_id, max_abs_error, l2_error):
    os.makedirs("logs", exist_ok=True)
    write_header = not os.path.exists(EVALS_CSV)
    with open(EVALS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVALS_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "name": name,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": run_id,
            "max_abs_error": f"{max_abs_error:.6e}",
            "l2_error": f"{l2_error:.6e}",
        })
    print(f"appended to {EVALS_CSV}")


def plot_evals(names=None, title="Evaluation comparison", out_path=None):
    if not os.path.exists(EVALS_CSV):
        sys.exit(f"no evals CSV found at {EVALS_CSV} — run eval.py --name first")

    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        sys.exit("numpy + matplotlib required. run: pip install numpy matplotlib")

    all_rows = []
    with open(EVALS_CSV, newline="") as f:
        for row in csv.DictReader(f):
            all_rows.append({
                "name":          row["name"],
                "max_abs_error": float(row["max_abs_error"]),
                "l2_error":      float(row["l2_error"]),
            })

    if names is not None:
        name_set = set(names)
        # preserve order of names list, pick last occurrence per name
        by_name = {}
        for r in all_rows:
            if r["name"] in name_set:
                by_name[r["name"]] = r
        rows = [by_name[n] for n in names if n in by_name]
    else:
        rows = all_rows

    if not rows:
        sys.exit("evals CSV is empty")

    row_names = [r["name"] for r in rows]
    max_errs  = [r["max_abs_error"] for r in rows]
    l2_errs   = [r["l2_error"] for r in rows]
    x         = np.arange(len(rows))
    width     = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.9 + 2), 5))
    ax.bar(x - width / 2, max_errs, width, label="max abs error", color="tab:red",   alpha=0.8)
    ax.bar(x + width / 2, l2_errs,  width, label="L2 error",      color="tab:blue",  alpha=0.8)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(row_names, rotation=30, ha="right")
    ax.set_ylabel("error (log scale)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    os.makedirs("plots", exist_ok=True)
    out = out_path or os.path.join("plots", "evals_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved PINN run.")
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help="Path to run_*.pt file. Defaults to latest in runs/.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=100,
        help="Eval grid resolution per axis (default 100).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Label for this evaluation (required to log results to evals.csv).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot evals.csv comparison bar chart instead of running evaluation.",
    )
    args = parser.parse_args()

    if args.plot:
        plot_evals()
        return

    if args.name is None:
        sys.exit("--name is required. e.g.: python eval.py --name baseline")

    run_path = resolve_run_path(args.run)
    bundle = load_run(run_path)
    run_id = bundle.get("run_id", os.path.splitext(os.path.basename(run_path))[0].replace("run_", ""))

    model = PINN(**bundle["model_config"])
    model.load_state_dict(bundle["model_state"])

    pde = build_pde_from_bundle(bundle)
    result = evaluate(model, pde, device="cpu", grid=args.grid)

    if "max_abs_error" in result:
        print(f"max abs error: {result['max_abs_error']:.4e}")
        print(f"L2 error:      {result['l2_error']:.4e}")
        append_eval_result(args.name, run_id, result["max_abs_error"], result["l2_error"])
    else:
        print("no analytical solution — prediction only, nothing logged")


if __name__ == "__main__":
    main()
