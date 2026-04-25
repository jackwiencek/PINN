import argparse
import csv
import glob
import os
import sys
import torch

from model import PINN
from eval import evaluate, build_pde_from_bundle


def load_run(run_path):
    if not os.path.exists(run_path):
        sys.exit(f"run file not found: {run_path}")
    print(f"loading {run_path}")
    return torch.load(run_path, map_location="cpu", weights_only=False)


def resolve_run_path(arg):
    if arg is not None:
        return arg
    candidates = sorted(glob.glob(os.path.join("runs", "run_*.pt")))
    if not candidates:
        sys.exit("no runs found in runs/ — pass a path or run train.py first")
    return candidates[-1]


def load_perf_csv(csv_path):
    """Load perf CSV (Python or C++ generated). Returns list of dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "epoch":        int(row["epoch"]),
                "phase":        row["phase"],
                "wall_time_s":  float(row["wall_time_s"]),
                "epoch_time_s": float(row["epoch_time_s"]),
            })
    return rows


def plot_perf_standalone(perf_rows, run_id):
    """Performance-only plot for C++ (or any) CSV log — no model needed."""
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)
    adam_rows  = [r for r in perf_rows if r["phase"] == "adam"]
    lbfgs_rows = [r for r in perf_rows if r["phase"] == "lbfgs"]
    adam_n = len(adam_rows)

    fig, ax = plt.subplots(figsize=(8, 4))
    if adam_rows:
        ax.plot([r["epoch"] for r in adam_rows],
                [r["epoch_time_s"] * 1000 for r in adam_rows],
                color="tab:blue", linewidth=0.8, alpha=0.7, label="Adam (ms/epoch)")
    if lbfgs_rows:
        lx = [adam_n + r["epoch"] for r in lbfgs_rows]
        ax.plot(lx, [r["epoch_time_s"] * 1000 for r in lbfgs_rows],
                color="tab:green", linewidth=0.8, alpha=0.9, label="L-BFGS (ms/step)")
        if adam_rows:
            ax.axvline(adam_n, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
            ax.axvspan(adam_n, adam_n + len(lbfgs_rows), alpha=0.08, color="tab:green", zorder=0)
    ax.set_xlabel("step")
    ax.set_ylabel("time per step (ms)")
    ax.set_title(f"Performance — {run_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join("plots", f"perf_{run_id}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_loss_curves_to_path(bundle, label, out_path):
    """Single-run loss curve: total + PDE + IC + BC, Adam/L-BFGS boundary. No perf subplot."""
    import matplotlib.pyplot as plt

    total    = bundle["total_loss"]
    pde_loss = bundle["pde_loss"]
    ic       = bundle["ic_loss"]
    bc       = bundle["bc_loss"]
    epochs_x = range(len(total))

    adam_epochs  = bundle.get("adam_epochs", bundle.get("epochs", len(total)))
    lbfgs_epochs = bundle.get("lbfgs_epochs", 0)
    has_lbfgs    = lbfgs_epochs > 0 and adam_epochs < len(total)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs_x, total,    label="total", linewidth=1.5)
    ax.semilogy(epochs_x, pde_loss, label="PDE",   alpha=0.7)
    ax.semilogy(epochs_x, ic,       label="IC",    alpha=0.7)
    ax.semilogy(epochs_x, bc,       label="BC",    alpha=0.7)
    if has_lbfgs:
        ax.axvspan(adam_epochs, len(total), alpha=0.08, color="tab:green", zorder=0)
        ax.axvline(adam_epochs, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ymin, ymax = ax.get_ylim()
        y_text = ymax / (ymax / ymin) ** 0.05
        ax.text(adam_epochs * 0.5, y_text, "Adam",   ha="center", va="top", fontsize=9, alpha=0.7)
        ax.text((adam_epochs + len(total)) / 2, y_text, "L-BFGS", ha="center", va="top", fontsize=9, alpha=0.7)
    ax.set_xlabel("step (Adam epoch | L-BFGS outer step)" if has_lbfgs else "epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"Loss curves — {label}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot PINN training run.")
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help="Path to run_*.pt file. Defaults to latest in runs/.",
    )
    parser.add_argument(
        "--perf-log",
        default=None,
        metavar="CSV",
        help="Standalone perf_*.csv (e.g. from C++ trainer). Skips model-dependent plots.",
    )
    args = parser.parse_args()

    if args.perf_log:
        perf_rows = load_perf_csv(args.perf_log)
        run_id = os.path.splitext(os.path.basename(args.perf_log))[0].replace("perf_", "")
        plot_perf_standalone(perf_rows, run_id)
        return

    run_path = resolve_run_path(args.run)
    bundle = load_run(run_path)
    run_id = bundle.get("run_id") or os.path.splitext(os.path.basename(run_path))[0].replace("run_", "")

    try:
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
    except ImportError:
        sys.exit("numpy + matplotlib required. run: pip install numpy matplotlib")

    os.makedirs("plots", exist_ok=True)

    # --- Loss curves ---
    total = bundle["total_loss"]
    pde_loss = bundle["pde_loss"]
    ic = bundle["ic_loss"]
    bc = bundle["bc_loss"]
    epochs_x = range(len(total))

    # Adam/L-BFGS phase boundary (fallback for old bundles without these keys)
    adam_epochs = bundle.get("adam_epochs", bundle.get("epochs", len(total)))
    lbfgs_epochs = bundle.get("lbfgs_epochs", 0)
    has_lbfgs = lbfgs_epochs > 0 and adam_epochs < len(total)

    perf_log = bundle.get("perf_log") or []
    has_perf = bool(perf_log)

    if has_perf:
        fig, (ax, ax_perf) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax_perf = None

    ax.semilogy(epochs_x, total, label="total", linewidth=1.5)
    ax.semilogy(epochs_x, pde_loss, label="PDE", alpha=0.7)
    ax.semilogy(epochs_x, ic, label="IC", alpha=0.7)
    ax.semilogy(epochs_x, bc, label="BC", alpha=0.7)
    if has_lbfgs:
        ax.axvspan(adam_epochs, len(total), alpha=0.08, color="tab:green", zorder=0)
        ax.axvline(adam_epochs, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ymin, ymax = ax.get_ylim()
        y_text = ymax / (ymax / ymin) ** 0.05
        ax.text(adam_epochs * 0.5, y_text, "Adam", ha="center", va="top", fontsize=9, alpha=0.7)
        ax.text((adam_epochs + len(total)) / 2, y_text, "L-BFGS", ha="center", va="top", fontsize=9, alpha=0.7)
    if ax_perf is None:
        ax.set_xlabel("step (Adam epoch | L-BFGS outer step)" if has_lbfgs else "epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"Loss curves — {run_id}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    if ax_perf is not None:
        adam_rows  = [r for r in perf_log if r["phase"] == "adam"]
        lbfgs_rows = [r for r in perf_log if r["phase"] == "lbfgs"]
        ax_perf.plot([r["epoch"] for r in adam_rows],
                     [r["epoch_time_s"] * 1000 for r in adam_rows],
                     color="tab:blue", linewidth=0.8, alpha=0.7, label="Adam (ms/epoch)")
        if lbfgs_rows:
            lx = [adam_epochs + r["epoch"] for r in lbfgs_rows]
            ax_perf.plot(lx, [r["epoch_time_s"] * 1000 for r in lbfgs_rows],
                         color="tab:green", linewidth=0.8, alpha=0.9, label="L-BFGS (ms/step)")
            if has_lbfgs:
                ax_perf.axvspan(adam_epochs, len(total), alpha=0.08, color="tab:green", zorder=0)
                ax_perf.axvline(adam_epochs, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        ax_perf.set_xlabel("step (Adam epoch | L-BFGS outer step)" if has_lbfgs else "epoch")
        ax_perf.set_ylabel("time per step (ms)")
        ax_perf.legend(fontsize=8)
        ax_perf.grid(True, alpha=0.3)

    fig.tight_layout()
    loss_path = os.path.join("plots", f"loss_curves_{run_id}.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"saved {loss_path}")

    # --- Rebuild model for solution heatmaps ---
    model = PINN(**bundle["model_config"])
    model.load_state_dict(bundle["model_state"])

    pde = build_pde_from_bundle(bundle)
    grid = 200
    result = evaluate(model, pde, device="cpu", grid=grid)
    u_pred = result["u_pred"].reshape(grid, grid).numpy()

    L = pde.x_max - pde.x_min
    T = pde.t_max

    # --- Prediction heatmap ---
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        u_pred.T,
        origin="lower",
        extent=[pde.x_min, pde.x_max, 0, T],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(f"u_pred(x, t) — {run_id}")
    fig.colorbar(im, ax=ax, label="u")
    fig.tight_layout()
    pred_path = os.path.join("plots", f"heat_pred_{run_id}.png")
    fig.savefig(pred_path, dpi=150)
    plt.close(fig)
    print(f"saved {pred_path}")

    # --- Error heatmap (only if analytical solution exists) ---
    if "u_exact" in result:
        u_exact = result["u_exact"].reshape(grid, grid).numpy()
        err = np.abs(u_pred - u_exact)

        err_floor = max(float(err.min()), 1e-12)
        err_ceiling = float(err.max()) + 1e-12
        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(
            err.T,
            origin="lower",
            extent=[pde.x_min, pde.x_max, 0, T],
            aspect="auto",
            cmap="magma",
            norm=LogNorm(vmin=err_floor, vmax=err_ceiling),
        )
        ax.set_xlabel("x")
        ax.set_ylabel("t")
        ax.set_title(f"|u_pred - u_exact| — {run_id}")
        fig.colorbar(im, ax=ax, label="abs error (log)")
        fig.tight_layout()
        err_path = os.path.join("plots", f"heat_error_{run_id}.png")
        fig.savefig(err_path, dpi=150)
        plt.close(fig)
        print(f"saved {err_path}")

        print(f"max abs error: {err.max():.4e}")
        print(f"L2 error:      {np.sqrt((err ** 2).mean()):.4e}")
    else:
        print("no analytical solution — error heatmap skipped")


if __name__ == "__main__":
    main()
