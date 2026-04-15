import argparse
import glob
import os
import sys
import torch

from model import PINN


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


def main():
    parser = argparse.ArgumentParser(description="Plot PINN training run.")
    parser.add_argument(
        "run",
        nargs="?",
        default=None,
        help="Path to run_*.pt file. Defaults to latest in runs/.",
    )
    args = parser.parse_args()

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
    pde = bundle["pde_loss"]
    ic = bundle["ic_loss"]
    bc = bundle["bc_loss"]
    epochs_x = range(len(total))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(epochs_x, total, label="total", linewidth=1.5)
    ax.semilogy(epochs_x, pde, label="PDE", alpha=0.7)
    ax.semilogy(epochs_x, ic, label="IC", alpha=0.7)
    ax.semilogy(epochs_x, bc, label="BC", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(f"Loss curves — {run_id}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    loss_path = os.path.join("plots", f"loss_curves_{run_id}.png")
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"saved {loss_path}")

    # --- Rebuild model for solution heatmaps ---
    model = PINN(**bundle["model_config"])
    model.load_state_dict(bundle["model_state"])
    model.eval()

    L = bundle["L"]
    T = bundle["T"]
    alpha = bundle["alpha"]

    with torch.no_grad():
        xs = torch.linspace(0.0, L, 200)
        ts = torch.linspace(0.0, T, 200)
        X, Tm = torch.meshgrid(xs, ts, indexing="ij")
        x_flat = X.reshape(-1, 1)
        t_flat = Tm.reshape(-1, 1)

        u_pred = model(x_flat, t_flat).reshape(200, 200).numpy()
        k = torch.pi / L
        u_exact = (
            torch.sin(k * x_flat) * torch.exp(-(k ** 2) * alpha * t_flat)
        ).reshape(200, 200).numpy()

    err = np.abs(u_pred - u_exact)

    # --- Prediction heatmap ---
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        u_pred.T,
        origin="lower",
        extent=[0, L, 0, T],
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

    # --- Error heatmap (log scale) ---
    err_floor = max(float(err.min()), 1e-12)
    err_ceiling = float(err.max()) + 1e-12
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        err.T,
        origin="lower",
        extent=[0, L, 0, T],
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


if __name__ == "__main__":
    main()
