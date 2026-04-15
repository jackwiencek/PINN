import argparse
import glob
import os
import sys
import torch

from model import PINN


def evaluate(model, L, T, alpha, device="cpu", grid=100):
    model.eval()
    with torch.no_grad():
        xs = torch.linspace(0.0, L, grid, device=device)
        ts = torch.linspace(0.0, T, grid, device=device)
        X, Tm = torch.meshgrid(xs, ts, indexing="ij")
        x_flat = X.reshape(-1, 1)
        t_flat = Tm.reshape(-1, 1)

        u_pred = model(x_flat, t_flat)
        k = torch.pi / L
        u_exact = torch.sin(k * x_flat) * torch.exp(-(k ** 2) * alpha * t_flat)

        max_err = (u_pred - u_exact).abs().max().item()
        l2_err = ((u_pred - u_exact) ** 2).mean().sqrt().item()

    return {
        "max_abs_error": max_err,
        "l2_error": l2_err,
        "u_pred": u_pred,
        "u_exact": u_exact,
        "X": X,
        "Tm": Tm,
    }


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


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved PINN run against the analytical 1D heat solution.")
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
    args = parser.parse_args()

    run_path = resolve_run_path(args.run)
    bundle = load_run(run_path)

    model = PINN(**bundle["model_config"])
    model.load_state_dict(bundle["model_state"])

    result = evaluate(
        model,
        L=bundle["L"],
        T=bundle["T"],
        alpha=bundle["alpha"],
        device="cpu",
        grid=args.grid,
    )

    print(f"max abs error: {result['max_abs_error']:.4e}")
    print(f"L2 error:      {result['l2_error']:.4e}")


if __name__ == "__main__":
    main()
