import argparse
import glob
import os
import sys
import torch

from model import PINN
from physics import HeatEquation1D, ViscousBurgers1D


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
    args = parser.parse_args()

    run_path = resolve_run_path(args.run)
    bundle = load_run(run_path)

    model = PINN(**bundle["model_config"])
    model.load_state_dict(bundle["model_state"])

    pde = build_pde_from_bundle(bundle)
    result = evaluate(model, pde, device="cpu", grid=args.grid)

    if "max_abs_error" in result:
        print(f"max abs error: {result['max_abs_error']:.4e}")
        print(f"L2 error:      {result['l2_error']:.4e}")
    else:
        print("no analytical solution — prediction only")


if __name__ == "__main__":
    main()
