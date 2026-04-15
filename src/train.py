import datetime
import os
import time
import torch
import torch.nn as nn
from model import PINN
from sampling import sample_collocation, sample_ic, sample_bc
from physics import compute_loss
from eval import evaluate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    L = 1.0
    T = 1.0
    alpha = 1.0
    epochs = 10000
    model_config = {"input_dim": 2, "hidden_layers": 6, "neurons": 40}

    model = PINN(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # All points, requires_grad = True for collocation
    # fixed for now
    x_f, t_f = sample_collocation(5000, L, T, device)
    x_ic, t_ic, u_ic = sample_ic(200, L, device)
    x_bc_left, x_bc_right, t_bc, u_bc_left, u_bc_right = sample_bc(200, L, T, device)

    total_loss_list = []
    col_loss_list = []
    ic_loss_list = []
    bc_loss_list = []

    ckpt_path = "ckpt.pt"
    start_epoch = 0
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")

    t_start = time.time()

    # Training Loop
    for epoch in range(start_epoch, epochs):
        optimizer.zero_grad()

        total_loss, (l_pde, l_ic, l_bc) = compute_loss(
            model, x_f, t_f, x_ic, t_ic, u_ic,
            x_bc_left, x_bc_right, t_bc, u_bc_left, u_bc_right, alpha
        )

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # .item() detaches — avoids holding the autograd graph in a list
        total_loss_list.append(total_loss.item())
        col_loss_list.append(l_pde.item())
        ic_loss_list.append(l_ic.item())
        bc_loss_list.append(l_bc.item())

        if epoch % 100 == 0:
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            print(
                f"epoch {epoch}: total={total_loss.item():.4e} "
                f"pde={l_pde.item():.4e} ic={l_ic.item():.4e} bc={l_bc.item():.4e} "
                f"lr={lr:.2e} elapsed={elapsed:.1f}s"
            )

        if (epoch + 1) % 500 == 0:
            x_f, t_f = sample_collocation(5000, L, T, device)
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": optimizer.state_dict(),
                    "sched": scheduler.state_dict(),
                    "epoch": epoch,
                },
                ckpt_path,
            )

    total_time = time.time() - t_start
    print(f"training done in {total_time:.1f}s ({total_time/60:.2f} min)")

    # --- analytical evaluation ---
    result = evaluate(model, L, T, alpha, device=device, grid=100)
    max_err = result["max_abs_error"]
    l2_err = result["l2_error"]
    print(f"max abs error: {max_err:.4e}")
    print(f"L2 error:      {l2_err:.4e}")

    # --- save run bundle (loss lists + model + metadata) ---
    os.makedirs("runs", exist_ok=True)
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = os.path.join("runs", f"run_{run_id}.pt")
    torch.save(
        {
            "run_id": run_id,
            "total_loss": total_loss_list,
            "pde_loss": col_loss_list,
            "ic_loss": ic_loss_list,
            "bc_loss": bc_loss_list,
            "model_state": model.state_dict(),
            "model_config": model_config,
            "L": L,
            "T": T,
            "alpha": alpha,
            "epochs": epochs,
            "max_abs_error": max_err,
            "l2_error": l2_err,
        },
        run_path,
    )
    print(f"run saved: {run_path}")


if __name__ == "__main__":
    main()
