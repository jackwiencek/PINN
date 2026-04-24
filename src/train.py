import datetime
import os
import time
import torch
from model import PINN
from sampling import sample_collocation, sample_ic, sample_bc
from physics import HeatEquation1D, compute_loss
from eval import evaluate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    pde = HeatEquation1D(L=1.0, T=1.0, alpha=1.0)
    epochs = 10000
    model_config = {"input_dim": 2, "hidden_layers": 6, "neurons": 40}

    # L-BFGS fine-tune phase (runs after Adam)
    lbfgs_epochs = 500
    lbfgs_lr = 1.0
    lbfgs_max_iter = 20
    lbfgs_history_size = 50
    lbfgs_tol_grad = 1e-7
    lbfgs_tol_change = 1e-9

    model = PINN(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # All points, requires_grad = True for collocation
    # fixed for now
    x_f, t_f = sample_collocation(5000, pde, device)
    x_ic, t_ic, _ = sample_ic(200, pde, device)
    t_bc = sample_bc(200, pde, device)

    total_loss_list = []
    col_loss_list = []
    ic_loss_list = []
    bc_loss_list = []

    # NOTE: ckpt only saved during Adam phase; resume after L-BFGS start unsupported
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
            model, pde, x_f, t_f, x_ic, t_ic, None, t_bc
        )

        total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
            x_f, t_f = sample_collocation(5000, pde, device)
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

    # --- L-BFGS fine-tune (full-batch, frozen collocation) ---
    lbfgs = torch.optim.LBFGS(
        model.parameters(),
        lr=lbfgs_lr,
        max_iter=lbfgs_max_iter,
        history_size=lbfgs_history_size,
        tolerance_grad=lbfgs_tol_grad,
        tolerance_change=lbfgs_tol_change,
        line_search_fn="strong_wolfe",
    )
    _last = {}
    t_lbfgs = time.time()
    for lb_epoch in range(lbfgs_epochs):
        def closure():
            lbfgs.zero_grad()
            total, (l_pde, l_ic, l_bc) = compute_loss(
                model, pde, x_f, t_f, x_ic, t_ic, None, t_bc
            )
            total.backward()
            _last["t"], _last["p"], _last["i"], _last["b"] = total, l_pde, l_ic, l_bc
            return total
        lbfgs.step(closure)
        total_loss_list.append(_last["t"].item())
        col_loss_list.append(_last["p"].item())
        ic_loss_list.append(_last["i"].item())
        bc_loss_list.append(_last["b"].item())
        if lb_epoch % 10 == 0:
            print(
                f"lbfgs {lb_epoch}: total={_last['t'].item():.4e} "
                f"pde={_last['p'].item():.4e} ic={_last['i'].item():.4e} "
                f"bc={_last['b'].item():.4e} elapsed={time.time()-t_lbfgs:.1f}s"
            )

    # --- analytical evaluation ---
    result = evaluate(model, pde, device=device, grid=100)
    if "max_abs_error" in result:
        max_err = result["max_abs_error"]
        l2_err = result["l2_error"]
        print(f"max abs error: {max_err:.4e}")
        print(f"L2 error:      {l2_err:.4e}")
    else:
        max_err = None
        l2_err = None

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
            "pde_class": type(pde).__name__,
            "pde_params": {"L": pde.x_max, "T": pde.t_max, "alpha": pde.alpha},
            # backward compat keys
            "L": pde.x_max,
            "T": pde.t_max,
            "alpha": pde.alpha,
            "epochs": epochs,
            "adam_epochs": epochs,
            "lbfgs_epochs": lbfgs_epochs,
            "max_abs_error": max_err,
            "l2_error": l2_err,
        },
        run_path,
    )
    print(f"run saved: {run_path}")


if __name__ == "__main__":
    main()
