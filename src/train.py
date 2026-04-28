import csv
import datetime
import os
import time
import torch
from model import PINN
from sampling import sample_collocation, sample_ic, sample_bc
from physics import HeatEquation1D, compute_loss, ViscousBurgers1D
from eval import evaluate, append_eval_result


def train(
    pde_type: str = "burgers",
    use_lbfgs: bool = True,
    use_resampling: bool = True,
    run_name: str = "baseline",
    use_checkpoint: bool = False,
    num_threads: int = 8,
):
    torch.set_num_threads(num_threads)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{run_name}] device: {device}  threads: {num_threads}")

    if pde_type == "heat":
        pde = HeatEquation1D()
    else:
        pde = ViscousBurgers1D()

    epochs = 10000
    model_config = {"input_dim": 2, "hidden_layers": 6, "neurons": 40}

    lbfgs_epochs = 500
    lbfgs_lr = 1.0
    lbfgs_max_iter = 20
    lbfgs_history_size = 50
    lbfgs_tol_grad = 1e-7
    lbfgs_tol_change = 1e-9

    NUM_CORES = num_threads

    run_id = f"{run_name}_{datetime.datetime.now().strftime('%H%M%S')}"
    os.makedirs("py_logs", exist_ok=True)
    perf_csv_path = os.path.join("py_logs", f"perf_{run_id}.csv")
    _perf_file = open(perf_csv_path, "w", newline="")
    _perf_writer = csv.writer(_perf_file)
    _perf_writer.writerow(["epoch", "phase", "wall_time_s", "epoch_time_s"])
    perf_log = []

    model = PINN(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

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
    if use_checkpoint and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["opt"])
        scheduler.load_state_dict(ckpt["sched"])
        start_epoch = ckpt["epoch"] + 1
        print(f"resumed from epoch {start_epoch}")

    t_start = time.time()
    t_epoch_start = time.time()

    for epoch in range(start_epoch, epochs):
        optimizer.zero_grad()

        total_loss, (l_pde, l_ic, l_bc) = compute_loss(
            model, pde, x_f, t_f, x_ic, t_ic, None, t_bc
        )

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss_list.append(total_loss.item())
        col_loss_list.append(l_pde.item())
        ic_loss_list.append(l_ic.item())
        bc_loss_list.append(l_bc.item())

        _now = time.time()
        _epoch_time = _now - t_epoch_start
        _wall_time = _now - t_start
        t_epoch_start = _now
        _perf_writer.writerow([epoch, "adam", round(_wall_time, 6), round(_epoch_time, 6)])
        perf_log.append({"epoch": epoch, "phase": "adam",
                         "wall_time_s": _wall_time, "epoch_time_s": _epoch_time})

        if epoch % 100 == 0:
            elapsed = time.time() - t_start
            lr = scheduler.get_last_lr()[0]
            print(
                f"[{run_name}] epoch {epoch}: total={total_loss.item():.4e} "
                f"pde={l_pde.item():.4e} ic={l_ic.item():.4e} bc={l_bc.item():.4e} "
                f"lr={lr:.2e} elapsed={elapsed:.1f}s"
            )

        if (epoch + 1) % 500 == 0:
            if use_resampling:
                x_f, t_f = sample_collocation(5000, pde, device)
            if use_checkpoint:
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
    print(f"[{run_name}] Adam done in {total_time:.1f}s ({total_time/60:.2f} min)")

    if use_lbfgs:
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
        t_lbfgs_step = time.time()
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

            _now = time.time()
            _step_time = _now - t_lbfgs_step
            _wall_time = _now - t_lbfgs
            t_lbfgs_step = _now

            total_loss_list.append(_last["t"].item())
            col_loss_list.append(_last["p"].item())
            ic_loss_list.append(_last["i"].item())
            bc_loss_list.append(_last["b"].item())

            _perf_writer.writerow([lb_epoch, "lbfgs", round(_wall_time, 6), round(_step_time, 6)])
            perf_log.append({"epoch": lb_epoch, "phase": "lbfgs",
                             "wall_time_s": _wall_time, "epoch_time_s": _step_time})

            if lb_epoch % 10 == 0:
                print(
                    f"[{run_name}] lbfgs {lb_epoch}: total={_last['t'].item():.4e} "
                    f"pde={_last['p'].item():.4e} ic={_last['i'].item():.4e} "
                    f"bc={_last['b'].item():.4e} elapsed={time.time()-t_lbfgs:.1f}s"
                )

    _perf_file.close()
    print(f"perf log: {perf_csv_path}")

    result = evaluate(model, pde, device=device, grid=100)
    if "max_abs_error" in result:
        max_err = result["max_abs_error"]
        l2_err = result["l2_error"]
        print(f"[{run_name}] max abs error: {max_err:.4e}")
        print(f"[{run_name}] L2 error:      {l2_err:.4e}")
        append_eval_result(run_name, run_id, max_err, l2_err)
    else:
        max_err = None
        l2_err = None

    os.makedirs("runs", exist_ok=True)
    run_path = os.path.join("runs", f"run_{run_id}.pt")

    pde_class = type(pde).__name__
    if isinstance(pde, HeatEquation1D):
        pde_params = {"L": pde.x_max, "T": pde.t_max, "alpha": pde.alpha}
        compat_keys = {"L": pde.x_max, "T": pde.t_max, "alpha": pde.alpha}
    else:
        assert isinstance(pde, ViscousBurgers1D)
        pde_params = {"x_min": pde.x_min, "x_max": pde.x_max, "T": pde.t_max, "nu": pde.nu}
        compat_keys = {"x_min": pde.x_min, "x_max": pde.x_max, "T": pde.t_max, "nu": pde.nu}

    torch.save(
        {
            "run_id": run_id,
            "run_name": run_name,
            "total_loss": total_loss_list,
            "pde_loss": col_loss_list,
            "ic_loss": ic_loss_list,
            "bc_loss": bc_loss_list,
            "model_state": model.state_dict(),
            "model_config": model_config,
            "pde_class": pde_class,
            "pde_params": pde_params,
            **compat_keys,
            "epochs": epochs,
            "adam_epochs": epochs,
            "lbfgs_epochs": lbfgs_epochs if use_lbfgs else 0,
            "max_abs_error": max_err,
            "l2_error": l2_err,
            "perf_log": perf_log,
            "num_cores": NUM_CORES,
            "use_lbfgs": use_lbfgs,
            "use_resampling": use_resampling,
        },
        run_path,
    )
    print(f"run saved: {run_path}")
    return run_id, run_path


def main():
    torch.set_num_interop_threads(1)
    train(
        pde_type="burgers",
        use_lbfgs=True,
        use_resampling=True,
        run_name="baseline",
        use_checkpoint=True,
    )


if __name__ == "__main__":
    main()
