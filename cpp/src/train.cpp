#include "train.hpp"
#include "model.hpp"
#include "physics.hpp"
#include "sampling.hpp"
#include "eval.hpp"

#include <torch/torch.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

static std::string now_hhmmss() {
    auto tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    std::ostringstream s;
    s << std::put_time(&tm_buf, "%H%M%S");
    return s.str();
}

// Inline cosine annealing: lr = 0.5 * base_lr * (1 + cos(pi * epoch / T_max))
static void cosine_anneal_lr(torch::optim::Adam& opt, double base_lr, int epoch, int T_max) {
    double lr = 0.5 * base_lr * (1.0 + std::cos(M_PI * epoch / T_max));
    for (auto& pg : opt.param_groups()) {
        static_cast<torch::optim::AdamOptions&>(pg.options()).lr(lr);
    }
}

TrainResult train(const std::string& pde_type,
                  bool               use_lbfgs,
                  bool               use_resampling,
                  const std::string& run_name,
                  int                num_threads) {

    torch::set_num_threads(num_threads);
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::printf("[%s] device: %s  threads: %d\n",
                run_name.c_str(),
                torch::cuda::is_available() ? "cuda" : "cpu",
                num_threads);

    std::unique_ptr<PDEProblem> pde;
    if (pde_type == "heat") {
        pde = std::make_unique<HeatEquation1D>();
    } else {
        pde = std::make_unique<ViscousBurgers1D>();
    }

    const int epochs       = 10000;
    const int lbfgs_epochs = 500;
    const double base_lr   = 1e-3;
    const int N_col        = 5000;
    const int N_ic         = 200;
    const int N_bc         = 200;

    auto run_id = run_name + "_" + now_hhmmss();
    fs::create_directories("c_logs");
    std::string perf_csv = "c_logs/perf_" + run_id + ".csv";
    std::ofstream perf_file(perf_csv);
    if (!perf_file) throw std::runtime_error("cannot open " + perf_csv);
    perf_file << "epoch,phase,wall_time_s,epoch_time_s\n";

    std::string loss_csv = "c_logs/loss_" + run_id + ".csv";
    std::ofstream loss_file(loss_csv);
    if (!loss_file) throw std::runtime_error("cannot open " + loss_csv);
    loss_file << "epoch,phase,total_loss,pde_loss,ic_loss,bc_loss\n";

    PINN model;
    model.to(device);

    torch::optim::Adam adam(model.parameters(), torch::optim::AdamOptions(base_lr));

    auto [x_f, t_f] = sample_collocation(N_col, *pde, device);
    auto [x_ic, t_ic] = sample_ic(N_ic, *pde, device);
    auto t_bc = sample_bc(N_bc, *pde, device);

    auto t_start      = Clock::now();
    auto t_epoch_start = t_start;

    // -----------------------------------------------------------------------
    // Adam phase
    // -----------------------------------------------------------------------
    for (int epoch = 0; epoch < epochs; ++epoch) {
        model.train();
        adam.zero_grad();

        auto loss = compute_loss(model, *pde, x_f, t_f, x_ic, t_ic, t_bc);
        loss.total.backward();
        adam.step();
        cosine_anneal_lr(adam, base_lr, epoch, epochs);

        loss_file << epoch << ",adam,"
                  << std::scientific << std::setprecision(6)
                  << loss.total.item<double>() << ","
                  << loss.l_pde.item<double>()  << ","
                  << loss.l_ic.item<double>()   << ","
                  << loss.l_bc.item<double>()   << "\n";

        auto now        = Clock::now();
        double wall     = Seconds(now - t_start).count();
        double ep_time  = Seconds(now - t_epoch_start).count();
        t_epoch_start   = now;

        perf_file << epoch << ",adam,"
                  << std::fixed << std::setprecision(6) << wall << ","
                  << ep_time << "\n";

        if (epoch % 100 == 0) {
            std::printf("[%s] epoch %d: total=%.4e pde=%.4e ic=%.4e bc=%.4e elapsed=%.1fs\n",
                        run_name.c_str(), epoch,
                        loss.total.item<double>(),
                        loss.l_pde.item<double>(),
                        loss.l_ic.item<double>(),
                        loss.l_bc.item<double>(),
                        wall);
        }

        if ((epoch + 1) % 500 == 0 && use_resampling) {
            auto [xf2, tf2] = sample_collocation(N_col, *pde, device);
            x_f = xf2;
            t_f = tf2;
        }
    }

    double adam_total = Seconds(Clock::now() - t_start).count();
    std::printf("[%s] Adam done in %.1fs (%.2f min)\n",
                run_name.c_str(), adam_total, adam_total / 60.0);

    // -----------------------------------------------------------------------
    // L-BFGS phase
    // -----------------------------------------------------------------------
    if (use_lbfgs) {
        torch::optim::LBFGSOptions lbfgs_opts(1.0);
        lbfgs_opts.max_iter(20);
        lbfgs_opts.history_size(50);
        lbfgs_opts.tolerance_grad(1e-7);
        lbfgs_opts.tolerance_change(1e-9);
        lbfgs_opts.line_search_fn("strong_wolfe");
        torch::optim::LBFGS lbfgs(model.parameters(), lbfgs_opts);

        struct Last { torch::Tensor t, p, i, b; } last;

        auto t_lbfgs       = Clock::now();
        auto t_lbfgs_step  = t_lbfgs;

        for (int lb = 0; lb < lbfgs_epochs; ++lb) {
            auto closure = [&]() -> torch::Tensor {
                lbfgs.zero_grad();
                auto lc = compute_loss(model, *pde, x_f, t_f, x_ic, t_ic, t_bc);
                lc.total.backward();
                last = {lc.total, lc.l_pde, lc.l_ic, lc.l_bc};
                return lc.total;
            };
            lbfgs.step(closure);

            loss_file << lb << ",lbfgs,"
                      << std::scientific << std::setprecision(6)
                      << last.t.item<double>() << ","
                      << last.p.item<double>() << ","
                      << last.i.item<double>() << ","
                      << last.b.item<double>() << "\n";

            auto now       = Clock::now();
            double wall    = Seconds(now - t_lbfgs).count();
            double step_t  = Seconds(now - t_lbfgs_step).count();
            t_lbfgs_step   = now;

            perf_file << lb << ",lbfgs,"
                      << std::fixed << std::setprecision(6) << wall << ","
                      << step_t << "\n";

            if (lb % 10 == 0) {
                std::printf("[%s] lbfgs %d: total=%.4e pde=%.4e ic=%.4e bc=%.4e elapsed=%.1fs\n",
                            run_name.c_str(), lb,
                            last.t.item<double>(),
                            last.p.item<double>(),
                            last.i.item<double>(),
                            last.b.item<double>(),
                            wall);
            }
        }
    }

    perf_file.close();
    loss_file.close();
    std::cout << "perf log:  " << perf_csv << "\n";
    std::cout << "loss log:  " << loss_csv << "\n";

    // -----------------------------------------------------------------------
    // Evaluation
    // -----------------------------------------------------------------------
    auto result = evaluate(model, *pde, device);
    if (result.max_abs_error.has_value()) {
        std::printf("[%s] max abs error: %.4e\n", run_name.c_str(), result.max_abs_error.value());
        std::printf("[%s] L2 error:      %.4e\n", run_name.c_str(), result.l2_error.value());
        append_eval_result(run_name, run_id,
                           result.max_abs_error.value(), result.l2_error.value());
    }

    // -----------------------------------------------------------------------
    // Save model state
    // -----------------------------------------------------------------------
    fs::create_directories("c_runs");
    std::string run_path = "c_runs/run_" + run_id + ".pt";
    torch::serialize::OutputArchive archive;
    model.save(archive);
    archive.save_to(run_path);
    std::cout << "run saved: " << run_path << "\n";

    return {run_id, run_path};
}
