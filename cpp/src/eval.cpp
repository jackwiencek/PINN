#include "eval.hpp"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

static const std::string EVALS_CSV = "c_logs/evals.csv";
static const std::string EVALS_HEADER = "name,timestamp,run_id,max_abs_error,l2_error\n";

EvalResult evaluate(PINN& model, PDEProblem& pde, torch::Device dev, int grid) {
    model.eval();
    torch::NoGradGuard no_grad;

    auto xs = torch::linspace(pde.x_min, pde.x_max, grid,
                               torch::TensorOptions().dtype(torch::kFloat32).device(dev));
    auto ts = torch::linspace(0.0, pde.t_max, grid,
                               torch::TensorOptions().dtype(torch::kFloat32).device(dev));

    auto grids = torch::meshgrid({xs, ts}, "ij");
    auto X  = grids[0];
    auto Tm = grids[1];

    auto x_flat = X.reshape({-1, 1});
    auto t_flat = Tm.reshape({-1, 1});

    auto u_pred = model.forward(x_flat, t_flat);

    EvalResult result;
    result.u_pred = u_pred;
    result.X = X;
    result.Tm = Tm;

    auto u_exact_opt = pde.analytical_solution(x_flat, t_flat);
    if (u_exact_opt.has_value()) {
        auto& u_exact = u_exact_opt.value();
        auto diff = u_pred - u_exact;
        result.u_exact = u_exact;
        result.max_abs_error = diff.abs().max().item<double>();
        result.l2_error = (diff.norm() / u_exact.norm()).item<double>();
    }

    model.train();
    return result;
}

void append_eval_result(const std::string& name, const std::string& run_id,
                         double max_abs_error, double l2_error) {
    fs::create_directories("c_logs");

    bool write_header = !fs::exists(EVALS_CSV);
    std::ofstream f(EVALS_CSV, std::ios::app);
    if (!f) throw std::runtime_error("cannot open " + EVALS_CSV);

    if (write_header) f << EVALS_HEADER;

    // timestamp YYYY-MM-DDTHH:MM:SS
    auto now = std::chrono::system_clock::now();
    auto tt  = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
#ifdef _WIN32
    localtime_s(&tm_buf, &tt);
#else
    localtime_r(&tt, &tm_buf);
#endif
    std::ostringstream ts;
    ts << std::put_time(&tm_buf, "%Y-%m-%dT%H:%M:%S");

    char max_buf[32], l2_buf[32];
    std::snprintf(max_buf, sizeof(max_buf), "%.6e", max_abs_error);
    std::snprintf(l2_buf,  sizeof(l2_buf),  "%.6e", l2_error);

    f << name << "," << ts.str() << "," << run_id << ","
      << max_buf << "," << l2_buf << "\n";

    std::cout << "appended to " << EVALS_CSV << "\n";
}

std::string resolve_run_path(const std::optional<std::string>& arg) {
    if (arg.has_value()) return arg.value();

    std::vector<std::string> candidates;
    if (fs::exists("c_runs")) {
        for (auto& p : fs::directory_iterator("c_runs")) {
            auto s = p.path().string();
            if (s.find("run_") != std::string::npos &&
                p.path().extension() == ".pt") {
                candidates.push_back(s);
            }
        }
    }
    if (candidates.empty())
        throw std::runtime_error("no runs found in c_runs/ — pass a path or run pinn_experiments first");

    std::sort(candidates.begin(), candidates.end());
    return candidates.back();
}

std::unique_ptr<PDEProblem> build_pde_from_class(const std::string& pde_class) {
    if (pde_class == "ViscousBurgers1D")
        return std::make_unique<ViscousBurgers1D>();
    return std::make_unique<HeatEquation1D>();
}
