#include "eval.hpp"
#include "model.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

// CLI: pinn_eval <run_path> --name <label> [--grid N]
// Loads C++ model state-dict, recreates PINN + PDE from sibling .json, evaluates.
int main(int argc, char* argv[]) {
    std::string run_path;
    std::string name;
    int grid = 100;
    std::string pde_class = "HeatEquation1D";

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--name" && i + 1 < argc) {
            name = argv[++i];
        } else if (a == "--grid" && i + 1 < argc) {
            grid = std::stoi(argv[++i]);
        } else if (a == "--pde" && i + 1 < argc) {
            pde_class = argv[++i];
        } else if (a[0] != '-') {
            run_path = a;
        }
    }

    if (run_path.empty()) {
        try { run_path = resolve_run_path(); }
        catch (const std::exception& e) {
            std::cerr << e.what() << "\n";
            return 1;
        }
    }
    if (name.empty()) {
        std::cerr << "--name is required. e.g.: pinn_eval c_runs/run_x.pt --name baseline\n";
        return 1;
    }

    auto dev = torch::kCPU;

    PINN model;
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(run_path);
        model.load(archive);
        std::cout << "loaded " << run_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "failed to load model: " << e.what() << "\n";
        return 1;
    }

    auto pde = build_pde_from_class(pde_class);
    auto result = evaluate(model, *pde, dev, grid);

    if (result.max_abs_error.has_value()) {
        std::printf("max abs error: %.4e\n", result.max_abs_error.value());
        std::printf("L2 error:      %.4e\n", result.l2_error.value());

        // derive run_id from filename
        auto fname = run_path;
        auto slash = fname.find_last_of("/\\");
        if (slash != std::string::npos) fname = fname.substr(slash + 1);
        if (fname.substr(0, 4) == "run_") fname = fname.substr(4);
        if (fname.size() > 3 && fname.substr(fname.size() - 3) == ".pt")
            fname = fname.substr(0, fname.size() - 3);

        append_eval_result(name, fname, result.max_abs_error.value(), result.l2_error.value());
    } else {
        std::cout << "no analytical solution — prediction only, nothing logged\n";
    }

    return 0;
}
