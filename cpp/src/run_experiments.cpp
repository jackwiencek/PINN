#include <torch/torch.h>
#include "train.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct ExperimentConfig {
    std::string pde_type;
    bool        use_lbfgs;
    bool        use_resampling;
    std::string run_name;
};

static const std::vector<ExperimentConfig> EXPERIMENTS = {
    {"heat",    false, false, "heat_adam"},
    {"heat",    false, true,  "heat_adam_resample"},
    {"heat",    true,  false, "heat_adam_lbfgs"},
    {"heat",    true,  true,  "heat_adam_resample_lbfgs"},
    {"burgers", false, false, "burgers_adam"},
    {"burgers", false, true,  "burgers_adam_resample"},
    {"burgers", true,  false, "burgers_adam_lbfgs"},
    {"burgers", true,  true,  "burgers_adam_resample_lbfgs"},
};

int main() {
    torch::set_num_interop_threads(1);

    // Collect manifest entries: run_name -> {run_id, run_path}
    std::vector<std::pair<std::string, TrainResult>> manifest;

    for (auto& cfg : EXPERIMENTS) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "starting: " << cfg.run_name << "\n";
        std::cout << std::string(60, '=') << "\n";

        auto result = train(cfg.pde_type, cfg.use_lbfgs, cfg.use_resampling,
                            cfg.run_name, /*num_threads=*/8);
        manifest.push_back({cfg.run_name, result});
    }

    // Write manifest JSON to c_logs/experiment_manifest.json
    std::string manifest_path = "c_logs/experiment_manifest.json";
    std::ofstream mf(manifest_path);
    if (!mf) {
        std::cerr << "warning: cannot write " << manifest_path << "\n";
    } else {
        mf << "{\n";
        for (std::size_t i = 0; i < manifest.size(); ++i) {
            auto& [name, res] = manifest[i];
            mf << "  \"" << name << "\": {\"run_id\": \"" << res.run_id
               << "\", \"run_path\": \"" << res.run_path << "\"}";
            if (i + 1 < manifest.size()) mf << ",";
            mf << "\n";
        }
        mf << "}\n";
        std::cout << "\nmanifest saved: " << manifest_path << "\n";
    }

    std::cout << "\nall done.\n";
    std::cout << "perf CSVs in: c_logs/\n";
    return 0;
}
