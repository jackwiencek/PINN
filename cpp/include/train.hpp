#pragma once
#include <string>

struct TrainResult {
    std::string run_id;
    std::string run_path;
};

TrainResult train(const std::string& pde_type     = "burgers",
                  bool               use_lbfgs     = true,
                  bool               use_resampling = true,
                  const std::string& run_name       = "baseline",
                  int                num_threads    = 8);
