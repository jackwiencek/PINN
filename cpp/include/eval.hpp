#pragma once
#include <torch/torch.h>
#include <optional>
#include <string>
#include "model.hpp"
#include "physics.hpp"

struct EvalResult {
    torch::Tensor u_pred;
    torch::Tensor X;
    torch::Tensor Tm;
    std::optional<torch::Tensor> u_exact;
    std::optional<double> max_abs_error;
    std::optional<double> l2_error;
};

EvalResult evaluate(PINN& model, PDEProblem& pde, torch::Device dev, int grid = 100);

void append_eval_result(const std::string& name, const std::string& run_id,
                         double max_abs_error, double l2_error);

std::string resolve_run_path(const std::optional<std::string>& arg = std::nullopt);

std::unique_ptr<PDEProblem> build_pde_from_class(const std::string& pde_class);
