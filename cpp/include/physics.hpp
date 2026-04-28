#pragma once
#include <torch/torch.h>
#include <optional>
#include <tuple>
#include "model.hpp"

// ---------------------------------------------------------------------------
// Abstract PDE base
// ---------------------------------------------------------------------------
struct PDEProblem {
    double x_min{0.0};
    double x_max{1.0};
    double t_max{1.0};

    virtual ~PDEProblem() = default;

    virtual torch::Tensor residual(PINN& model,
                                   torch::Tensor x,
                                   torch::Tensor t) = 0;

    virtual torch::Tensor initial_condition(torch::Tensor x) = 0;

    // Returns list of {type, x_val, value} for Dirichlet BCs.
    struct BC { double x_val; double value; };
    virtual std::vector<BC> boundary_conditions() = 0;

    virtual std::optional<torch::Tensor> analytical_solution(torch::Tensor x,
                                                              torch::Tensor t) {
        return std::nullopt;
    }
};

// ---------------------------------------------------------------------------
// Heat Equation: u_t = alpha * u_xx
// ---------------------------------------------------------------------------
class HeatEquation1D : public PDEProblem {
public:
    explicit HeatEquation1D(double L = 1.0, double T = 1.0, double alpha = 1.0);

    torch::Tensor residual(PINN& model, torch::Tensor x, torch::Tensor t) override;
    torch::Tensor initial_condition(torch::Tensor x) override;
    std::vector<BC> boundary_conditions() override;
    std::optional<torch::Tensor> analytical_solution(torch::Tensor x, torch::Tensor t) override;

    double alpha;
};

// ---------------------------------------------------------------------------
// Viscous Burgers: u_t + u*u_x = nu*u_xx
// ---------------------------------------------------------------------------
class ViscousBurgers1D : public PDEProblem {
public:
    explicit ViscousBurgers1D(double x_min = -1.0, double x_max = 1.0,
                               double T = 0.99, double nu = 0.01 / M_PI);

    torch::Tensor residual(PINN& model, torch::Tensor x, torch::Tensor t) override;
    torch::Tensor initial_condition(torch::Tensor x) override;
    std::vector<BC> boundary_conditions() override;
    std::optional<torch::Tensor> analytical_solution(torch::Tensor x, torch::Tensor t) override;

    double nu;
};

// ---------------------------------------------------------------------------
// Loss computation
// ---------------------------------------------------------------------------
struct LossComponents {
    torch::Tensor total;
    torch::Tensor l_pde;
    torch::Tensor l_ic;
    torch::Tensor l_bc;
};

LossComponents compute_loss(PINN& model, PDEProblem& pde,
                             torch::Tensor x_f, torch::Tensor t_f,
                             torch::Tensor x_ic, torch::Tensor t_ic,
                             torch::Tensor t_bcs,
                             std::tuple<double, double, double> lambdas = {1.0, 1.0, 1.0});
