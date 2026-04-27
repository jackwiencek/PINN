#include "physics.hpp"
#include "../third_party/hermite_nodes_50.hpp"
#include <cmath>

// ---------------------------------------------------------------------------
// HeatEquation1D
// ---------------------------------------------------------------------------
HeatEquation1D::HeatEquation1D(double L, double T, double alpha_)
    : alpha(alpha_) {
    x_min = 0.0;
    x_max = L;
    t_max = T;
}

torch::Tensor HeatEquation1D::residual(PINN& model, torch::Tensor x, torch::Tensor t) {
    auto u = model.forward(x, t);
    auto ones_u = torch::ones_like(u);
    auto u_t = torch::autograd::grad({u}, {t}, {ones_u}, /*retain_graph=*/true,
                                      /*create_graph=*/true)[0];
    auto u_x = torch::autograd::grad({u}, {x}, {ones_u}, /*retain_graph=*/true,
                                      /*create_graph=*/true)[0];
    auto u_xx = torch::autograd::grad({u_x}, {x}, {torch::ones_like(u_x)},
                                       /*retain_graph=*/true, /*create_graph=*/true)[0];
    return u_t - alpha * u_xx;
}

torch::Tensor HeatEquation1D::initial_condition(torch::Tensor x) {
    return torch::sin(M_PI * x / x_max);
}

std::vector<PDEProblem::BC> HeatEquation1D::boundary_conditions() {
    return {{x_min, 0.0}, {x_max, 0.0}};
}

std::optional<torch::Tensor> HeatEquation1D::analytical_solution(
    torch::Tensor x, torch::Tensor t) {
    double k = M_PI / x_max;
    return torch::sin(k * x) * torch::exp(-(k * k) * alpha * t);
}

// ---------------------------------------------------------------------------
// ViscousBurgers1D
// ---------------------------------------------------------------------------
ViscousBurgers1D::ViscousBurgers1D(double xmin, double xmax, double T, double nu_)
    : nu(nu_) {
    x_min = xmin;
    x_max = xmax;
    t_max = T;
}

torch::Tensor ViscousBurgers1D::residual(PINN& model, torch::Tensor x, torch::Tensor t) {
    auto u = model.forward(x, t);
    auto ones_u = torch::ones_like(u);
    auto u_t = torch::autograd::grad({u}, {t}, {ones_u}, /*retain_graph=*/true,
                                      /*create_graph=*/true)[0];
    auto u_x = torch::autograd::grad({u}, {x}, {ones_u}, /*retain_graph=*/true,
                                      /*create_graph=*/true)[0];
    auto u_xx = torch::autograd::grad({u_x}, {x}, {torch::ones_like(u_x)},
                                       /*retain_graph=*/true, /*create_graph=*/true)[0];
    return u_t + u * u_x - nu * u_xx;
}

torch::Tensor ViscousBurgers1D::initial_condition(torch::Tensor x) {
    return -torch::sin(M_PI * x);
}

std::vector<PDEProblem::BC> ViscousBurgers1D::boundary_conditions() {
    return {{x_min, 0.0}, {x_max, 0.0}};
}

std::optional<torch::Tensor> ViscousBurgers1D::analytical_solution(
    torch::Tensor x, torch::Tensor t) {
    // Cole-Hopf transform + 50-node Gauss-Hermite quadrature.
    // Mirrors src/physics.py:84-111 exactly, including per-row max(A) subtraction.
    auto dev = x.device();

    auto z_arr = hermite_50::nodes;
    auto w_arr = hermite_50::weights;

    auto z = torch::tensor(std::vector<double>(z_arr.begin(), z_arr.end()),
                            torch::TensorOptions().dtype(torch::kFloat64).device(dev));
    auto w = torch::tensor(std::vector<double>(w_arr.begin(), w_arr.end()),
                            torch::TensorOptions().dtype(torch::kFloat64).device(dev));

    auto x_f = x.to(torch::kFloat64).squeeze(-1);  // (M,)
    auto t_f = t.to(torch::kFloat64).squeeze(-1);  // (M,)

    auto sqrt_4nut = torch::sqrt(4.0 * nu * t_f).unsqueeze(1);           // (M, 1)
    auto y = x_f.unsqueeze(1) - z.unsqueeze(0) * sqrt_4nut;              // (M, Nq)

    auto A = -torch::cos(M_PI * y) / (2.0 * M_PI * nu);
    auto A_max = std::get<0>(A.max(/*dim=*/1, /*keepdim=*/true));
    auto expA = torch::exp(A - A_max);

    auto num = -(w * torch::sin(M_PI * y) * expA).sum(/*dim=*/1);
    auto den =  (w * expA).sum(/*dim=*/1);

    return (num / den).to(x.dtype()).unsqueeze(-1);
}

// ---------------------------------------------------------------------------
// compute_loss
// ---------------------------------------------------------------------------
LossComponents compute_loss(PINN& model, PDEProblem& pde,
                             torch::Tensor x_f, torch::Tensor t_f,
                             torch::Tensor x_ic, torch::Tensor t_ic,
                             torch::Tensor t_bcs,
                             std::tuple<double, double, double> lambdas) {
    auto [lam_pde, lam_ic, lam_bc] = lambdas;

    // PDE residual
    auto res = pde.residual(model, x_f, t_f);
    auto l_pde = torch::mean(res * res);

    // IC loss
    auto u_pred_ic = model.forward(x_ic, t_ic);
    auto u_ic = pde.initial_condition(x_ic);
    auto l_ic = torch::mean(torch::pow(u_pred_ic - u_ic, 2));

    // BC loss
    auto bcs = pde.boundary_conditions();
    std::vector<torch::Tensor> bc_losses;
    int64_t N = t_bcs.size(0);
    for (auto& bc : bcs) {
        auto x_bc = torch::full({N, 1}, bc.x_val,
                                 torch::TensorOptions().dtype(t_bcs.dtype()).device(t_bcs.device()));
        auto u_target = torch::full({N, 1}, bc.value,
                                     torch::TensorOptions().dtype(t_bcs.dtype()).device(t_bcs.device()));
        auto u_pred_bc = model.forward(x_bc, t_bcs);
        bc_losses.push_back(torch::mean(torch::pow(u_pred_bc - u_target, 2)));
    }
    torch::Tensor l_bc;
    if (!bc_losses.empty()) {
        l_bc = torch::stack(bc_losses).mean();
    } else {
        l_bc = torch::zeros({}, x_f.options());
    }

    auto total = lam_pde * l_pde + lam_ic * l_ic + lam_bc * l_bc;
    return {total, l_pde, l_ic, l_bc};
}
