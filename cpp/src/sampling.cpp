#include "sampling.hpp"

std::pair<torch::Tensor, torch::Tensor>
sample_collocation(int N, PDEProblem& pde, torch::Device dev) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto x = (torch::rand({N, 1}, opts) * (pde.x_max - pde.x_min) + pde.x_min);
    auto t = (torch::rand({N, 1}, opts) * pde.t_max);
    x.set_requires_grad(true);
    t.set_requires_grad(true);
    return {x, t};
}

std::pair<torch::Tensor, torch::Tensor>
sample_ic(int N, PDEProblem& pde, torch::Device dev) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    auto x = torch::rand({N, 1}, opts) * (pde.x_max - pde.x_min) + pde.x_min;
    auto t = torch::zeros({N, 1}, opts);
    return {x, t};
}

torch::Tensor
sample_bc(int N, PDEProblem& pde, torch::Device dev) {
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(dev);
    return torch::rand({N, 1}, opts) * pde.t_max;
}
