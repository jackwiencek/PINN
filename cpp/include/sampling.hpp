#pragma once
#include <torch/torch.h>
#include <tuple>
#include "physics.hpp"

std::pair<torch::Tensor, torch::Tensor>
sample_collocation(int N, PDEProblem& pde, torch::Device dev);

std::pair<torch::Tensor, torch::Tensor>
sample_ic(int N, PDEProblem& pde, torch::Device dev);

torch::Tensor
sample_bc(int N, PDEProblem& pde, torch::Device dev);
