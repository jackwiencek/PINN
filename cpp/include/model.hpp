#pragma once
#include <torch/torch.h>

class PINN : public torch::nn::Module {
public:
    explicit PINN(int64_t input_dim = 2, int64_t hidden_layers = 6, int64_t neurons = 40);

    torch::Tensor forward(torch::Tensor x, torch::Tensor t);

private:
    torch::nn::ModuleList layers_;
    torch::nn::Linear output_layer_{nullptr};
    int64_t hidden_layers_;
    int64_t neurons_;

    void init_weights_();
};
