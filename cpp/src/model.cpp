#include "model.hpp"

PINN::PINN(int64_t input_dim, int64_t hidden_layers, int64_t neurons)
    : hidden_layers_(hidden_layers), neurons_(neurons) {

    // Input layer
    layers_->push_back(register_module("layer_0", torch::nn::Linear(input_dim, neurons)));

    // Hidden layers
    for (int64_t i = 0; i < hidden_layers; ++i) {
        layers_->push_back(register_module(
            "layer_" + std::to_string(i + 1),
            torch::nn::Linear(neurons, neurons)));
    }

    output_layer_ = register_module("output_layer", torch::nn::Linear(neurons, 1));

    init_weights_();
}

void PINN::init_weights_() {
    for (auto& m : modules(/*include_self=*/false)) {
        if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(m.get())) {
            torch::nn::init::xavier_normal_(linear->weight);
            torch::nn::init::zeros_(linear->bias);
        }
    }
}

torch::Tensor PINN::forward(torch::Tensor x, torch::Tensor t) {
    auto xt = torch::cat({x, t}, /*dim=*/1);
    for (auto& layer : *layers_) {
        xt = torch::tanh(layer->as<torch::nn::Linear>()->forward(xt));
    }
    return output_layer_->forward(xt);
}
