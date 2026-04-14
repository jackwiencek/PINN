import torch
import torch.nn as nn


class PINN(nn.Module):
    def __init__(self, input_dim=2, hidden_layers=6, neurons=40):
        super().__init__()

        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(input_dim, neurons))
        
        # Add 'n' hidden layers
        for _ in range(hidden_layers):
            self.layers.append(nn.Linear(neurons, neurons))
            
        # Output layer
        self.output_layer = nn.Linear(neurons, 1)

        #Activation function
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.output_layer(x)
    
device = "cuda" if torch.cuda.is_available() else "cpu"

