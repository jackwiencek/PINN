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

        #set initial weights
        self.apply(self._init_weights)

        #Activation function
        self.activation = nn.Tanh()

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            # This is the "Xavier Normal" magic line
            nn.init.xavier_normal_(m.weight)
            # Set biases to zero to start clean
            nn.init.zeros_(m.bias)

    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)   # shape (N, 2)
        for layer in self.layers:
            xt = self.activation(layer(xt))
        return self.output_layer(xt)
    

