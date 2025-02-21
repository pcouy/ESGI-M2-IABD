import torch
from torch import nn
from einops import rearrange

class LinearNeuralStack(nn.Module):
    """
    Implémente des couches de sorties linéaires (*fully connected*) simples
    """
    def __init__(self, layers, in_dim, n_actions, activation=nn.ReLU, initial_biases=None):
        super().__init__()
        linear_layers = []
        for n in layers:
            linear_layers.append(nn.Linear(in_dim, n))
            linear_layers.append(activation())
            in_dim = n
        
        last_layer = nn.Linear(in_dim, n_actions)
        
        self.layers = nn.Sequential(
            *linear_layers,
            last_layer
        )

        self.apply(self._init_weights)
        
    def forward(self, x):
        return self.layers(x)

    def log_tensorboard(self, tensorboard, step, name="linear"):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                tensorboard.add_histogram(f"{name}/{i}/weights", layer.weight, step)
                tensorboard.add_histogram(f"{name}/{i}/biases", layer.bias, step)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module == self.layers[-1]:  # Output layer
                # Smaller initialization for better initial estimates
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)
            else:  # Hidden layers
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(module.bias, 0.1)
