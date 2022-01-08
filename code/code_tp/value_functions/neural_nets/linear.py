import torch
from torch import nn
from einops import rearrange

class LinearNeuralStack(nn.Module):
    def __init__(self, layers, in_dim, n_actions, activation=nn.ReLU):
        super().__init__()
        linear_layers = []
        for n in layers:
            linear_layers.append(nn.Linear(in_dim, n))
            linear_layers.append(activation())
            in_dim = n
        
        last_layer = nn.Linear(in_dim, n_actions)
        last_layer.weight.data.uniform_(-1e-3,1e-3)
        last_layer.bias.data.uniform_(-1e-3,1e-3)
        
        self.layers = nn.Sequential(
            *linear_layers,
            last_layer
        )
        
    def forward(self, x):
        return self.layers(x)