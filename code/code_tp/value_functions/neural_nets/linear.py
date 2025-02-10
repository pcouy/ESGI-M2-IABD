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
        last_layer.weight.data.uniform_(-1e-3,1e-3)
        if initial_biases is None:
            last_layer.bias.data.uniform_(-1e-3,1e-3)
        elif type(initial_biases) is int:
            last_layer.bias.data = torch.Tensor([initial_biases for _ in range(n_actions)])
        else:
            last_layer.bias.data = torch.Tensor(initial_biases)
        
        self.layers = nn.Sequential(
            *linear_layers,
            last_layer
        )
        
    def forward(self, x):
        return self.layers(x)
