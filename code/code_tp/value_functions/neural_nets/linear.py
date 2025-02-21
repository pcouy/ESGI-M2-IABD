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
        nn.init.orthogonal_(last_layer.weight, 1e-2)
        if initial_biases is None:
            nn.init.xavier_uniform_(last_layer.weight, 1e-2)
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

    def log_tensorboard(self, tensorboard, step, name="linear"):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                tensorboard.add_histogram(f"{name}/{i}/weights", layer.weight, step)
                tensorboard.add_histogram(f"{name}/{i}/biases", layer.bias, step)
