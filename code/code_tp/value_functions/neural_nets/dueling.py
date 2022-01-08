from .convolutional import ConvolutionalNN
from .linear import LinearNeuralStack
import torch
from torch import nn
from einops import rearrange
import copy

class DuelingOutputStack(nn.Module):
    def __init__(self, layers, in_dim, n_actions, activation=nn.ReLU):
        super().__init__()
        # set advantage layer
        self.advantage_layer = LinearNeuralStack(layers, in_dim, n_actions, activation)

        # set value layer
        self.value_layer = LinearNeuralStack(layers, in_dim, 1, activation)
        
    def forward(self, x):
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q