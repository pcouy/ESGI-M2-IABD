from .convolutional import ConvolutionalNN
from .linear import LinearNeuralStack
import torch
from torch import nn
from einops import rearrange
import copy

class DuelingOutputStack(nn.Module):
    """
    Implémente les couches de sorties du réseau de neurones telles que décrites
    dans [Dueling Network Architectures for Deep Reinforcement Learning](http://arxiv.org/abs/1511.06581)
    """
    def __init__(self, layers, in_dim, n_actions, activation=nn.ReLU, initial_biases=None):
        super().__init__()
        # set advantage layer
        self.advantage_layer = LinearNeuralStack(layers, in_dim, n_actions, activation, initial_biases)

        # set value layer
        self.value_layer = LinearNeuralStack(layers, in_dim, 1, activation)
        
    def forward(self, x):
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)
        
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q

    def log_tensorboard(self, tensorboard, step, **kwargs):
       self.advantage_layer.log_tensorboard(tensorboard, step, name="advantage", **kwargs)
       self.value_layer.log_tensorboard(tensorboard, step, name="value", **kwargs)
