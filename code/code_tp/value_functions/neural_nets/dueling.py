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

    def __init__(
        self,
        layers,
        in_dim,
        n_actions,
        n_atoms=1,
        activation=nn.ReLU,
        initial_biases=None,
        identify_mean=True,
        streams_class=LinearNeuralStack,
        streams_kwargs={},
        device=None,
    ):
        super().__init__()
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.identify_mean = identify_mean
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        # set advantage layer
        self.advantage_layer = streams_class(
            layers,
            in_dim,
            n_actions,
            n_atoms=n_atoms,
            activation=activation,
            initial_biases=initial_biases,
            device=self.device,
            **streams_kwargs,
        )

        # set value layer
        self.value_layer = streams_class(
            layers,
            in_dim,
            1,
            n_atoms=n_atoms,
            activation=activation,
            initial_biases=initial_biases,
            device=self.device,
            **streams_kwargs,
        )

    def forward(self, x):
        value = self.value_layer(x).view(-1, 1, self.n_atoms)
        advantage = self.advantage_layer(x).view(-1, self.n_actions, self.n_atoms)
        if self.identify_mean:
            q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q = value + advantage - advantage.max(dim=-1, keepdim=True)[0]

        return q

    def log_tensorboard(self, tensorboard, step, **kwargs):
        self.advantage_layer.log_tensorboard(
            tensorboard, step, name="advantage", **kwargs
        )
        self.value_layer.log_tensorboard(tensorboard, step, name="value", **kwargs)

    def reset_noise(self, strength=1):
        if hasattr(self.advantage_layer, "reset_noise"):
            self.advantage_layer.reset_noise(strength)
        if hasattr(self.value_layer, "reset_noise"):
            self.value_layer.reset_noise(strength)
