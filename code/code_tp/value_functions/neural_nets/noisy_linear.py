import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self, mu_range=None):
        """Reset trainable network parameters (factorized gaussian noise)."""
        if mu_range is None:
            mu_range = 1 / math.sqrt(self.in_features)
        else:
            mu_range = mu_range / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self, strength=1):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(strength * epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(strength * epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class NoisyLinearNeuralStack(nn.Module):
    """
    Implémente des couches de sorties linéaires (*fully connected*) simples
    """

    def __init__(
        self,
        layers,
        in_dim,
        n_actions,
        activation=nn.ReLU,
        initial_biases=None,
        std_init=0.5,
    ):
        super().__init__()
        linear_layers = []
        self.n_actions = n_actions
        for n in layers:
            linear_layers.append(NoisyLinear(in_dim, n, std_init=std_init))
            linear_layers.append(activation())
            in_dim = n

        last_layer = NoisyLinear(in_dim, n_actions, std_init=std_init)
        last_layer.reset_parameters(mu_range=1e-3)
        self.layers = nn.Sequential(*linear_layers, last_layer)

        self.last_activation = None

    def forward(self, x):
        self.last_activation = self.layers(x)
        return self.last_activation

    def reset_noise(self, strength=1):
        """Reset all noisy layers."""
        for layer in self.layers:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise(strength)

    def log_tensorboard(self, tensorboard, step, name="linear", action_mapper=str):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, NoisyLinear):
                tensorboard.add_histogram(
                    f"nn_params/{name}_{i}_weights", layer.weight_mu, step
                )
                tensorboard.add_histogram(
                    f"nn_params/{name}_{i}_biases", layer.bias_mu, step
                )
                tensorboard.add_histogram(
                    f"nn_params/{name}_{i}_weights_sigma", layer.weight_sigma, step
                )
                tensorboard.add_histogram(
                    f"nn_params/{name}_{i}_biases_sigma", layer.bias_sigma, step
                )
                tensorboard.add_scalar(
                    f"nn_noise/{name}_{i}_weights_sigma_norm",
                    layer.weight_sigma.norm(),
                    step,
                )
                tensorboard.add_scalar(
                    f"nn_noise/{name}_{i}_biases_sigma_norm",
                    layer.bias_sigma.norm(),
                    step,
                )
                # Log gradients if they exist
                if layer.weight_mu.grad is not None:
                    tensorboard.add_histogram(
                        f"nn_grads/{name}_{i}_weight_grads", layer.weight_mu.grad, step
                    )
                if layer.bias_mu.grad is not None:
                    tensorboard.add_histogram(
                        f"nn_grads/{name}_{i}_bias_grads", layer.bias_mu.grad, step
                    )
                if layer.weight_sigma.grad is not None:
                    tensorboard.add_histogram(
                        f"nn_grads/{name}_{i}_weight_sigma_grads",
                        layer.weight_sigma.grad,
                        step,
                    )
                if layer.bias_sigma.grad is not None:
                    tensorboard.add_histogram(
                        f"nn_grads/{name}_{i}_bias_sigma_grads",
                        layer.bias_sigma.grad,
                        step,
                    )
        if self.last_activation is not None:
            tensorboard.add_histogram(f"{name}_activation", self.last_activation, step)
            if self.n_actions > 1:
                for i in range(self.n_actions):
                    tensorboard.add_histogram(
                        f"{name}_activation/{action_mapper(i)}",
                        self.last_activation[:, i],
                        step,
                    )
