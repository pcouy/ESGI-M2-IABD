import torch
from torch import nn
from einops import rearrange


class LinearNeuralStack(nn.Module):
    """
    Implémente des couches de sorties linéaires (*fully connected*) simples
    """

    def __init__(
        self, layers, in_dim, n_actions, activation=nn.ReLU, initial_biases=None
    ):
        super().__init__()
        linear_layers = []
        for n in layers:
            linear_layers.append(nn.Linear(in_dim, n))
            linear_layers.append(activation())
            in_dim = n

        self.n_actions = n_actions
        last_layer = nn.Linear(in_dim, n_actions)

        self.layers = nn.Sequential(*linear_layers, last_layer)

        self.apply(self._init_weights)
        self.last_activation = None

    def forward(self, x):
        self.last_activation = self.layers(x)
        return self.last_activation

    def log_tensorboard(self, tensorboard, step, name="linear", action_mapper=str):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                tensorboard.add_histogram(
                    f"nn_params/{name}_{i}_weights", layer.weight, step
                )
                tensorboard.add_histogram(
                    f"nn_params/{name}_{i}_biases", layer.bias, step
                )
                # Log gradients if they exist
                if layer.weight.grad is not None:
                    tensorboard.add_histogram(
                        f"nn_grads/{name}_{i}_weight_grads", layer.weight.grad, step
                    )
                if layer.bias.grad is not None:
                    tensorboard.add_histogram(
                        f"nn_grads/{name}_{i}_bias_grads", layer.bias.grad, step
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

    def _init_weights(self, module):
        module.weight.data.uniform(3e-3, 3e-3)
        module.bias.data.uniform(3e-3, 3e-3)
        print("Linear init")
        return
        if isinstance(module, nn.Linear):
            if module == self.layers[-1]:  # Output layer
                # Smaller initialization for better initial estimates
                nn.init.orthogonal_(module.weight, gain=0.001)
                nn.init.zeros_(module.bias)
            else:  # Hidden layers
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                nn.init.constant_(module.bias, 0.1)
