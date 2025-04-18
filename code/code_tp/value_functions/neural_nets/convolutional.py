import torch
from torch import nn
from einops import rearrange
from .linear import LinearNeuralStack
from .conv_with_time import SpacioTemporalConv


class ConvolutionalTorso(nn.Module):
    def __init__(
        self,
        img_shape,
        n_filters=None,
        kernel_size=4,
        stride=2,
        padding=0,
        dilation=1,
        activation=nn.Tanh,
        pooling=None,
        device=None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_shape = img_shape

        layers = []
        if n_filters is None:
            n_filters = [16, 16]

        if type(kernel_size) is int:
            kernel_sizes = [kernel_size for _ in n_filters]
        elif type(kernel_size) is list:
            kernel_sizes = kernel_size

        if type(pooling) is str or pooling is None:
            poolings = [pooling for _ in n_filters]
        elif type(pooling) is list:
            poolings = pooling

        if type(padding) is int:
            paddings = [padding for _ in n_filters]
        elif type(padding) is list:
            paddings = padding

        if type(stride) is int:
            strides = [stride for _ in n_filters]
        elif type(stride) is list:
            strides = stride

        n_filters = [img_shape[-1]] + n_filters

        for n_in, n_out, kernel_size, padding, stride, pooling in zip(
            n_filters[:-1], n_filters[1:], kernel_sizes, paddings, strides, poolings
        ):
            i_layer = 0
            if len(layers) == 0 and len(img_shape) == 4:
                conv_class = SpacioTemporalConv
                additional_args = {"time_size": img_shape[-2], "activation": activation}
            else:
                conv_class = nn.Conv2d
                additional_args = {}

            if pooling is not None:
                layers.append(
                    conv_class(
                        n_in,
                        n_out,
                        kernel_size,
                        1,
                        padding,
                        dilation,
                        **additional_args,
                        device=self.device,
                    )
                )

                if pooling == "max":
                    layers.append(nn.MaxPool2d(stride + 1, stride, 1))
                elif pooling == "avg":
                    layers.append(nn.AvgPool2d(stride + 1, stride, 1))
                else:
                    print("Pooling type {} unkwown :  no pooling used")
            else:
                layers.append(
                    conv_class(
                        n_in,
                        n_out,
                        kernel_size,
                        stride,
                        padding,
                        dilation,
                        **additional_args,
                        device=self.device,
                    )
                )

            layers.append(activation())

        layers.append(nn.Flatten())
        self.conv_stack = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def forward(self, x):
        if len(x.shape) == 4:
            return self.conv_stack(rearrange(x, "b w h c -> b c w h"))
        if len(x.shape) == 5:
            return self.conv_stack(rearrange(x, "b w h t c -> b c w h t"))
        if len(x.shape) == 3:
            return self.conv_stack(rearrange([x], "b w h c -> b c w h"))[0]
        raise ValueError(f"Input shape {x.shape} not supported")

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # Kaiming initialization for conv layers
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                # Initialize bias to small positive values to prevent dead ReLUs
                nn.init.constant_(module.bias, 0.1)
        if isinstance(module, SpacioTemporalConv):
            for layer in module.layers:
                if getattr(layer, "weight", None) is not None:
                    nn.init.kaiming_normal_(
                        layer.weight, mode="fan_out", nonlinearity="relu"
                    )
                if getattr(layer, "bias", None) is not None:
                    nn.init.constant_(layer.bias, 0.1)

    def log_tensorboard(self, tensorboard, step):
        for i, layer in enumerate(self.conv_stack):
            if isinstance(layer, nn.Conv2d):
                # Log weights and biases
                tensorboard.add_histogram(
                    f"nn_params/conv_stack_{i}_weights", layer.weight, step
                )
                tensorboard.add_histogram(
                    f"nn_params/conv_stack_{i}_biases", layer.bias, step
                )
                # Log gradients if they exist
                if layer.weight.grad is not None:
                    tensorboard.add_histogram(
                        f"nn_grads/conv_stack_{i}_weight_grads", layer.weight.grad, step
                    )
                if layer.bias.grad is not None:
                    tensorboard.add_histogram(
                        f"nn_grads/conv_stack_{i}_bias_grads", layer.bias.grad, step
                    )
            elif isinstance(layer, SpacioTemporalConv):
                for j, inner_layer in enumerate(layer.layers):
                    if getattr(inner_layer, "weight", None) is not None:
                        tensorboard.add_histogram(
                            f"nn_params/conv_stack_{i}-{j}_weights",
                            inner_layer.weight,
                            step,
                        )
                    else:
                        continue
                    if getattr(inner_layer, "bias", None) is not None:
                        tensorboard.add_histogram(
                            f"nn_params/conv_stack_{i}-{j}_biases",
                            inner_layer.bias,
                            step,
                        )
                    else:
                        continue
                    if inner_layer.weight.grad is not None:
                        tensorboard.add_histogram(
                            f"nn_grads/conv_stack_{i}-{j}_weight_grads",
                            inner_layer.weight.grad,
                            step,
                        )
                    if inner_layer.bias.grad is not None:
                        tensorboard.add_histogram(
                            f"nn_grads/conv_stack_{i}-{j}_bias_grads",
                            inner_layer.bias.grad,
                            step,
                        )


class ConvolutionalNN(nn.Module):
    """
    Implémente un réseau de neurones convolutionnel en PyTorch.
    """

    def __init__(
        self,
        img_shape,
        n_actions,
        n_atoms=1,
        torso_class=ConvolutionalTorso,
        torso_args={},
        activation=nn.Tanh,
        output_stack_class=LinearNeuralStack,
        output_stack_args={"layers": [256]},
        embedding_dim=None,
        embedding_size=None,
        device=None,
    ):
        """
        * `img_shape` est un tuple représentant la forme d'une image passée en entrée du réseau
        de neurones (telle que donnée par np_arrah.shape)
        * `n_actions` est le nombre d'actions à évaluer, càd le nombre de sorties du réseau de
        neurones
        * L'argument `n_filters` doit être une liste contenant un élément par couche
        convolutionnelle.
        * Les arguments ` kernel_size`, ` stride`, `padding` et `pooling` peuvent être des entiers
        ou des listes. S'il s'agit d'un entier, la valeur de cet entier sera utilisée
        pour toutes les couches convolutionnelles. Si la valeur est une liste, la liste doit
        contenir le même nombre d'éléments que `n_filters`
        * `dilation` est un entier
        * `activation` est la fonction d'activation utilisée entre chaque couche convolutionnelle.
        Il doit s'agir d'une fonction d'activation PyTorch
        * `pooling` permet de remplacer le *stride* des convolutions par une couche de
        pooling en sortie si sa valeur est différente de `None`. Les valeurs possibles sont :
        `None`, `"max"` et `"avg"`
        * `output_stack_class` et `output_stack_args` sont utilisés pour instancier le module Torch
        qui fera le lien entre la sortie des couches convolutionnelles et la sortie à `n_actions`
        éléments du DQN
        * `embedding_dim` est la dimension du vecteur d'embedding pour l'entrée secondaire. Si None,
        aucun embedding n'est utilisé
        * `embedding_size` est la taille du vocabulaire pour l'embedding (nombre de valeurs discrètes
        possibles). Doit être spécifié si embedding_dim est utilisé

        Si vous ne comprenez pas la signification d'un argument, la documentation de PyTorch
        et/ou un tutoriel sur les réseaux de neurones convolutionnels devrait éclaircir le
        rôle de chacun de ces paramètres
        """
        super().__init__()
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_shape = img_shape
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.conv_stack = torso_class(img_shape, activation=activation, **torso_args)

        # Get the output size of the torso
        x = torch.rand((1, *img_shape), device=self.device)
        y = self.conv_stack(x)
        in_size = y.shape[-1]

        # Add embedding layer if specified
        self.embedding = None
        if embedding_dim is not None and embedding_size is not None:
            self.embedding = nn.Embedding(embedding_size, embedding_dim, device=self.device)
            in_size += embedding_dim  # Increase input size for the linear stack

        self.last_layers = self._init_output_stack(
            output_stack_class, in_size, n_actions, activation, output_stack_args
        )

    def _init_output_stack(
        self, output_stack_class, in_size, n_actions, activation, output_stack_args
    ):
        last_layers = output_stack_class(
            in_dim=in_size,
            n_actions=n_actions,
            n_atoms=self.n_atoms,
            activation=activation,
            **output_stack_args,
            device=self.device,
        )
        return last_layers

    def forward(self, x, secondary_input=None):
        if x.device != self.device:
            x.to(self.device)
        if secondary_input is not None and secondary_input.device != self.device:
            secondary_input.to(self.device)
        conv_out = self.conv_stack(x)

        # Handle secondary input if provided
        if secondary_input is not None and self.embedding is not None:
            embedded = self.embedding(secondary_input)
            if len(conv_out.shape) == 1:  # Single sample case
                conv_out = torch.cat([conv_out, embedded.squeeze(0)], dim=0)
            else:  # Batch case
                conv_out = torch.cat([conv_out, embedded], dim=1)

        # Add 1 to the output so the output layers targets are closer to 0
        # (should work better with reward scaling buffer, which targets an avergage
        # of 1 for discounted returns)
        return self.last_layers(conv_out)

    def reset_noise(self, strength=1):
        if callable(getattr(self.last_layers, "reset_noise", None)):
            self.last_layers.reset_noise(strength)

    def log_tensorboard(self, tensorboard, step, action_mapper=str, n_actions=None):
        if n_actions is None:
            n_actions = self.n_actions
        if self.embedding is not None:
            tensorboard.add_embedding(
                self.embedding.weight,
                global_step=step,
                metadata=[action_mapper(i) for i in range(n_actions)],
            )
        if callable(getattr(self.last_layers, "log_tensorboard", None)):
            self.last_layers.log_tensorboard(
                tensorboard, step, action_mapper=action_mapper
            )
        if callable(getattr(self.conv_stack, "log_tensorboard", None)):
            self.conv_stack.log_tensorboard(tensorboard, step)
