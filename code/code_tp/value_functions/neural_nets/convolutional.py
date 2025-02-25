import torch
from torch import nn
from einops import rearrange
from .linear import LinearNeuralStack
from .conv_with_time import SpacioTemporalConv

class ConvolutionalNN(nn.Module):
    """
    Implémente un réseau de neurones convolutionnel en PyTorch.
    """
    def __init__(self,
                 img_shape,
                 n_actions,
                 n_filters=None,
                 kernel_size=4,
                 stride=2,
                 padding=0,
                 dilation=1,
                 activation=nn.Tanh,
                 pooling=None,
                 output_stack_class=LinearNeuralStack,
                 output_stack_args={ 'layers': [256] },
                 embedding_dim=None,
                 embedding_size=None
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
        self.img_shape = img_shape
        self.n_actions = n_actions
        layers = []
        if n_filters is None:
            n_filters = [16,16]

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
                additional_args = {"time_size": img_shape[-2]}
            else:
                conv_class = nn.Conv2d
                additional_args = {}
            
            if pooling is not None:
                layers.append(conv_class(
                    n_in,
                    n_out,
                    kernel_size,
                    1,
                    padding,
                    dilation,
                    **additional_args,
                ))
                
                if pooling == "max":
                    layers.append(nn.MaxPool2d(stride+1, stride, 1))
                elif pooling == "avg":
                    layers.append(nn.AvgPool2d(stride+1, stride, 1))
                else:
                    print("Pooling type {} unkwown :  no pooling used")
            else:
                layers.append(conv_class(
                    n_in,
                    n_out,
                    kernel_size,
                    stride,
                    padding,
                    dilation,
                    **additional_args,
                ))
            
            layers.append(activation())
            
        layers.append(nn.Flatten())
        conv_stack = nn.Sequential(*layers)

        x = torch.rand(img_shape)
        if len(img_shape) == 3:
            y=conv_stack(rearrange([x], 'b w h c -> b c w h'))
        else:
            y=conv_stack(rearrange([x], 'b w h t c -> b c w h t'))

        in_size = y.shape[-1]
        
        # Add embedding layer if specified
        self.embedding = None
        if embedding_dim is not None and embedding_size is not None:
            self.embedding = nn.Embedding(embedding_size, embedding_dim)
            in_size += embedding_dim  # Increase input size for the linear stack

        last_layers = output_stack_class(
            in_dim=in_size,
            n_actions=n_actions,
            activation=activation,
            **output_stack_args
        )
        
        self.conv_stack = conv_stack
        self.last_layers = last_layers
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # Kaiming initialization for conv layers
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                # Initialize bias to small positive values to prevent dead ReLUs
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x, secondary_input=None):
        if len(x.shape) == 4:
            conv_out = self.conv_stack(rearrange(x, "b w h c -> b c w h"))
        elif len(x.shape) == 5:
            conv_out = self.conv_stack(rearrange(x, "b w h t c -> b c w h t"))
        elif len(x.shape) == 3:
            conv_out = self.conv_stack(rearrange([x], "b w h c -> b c w h"))[0]
            
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
        return self.last_layers(conv_out) + 1

    def log_tensorboard(self, tensorboard, step, action_mapper=str):
        if self.embedding is not None:
            tensorboard.add_embedding(
                self.embedding.weight,
                global_step=step,
                metadata=[action_mapper(i) for i in range(self.n_actions)],
            )
        if callable(getattr(self.last_layers, "log_tensorboard", None)):
            self.last_layers.log_tensorboard(tensorboard, step, action_mapper=action_mapper)
        for i, layer in enumerate(self.conv_stack):
            if isinstance(layer, nn.Conv2d):
                tensorboard.add_histogram(f"nn_params/conv_stack_{i}_weights", layer.weight, step)
                tensorboard.add_histogram(f"nn_params/conv_stack_{i}_biases", layer.bias, step)
