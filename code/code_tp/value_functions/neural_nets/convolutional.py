import torch
from torch import nn
from einops import rearrange
from .linear import LinearNeuralStack

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
                 output_stack_args={ 'layers': [256] }
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

        if type(pooling) is int or pooling is None:
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
            
            if pooling is not None:
                layers.append(nn.Conv2d(
                    n_in,
                    n_out,
                    kernel_size,
                    1,
                    padding,
                    dilation
                ))
                
                if pooling == "max":
                    layers.append(nn.MaxPool2d(stride+1, stride, 1))
                elif pooling == "avg":
                    layers.append(nn.AvgPool2d(stride+1, stride, 1))
                else:
                    print("Pooling type {} unkwown :  no pooling used")
            else:
                layers.append(nn.Conv2d(
                    n_in,
                    n_out,
                    kernel_size,
                    stride,
                    padding,
                    dilation
                ))
            
            layers.append(activation())
            
        layers.append(nn.Flatten())
        conv_stack = nn.Sequential(*layers)

        x = torch.rand(img_shape)
        y=conv_stack(rearrange([x], 'b w h c -> b c w h'))

        in_size = y.shape[-1]
        last_layers = output_stack_class(
            in_dim=in_size,
            n_actions=n_actions,
            activation=activation,
            **output_stack_args
        )
        
        self.layers = nn.Sequential(
            conv_stack,
            last_layers
        )

    def forward(self, x):
        if len(x.shape) == 4:
            return self.layers(rearrange(x, "b w h c -> b c w h"))
        elif len(x.shape) == 3:
            return self.layers(rearrange([x], "b w h c -> b c w h"))[0]
