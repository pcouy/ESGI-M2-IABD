import numpy as np
import matplotlib.pyplot as plt
from .base import DiscreteQFunction
import torch
from torch import nn
from einops import rearrange

class ConvolutionalNN(nn.Module):
    def __init__(self,
                 img_shape,
                 n_actions,
                 n_filters=None,
                 kernel_size=4,
                 stride=2,
                 padding=0,
                 dilation=1,
                 hidden_linear=[],
                 activation=nn.Tanh,
                 pooling=None
            ):
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

        if type(pooling) is int:
            poolings = [pooling for _ in n_filters]
        elif type(pooling) is list:
            poolings = pooling

        if type(padding) is int:
            paddings = [padding for _ in n_filters]
        elif type(padding) is list:
            paddings = padding

        n_filters = [img_shape[-1]] + n_filters

        for n_in, n_out, kernel_size, padding, pooling in zip(
                n_filters[:-1], n_filters[1:], kernel_sizes, paddings, poolings
        ):
            i_layer = 0
            layers.append(nn.Conv2d(
                n_in,
                n_out,
                kernel_size,
                stride,
                padding,
                dilation
            ))
            if pooling is not None:
                if pooling == "max":
                    layers.append(nn.MaxPool2d(3, 2, 1))
                elif pooling == "avg":
                    layers.append(nn.AvgPool2d(3, 2, 1))
                else:
                    print("Pooling type {} unkwown :  no pooling used")
            layers.append(activation())
        layers.append(nn.Flatten())
        conv_stack = nn.Sequential(*layers)

        x = torch.rand(img_shape)
        y=conv_stack(rearrange([x], 'b w h c -> b c w h'))

        last_layers = []
        in_size = y.shape[-1]
        for l in hidden_linear:
            last_layers.append(nn.Linear(in_size, l))
            last_layers.append(activation())
            in_size = l

        last_layer = nn.Linear(in_size, n_actions)
        last_layer.weight.data.uniform_(-1e-3,1e-3)
        last_layer.bias.data.uniform_(-1e-3,1e-3)

        self.layers = nn.Sequential(
            conv_stack,
            *last_layers,
            last_layer
        )

    def forward(self, x):
        if len(x.shape) == 4:
            return self.layers(rearrange(x, "b w h c -> b c w h"))
        elif len(x.shape) == 3:
            return self.layers(rearrange([x], "b w h c -> b c w h"))[0]



class ConvolutionalQFunction(DiscreteQFunction):
    def __init__(self, env, nn_args, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = ConvolutionalNN(
            env.observation_space.shape, env.action_space.n, **nn_args
        ).to(self.device)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=self.lr)
        self.stats.update({
            'nn_loss': {
                'x_label': 'step',
                'data': []
            }
        })
        self.init_args = locals()
        print(self.device)

    def from_state(self, state):
        return self.from_state_batch(rearrange(state,"a b c -> 1 a b c"))[0]

    def from_state_batch(self, states):
        return self.nn(torch.tensor(states, dtype=torch.float32, device=self.device))

    def __call__(state, action):
        return self.call_batch(rearrange(action, "a b c -> 1 a b c"))[0]

    def call_batch(self, states, actions):
        values = self.nn(torch.tensor(states, dtype=torch.float32, device=self.device))\
                    .gather(-1,
                        torch.tensor(actions, dtype=torch.int64, device=self.device)\
                            .reshape((len(actions),1))
                    )
        return values

    def update(self, state, action, target_value):
        return self.update_batch([state], [action], [target_value])[0]

    def update_batch(self, states, actions, target_values, is_weights=None):
        if is_weights is None:
            is_weights = torch.ones((states.shape[0],), dtype=torch.float32)
        else:
            is_weights = torch.tensor(is_weights, dtype=torch.float32)
        target_values = torch.tensor(target_values,dtype=torch.float32, device=self.device).detach()
        pred_values = self.call_batch(states, actions)
        pred_error_indiv = torch.abs(pred_values[:,0] - target_values)
        pred_error = (
            is_weights * nn.functional.mse_loss(pred_values[:,0], target_values, reduction='none')
        ).mean()

        self.optim.zero_grad()
        pred_error.backward()
        self.optim.step()
        self.agent.log_data("nn_loss", pred_error.clone().cpu().item())
        return pred_error_indiv.detach().cpu().numpy()

    def best_action_value_from_state(self, state):
        maxa, maxv = self.best_action_value_from_state_batch(torch.tensor([state]))
        return maxa[0], maxv[0]

    def best_action_value_from_state_batch(self, states):
        values = self.from_state_batch(states)
        m = values.max(axis=1)
        return m.indices, m.values

    def export_f(self):
        return self.nn.state_dict()

    def import_f(self, d):
        self.nn.load_state_dict(d)
