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
                 activation=nn.Tanh,
            ):
        super().__init__()
        self.img_shape = img_shape
        self.n_actions = n_actions
        layers = []
        if n_filters is None:
            n_filters = [16,16]
        n_filters = [img_shape[-1]] + n_filters
        for n_in, n_out in zip(n_filters[:-1], n_filters[1:]):
            i_layer = 0
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

        last_layer = nn.Linear(y.shape[-1], n_actions)
        last_layer.weight.data.uniform_(-1e-3,1e-3)
        last_layer.bias.data.uniform_(-1e-3,1e-3)

        self.layers = nn.Sequential(
            conv_stack,
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
        self.nn = ConvolutionalNN(env.observation_space.shape, env.action_space.n, **nn_args)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=self.lr)
        self.stats.update({
            'nn_loss': {
                'x_label': 'step',
                'data': []
            }
        })

    def from_state(self, state):
        return self.from_state_batch(rearrange(state,"a b c -> 1 a b c"))[0]

    def from_state_batch(self, states):
        return self.nn(torch.tensor(states, dtype=torch.float32))

    def __call__(state, action):
        return self.call_batch(rearrange(action, "a b c -> 1 a b c"))[0]

    def call_batch(self, states, actions):
        values = self.nn(torch.tensor(states, dtype=torch.float32))\
                    .gather(-1, torch.tensor(actions, dtype=torch.int64).reshape((len(actions),1)))
        return values

    def update(self, state, action, target_value):
        self.update_batch([state], [action], [target_value])

    def update_batch(self, states, actions, target_values):
        pred_values = self.call_batch(states, actions)
        pred_error = nn.functional.smooth_l1_loss(
            pred_values[:,0],
            torch.tensor(target_values,dtype=torch.float32).detach()
        )
        self.optim.zero_grad()
        pred_error.backward()
        self.optim.step()
        self.stats['nn_loss']['data'].append(pred_error.clone().detach().item())
        del pred_error

    def best_action_value_from_state(self, state):
        maxa, maxv = self.best_action_value_from_state_batch(torch.tensor([state]))
        return maxa[0], maxv[0]

    def best_action_value_from_state_batch(self, states):
        values = self.from_state_batch(states)
        m = values.max(axis=1)
        return m.indices, m.values

