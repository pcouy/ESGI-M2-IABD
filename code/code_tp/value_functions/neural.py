import numpy as np
import matplotlib.pyplot as plt
from .base import DiscreteQFunction
import torch
from torch import nn
from einops import rearrange
from .neural_nets.convolutional import ConvolutionalNN

class ConvolutionalQFunction(DiscreteQFunction):
    def __init__(self, env, nn_args, *args, nn_class=ConvolutionalNN, gradient_clipping=None, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = nn_class(
            img_shape=env.observation_space.shape,
            n_actions=env.action_space.n,
            **nn_args
        ).to(self.device)
        print(self.nn)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=self.lr)

        if gradient_clipping is not None:
            nn.utils.clip_grad_norm_(self.nn.parameters(), gradient_clipping)

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

    def __call__(self, state, action):
        return self.call_batch([state], rearrange(action, "a b c -> 1 a b c"))[0]

    def call_batch(self, states, actions):
        values = self.nn(torch.tensor(states, dtype=torch.float32, device=self.device))\
                    .gather(-1,
                        torch.tensor(actions, dtype=torch.int64, device=self.device)\
                            .reshape((len(actions),1))
                    )
        return values

    def update(self, state, action, target_value, is_weight=None):
        return self.update_batch([state], [action], [target_value])[0]

    def update_batch(self, states, actions, target_values, is_weights=None):
        if is_weights is None:
            is_weights = torch.ones((states.shape[0],), dtype=torch.float32, device=self.device)
        else:
            is_weights = torch.tensor(is_weights, dtype=torch.float32, device=self.device)
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
