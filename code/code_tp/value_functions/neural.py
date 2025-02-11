import numpy as np
import matplotlib.pyplot as plt
from .base import DiscreteQFunction
import torch
from torch import nn
from einops import rearrange
from .neural_nets.convolutional import ConvolutionalNN

class ConvolutionalQFunction(DiscreteQFunction):
    """
    Implémente une approximation de la fonction de valeur basée sur un réseau de neurones
    convolutionnel en PyTorch (voir `code_tp.value_functions.neural_nets` pour plus
    de détails)
    """
    def to_tensor(self, x, dtype=torch.float32):
        """Convert input to tensor and move to correct device"""
        if isinstance(x, torch.Tensor):
            if x.device == self.device:
                return x.clone()
            return x.clone().detach().to(self.device)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def __init__(
        self,
        env,
        nn_args,
        *args,
        nn_class=ConvolutionalNN,
        gradient_clipping=None,
        use_prev_action=False,
        prev_action_embedding_dim=16,  # Default embedding dimension when use_prev_action is True
        **kwargs
    ):
        if use_prev_action:
            nn_args = nn_args.copy()
            nn_args['embedding_dim'] = prev_action_embedding_dim
            nn_args['embedding_size'] = env.action_space.n
            
        super().__init__(env, *args, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nn = nn_class(
            img_shape=env.observation_space.shape,
            n_actions=env.action_space.n,
            **nn_args
        ).to(self.device)
        self.nn.share_memory()
        print(self.nn)
        self.optim = torch.optim.Adam(self.nn.parameters(), lr=self.lr)

        self.has_time_dim = len(env.observation_space.shape) == 4
        self.use_prev_action = use_prev_action

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

    def add_batch_dim(self, tensor):
        if not self.has_time_dim:
            return rearrange(tensor, "a b c -> 1 a b c")
        else:
            return rearrange(tensor, "a b c d -> 1 a b c d")

    def from_state(self, state, prev_action=None):
        return self.from_state_batch(self.add_batch_dim(state), 
                                   None if prev_action is None else [prev_action])[0]

    def from_state_batch(self, states, prev_actions=None):
        states = self.to_tensor(states)
        
        if self.use_prev_action and prev_actions is not None:
            # Ensure prev_actions are long (int64) for embedding layer
            prev_actions = self.to_tensor(prev_actions, dtype=torch.long)
        else:
            prev_actions = None
        
        return self.nn(states, prev_actions)

    def __call__(self, state, action, prev_action=None):
        return self.call_batch([state], [action], 
                             None if prev_action is None else [prev_action])[0]

    def call_batch(self, states, actions, prev_actions=None):
        states = self.to_tensor(states)
        # Use long for action indices
        actions = self.to_tensor(actions, dtype=torch.long)
        
        if prev_actions is not None:
            # Ensure prev_actions are long for embedding layer
            prev_actions = self.to_tensor(prev_actions, dtype=torch.long)
        
        values = self.nn(states, prev_actions).gather(-1, actions.reshape((len(actions),1)))
        return values

    def update(self, state, action, target_value, prev_action=None, is_weight=None):
        return self.update_batch([state], [action], [target_value], 
                               None if prev_action is None else [prev_action])[0]

    def update_batch(self, states, actions, target_values, prev_actions=None, is_weights=None):
        if is_weights is None:
            is_weights = torch.ones((states.shape[0],), dtype=torch.float32, device=self.device)
        else:
            is_weights = self.to_tensor(is_weights)
        target_values = self.to_tensor(target_values).detach()
        
        self.optim.zero_grad(set_to_none=True)

        pred_values = self.call_batch(states, actions, prev_actions)
        pred_error_indiv = torch.abs(pred_values[:,0] - target_values)
        pred_error = (
            is_weights * nn.functional.mse_loss(pred_values[:,0], target_values, reduction='none')
        ).mean()

        pred_error.backward()
        self.optim.step()
        self.agent.log_data("nn_loss", pred_error.clone().cpu().item())
        return pred_error_indiv.detach().cpu().numpy()

    def best_action_value_from_state(self, state, prev_action=None):
        maxa, maxv = self.best_action_value_from_state_batch(
            torch.tensor([state]),
            None if prev_action is None else [prev_action]
        )
        return maxa[0], maxv[0]

    def best_action_value_from_state_batch(self, states, prev_actions=None):
        values = self.from_state_batch(states, prev_actions)
        m = values.max(axis=1)
        return m.indices, m.values

    def export_f(self):
        return self.nn.state_dict()

    def import_f(self, d):
        self.nn.load_state_dict(d)
