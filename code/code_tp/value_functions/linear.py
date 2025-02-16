import numpy as np
import matplotlib.pyplot as plt
from .base import DiscreteQFunction
import os
import copy

class LinearQValue(DiscreteQFunction):
    def __init__(self, env, default_value=0, use_prev_action=False, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.use_prev_action = use_prev_action
        # Add space for one-hot encoded previous action if enabled
        extra_dims = env.action_space.n if use_prev_action else 0
        self.weights = default_value*np.ones((self.action_space.n, (self.observation_space.shape[0] + 1 + extra_dims)))
        self.init_args = locals()

    def add_bias(self, state):
#        if type(state) is str:
#            state = np.zeros((self.weights.shape[1],))
        if state.shape[0] < self.weights.shape[1]:
            return np.concatenate(([1], state))
        else:
            return state

    def add_bias_and_prev_action(self, state, prev_action=None):
        if not self.use_prev_action or prev_action is None:
            return self.add_bias(state)
        
        # Create one-hot encoding for previous action
        prev_action_one_hot = np.zeros(self.action_space.n)
        prev_action_one_hot[prev_action] = 1
        
        # Concatenate state with bias and one-hot encoded previous action
        if state.shape[0] < self.weights.shape[1] - self.action_space.n:
            return np.concatenate(([1], state, prev_action_one_hot))
        else:
            return np.concatenate((state, prev_action_one_hot))

    def __call__(self, state, action, prev_action=None):
        state = self.add_bias_and_prev_action(state, prev_action)
        return np.matmul(self.weights[action], state)

    def from_state(self, state, prev_action=None):
        state = self.add_bias_and_prev_action(state, prev_action)
        return np.matmul(self.weights, state)

    def update(self, state, action, target_value, prev_action=None, is_weight=None):
        if is_weight is None:
            is_weight = 1
        state = self.add_bias_and_prev_action(state, prev_action)
        Q = self(state, action, prev_action)
        self.weights[action] = self.weights[action] + is_weight*self.lr*(target_value-Q)*state
        super().update(state, action, target_value)
        return np.abs(target_value-Q)

    def export_f(self):
        return copy.deepcopy(self.weights)

    def import_f(self, d):
        self.weights = copy.deepcopy(d)

    def mix_with(self, other, tau=0.001):
        self.weights = tau * other.weights + (1 - tau) * other.weights
        
