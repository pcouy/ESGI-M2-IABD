import numpy as np
import matplotlib.pyplot as plt
from .base import DiscreteQFunction
import os
import copy

class LinearQValue(DiscreteQFunction):
    def __init__(self, env, default_value=0, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.weights = default_value*np.ones((self.action_space.n, (self.observation_space.shape[0]+1)))
        self.init_args = locals()

    def add_bias(self, state):
#        if type(state) is str:
#            state = np.zeros((self.weights.shape[1],))
        if state.shape[0] < self.weights.shape[1]:
            return np.concatenate(([1], state))
        else:
            return state

    def __call__(self, state, action):
        state = self.add_bias(state)
        return np.matmul(self.weights[action], state)

    def from_state(self, state):
        state = self.add_bias(state)
        return np.matmul(self.weights, state)

    def update(self, state, action, target_value, is_weight=None):
        if is_weight is None:
            is_weight = 1
        state = self.add_bias(state)
        Q = self(state, action)
        self.weights[action] = self.weights[action] + is_weight*self.lr*(target_value-Q)*state
        super().update(state, action, target_value)
        return np.abs(target_value-Q)

    def export_f(self):
        return copy.deepcopy(self.weights)

    def import_f(self, d):
        self.weights = copy.deepcopy(d)
