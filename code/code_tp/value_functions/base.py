import numpy as np
import gym

def enum_discrete(space):
    return range(space.n)

class ValueFunction:
    def __init__(self, env, lr=0.1, lr_decay=0, lr_min=1e-5):
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.stats = {
            "lr": {
                "x_label": "step",
                "data": []
            }
        }

    def __call__(self, state):
        raise NotImplementedError

    def update(self, state, action, target_value):
        self.lr = max(self.lr*(1-self.lr_decay), self.lr_min)
        self.stats["lr"]["data"].append(self.lr)

class DiscreteQFunction(ValueFunction):
    def __init__(self, env, *args, **kwargs):
        assert isinstance(env.action_space, gym.spaces.Discrete) or \
            isinstance(env.action_space, gym.spaces.MultiDiscrete)
        super().__init__(env, *args, **kwargs)

    def enum_actions(self):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return enum_discrete(self.action_space)
        else:
            raise NotImplementedError

    def from_state(self, state):
        return {action: self(state, action) for action in self.enum_actions()}

    def __call__(self, state, action):
        raise NotImplementedError

