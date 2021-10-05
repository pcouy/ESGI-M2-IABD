from .base import RandomAgent
import matplotlib.pyplot as plt
import numpy as np
import os

class TabularValueAgent(RandomAgent):
    def __init__(self, env, gamma=0.99, epsilon=0.05, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        self.known_states = dict()
        self.gamma = gamma
        self.epsilon = epsilon

        self.n_known_states = [0]

    def train_with_transition(self, state, action, next_state, reward, done, infos):
        if self.state_to_key(state) not in self.known_states:
            self.known_states[self.state_to_key(state)] = {k:1 for k in range(self.env.action_space.n)}

        if self.state_to_key(next_state) not in self.known_states:
            self.known_states[self.state_to_key(next_state)] = {k:1 for k in range(self.env.action_space.n)}

        next_value, _ = max((v,a) for a,v in self.known_states[self.state_to_key(next_state)].items())
        target_value = reward + self.gamma*next_value
        current_value = self.known_states[self.state_to_key(state)][action]
        self.known_states[self.state_to_key(state)][action] = current_value + 0.01*(target_value-current_value)

    def export_tabular(self):
        return self.known_states

    def import_tabular(self, table):
        self.known_states.update(table)

    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if self.state_to_key(state) not in self.known_states:
            self.known_states[self.state_to_key(state)] = {k:1 for k in range(self.env.action_space.n)}

        if self.test or np.random.uniform(0,1) > epsilon:
            _, action = max((v,a) for a,v in self.known_states[self.state_to_key(state)].items())
        else:
            action = super().select_action()

        return action

    def state_to_key(self, state):
        return str(state)

    def step(self, action):
        self.n_known_states.append(len(self.known_states))
        return super().step(action)

    def show_known_states(self, save_dir=None):
        if save_dir is None:
            save_dir = self.save_dir
        fig, ax = plt.subplots()
        ax.plot(self.n_known_states)
        fig.tight_layout()
        os.makedirs(self.save_dir, exist_ok=True)
        fig.savefig(os.path.join(
            self.save_dir,
            "n_known_states.png"
        ))
        plt.close(fig)

    def train(self, n_episodes=1000, test_interval=50, 
              train_callbacks=None, test_callbacks=None, 
              *args, **kwargs):
        if train_callbacks is None:
            train_callbacks = []
        if test_callbacks is None:
            test_callbacks = []
        if self.show_known_states not in test_callbacks:
            test_callbacks+= [self.show_known_states]
        super().train(n_episodes, test_interval, train_callbacks, test_callbacks, *args, **kwargs)
