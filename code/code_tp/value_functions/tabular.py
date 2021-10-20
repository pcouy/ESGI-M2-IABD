import numpy as np
import matplotlib.pyplot as plt
from .base import DiscreteQFunction
import os
import copy

class TabularQValue(DiscreteQFunction):
    """Fonction de valeur tabulaire"""
    def __init__(self, env, default_value=0, *args, **kwargs):
        """
        * `default_value`: Valeur d'initialisation *a priori* des états-actions
        """
        self.known_states = dict()
        self.n_known_states = [0]
        self.visit_count = dict()
        self.default_value = default_value
        super().__init__(env, *args, **kwargs)
        self.stats.update({
            "n_known_states": {
                "x_label": "Steps",
                "data": [0]
            }
        })
        self.init_args = locals()

    def __call__(self, state, action):
        return self.from_state(str(state))[action]

    def create_state(self, state):
        self.known_states[str(state)] = {k:self.default_value for k in self.enum_actions()}
        self.visit_count[str(state)] = {k:0 for k in self.enum_actions()}

    def from_state(self, state):
        if str(state) not in self.known_states:
            self.create_state(state)
        val_dict = self.known_states[str(state)]
        r_values = np.zeros((len(val_dict),))
        for i in range(len(val_dict)):
            r_values[i] = val_dict[i]
        return r_values

    def update(self, state, action, target_value):
        Q = self(state, action)
        self.known_states[str(state)][action] = (1-self.lr)*Q + self.lr*target_value
        self.agent.log_data("n_known_states", len(self.known_states))
        self.visit_count[str(state)][action]+= 1
        super().update(str(state), action, target_value)
        return np.abs(target_value-Q)

    def show_known_states(self, save_dir):
        fig, ax = plt.subplots()
        ax.plot(self.n_known_states)
        fig.tight_layout()
        fig.savefig(os.path.join(
            save_dir,
            "n_known_states.png"
        ))
        plt.close(fig)

    def export_f(self):
        return copy.deepcopy(self.known_states)

    def import_f(self, d):
        self.known_states.update(d)
