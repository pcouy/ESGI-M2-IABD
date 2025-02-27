import numpy as np
import matplotlib.pyplot as plt
from .base import DiscreteQFunction
import os
import copy


class TabularQValue(DiscreteQFunction):
    """Fonction de valeur tabulaire"""

    def __init__(self, env, default_value=0, use_prev_action=False, *args, **kwargs):
        """
        * `default_value`: Valeur d'initialisation *a priori* des états-actions
        * `use_prev_action`: Si True, l'action précédente est incluse dans la clé d'état
        """
        self.known_states = dict()
        self.n_known_states = [0]
        self.visit_count = dict()
        self.default_value = default_value
        self.use_prev_action = use_prev_action
        super().__init__(env, *args, **kwargs)
        self.stats.update({"n_known_states": {"x_label": "Steps", "data": [0]}})
        self.init_args = locals()

    def get_state_key(self, state, prev_action=None):
        if not self.use_prev_action or prev_action is None:
            return str(state)
        return f"{str(state)}|prev_{prev_action}"

    def __call__(self, state, action, prev_action=None):
        return self.from_state(state, prev_action)[action]

    def create_state(self, state_key):
        self.known_states[state_key] = {
            k: self.default_value for k in self.enum_actions()
        }
        self.visit_count[state_key] = {k: 0 for k in self.enum_actions()}

    def from_state(self, state, prev_action=None):
        state_key = self.get_state_key(state, prev_action)
        if state_key not in self.known_states:
            self.create_state(state_key)
        val_dict = self.known_states[state_key]
        r_values = np.zeros((len(val_dict),))
        for i in range(len(val_dict)):
            r_values[i] = val_dict[i]
        return r_values

    def update(self, state, action, target_value, prev_action=None, is_weight=None):
        state_key = self.get_state_key(state, prev_action)
        Q = self(state, action, prev_action)
        if is_weight is None:
            lr = self.lr
        else:
            lr = self.lr * (is_weight ** (1 / 10))
        self.known_states[state_key][action] = (1 - lr) * Q + lr * target_value
        self.agent.log_data("n_known_states", len(self.known_states))
        self.visit_count[state_key][action] += 1
        super().update(state, action, target_value)
        return np.abs(target_value - Q)

    def show_known_states(self, save_dir):
        fig, ax = plt.subplots()
        ax.plot(self.n_known_states)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "n_known_states.png"))
        plt.close(fig)

    def export_f(self):
        return copy.deepcopy(self.known_states)

    def import_f(self, d):
        self.known_states.update(d)
