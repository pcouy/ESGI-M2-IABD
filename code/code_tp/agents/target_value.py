import numpy as np
from .base import QLearningAgent
import copy

class TargetValueAgent(QLearningAgent):
    """
    Agent qui utilise une fonction de valeur différente pour déterminer la valeur cible
    lors d'un update. (http://arxiv.org/abs/1312.5602)
    """
    def __init__(self, env, target_update, *args, **kwargs):
        """
        * `target_update`: Fréquence de mise à jour de la fonction de valeur cible
        """
        super().__init__(env, *args, **kwargs)
        self.target_update = target_update
        self.target_value_function = self.value_function.clone()

    def train_with_transition(self, *args, **kwargs):
        #print("Training from TargetValueAgent")
        super().train_with_transition(*args, **kwargs)
        if self.training_steps % self.target_update == 0:
            self.target_value_function.import_f(
                self.value_function.export_f()
            )

    def eval_state(self, state):
        return self.target_value_function.best_action_value_from_state(state)

    def eval_state_batch(self, states):
        return self.target_value_function.best_action_value_from_state_batch(states)

class DoubleQLearning(TargetValueAgent):
    def eval_state_batch(self, states):
        selected_actions, _ = self.value_function.best_action_value_from_state_batch(states)
        values = self.target_value_function.call_batch(states, selected_actions)
        values = values.flatten()
        return selected_actions, values
    
    def eval_state(self, state):
        selected_action, _ = self.value_function.best_action_value_from_state(state)
        value = self.target_value_function(state, selected_action)
        return selected_actions, values