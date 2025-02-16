import numpy as np
from .base import QLearningAgent
import copy
import torch

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
        self.target_value_function.nn.eval() #On économise le calcul des gradients sur la fonction cible
        self.discrete_updates = (target_update >= 1)

    def train_with_transition(self, state, action, next_state, reward, done, infos, prev_action=None):
        #print("Training from TargetValueAgent")
        super().train_with_transition(state, action, next_state, reward, done, infos, prev_action)
        if self.discrete_updates and self.training_steps % self.target_update == 0:
            self.target_value_function.import_f(
                self.value_function.export_f()
            )
        else:
            self.target_value_function.mix_with(self.value_function, tau=self.target_update)

    @torch.no_grad()
    def eval_state(self, state, prev_action=None):
        return self.target_value_function.best_action_value_from_state(state, prev_action)

    @torch.no_grad()
    def eval_state_batch(self, states, prev_actions=None):
        return self.target_value_function.best_action_value_from_state_batch(states, prev_actions)

class DoubleQLearning(TargetValueAgent):
    """
    Modifie `TargetValueAgent` pour utiliser la variante proposée par l'article
    [Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)
    """
    @torch.no_grad()
    def eval_state_batch(self, states, prev_actions=None):
        selected_actions, _ = self.value_function.best_action_value_from_state_batch(states, prev_actions)
        values = self.target_value_function.call_batch(states, selected_actions, prev_actions)
        values = values.flatten()
        return selected_actions, values
    
    @torch.no_grad()
    def eval_state(self, state, prev_action=None):
        selected_action, _ = self.value_function.best_action_value_from_state(state, prev_action)
        value = self.target_value_function(state, selected_action, prev_action)
        return selected_action, value
