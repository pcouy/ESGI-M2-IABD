import numpy as np
import torch
from einops import rearrange

class RandomPolicy:
    """
    Politique aléatoire, sert de classe de base à toutes les politiques.
    Un agent suivant cette politique est équivalent à un `RandomAgent` dans son comportement
    """
    def __init__(self, value_function):
        self.value_function = value_function
        self.stats = {}
        self.agent = None

    def __call__(self, state):
        """
        Prend un état en argument, retourne une action
        """
        return self.value_function.action_space.sample()
    
    def batch_call(self, state_batch):
        """
        Prend un batch d'état en argument, retourne un batch de valeurs
        """
        return self(state_batch)

    def test(self, state):
        """
        Prend un état en argument, retourne une action.
        Utilisée durant les épisodes d'évaluation, le comportement par défaut est d'appliquer
        la politique définie dans `__call__`
        """
        return self(state)

    def update(self):
        """
        Méthode appelée à la fin de chaque pas de temps pour éventuellement mettre
        à jour la politique
        """
        pass

class GreedyQPolicy(RandomPolicy):
    """
    Implémente la politique *greedy* sur une fonction de valeur
    """
    def __init__(self, value_function):
        super().__init__(value_function)
        self.stats.update({
            'predicted_value': {
                'x_label': 'step',
                'data': []
            }
        })
        
    def best_action_value_from_values(self, values):
        if type(values) is torch.Tensor:
            values = rearrange(values, 'a -> 1 a')
            maxa, maxv = self.best_action_value_from_values_batch(values)
            return maxa[0], maxv[0]
        else:
            maxv, maxa = max((v,a) for a,v in enumerate(values))
            return maxa, maxv
    
    def best_action_value_from_values_batch(self, values_batch):
        if type(values_batch) is torch.Tensor:
            m = values_batch.max(axis=1)
            return m.indices, m.values
        else:
            maxv = values_batch.max(axis=1)
            maxa = values_batch.argmax(axis=1)
            return maxa, maxv
            

    def __call__(self, state, prev_action=None):
        values = self.value_function.from_state(state, prev_action)
        action, value = self.best_action_value_from_values(values)
        if type(value) is torch.Tensor:
            value = value.clone().detach().item()
        if type(values) is torch.Tensor:
            values = values.clone().detach().cpu().numpy()

        actions = [k for k,v in enumerate(values) if v == value]

        self.agent.log_data("predicted_value", value)
        return np.random.choice(actions)
    
    def batch_call(self, state_batch, prev_actions=None):
        values_batch = self.value_function.from_state_batch(state_batch, prev_actions)
        action_batch, value_batch = self.best_action_value_from_values_batch(values_batch)
        if type(value_batch) is torch.Tensor:
            value_batch = value_batch.clone().detach().item()
        if type(action_batch) is torch.Tensor:
            action_batch = action_batch.clone().detach().cpu().numpy()
        
        self.agent.log_data("predicted_value", value_batch.mean())
        return action_batch
        

class EGreedyPolicy(RandomPolicy):
    """
    Implémente une politique *epsilon-greedy* qui choisit une action aléatoire avec la
    probabilité `epsilon` et l'action définie par la politique *greedy* le reste du temps.

    `epsilon_decay` et `epsilon_min` permettent de faire varier la valeur donnée à
    `epsilon` au cours de l'entrainement
    """
    def __init__(self, value_function, greedy_policy_class=GreedyQPolicy, epsilon=0.05, epsilon_decay=0, epsilon_min=0.05, epsilon_test=0):
        self.greedy_policy = greedy_policy_class(value_function)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_test = epsilon_test
        super().__init__(value_function)
        self.stats.update({
            "epsilon": {
                "x_label": "step",
                "data": []
            }
        })

    def __call__(self, state, prev_action=None, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.uniform(0,1) > epsilon:
            self.greedy_policy.agent = self.agent
            action = self.greedy_policy(state, prev_action)
            self.stats.update(self.greedy_policy.stats)
        else:
            action = super().__call__(state)

        return action
    
    def batch_call(self, state_batch, prev_actions=None, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
            
        greedy_action_batch = self.greedy_policy.batch_call(state_batch, prev_actions)
        random_action_batch = super().batch_call(state_batch)
        random_mask = np.random.uniform(0, 1, (state_batch.shape[0],))
        action_batch = np.where(random_mask > epsilon,
                                greedy_action_batch,
                                random_action_batch)
        return action_batch

    def test(self, state, prev_action=None):
        return self(state, prev_action, epsilon=self.epsilon_test)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon*(1-self.epsilon_decay), self.epsilon_min)

    def update(self):
        self.update_epsilon()
        self.agent.log_data("epsilon", self.epsilon)
        super().update()

class CosineEGreedyPolicy(EGreedyPolicy):
    """
    Fait varier epsilon comme un cosinus du pas de temps

    Réutilise les paramètres de `EGreedyPolicy` pour définir le cosinus :

    * `T`: Période du cosinus (en pas de temps)
    * `epsilon`: Valeur max d'epsilon
    * `epsilon_min`: Valeur max d'epsilon
    * `epsilon_decay`: Décroissance de epsilon_max

    $\epsilon = (\epsilon_{max}-\epsilon_{min})/2 \cos(\frac{2\Pi}{\epsilon_{decay}} t) + \epsilon_{min}$
    """
    def __init__(self, T, epsilon_max_final=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon_max = self.epsilon
        if epsilon_max_final is None:
            epsilon_max_final = 0
        self.epsilon_max_final = max(self.epsilon_min, epsilon_max_final)
        self.t = 0
        self.T = T

    def update_epsilon(self):
        self.epsilon = (self.epsilon_max-self.epsilon_min)/2 * np.cos(self.t*2*np.pi/self.T) +\
                (self.epsilon_max + self.epsilon_min)/2
        self.epsilon_max = max(self.epsilon_max*(1-self.epsilon_decay), self.epsilon_max_final)
        self.t+= 1

