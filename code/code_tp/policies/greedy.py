import numpy as np
import torch

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

    def __call__(self, state):
        values = self.value_function.from_state(state)
        action, value = self.value_function.best_action_value_from_state(state)
        if type(value) is torch.Tensor:
            value = value.clone().detach().item()
        if type(values) is torch.Tensor:
            values = values.clone().detach().cpu().numpy()

        actions = [k for k,v in enumerate(values) if v == value]


        self.stats['predicted_value']['data'].append(value)
        return np.random.choice(actions)

class EGreedyPolicy(RandomPolicy):
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

    def __call__(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.uniform(0,1) > epsilon:
            action = self.greedy_policy(state)
            self.stats.update(self.greedy_policy.stats)
        else:
            action = super().__call__(state)

        return action

    def test(self, state):
        return self(state, epsilon=self.epsilon_test)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon*(1-self.epsilon_decay), self.epsilon_min)

    def update(self):
        self.update_epsilon()
        self.stats["epsilon"]["data"].append(self.epsilon)
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
    def __init__(self, T, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon_max = self.epsilon
        self.t = 0
        self.T = T

    def update_epsilon(self):
        self.epsilon = (self.epsilon_max-self.epsilon_min)/2 * np.cos(self.t*2*np.pi/self.T) +\
                (self.epsilon_max/2 + self.epsilon_min)
        self.epsilon_max = max(self.epsilon_max*(1-self.epsilon_decay), self.epsilon_min)
        self.t+= 1

