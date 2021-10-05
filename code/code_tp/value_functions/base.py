import numpy as np
import gym

def enum_discrete(space):
    """Fonction utilitaire pour obtenir un itérateur sur les actions d'un espace `gym.spaces.Discrete`"""
    return range(space.n)

class ValueFunction:
    """
    Classe de base des fonctions de valeur de type $v(s)$
    """
    def __init__(self, env, lr=0.1, lr_decay=0, lr_min=1e-5):
        """
        * `env`: Environnement sur lequel porte la fonction de valeur. Est utilisé uniquement pour
                 déterminer les espaces des états et des actions.
        * `lr`: Taux d'apprentissage
        * `lr_decay`: Décroissance du taux d'apprentissage : à chaque pas de temps, le taux
                      d'apprentissage est multiplié par `1-lr_decay`
        * `lr_min`: Taux d'apprentissage minimum, à partir duquel celui-ci de décroit plus
        """
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_min = lr_min
        self.agent = None
        self.stats = {
            "lr": {
                "x_label": "step",
                "data": []
            }
        }

    def __call__(self, state):
        """
        Prend un état en entrée et renvoit sa valeur
        """
        raise NotImplementedError

    def update(self, state, action, target_value):
        """
        Met à jour la fonction de valeur à partir d'une transition expérimentée.
        Maintient également les paramètres de l'entrainement.
        Pour cette classe de base, s'occupe uniquement de la décroissance du taux d'apprentissage.

        * `state`: état de départ d'une transition
        * `action`: action effectuée depuis l'état `state`
        * `target_value`: valeur cible, déterminée par l'agent selon son algorithme d'apprentissage
        """
        self.lr = max(self.lr*(1-self.lr_decay), self.lr_min)
        self.stats["lr"]["data"].append(self.lr)

class DiscreteQFunction(ValueFunction):
    """
    Classe de base pour les fonctions de valeur de type $q(s,a)$ dans les espaces d'actions discrets
    """
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
        """Prend un état et renvoit un dictionnaire `action : valeur` pour cet état"""
        return {action: self(state, action) for action in self.enum_actions()}

    def __call__(self, state, action):
        """Prend un état et une action et renvoit leur valeur"""
        raise NotImplementedError

