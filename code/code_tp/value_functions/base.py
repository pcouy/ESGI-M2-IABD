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

    def call_batch(self, states):
        """
        Permet d'évaluer un batch d'états. Simple boucle qui sera surchargée pour tirer partie
        de l'évaluation par batch des fonctions neurales
        """
        return np.array([self(state) for state in states])

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

    def update_batch(self, states, actions, target_values):
        """
        Méthode mettant à jour l'agent sur un *batch* de transitions. Simple boucle pour appeler
        `self.update` par défaut. Est utile principalement pour les agents à *replay buffer*.

        Méthode qui sera surchargée avec les fonctions de valeur neurales, qui prennent en charge
        l'évaluation par batch.
        """
        for state, action, target_value in zip(states, actions, target_values):
            self.update(state, action, target_value)

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
        return np.array([self(state, action) for action in self.enum_actions()])

    def from_state_batch(self, states):
        action_values = []
        for state in states:
            action_values.append(self.from_state(state))
        return np.array(action_values)

    def best_action_value_from_state(self, state):
        values = self.from_state(state)
        maxv, maxa = max((v,a) for a,v in enumerate(values))
        return maxa, maxv

    def best_action_value_from_state_batch(self, states):
        values_batch = self.from_state_batch(states)
        maxv = values_batch.max(axis=1)
        maxa = values_batch.argmax(axis=1)
        return maxa, maxv

    def __call__(self, state, action):
        """Prend un état et une action et renvoit leur valeur"""
        raise NotImplementedError

    def call_batch(self, states, actions):
        return np.array([self(*args) for args in zip(states,actions)])
