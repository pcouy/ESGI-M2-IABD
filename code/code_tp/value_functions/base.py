import numpy as np
import gymnasium as gym


def enum_discrete(space):
    """Fonction utilitaire pour obtenir un itérateur sur les actions d'un espace `gym.spaces.Discrete`"""
    return range(space.n)


class ValueFunction:
    """
    Classe de base des fonctions de valeur de type $v(s)$
    """

    def __init__(self, env, lr=0.1, lr_decay=0, lr_min=1e-5, **kwargs):
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
        self.stats = {"lr": {"x_label": "step", "data": []}}
        self.init_args = kwargs
        self.init_args.update(locals())

    def __call__(self, state, prev_action=None):
        """
        Prend un état en entrée et renvoit sa valeur
        """
        raise NotImplementedError

    def call_batch(self, states, prev_actions=None):
        """
        Permet d'évaluer un batch d'états. Simple boucle qui sera surchargée pour tirer partie
        de l'évaluation par batch des fonctions neurales
        """
        if prev_actions is None:
            return np.array([self(state) for state in states])
        return np.array(
            [
                self(state, prev_action)
                for state, prev_action in zip(states, prev_actions)
            ]
        )

    def update(self, state, action, target_value, prev_action=None, is_weight=None):
        """
        Met à jour la fonction de valeur à partir d'une transition expérimentée.
        Maintient également les paramètres de l'entrainement.
        Pour cette classe de base, s'occupe uniquement de la décroissance du taux d'apprentissage.

        * `state`: état de départ d'une transition
        * `action`: action effectuée depuis l'état `state`
        * `target_value`: valeur cible, déterminée par l'agent selon son algorithme d'apprentissage
        * `prev_action`: action précédente (optionnel)
        * `is_weight`: poids d'importance sampling (optionnel)
        """
        self.lr = max(self.lr * (1 - self.lr_decay), self.lr_min)
        self.agent.log_data("lr", self.lr)

    def update_batch(
        self, states, actions, target_values, prev_actions=None, is_weights=None
    ):
        """
        Méthode mettant à jour l'agent sur un *batch* de transitions. Simple boucle pour appeler
        `self.update` par défaut. Est utile principalement pour les agents à *replay buffer*.

        Méthode qui sera surchargée avec les fonctions de valeur neurales, qui prennent en charge
        l'évaluation par batch.
        """
        if is_weights is None:
            is_weights = np.ones((states.shape[0],))
        if prev_actions is None:
            prev_actions = [None] * states.shape[0]

        return np.array(
            [
                self.update(state, action, target_value, prev_action, is_weight)
                for state, action, target_value, prev_action, is_weight in zip(
                    states, actions, target_values, prev_actions, is_weights
                )
            ]
        )

    def export_f(self):
        pass

    def import_f(self, d):
        pass

    def mix_with(self, other, tau=1):
        pass

    def clone(self, **clone_kwargs):
        init_args = self.init_args.copy()
        del init_args["self"]
        del init_args["__class__"]
        args = init_args["args"]
        del init_args["args"]
        kwargs = init_args["kwargs"]
        del init_args["kwargs"]
        for key in clone_kwargs:
            if key in init_args:
                del init_args[key]
            if key in kwargs:
                del kwargs[key]
        return type(self)(*args, **kwargs, **clone_kwargs, **init_args)


class DiscreteQFunction(ValueFunction):
    """
    Classe de base pour les fonctions de valeur de type $q(s,a)$ dans les espaces d'actions discrets
    """

    def __init__(self, env, *args, **kwargs):
        assert isinstance(env.action_space, gym.spaces.Discrete) or isinstance(
            env.action_space, gym.spaces.MultiDiscrete
        )
        super().__init__(env, *args, **kwargs)
        self.init_args = locals()
        self.last_result = None

    def enum_actions(self):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return enum_discrete(self.action_space)
        else:
            raise NotImplementedError

    def from_state(self, state, prev_action=None):
        """Prend un état et renvoit un dictionnaire `action : valeur` pour cet état"""
        self.last_result = np.array(
            [self(state, action, prev_action) for action in self.enum_actions()]
        )
        return self.last_result

    def from_state_batch(self, states, prev_actions=None):
        action_values = []
        for i, state in enumerate(states):
            prev_action = prev_actions[i] if prev_actions is not None else None
            action_values.append(self.from_state(state, prev_action))
        self.last_result = np.array(action_values)
        return self.last_result

    def best_action_value_from_state(self, state, prev_action=None):
        values = self.from_state(state, prev_action)
        maxv, maxa = max((v, a) for a, v in enumerate(values))
        return maxa, maxv

    def best_action_value_from_state_batch(self, states, prev_actions=None):
        values_batch = self.from_state_batch(states, prev_actions)
        maxv = values_batch.max(axis=1)
        maxa = values_batch.argmax(axis=1)
        return maxa, maxv

    def __call__(self, state, action, prev_action=None):
        """Prend un état et une action et renvoit leur valeur"""
        raise NotImplementedError

    def call_batch(self, states, actions, prev_actions=None):
        return np.array(
            [
                self(*args)
                for args in zip(
                    states,
                    actions,
                    [
                        prev_actions[i] if prev_actions is not None else None
                        for i in range(len(states))
                    ],
                )
            ]
        )
