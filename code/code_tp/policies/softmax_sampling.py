import numpy as np
from .greedy import EGreedyPolicy

class SoftmaxSamplingPolicy(EGreedyPolicy):
    """
    Pareil que la politique epsilon-greedy, mais l'action est tirée avec la probabilité :

    p(a) = softmax(5*(1-epsilon)*Q(s,a))

    epsilon=1 <=> Choix complètement équiprobable parmi les actions
    epsilon=0 <=> Politique greedy
    """
    def __call__(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        values = self.value_function.from_state(state)
        aux = np.exp(5 * ((1-self.epsilon)**2.5) * values)
        probas = aux/np.sum(aux)

        action = np.random.choice([x for x in range(len(probas))], p=probas)

        return action
