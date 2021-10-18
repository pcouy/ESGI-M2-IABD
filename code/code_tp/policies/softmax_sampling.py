import numpy as np
from .greedy import EGreedyPolicy
import torch

class SoftmaxSamplingPolicy(EGreedyPolicy):
    """
    Pareil que la politique epsilon-greedy, mais l'action est tirée avec la probabilité :

    p(a) = softmax(5*(1-epsilon)*Q(s,a))

    epsilon=1 <=> Choix complètement équiprobable parmi les actions
    epsilon=0 <=> Politique greedy
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats.update({
            'picked_proba':{
                'x_label': 'step',
                'data': []
            }
        })

    def __call__(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        values = self.value_function.from_state(state)
        if type(values) is torch.Tensor:
            values = values.clone().detach().cpu().numpy()

        if epsilon > 0.0015:
            try:
                aux = np.exp((1/epsilon - 1) * (values-np.max(values)))
                if np.max(aux) == np.inf:
                    aux[aux != np.inf] = 0
                    aux[aux == np.inf] = 1
                probas = aux/np.sum(aux)
                action = np.random.choice([x for x in range(len(probas))], p=probas)
                self.agent.log_data("picked_proba", probas[action])
            except:
                print(epsilon, values, aux, probas)
                action = self.greedy_policy(state)
                self.agent.log_data("picked_proba", 1)
        else:
            action = self.greedy_policy(state)
            self.agent.log_data("picked_proba", 1)

        self.agent.log_data("predicted_value", values[action])
        self.stats.update(self.greedy_policy.stats)

        return action
