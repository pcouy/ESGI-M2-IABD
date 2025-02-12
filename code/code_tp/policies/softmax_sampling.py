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
        biases = kwargs.pop("biases", None)
        self.biases_decay = kwargs.pop("biases_decay", 0.9999)
        super().__init__(*args, **kwargs)
        if biases is None:
            self.biases = np.array([0.0 for action in range(self.value_function.action_space.n)])
        else:
            self.biases = np.array(biases)

        self.stats.update({
            'picked_proba':{
                'x_label': 'step',
                'data': []
            }
        })

    def __call__(self, state, prev_action=None, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        values = self.value_function.from_state(state, prev_action)
        if type(values) is torch.Tensor:
            values = values.clone().detach().cpu().numpy()

        if epsilon > 0.0015:
            try:
                aux = np.exp((1/epsilon - 1) * (values+self.biases-np.max(values+self.biases)))
                self.biases = self.biases * self.biases_decay
                if np.any(np.isnan(aux)):
                    print("Warning: NaN values detected in softmax sampling probabilities")
                    print(f"Values: {values}")
                    print(f"Biases: {self.biases}")
                    print(f"Max values: {np.max(values)}")
                    print(f"Max biases: {np.max(self.biases)}")
                    aux[np.isnan(aux)] = 1  # Replace NaN values with 0
                if np.max(aux) == np.inf:
                    aux[aux != np.inf] = 0
                    aux[aux == np.inf] = 1
                probas = aux/np.sum(aux)
                action = np.random.choice([x for x in range(len(probas))], p=probas)
                self.agent.log_data("picked_proba", probas[action])
            except:
                print(epsilon, values, aux, probas)
                self.greedy_policy.agent = self.agent
                action = self.greedy_policy(state, prev_action)
                self.agent.log_data("picked_proba", 1)
        else:
            self.greedy_policy.agent = self.agent
            action = self.greedy_policy(state, prev_action)
            self.agent.log_data("picked_proba", 1)

        self.agent.log_data("predicted_value", values[action])
        self.stats.update(self.greedy_policy.stats)

        return action
    
    def call_batch(self, state_batch, epsilon=None):
        raise NotImplementedError
