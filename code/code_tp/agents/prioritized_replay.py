import random
import numpy as np
from .replay_buffer import ReplayBufferAgent, ReplayBuffer

"""
CREDITS TO https://github.com/rlcode/per

SumTree
a binary tree data structure where the parent’s value is the sum of its children
"""

class SumTree:
    """
    A binary tree data structure where the parent’s value is the sum of its children
    """
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and sample
    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and sample
    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Implémentation d'une mémoire des expériences passées avec échantillonage priorisé
    telle que décrit dans [l'article *Prioritized Experience Replay*](http://arxiv.org/abs/1511.05952)

    Modifie `ReplayBuffer` pour échantilloner la mémoire en tenant compte d'un critère de priorité
    """
    e = 0.01

    def __init__(self, obs_shape, max_size=100000, batch_size=32, default_error=10000, 
                    alpha=0.5, alpha_decrement_per_sampling=None,
                    beta=0, beta_increment_per_sampling=None,
                    total_samplings=None,
                    **kwargs):
        """
        * `alpha`: Hyperparamètre *priority* dans l'article. Une valeur de 0 correspond à donner la
        même priorité à toutes les transitions, *ie* à utiliser un *replay buffer* non priorisé
        * `beta`: Hyperparamètre *Importance-Sampling* dans l'article. Vise à compenser le fait que
        certaines transitions sont plus souvent utilisées dans l'entrainement du NN en modifiant la
        taille des updates (des mises à jours plus petites mais plus fréquentes correspondent à une
        recherche plus fine dans l'espace des paramètres du NN)
        * `alpha_decrement_per_sampling`: La valeur par défaut vise à atteindre `alpha=0` après
        `total_samplings` (voir ci-dessous) *batchs* échantillonés.
        * `beta_increment_per_sampling`: La valeur par défaut vise à atteindre `beta=1` après
        `total_samplings` (voir ci-dessous) *batchs* échantillonés.
        * `total_samplings`: Vaut `5*max_size` par défaut
        """
        self.alpha = alpha
        self.beta = beta

        if total_samplings is None:
            total_samplings = 5*max_size

        if alpha_decrement_per_sampling is None:
            self.alpha_decrement_per_sampling = self.alpha/total_samplings
        else:
            self.alpha_decrement_per_sampling = alpha_decrement_per_sampling

        if beta_increment_per_sampling is None:
            self.beta_increment_per_sampling = (1-self.beta)/total_samplings
        else:
            self.beta_increment_per_sampling = beta_increment_per_sampling

        self.tree = SumTree(max_size)
        self.default_error = default_error
        super().__init__(obs_shape, max_size, batch_size, **kwargs)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def store(self, *args, **kwargs):
        i = super().store(*args, **kwargs)
        p = self._get_priority(self.default_error)
        self.tree.add(p, i)

    def sample(self):
        batch = []
        idxs = []
        js = []
        segment = self.tree.total() / self.batch_size
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        self.alpha = np.max([0., self.alpha - self.alpha_decrement_per_sampling])

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, j) = self.tree.get(s)
            priorities.append(p)
            js.append(j)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        batch = (
                    self.normalize(self.states[js]),
                    self.actions[js],
                    self.normalize(self.next_states[js]),
                    self.rewards[js], self.dones[js], None
                )

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class PrioritizedReplayBufferAgent(ReplayBufferAgent):
    """
    Agent tirant profit du `PrioritizedReplayBuffer` en utilisant la **Différence Temporelle**
    (càd l'erreur entre la prédiction de la fonction de valeur et la valeur cible) comme
    critère de priorité
    """
    def train_one_batch(self):
        (states, actions, next_states, rewards, dones, infos), idxs, is_weights = self.replay_buffer.sample()

        #print(actions.shape)
        target_values = self.target_value_from_state_batch(next_states, rewards, dones)
        errors = self.value_function.update_batch(states, actions, target_values, is_weights)
        for idx, err in zip(idxs, errors):
            self.replay_buffer.update(idx, err)
