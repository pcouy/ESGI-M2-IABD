import random
import numpy as np
from .replay_buffer import ReplayBufferAgent, ReplayBuffer

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree:
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
    e = 0.01
    a = 0.6
    beta = 0.4
    beta_increment_per_sampling = 0.001

    def __init__(self, obs_shape, max_size=100000, batch_size=32, default_error=10000):
        self.tree = SumTree(max_size)
        self.default_error = default_error
        super().__init__(obs_shape, max_size, batch_size)

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

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
                    self.states[js], self.actions[js], self.next_states[js],
                    self.rewards[js], self.dones[js], None
                )

        return batch, idxs, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

class PrioritizedReplayBufferAgent(ReplayBufferAgent):
    def train_with_transition(self, state, action, next_state, reward, done, infos):
        #print("Training from ReplayBufferAgent")
        self.replay_buffer.store(state, action, next_state, reward, done, infos)
        if self.replay_buffer.ready():
            n_stored = min(self.replay_buffer.n_inserted, self.replay_buffer.max_size)
            update_interval = self.replay_buffer.max_size/n_stored
            if self.training_steps-self.last_update >= update_interval:
                (states, actions, next_states, rewards, dones, infos), idxs, is_weights =\
                    self.replay_buffer.sample()

                #print(actions.shape)
                target_values = self.target_value_from_state_batch(next_states, rewards, dones)
                errors = self.value_function.update_batch(states, actions, target_values)
                for idx, err in zip(idxs, errors):
                    self.replay_buffer.update(idx, err)
                self.last_update = self.training_steps
            self.policy.update()
