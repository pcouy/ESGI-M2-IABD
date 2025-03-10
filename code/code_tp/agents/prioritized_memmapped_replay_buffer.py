from .memmapped_replay_buffer import MemmappedReplayBuffer, _return_one_batch
from .prioritized_replay import PrioritizedReplayBuffer
import numpy as np
import os
import torch
import random
import time
import queue
import warnings


def _prioritized_sampling(
    tree,  # The priority sum tree
    batch_size,  # Batch size
    n_stored,  # Number of stored transitions
    dones,  # Done flags
    max_size,  # Maximum buffer size
    beta,  # Beta parameter for importance sampling
    beta_increment,  # How much to increment beta
    alpha,  # Alpha parameter for prioritization
    alpha_decrement,  # How much to decrement alpha
):
    """Helper function to sample transitions based on priorities"""
    batch_indices = []
    priorities = []
    segment = tree.total() / batch_size

    # Update alpha and beta parameters
    beta = np.min([1.0, beta + beta_increment])
    alpha = np.max([0.0, alpha - alpha_decrement])

    # Keep sampling until we have enough valid transitions
    while len(batch_indices) < batch_size:
        a = segment * len(batch_indices)
        b = segment * (len(batch_indices) + 1)
        s = random.uniform(a, b)
        idx, p, data_idx = tree.get(s)
        # Skip if it's the last state of an episode
        if data_idx >= n_stored - 1 or dones[(data_idx) % max_size]:
            continue
        priorities.append(p)
        batch_indices.append(data_idx)

    # Calculate importance sampling weights
    sampling_probabilities = priorities / tree.total()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        is_weights = np.power(tree.n_entries * sampling_probabilities, -beta)
        is_weights /= is_weights.max()
        if len(w) > 0 and issubclass(w[-1].category, RuntimeWarning):
            is_weights = None
        else:
            is_weights = torch.tensor(
                is_weights,
                dtype=torch.float32,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

    # Calculate tree indices for updates
    tree_idxs = [idx + tree.capacity - 1 for idx in batch_indices]

    return np.array(batch_indices), np.array(tree_idxs), is_weights, beta, alpha


class MemmappedSumTree:
    """
    A memory-mapped version of the SumTree that stores the tree structure on disk
    """

    def __init__(self, capacity, storage_path):
        self.capacity = capacity
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)

        # Create memory-mapped arrays for the tree and data indices
        tree_path = os.path.join(storage_path, "sum_tree.dat")
        data_path = os.path.join(storage_path, "data_indices.dat")

        self.tree = np.memmap(
            tree_path, mode="w+", shape=(2 * capacity - 1,), dtype=np.float32
        )
        self.data = np.memmap(data_path, mode="w+", shape=(capacity,), dtype=np.int32)

        self.write = 0
        self.n_entries = 0

        # Initialize arrays to zero
        self.tree[:] = 0
        self.data[:] = 0
        self.tree.flush()
        self.data.flush()

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

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

    def add(self, p, data_idx):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data_idx
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

        # Periodically flush to disk
        if self.write % 100 == 0:
            self.tree.flush()
            self.data.flush()

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])


class PrioritizedMemmappedReplayBuffer(MemmappedReplayBuffer):
    """
    A memory-mapped version of PrioritizedReplayBuffer that stores both experience data
    and the priority tree structure on disk
    """

    e = 0.01

    def __init__(
        self,
        default_error=10000,
        default_error_scale=20,
        alpha=0.5,
        alpha_decrement_per_sampling=None,
        min_alpha=0.5,
        beta=0,
        beta_increment_per_sampling=None,
        total_samplings=None,
        **kwargs
    ):

        # Initialize the base memory-mapped buffer
        super().__init__(**kwargs)

        # Priority-related parameters
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.beta = beta
        if total_samplings is None:
            total_samplings = 5 * self.max_size

        self.alpha_decrement_per_sampling = (
            alpha_decrement_per_sampling
            if alpha_decrement_per_sampling is not None
            else (self.alpha - self.min_alpha) / total_samplings
        )
        self.beta_increment_per_sampling = (
            beta_increment_per_sampling
            if beta_increment_per_sampling is not None
            else (1 - self.beta) / total_samplings
        )

        # Create memory-mapped sum tree in a subdirectory
        tree_storage_path = os.path.join(self.storage_path, "sum_tree")
        self.tree = MemmappedSumTree(self.max_size, tree_storage_path)
        self.default_error_scale = default_error_scale
        self.default_error = default_error

    @property
    def default_error(self):
        return self.default_error_scale * self.running_error_mean

    @default_error.setter
    def default_error(self, value):
        self.running_error_mean = value / self.default_error_scale

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.alpha

    def store(self, state, action, next_state, reward, done, infos, prev_action=None):
        idx = super().store(state, action, next_state, reward, done, infos, prev_action)
        p = self._get_priority(self.default_error)
        self.tree.add(p, idx)
        return idx

    def _preloader_gpu_thread_func(self):
        # This thread preloads prioritized batches directly to GPU
        while not self.stop_event.is_set():
            with self.n_inserted_lock:
                current_n = self.n_inserted
            n_stored = current_n if current_n < self.max_size else self.max_size
            if not self.ready():
                time.sleep(0.1)
                continue

            # Skip if queue is full
            if self.preload_queue.full():
                time.sleep(0.1)
                continue

            # Sample transitions using the helper function
            batch_indices, tree_idxs, is_weights, self.beta, self.alpha = (
                _prioritized_sampling(
                    self.tree,
                    self.batch_size,
                    n_stored,
                    self.dones,
                    self.max_size,
                    self.beta,
                    self.beta_increment_per_sampling,
                    self.alpha,
                    self.alpha_decrement_per_sampling,
                )
            )

            (
                batch_states,
                batch_actions,
                batch_next_states,
                batch_rewards,
                batch_dones,
                batch_prev_actions,
            ) = _return_one_batch(
                (
                    self.states,
                    self.actions,
                    self.rewards,
                    self.dones,
                    self.prev_actions,
                ),
                self.batch_size,
                current_n,
                self.max_size,
                batch_indices,
                n_steps=self.n_step,
                discount=self.gamma,
            )

            # Load and normalize the data
            batch_states = self.normalize(
                torch.from_numpy(batch_states).to(self.device, non_blocking=True)
            )
            batch_actions = torch.from_numpy(batch_actions).to(
                self.device, non_blocking=True
            )
            batch_next_states = self.normalize(
                torch.from_numpy(batch_next_states).to(self.device, non_blocking=True)
            )
            batch_rewards = torch.from_numpy(batch_rewards).to(
                self.device, non_blocking=True
            )
            batch_dones = torch.from_numpy(batch_dones).to(
                self.device, non_blocking=True
            )
            batch_prev_actions = torch.from_numpy(batch_prev_actions).to(
                self.device, non_blocking=True
            )

            try:
                self.preload_queue.put(
                    (
                        (
                            batch_states,
                            batch_actions,
                            batch_next_states,
                            batch_rewards,
                            batch_dones,
                            batch_prev_actions,
                        ),
                        tree_idxs,
                        is_weights,
                    ),
                    block=False,
                )
            except queue.Full:
                time.sleep(0.1)
                continue
            time.sleep(0.01)  # slight pause

    def sample(self, i=None, timeout=0):
        if i is not None:
            return super().sample(i=i, skip_episode_check=True)

        if self.preload_queue is not None:
            try:
                return self.preload_queue.get(timeout=timeout)
            except queue.Empty:
                print("Queue is empty")
                # If queue is empty, sample directly
                pass

        # Fallback to direct sampling if queue is empty or no preloader
        with self.n_inserted_lock:
            current_n = self.n_inserted
        n_stored = current_n if current_n < self.max_size else self.max_size

        # Sample transitions using the helper function
        batch_indices, tree_idxs, is_weights, self.beta, self.alpha = (
            _prioritized_sampling(
                self.tree,
                self.batch_size,
                n_stored,
                self.dones,
                self.max_size,
                self.beta,
                self.beta_increment_per_sampling,
                self.alpha,
                self.alpha_decrement_per_sampling,
            )
        )
        self.alpha = np.max([self.min_alpha, self.alpha])

        # Get the batch using parent class's sample method with skip_episode_check=True
        batch = super().sample(i=batch_indices, skip_episode_check=True)

        return batch, tree_idxs, is_weights

    def update(self, idx, error):
        self.running_error_mean = (1 - 1e-6) * self.running_error_mean + 1e-6 * error
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def close(self):
        super().close()
        if hasattr(self, "tree"):
            self.tree.tree.flush()
            self.tree.data.flush()
