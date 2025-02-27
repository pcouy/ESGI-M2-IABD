from .replay_buffer import ReplayBuffer
import numpy as np
import os
import torch
import multiprocessing
import time
import queue
import threading


def _return_one_batch(
    memmapped_arrays,  # Tuple of memmapped arrays (states, actions, rewards, dones, prev_actions)
    batch_size,
    current_n,
    max_size,
    idx=None,
):
    n_stored = current_n if current_n < max_size else max_size
    states, actions, rewards, dones, prev_actions = memmapped_arrays
    if idx is None:
        valid_indices = np.where(~dones[:n_stored])[0]
        idx = np.random.choice(valid_indices, size=batch_size, replace=True)
    next_idx = (idx + 1) % (max_size)

    batch_states = states[idx].copy()
    batch_actions = actions[idx].copy()
    batch_next_states = states[next_idx].copy()
    batch_rewards = rewards[idx].copy()
    batch_dones = dones[next_idx].copy()
    batch_prev_actions = prev_actions[idx].copy()

    return (
        batch_states,
        batch_actions,
        batch_next_states,
        batch_rewards,
        batch_dones,
        batch_prev_actions,
    )


def _preloader_func(
    storage_path,
    max_size,
    batch_size,
    obs_shape,
    n_inserted_val,
    preload_queue,
    stop_event,
):
    # Open memmapped arrays for reading
    states_path = os.path.join(storage_path, "states.dat")
    actions_path = os.path.join(storage_path, "actions.dat")
    rewards_path = os.path.join(storage_path, "rewards.dat")
    dones_path = os.path.join(storage_path, "dones.dat")
    prev_actions_path = os.path.join(storage_path, "prev_actions.dat")

    # Now states array contains both current and next states sequentially
    states = np.memmap(
        states_path, mode="r", shape=(max_size + 1, *obs_shape), dtype=np.uint8
    )
    actions = np.memmap(actions_path, mode="r", shape=(max_size,), dtype=np.int64)
    rewards = np.memmap(rewards_path, mode="r", shape=(max_size,), dtype=np.float32)
    dones = np.memmap(dones_path, mode="r", shape=(max_size,), dtype=bool)
    prev_actions = np.memmap(
        prev_actions_path, mode="r", shape=(max_size,), dtype=np.int64
    )

    while not stop_event.is_set():
        with n_inserted_val.get_lock():
            current_n = n_inserted_val.value
        n_stored = current_n if current_n < max_size else max_size
        if n_stored < batch_size:
            time.sleep(0.1)
            continue

        batch = _return_one_batch(
            (states, actions, rewards, dones, prev_actions),
            batch_size,
            current_n,
            max_size,
        )

        try:
            preload_queue.put(batch, block=False)
        except Exception:
            time.sleep(0.05)
            continue
        time.sleep(0.01)  # slight pause to yield CPU


class MemmappedReplayBuffer(ReplayBuffer):
    def __init__(
        self,
        obs_shape,
        max_size=100000,
        batch_size=32,
        storage_path="./replay_buffer",
        preload_batches=5,
        preload_on_gpu=True,
        flush_every=None,
        warmup_size=None,
    ):
        # Initialize parameters manually to avoid GPU allocation in parent's __init__
        self.flush_every = flush_every
        self.obs_shape = obs_shape
        self.max_size = max_size
        self.batch_size = batch_size
        self.max_obs_val = -99999
        self.min_obs_val = 99999
        self.norm_offset = 0
        self.norm_scale = 1
        self.warmup_size = (
            warmup_size
            if warmup_size is not None
            else 2 * self.batch_size * preload_batches
        )
        self.storage_path = storage_path
        os.makedirs(self.storage_path, exist_ok=True)
        print(f"MemmappedReplayBuffer initialized with warmup_size={self.warmup_size}")

        # Create memmapped arrays for disk storage
        states_path = os.path.join(self.storage_path, "states.dat")
        actions_path = os.path.join(self.storage_path, "actions.dat")
        rewards_path = os.path.join(self.storage_path, "rewards.dat")
        dones_path = os.path.join(self.storage_path, "dones.dat")
        prev_actions_path = os.path.join(self.storage_path, "prev_actions.dat")

        # Single states array with one extra slot for the last next_state
        self.states = np.memmap(
            states_path, mode="w+", shape=(max_size + 1, *obs_shape), dtype=np.uint8
        )
        self.actions = np.memmap(
            actions_path, mode="w+", shape=(max_size,), dtype=np.int64
        )
        self.rewards = np.memmap(
            rewards_path, mode="w+", shape=(max_size,), dtype=np.float32
        )
        self.dones = np.memmap(dones_path, mode="w+", shape=(max_size,), dtype=bool)
        self.prev_actions = np.memmap(
            prev_actions_path, mode="w+", shape=(max_size,), dtype=np.int64
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.preload_batches = preload_batches
        self.preload_on_gpu = preload_on_gpu
        if self.preload_on_gpu:
            # Use a threading based preloader that loads directly to GPU
            self.n_inserted = 0
            self.n_inserted_lock = threading.Lock()
            self.preload_queue = queue.Queue(maxsize=self.preload_batches)
            self.stop_event = threading.Event()
            self.preloader_thread = threading.Thread(
                target=self._preloader_gpu_thread_func
            )
            self.preloader_thread.daemon = True
            self.preloader_thread.start()
        else:
            # Fallback to original CPU preloading using multiprocessing
            self.n_inserted_val = multiprocessing.Value("i", 0)
            self.preload_queue = multiprocessing.Queue(maxsize=self.preload_batches)
            self.stop_event = multiprocessing.Event()
            self.preloader_process = multiprocessing.Process(
                target=_preloader_func,
                args=(
                    self.storage_path,
                    max_size,
                    batch_size,
                    obs_shape,
                    self.n_inserted_val,
                    self.preload_queue,
                    self.stop_event,
                ),
            )
            self.preloader_process.daemon = True
            self.preloader_process.start()

    def _preloader_gpu_thread_func(self):
        # This thread preloads batches directly to GPU
        while not self.stop_event.is_set():
            with self.n_inserted_lock:
                current_n = self.n_inserted
            if self.n_inserted < self.warmup_size * 0.8:
                time.sleep(0.1)
                continue

            # Skip if queue is full
            if self.preload_queue.full():
                time.sleep(0.1)
                continue

            (
                states,
                actions,
                next_states,
                rewards,
                dones,
                prev_actions,
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
            )

            # Load and normalize states - note we get next states from idx + 1
            batch_states = self.normalize(
                torch.from_numpy(states).to(self.device, non_blocking=True)
            )
            batch_actions = torch.from_numpy(actions).to(self.device, non_blocking=True)
            batch_next_states = self.normalize(
                torch.from_numpy(next_states).to(self.device, non_blocking=True)
            )
            batch_rewards = torch.from_numpy(rewards).to(self.device, non_blocking=True)
            batch_dones = torch.from_numpy(dones).to(
                self.device, non_blocking=True
            )  # Get dones from next index
            batch_prev_actions = torch.from_numpy(prev_actions).to(
                self.device, non_blocking=True
            )
            try:
                self.preload_queue.put(
                    (
                        batch_states,
                        batch_actions,
                        batch_next_states,
                        batch_rewards,
                        batch_dones,
                        batch_prev_actions,
                    ),
                    block=False,
                )
            except queue.Full:
                time.sleep(0.05)
                continue
            time.sleep(0.01)  # slight pause

    def store(self, state, action, next_state, reward, done, infos, prev_action=None):
        # Convert states to numpy arrays and ensure correct dtype
        if isinstance(state, torch.Tensor):
            state_np = state.cpu().numpy()
        else:
            state_np = state
        if isinstance(next_state, torch.Tensor):
            next_state_np = next_state.cpu().numpy()
        else:
            next_state_np = next_state

        state_np = state_np.astype(np.uint8)
        next_state_np = next_state_np.astype(np.uint8)

        # Update normalization statistics
        old_max = self.max_obs_val
        old_min = self.min_obs_val
        self.max_obs_val = max(self.max_obs_val, state_np.max(), next_state_np.max())
        self.min_obs_val = min(self.min_obs_val, state_np.min(), next_state_np.min())
        if old_max != self.max_obs_val or old_min != self.min_obs_val:
            self.norm_offset = (self.max_obs_val + self.min_obs_val) / 2
            self.norm_scale = (self.max_obs_val - self.min_obs_val) / 2

        if self.preload_on_gpu:
            with self.n_inserted_lock:
                idx = self.n_inserted % self.max_size
                self.n_inserted += 1
                # If this is the end of an episode, increment again to protect the final state
                if done:
                    self.n_inserted += 1
        else:
            with self.n_inserted_val.get_lock():
                idx = self.n_inserted_val.value % self.max_size
                self.n_inserted_val.value += 1
                # If this is the end of an episode, increment again to protect the final state
                if done:
                    self.n_inserted_val.value += 1

        # Store current state and next state sequentially
        self.states[idx] = state_np
        next_idx = (idx + 1) % (self.max_size + 1)
        self.states[next_idx] = next_state_np

        # Store other transition data, with done at next_idx % max_size since dones array is max_size long
        self.actions[idx] = int(action)
        self.rewards[idx] = float(reward)
        self.dones[(idx + 1) % self.max_size] = bool(done)  # Store done at next index
        self.prev_actions[idx] = int(prev_action) if prev_action is not None else 0

        # Optionally flush every 100 insertions
        if self.flush_every is not None and idx % self.flush_every == 0:
            self.flush()

        return idx

    def flush(self):
        self.states.flush()
        self.actions.flush()
        self.rewards.flush()
        self.dones.flush()
        self.prev_actions.flush()

    def ready(self):
        if self.preload_on_gpu:
            with self.n_inserted_lock:
                count = self.n_inserted
            return count > self.warmup_size
        else:
            with self.n_inserted_val.get_lock():
                count = self.n_inserted_val.value
            return count > self.warmup_size

    def wait_for_queue(self, timeout=10):
        """Wait until the preload queue is full or timeout is reached.

        Args:
            timeout: Maximum time to wait in seconds (default=10)

        Returns:
            bool: True if queue is full, False if timeout was reached
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            # First ensure the buffer has enough transitions to sample
            if not self.ready():
                time.sleep(0.1)
                continue

            # Then check if the queue is full
            if self.preload_queue.full():
                return True
            time.sleep(0.1)
        return False

    def sample(self, i=None, skip_episode_check=False, timeout=0):
        if i is not None:
            # For direct index sampling, verify indices are not from episode endings
            i = np.asarray(i)
            (
                batch_states,
                batch_actions,
                batch_next_states,
                batch_rewards,
                batch_dones,
                batch_prev_actions,
            ) = _return_one_batch(
                (self.states, self.actions, self.rewards, self.dones, self.prev_actions),
                self.batch_size,
                self.n_inserted,
                self.max_size,
                i,
            )
            
            if self.preload_on_gpu:
                batch_states = torch.from_numpy(batch_states).to(
                    self.device, non_blocking=True
                )
                batch_actions = torch.from_numpy(batch_actions).to(
                    self.device, non_blocking=True
                )
                batch_next_states = torch.from_numpy(batch_next_states).to(
                    self.device, non_blocking=True
                )
                batch_rewards = torch.from_numpy(batch_rewards).to(
                    self.device, non_blocking=True
                )
                batch_dones = torch.from_numpy(batch_dones).to(
                    self.device, non_blocking=True
                )  # Get dones from next index
                batch_prev_actions = torch.from_numpy(batch_prev_actions).to(
                    self.device, non_blocking=True
                )
            
            # Normalize states for direct index sampling
            batch_states = self.normalize(batch_states)
            batch_next_states = self.normalize(batch_next_states)

            if self.preload_on_gpu:
                return (
                    batch_states,
                    batch_actions,
                    batch_next_states,
                    batch_rewards,
                    batch_dones,
                    batch_prev_actions,
                )
            else:
                return (
                    batch_states,
                    torch.from_numpy(batch_actions).to(self.device),
                    batch_next_states,
                    torch.from_numpy(batch_rewards).to(self.device),
                    torch.from_numpy(batch_dones).to(self.device),
                    torch.from_numpy(batch_prev_actions).to(self.device),
                )
        else:
            if self.preload_queue is not None:
                try:
                    # Preloaded batches are already normalized and on GPU
                    return self.preload_queue.get(timeout=timeout)
                except queue.Empty:
                    print("Queue is empty")
                    if self.preload_on_gpu:
                        with self.n_inserted_lock:
                            current_n = self.n_inserted

                        (
                            states,
                            actions,
                            next_states,
                            rewards,
                            dones,
                            prev_actions,
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
                        )

                        batch_states = self.normalize(
                            torch.from_numpy(states).to(self.device, non_blocking=True)
                        )
                        batch_actions = torch.from_numpy(actions).to(
                            self.device, non_blocking=True
                        )
                        batch_next_states = self.normalize(
                            torch.from_numpy(next_states).to(
                                self.device, non_blocking=True
                            )
                        )
                        batch_rewards = torch.from_numpy(rewards).to(
                            self.device, non_blocking=True
                        )
                        batch_dones = torch.from_numpy(dones).to(
                            self.device, non_blocking=True
                        )  # Get dones from next index
                        batch_prev_actions = torch.from_numpy(prev_actions).to(
                            self.device, non_blocking=True
                        )
                        return (
                            batch_states,
                            batch_actions,
                            batch_next_states,
                            batch_rewards,
                            batch_dones,
                            batch_prev_actions,
                        )
                    else:
                        with self.n_inserted_val.get_lock():
                            current_n = (
                                self.n_inserted_val.value
                                if self.n_inserted_val.value < self.max_size
                                else self.max_size
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
                        )

            # Normalize states for non-preloaded batches
            batch_states = self.normalize(batch_states)
            batch_next_states = self.normalize(batch_next_states)

            if self.preload_on_gpu:
                return (
                    batch_states,
                    batch_actions,
                    batch_next_states,
                    batch_rewards,
                    batch_dones,
                    batch_prev_actions,
                )
            else:
                return (
                    batch_states,
                    torch.from_numpy(batch_actions).to(self.device),
                    batch_next_states,
                    torch.from_numpy(batch_rewards).to(self.device),
                    torch.from_numpy(batch_dones).to(self.device),
                    torch.from_numpy(batch_prev_actions).to(self.device),
                )

    def close(self):
        if self.preload_on_gpu:
            if hasattr(self, "preloader_thread") and self.preloader_thread is not None:
                self.stop_event.set()
                self.preloader_thread.join()
        else:
            if (
                hasattr(self, "preloader_process")
                and self.preloader_process is not None
            ):
                self.stop_event.set()
                self.preloader_process.join()

    def __del__(self):
        self.close()
