from .base import QLearningAgent
import numpy as np
import torch

import time
import threading


class ReplayBuffer:
    """
    Implémentation d'une mémoire des expériences passées, telle que décrit dans
    [l'article sur le DQN](http://arxiv.org/abs/1312.5602)
    """

    def __init__(
        self, obs_shape, max_size=100000, batch_size=32, warmup_size=None, **kwargs
    ):
        """
        * `obs_shape` : Taille d'un tableau numpy contenant une observation
        * `max_size` : Nombre de transitions conservées en mémoire
        * `batch_size` : Nombre de transitions échantillonnées depuis la mémoire
        * `warmup_size` : Nombre de transitions à stocker avant de commencer à échantillonner
        """
        # Pre-allocate memory on GPU if using CUDA
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.states = torch.zeros(
            (max_size, *obs_shape), dtype=torch.uint8, device=device
        )
        self.actions = torch.zeros((max_size,), dtype=torch.long, device=device)
        self.next_states = torch.zeros(
            (max_size, *obs_shape), dtype=torch.uint8, device=device
        )
        self.rewards = torch.zeros((max_size,), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_size,), dtype=torch.bool, device=device)
        self.prev_actions = torch.zeros((max_size,), dtype=torch.long, device=device)
        self.device = device

        self.max_size = max_size
        self.batch_size = batch_size
        self.n_inserted = 0
        self.max_obs_val = -99999
        self.min_obs_val = 99999
        self.norm_offset = 0
        self.norm_scale = 1
        self.warmup_size = (
            warmup_size if warmup_size is not None else self.batch_size * 10
        )
        print(f"ReplayBuffer initialized with warmup_size={self.warmup_size}")

    def ready(self):
        """
        Indique si la mémoire contient au minimum 10 *batchs* de transitions
        """
        return self.n_inserted > self.warmup_size

    def store(self, state, action, next_state, reward, done, infos, prev_action=None):
        """
        Enregistre  une transition en mémoire (écrase les anciennes si la taille
        maximum est atteinte)

        Les paramètres représentent la transition à stocker
        """
        # Convert numpy arrays to torch tensors and move to correct device
        state = torch.from_numpy(state).to(self.device)
        next_state = torch.from_numpy(next_state).to(self.device)

        old_max = self.max_obs_val
        old_min = self.min_obs_val
        self.max_obs_val = max(self.max_obs_val, state.max().item())
        self.min_obs_val = min(self.min_obs_val, state.min().item())

        # Update normalization constants only if min/max changed
        if old_max != self.max_obs_val or old_min != self.min_obs_val:
            self.norm_offset = (self.max_obs_val + self.min_obs_val) / 2
            self.norm_scale = (self.max_obs_val - self.min_obs_val) / 2

        i = self.n_inserted % self.max_size
        self.states[i] = state
        self.actions[i] = action
        self.next_states[i] = next_state
        self.rewards[i] = float(reward)  # Convert reward to float
        self.dones[i] = bool(done)  # Ensure done is boolean
        self.prev_actions[i] = prev_action if prev_action is not None else 0

        self.n_inserted += 1
        return i

    def normalize(self, state):
        """
        Recadre les valeurs d'entrées entre -1 et 1
        """
        # Handle both numpy arrays and torch tensors
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        return (state.float() - self.norm_offset) / self.norm_scale

    def denormalize(self, state):
        """
        Recadre les valeurs d'entrées entre -1 et 1
        """
        # Handle both numpy arrays and torch tensors
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).to(self.device)
        return state * self.norm_scale + self.norm_offset

    def sample(self, i=None):
        """
        Renvoit un tuple de tableaux numpy contenant `batch_size` transitions dans
        le format suivant :

        * Les éléments du tuple renvoyé sont dans le même ordre que les paramètres de `self.store(...)`
        * Pour chaque élément du tuple, la première dimension du tableau correspond aux différents éléments d'un batch
        """
        # Sample directly on GPU
        n_stored = min(self.n_inserted, self.max_size)
        if i is None:
            i = torch.randint(0, n_stored, size=(self.batch_size,), device=self.device)
        elif isinstance(i, np.ndarray):
            i = torch.from_numpy(i).to(self.device)
        elif isinstance(i, torch.Tensor):
            i = i.to(self.device)

        return (
            self.normalize(self.states[i]),
            self.actions[i],
            self.normalize(self.next_states[i]),
            self.rewards[i],
            self.dones[i],
            self.prev_actions[i],
        )

    def log_tensorboard(self, tensorboard, step):
        pass

    @property
    def reward_scaling_factor(self):
        return 1


class ReplayBufferAgent(QLearningAgent):
    """
    Agent de base utilisant un *replay buffer* (http://arxiv.org/abs/1312.5602)

    La mise à jour de l'agent (dans `self.train_with_transition(...)`) est modifiée
    par rapport à l'agent *Q-learning* de la manière suivante :

    * Lorsque la méthode est appelée, la transition est stockée dans la mémoire de transitions (*replay buffer*)
    * Si le *buffer* est prêt, on échantillonne périodiquement une *batch* de transitions qu'on utilise  ensuite
    pour mettre à jour l'approximation de la fonction de valeur
    """

    def __init__(
        self,
        env,
        replay_buffer_class,
        replay_buffer_args={},
        update_interval=1,
        batches_per_update=1,
        **kwargs,
    ):
        """
        * `env`: Environnement gym dans lequel l'agent va évoluer
        * `replay_buffer_class`: Classe implémentant le *replay buffer*
        * `replay_buffer_args`: Arguments à passer au constructeur du *replay buffer*
        * `update_interval`: Interval
        * `kwargs`: Dictionnaire d'arguments passés au constructeur de la classe parente
        """
        print(kwargs)
        super().__init__(env, **kwargs)
        self.replay_buffer = replay_buffer_class(
            obs_shape=self.env.observation_space.shape, **replay_buffer_args
        )
        self.update_interval = update_interval
        self.batches_per_update = batches_per_update
        self.last_update = 0

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value
        self.log_data("gamma", value, accumulate=True, test=False)
        if hasattr(self, "replay_buffer") and hasattr(self.replay_buffer, "gamma"):
            self.replay_buffer.gamma = value

    def train_with_transition(
        self, state, action, next_state, reward, done, infos, prev_action=None
    ):
        # print("Training from ReplayBufferAgent")
        self.replay_buffer.store(
            state, action, next_state, reward, done, infos, prev_action
        )
        self.replay_buffer.log_tensorboard(self.tensorboard, self.training_steps)
        if self.replay_buffer.ready():
            # n_stored = min(self.replay_buffer.n_inserted, self.replay_buffer.max_size)
            # update_interval = self.replay_buffer.max_size/n_stored
            update_interval = self.update_interval
            if (
                update_interval > 0
                and self.training_steps - self.last_update >= update_interval
            ):
                for _ in range(self.batches_per_update):
                    self.train_one_batch()
                self.last_update = self.training_steps
            self.policy.update()
            self.policy.value_scaling = self.replay_buffer.reward_scaling_factor

    def select_action(self, state, prev_action=None, **kwargs):
        if not self.replay_buffer.ready():
            return self.env.action_space.sample()
        update_epsilon = kwargs.pop("update_epsilon", self.replay_buffer.ready())
        return super().select_action(
            self.replay_buffer.normalize(state),
            prev_action,
            update_epsilon=update_epsilon,
            **kwargs,
        )

    def train_one_batch(self):
        states, actions, next_states, rewards, dones, prev_actions = (
            self.replay_buffer.sample()
        )
        # print(actions.shape)
        target_values = self.target_value_from_state_batch(
            next_states,
            rewards,
            dones,
            actions,
            getattr(self.replay_buffer, "n_step", 1),
        )
        self.value_function.update_batch(states, actions, target_values, prev_actions)

    def train(self, *args, **kwargs):
        if self.update_interval > 0:
            return super().train(*args, **kwargs)
        n_episodes = kwargs.get("n_episodes", 1000 if len(args) == 0 else args[0])

        self.experience_process = threading.Thread(
            target=super().train, args=args, kwargs=kwargs
        )
        self.experience_process.start()
        self.neural_process = threading.Thread(
            target=self.parallel_neural_training, args=(n_episodes,)
        )
        self.neural_process.start()

        try:
            self.experience_process.join()
            self.neural_process.join()
        except KeyboardInterrupt:
            self.should_stop = True
            try:
                self.replay_buffer.close()
            except (AttributeError, RuntimeError):
                pass
            self.experience_process.join()
            self.neural_process.join()

    def parallel_neural_training(self, max_episodes):
        while not self.replay_buffer.ready():
            time.sleep(0.2)

        trained_batches = 0
        while self.training_episodes < max_episodes and not self.should_stop:
            if trained_batches > self.training_steps // abs(self.update_interval):
                time.sleep(0.2)
                continue
            self.train_one_batch()
            trained_batches += 1
