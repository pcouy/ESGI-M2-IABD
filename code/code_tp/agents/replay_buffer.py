from .base import Agent
import numpy as np

class ReplayBuffer:
    """
    Implémentation d'une mémoire des expériences passées, telle que décrit dans
    [l'article sur le DQN](http://arxiv.org/abs/1312.5602)
    """
    def __init__(self, obs_shape, max_size=100000, batch_size=32):
        """
        * `obs_shape` : Taille d'un tableau numpy contenant une observation
        * `max_size` : Nombre de transitions conservées en mémoire
        * `batch_size` : Nombre de transitions échantillonnées depuis la mémoire
        """
        self.states = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((max_size,), dtype=np.int8)
        self.next_states = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.rewards = np.zeros((max_size,), dtype=np.float16)
        self.dones = np.zeros((max_size,), dtype=np.bool)

        self.max_size = max_size
        self.batch_size = batch_size
        self.n_inserted = 0
        self.max_obs_val = -99999
        self.min_obs_val = 99999

    def ready(self):
        """
        Indique si la mémoire contient au minimum 10 *batchs* de transitions
        """
        return self.n_inserted > self.batch_size*10

    def store(self, state, action, next_state, reward, done, infos):
        """
        Enregistre  une transition en mémoire (écrase les anciennes si la taille
        maximum est atteinte)

        Les paramètres représentent la transition à stocker
        """
        self.max_obs_val = max(self.max_obs_val, state.max())
        self.min_obs_val = min(self.min_obs_val, state.min())
        i = self.n_inserted % self.max_size
        self.states[i] = state
        self.actions[i] = action
        self.next_states[i] = next_state
        self.rewards[i] = reward
        self.dones[i] = done

        self.n_inserted+= 1
        return i

    def normalize(self, state):
        """
        Recadre les valeurs d'entrées entre 0 et 1
        """
        return (2*state - (self.max_obs_val + self.min_obs_val)) / (self.max_obs_val - self.min_obs_val)

    def sample(self):
        """
        Renvoit un tuple de tableaux numpy contenant `batch_size` transitions dans
        le format suivant :

        * Les éléments du tuple renvoyé sont dans le même ordre que les paramètres de `self.store(...)`
        * Pour chaque élément du tuple, la première dimension du tableau correspond aux différents éléments d'un batch
        """
        n_stored = min(self.n_inserted, self.max_size)
        i = np.random.randint(0, n_stored, size=(self.batch_size,))

        return self.normalize(self.states[i]),\
                self.actions[i],\
                self.normalize(self.next_states[i]),\
                self.rewards[i], self.dones[i], None

class ReplayBufferAgent(Agent):
    """
    Agent de base utilisant un *replay buffer* (http://arxiv.org/abs/1312.5602)

    La mise à jour de l'agent (dans `self.train_with_transition(...)`) est modifiée
    par rapport à l'agent *Q-learning* de la manière suivante :

    * Lorsque la méthode est appelée, la transition est stockée dans la mémoire de transitions (*replay buffer*)
    * Si le *buffer* est prêt, on échantillonne périodiquement une *batch* de transitions qu'on utilise  ensuite
    pour mettre à jour l'approximation de la fonction de valeur
    """
    def __init__(self, env, replay_buffer_class, replay_buffer_args={}, update_interval=1, **kwargs):
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
            obs_shape = self.env.observation_space.shape,
            **replay_buffer_args
        )
        self.update_interval = update_interval
        self.last_update = 0

    def train_with_transition(self, state, action, next_state, reward, done, infos):
        #print("Training from ReplayBufferAgent")
        self.replay_buffer.store(state, action, next_state, reward, done, infos)
        if self.replay_buffer.ready():
            n_stored = min(self.replay_buffer.n_inserted, self.replay_buffer.max_size)
            #update_interval = self.replay_buffer.max_size/n_stored
            update_interval = self.update_interval
            if self.training_steps-self.last_update >= update_interval:
                states, actions, next_states, rewards, dones, infos = self.replay_buffer.sample()
                #print(actions.shape)
                target_values = self.target_value_from_state_batch(next_states, rewards, dones)
                self.value_function.update_batch(states, actions, target_values)
                self.last_update = self.training_steps
            self.policy.update()
