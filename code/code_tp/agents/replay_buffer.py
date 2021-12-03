from .base import Agent
import numpy as np

class ReplayBuffer:
    def __init__(self, obs_shape, max_size=100000, batch_size=32, normalize=True):
        self.states = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((max_size,), dtype=np.uint8)
        self.next_states = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.rewards = np.zeros((max_size,), dtype=np.float16)
        self.dones = np.zeros((max_size,), dtype=np.bool)

        self.max_size = max_size
        self.batch_size = batch_size
        self.n_inserted = 0
        self.normalize = normalize

    def ready(self):
        return self.n_inserted > self.batch_size*10

    def store(self, state, action, next_state, reward, done, infos):
        i = self.n_inserted % self.max_size
        self.states[i] = state
        self.actions[i] = action
        self.next_states[i] = next_state
        self.rewards[i] = reward
        self.dones[i] = done

        self.n_inserted+= 1
        return i

    def sample(self):
        n_stored = min(self.n_inserted, self.max_size)
        i = np.random.randint(0, n_stored, size=(self.batch_size,))

        return self.states[i], self.actions[i], self.next_states[i],\
                self.rewards[i], self.dones[i], None

class ReplayBufferAgent(Agent):
    def __init__(self, env, replay_buffer_class, replay_buffer_args={}, update_interval=1, **kwargs):
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
