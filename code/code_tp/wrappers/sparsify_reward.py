import gymnasium as gym
import numpy as np


class SparsifyReward(gym.RewardWrapper):
    def __init__(self, env, one_every=10, noise_level=0.0):
        super().__init__(env)
        self.one_every = one_every
        self.inner_reward = 0
        self.noise_level = noise_level

    def reward(self, reward):
        self.inner_reward += reward
        noise = np.clip(
            np.random.normal(0, self.noise_level),
            -self.one_every / 2,
            self.one_every / 2,
        )
        step_reward = 0
        while self.inner_reward > self.one_every + noise:
            self.inner_reward -= self.one_every + noise
            step_reward += 1
        return step_reward
