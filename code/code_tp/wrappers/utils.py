import gym
import numpy as np

class LogScaleObs(gym.ObservationWrapper):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        self.o = self.env.observation_space
        self.observation_space = gym.spaces.Box(
            low=self.t(self.o.low),
            high=self.t(self.o.high),
            shape=self.o.shape
        )

    def t(self, x):
        if type(x) is str:
            x = np.zeros_like(self.o.low)
        x = x.astype(float)
        l = np.min(np.stack((np.abs(self.o.low), np.abs(self.o.high), np.zeros_like(self.o.low)), axis=1), axis=1)
        h = np.max(np.stack((np.abs(self.o.low), np.abs(self.o.high)), axis=1), axis=1)
        return np.sign(x)*np.log(np.abs(x)-l+1)/np.log(h-l+1)

    def observation(self, s):
        return self.t(s)


class TabularObservation(gym.ObservationWrapper):
    def __init__(self, env, n_levels, feature_wrapper=None, *args, **kwargs):
        if feature_wrapper is not None:
            env = feature_wrapper(env)
        super().__init__(env, *args, **kwargs)
        assert isinstance(self.env.observation_space, gym.spaces.Box)
        self.old_obs_space = self.observation_space
        if type(n_levels) is not np.ndarray:
            n_levels = n_levels * np.ones_like(self.old_obs_space.sample(), dtype=int)
        self.n_levels = n_levels
        self.observation_space = gym.spaces.MultiDiscrete(self.n_levels)

    def observation(self, s):
        r = np.zeros(self.observation_space.shape, dtype=int)
        for i, (low, high, n, obs) in enumerate(zip(
                self.old_obs_space.low,
                self.old_obs_space.high,
                self.n_levels,
                s
        )):
            assert high >= low
            try:
                r[i] = int(np.round((n-1) * ((obs - low) / (high - low + 1e-6))))
            except:
                print(i, n, obs, low, high)
                r[i] = 0
        return r

class BoredomWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return reward - 0.01
