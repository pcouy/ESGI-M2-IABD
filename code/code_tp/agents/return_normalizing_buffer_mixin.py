import torch
import numpy as np
import math


class TorchRunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), dtype=torch.float32):
        """Tracks the mean, variance and count of values."""
        self.mean = torch.zeros(shape, dtype=dtype)
        self.var = torch.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        if isinstance(x, np.ndarray):
            _x = torch.from_numpy(x)
        elif isinstance(x, float):
            _x = torch.tensor([x])
        else:
            _x = x

        batch_count = _x.shape[0]
        batch_mean = torch.mean(_x, dim=0)
        if batch_count > 1:
            batch_var = torch.var(_x, dim=0)
        else:
            batch_var = torch.zeros(())
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


class ReturnNormalizingBufferMixin:
    OVERRIDE_VALUE_SCALING = True

    def __init__(self, *args, **kwargs):
        self.gamma = kwargs.get("gamma", 0.99)  # discount factor
        self.avg_episode_length = kwargs.pop("init_avg_episode_length", 1000)
        super().__init__(*args, **kwargs)
        self._n_dones = 1
        self.reward_rms = TorchRunningMeanStd(epsilon=self.avg_episode_length)
        self.last_tensorboard_step = 0

    @property
    def mean_reward(self):
        return self.reward_rms.mean

    @property
    def var_reward(self):
        return self.reward_rms.var

    @property
    def std_reward(self):
        return torch.sqrt(self.var_reward)

    @property
    def reward_to_value_factor(self):
        return (
            1
            - (self.gamma * (1 - self.gamma**self.avg_episode_length))
            / (self.avg_episode_length * (1 - self.gamma))
        ) / (1 - self.gamma)

    @property
    def lambda_reward(self):
        return (
            self.std_reward
            * math.sqrt(
                1 / (1 - self.gamma**2)
                - self.gamma**2
                / self.avg_episode_length
                * (1 - self.gamma ** (2 * self.avg_episode_length))
                / ((1 - self.gamma**2) ** 2)
            )
            + 1e-8
        )

    @property
    def n_dones(self):
        return self._n_dones

    @n_dones.setter
    def n_dones(self, value):
        self._n_dones = value
        self.avg_episode_length = self.n_inserted / self.n_dones

    def scale_reward(self, reward):
        return (reward - self.mean_reward) / self.lambda_reward

    def unscale_reward(self, scaled_reward):
        return scaled_reward * self.lambda_reward + self.mean_reward

    def unscale_value(self, value):
        return (
            value * self.lambda_reward + self.mean_reward * self.reward_to_value_factor
        )

    def store(self, *transition):
        result = super().store(*transition)
        done = transition[4] if len(transition) > 4 else False
        reward = transition[3]
        if done:
            self.n_dones += 1
        self.reward_rms.update(reward)
        return result

    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        # Convert samples to list if it's a tuple
        if isinstance(samples, tuple):
            samples = list(samples)

        if len(samples) > 3:
            samples[3] = self.scale_reward(samples[3])
        else:
            # Convert inner tuple to list if needed
            if isinstance(samples[0], tuple):
                samples[0] = list(samples[0])
            samples[0][3] = self.scale_reward(samples[0][3])
        return samples

    def log_tensorboard(self, tensorboard, step):
        if step - self.last_tensorboard_step > 1000:
            self.last_tensorboard_step = step
            tensorboard.add_scalar("reward_scaling/reward_mean", self.mean_reward, step)
            tensorboard.add_scalar("reward_scaling/reward_std", self.std_reward, step)
            tensorboard.add_scalar(
                "reward_scaling/reward_lambda", self.lambda_reward, step
            )
            tensorboard.add_scalar(
                "reward_scaling/reward_to_value_factor",
                self.reward_to_value_factor,
                step,
            )
            tensorboard.add_scalar(
                "reward_scaling/episode_length_avg", self.avg_episode_length, step
            )
            tensorboard.add_scalar("reward_scaling/episode_n_dones", self.n_dones, step)
            tensorboard.add_scalar("reward_scaling/discount_rate", self.gamma, step)
            tensorboard.add_scalar(
                "reward_scaling/discount_rate_avg",
                self.reward_to_value_factor / self.avg_episode_length,
                step,
            )
