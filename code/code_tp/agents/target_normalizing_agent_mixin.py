from gymnasium.wrappers.utils import RunningMeanStd
import torch
import numpy as np

class SlidingMeanStd(RunningMeanStd):
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, *args, moments_lr=0.001, **kwargs):
        super().__init__(*args, **kwargs)
        self.moments_lr = moments_lr

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        self.mean = self.mean + self.moments_lr * (new_mean - self.mean)
        self.var = self.var + self.moments_lr * (new_var - self.var)
        self.count = self.count + self.moments_lr * (new_count - self.count)

class TargetNormalizingAgentMixin:
    def __init__(self, *args, initial_returns_mean=0.0, initial_returns_var=1.0, initial_returns_count=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.running_target_value = RunningMeanStd(epsilon=initial_returns_count)
        self.running_target_value.mean+= initial_returns_mean
        self.running_target_value.var*= initial_returns_var

    @torch.compiler.disable(recursive=True)
    def update_running_target_value(self, value):
        if isinstance(value, torch.Tensor):
            _value = value.detach().cpu().numpy()
        elif isinstance(value, float):
            _value = np.array([value])
        elif isinstance(value, list):
            _value = np.array(value)
        else:
            _value = value
        self.running_target_value.update(_value)
        self.log_data("running_target_value/mean", self.running_target_value.mean, test=False)
        self.log_data("running_target_value/var", self.running_target_value.var, test=False)

    def target_value_from_state(self, *args, **kwargs):
        target = super().target_value_from_state(*args, **kwargs)
        self.update_running_target_value(target)
        return target
    
    def target_value_from_state_batch(self, *args, **kwargs):
        targets = super().target_value_from_state_batch(*args, **kwargs)
        self.update_running_target_value(targets)
        return targets
    
    def scale_target(self, target):
        return (target - self.running_target_value.mean) / (np.sqrt(self.running_target_value.var) + 1e-4)
    
    def unscale_target(self, target):
        return target * np.sqrt(self.running_target_value.var) + self.running_target_value.mean