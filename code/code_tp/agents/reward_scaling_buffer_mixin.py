class RewardScalingBufferMixin:
    """
    Mixin to scale rewards based on a moving average of discounted returns.

    Example :
    class Buffer(RewardScalingBufferMixin, ReplayBuffer):
        pass
    """
    def __init__(self, *args, **kwargs):
        self.gamma = kwargs.pop("gamma", 0.99)  # discount factor
        super().__init__(*args, **kwargs)
        self.n_dones = 0
        self.avg_episode_length = 1000
        self.average_reward = 0
        

    def store(self, *transition):
        result = super().store(*transition)
        done = transition[4] if len(transition) > 4 else False
        if done:
            self.n_dones += 1
            self.avg_episode_length = self.n_inserted / self.n_dones
        self.average_reward = (self.n_inserted * self.average_reward + transition[3]) / (self.n_inserted + 1)
        return result

    @property
    def reward_scaling_factor(self):
        return self.average_reward * (1-self.gamma**self.avg_episode_length)/(1-self.gamma)
    
    def sample(self, *args, **kwargs):
        samples = super().sample(*args, **kwargs)
        # Convert samples to list if it's a tuple
        if isinstance(samples, tuple):
            samples = list(samples)
        
        if len(samples) > 3:
            samples[3] = samples[3] / self.reward_scaling_factor
        else:
            # Convert inner tuple to list if needed
            if isinstance(samples[0], tuple):
                samples[0] = list(samples[0])
            samples[0][3] = samples[0][3] / self.reward_scaling_factor
        return samples
    
    def log_tensorboard(self, tensorboard, step):
        tensorboard.add_scalar('reward_scaling_factor', self.reward_scaling_factor, step)
