class AnnealedDiscountMixin:
    def __init__(self, *args, **kwargs):
        self.max_gamma = kwargs.pop("max_gamma", 0.99)
        self.start_gamma = kwargs.pop("start_gamma", 0.99)
        self.gamma_annealing_rate = kwargs.pop("gamma_annealing_rate", 0.0001)
        super().__init__(*args, **kwargs)

    def train_with_transition(self, *args, **kwargs):
        super().train_with_transition(*args, **kwargs)
        self.gamma += (self.max_gamma - self.gamma) * self.gamma_annealing_rate
