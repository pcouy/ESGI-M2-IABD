import numpy as np


class DecisionRateAgentMixin:
    def __init__(self, *args, **kwargs):
        self.decision_rate = kwargs.pop("decision_rate", 1)
        self.target_decision_rate = kwargs.pop("target_decision_rate", 1)
        self.decision_rate_lr = kwargs.pop("decision_rate_lr", 1e-6)
        self.last_selected_action = None
        self.action_held_for = 0
        super().__init__(*args, **kwargs)

    def select_action(self, *args, **kwargs):
        if self.test:
            return super().select_action(*args, **kwargs)

        self.decision_rate += (
            self.target_decision_rate - self.decision_rate
        ) * self.decision_rate_lr
        self.log_data("decision_rate", self.decision_rate)

        if (
            self.last_selected_action is None
            or np.random.uniform() < self.decision_rate
        ):
            self.last_selected_action = super().select_action(*args, **kwargs)
            self.action_held_for = 0
        else:
            self.action_held_for += 1

        self.log_data("decision_rate/action_held_for", self.action_held_for)

        return self.last_selected_action

    def episode_end(self, *args, **kwargs):
        self.last_selected_action = None
        self.action_held_for = 0
        return super().episode_end(*args, **kwargs)
