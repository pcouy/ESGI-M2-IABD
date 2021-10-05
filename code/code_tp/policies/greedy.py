import numpy as np

class RandomPolicy:
    def __init__(self, value_function):
        self.value_function = value_function
        self.stats = {}

    def __call__(self, state):
        return self.value_function.action_space.sample()

    def test(self, state):
        return self(state)

    def update(self):
        pass

class GreedyQPolicy(RandomPolicy):
    def __init__(self, value_function):
        super().__init__(value_function)
        self.stats.update({
            'predicted_value': {
                'x_label': 'step',
                'data': []
            }
        })

    def __call__(self, state):
        values = self.value_function.from_state(state)
        if type(values) is not dict:
            values = {k:v for k,v in enumerate(values)}
        value, _ = max((v,a) for a,v in values.items())

        actions = [k for k,v in values.items() if v==value]

        self.stats['predicted_value']['data'].append(value)
        return np.random.choice(actions)

class EGreedyPolicy(RandomPolicy):
    def __init__(self, value_function, greedy_policy_class=GreedyQPolicy, epsilon=0.05, epsilon_decay=0, epsilon_min=0.05, epsilon_test=0):
        self.greedy_policy = greedy_policy_class(value_function)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_test = epsilon_test
        super().__init__(value_function)
        self.stats.update({
            "epsilon": {
                "x_label": "step",
                "data": []
            }
        })

    def __call__(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.uniform(0,1) > epsilon:
            action = self.greedy_policy(state)
            self.stats.update(self.greedy_policy.stats)
        else:
            action = super().__call__(state)

        return action

    def test(self, state):
        return self(state, epsilon=self.epsilon_test)

    def update(self):
        self.epsilon = max(self.epsilon*(1-self.epsilon_decay), self.epsilon_min)
        self.stats["epsilon"]["data"].append(self.epsilon)
        super().update()
