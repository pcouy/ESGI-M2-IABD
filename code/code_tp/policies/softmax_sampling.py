import numpy as np
from .greedy import EGreedyPolicy
import torch


class SoftmaxSamplingPolicy(EGreedyPolicy):
    """
    Pareil que la politique epsilon-greedy, mais l'action est tirée avec la probabilité :

    p(a) = softmax(5*(1-epsilon)*Q(s,a))

    L'epsilon et l'entropie cible sont ajustés automatiquement au cours de l'apprentissage.
    Plus l'entropie est élevée, plus la politique est exploratoire.
    """

    def __init__(
        self,
        *args,
        target_entropy=None,
        entropy_lr=0.01,
        epsilon_lr=0.01,
        min_epsilon=0.0015,
        target_entropy_decay=0.9999,
        final_target_entropy=0.1,
        value_scaling=1,
        **kwargs,
    ):
        biases = kwargs.pop("biases", None)
        self.biases_decay = kwargs.pop("biases_decay", 0.9999)
        self.final_target_entropy = final_target_entropy
        self.target_entropy_decay = target_entropy_decay
        self.entropy_lr = entropy_lr
        self.epsilon_lr = epsilon_lr
        self.min_epsilon = min_epsilon
        self.value_scaling = value_scaling
        # Override parent's epsilon parameters to prevent automatic decay
        kwargs["epsilon_decay"] = 0
        kwargs["epsilon_min"] = min_epsilon

        super().__init__(*args, **kwargs)
        self.target_entropy = (
            target_entropy
            if target_entropy is not None
            else self.value_function.action_space.n
        )
        self.initial_target_entropy = self.target_entropy
        self.running_entropy = self.target_entropy

        if biases is None:
            self.biases = np.array(
                [0.0 for action in range(self.value_function.action_space.n)]
            )
        else:
            self.biases = np.array(biases)

        self.stats.update(
            {
                "picked_proba": {"x_label": "step", "data": []},
                "entropy": {"x_label": "step", "data": []},
                "running_entropy": {"x_label": "step", "data": []},
                "target_entropy": {"x_label": "step", "data": []},
            }
        )

    def update_epsilon(self):
        """Override parent's epsilon update to prevent automatic decay"""
        pass

    def update(self):
        """Update target entropy and log stats"""
        # Decay target entropy
        if not self.agent.test:
            self.target_entropy = max(
                np.exp(self.final_target_entropy),
                self.target_entropy - self.target_entropy_decay,
            )
            self.agent.log_data("target_exp_entropy", self.target_entropy)
        super().update()

    def __call__(self, state, prev_action=None, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon

        with torch.no_grad():
            values = self.value_function.from_state(state, prev_action)
            if type(values) is torch.Tensor:
                values = values.clone().detach().cpu().numpy()

        if epsilon >= self.min_epsilon:
            try:
                # Add numerical stability by scaling values
                scaled_values = (
                    values + self.biases - np.max(values + self.biases)
                ) * self.value_scaling
                aux = np.exp((1 / epsilon - 1) * scaled_values)
                self.biases = self.biases * self.biases_decay

                # Handle NaN and Inf values
                if np.any(np.isnan(aux)) or np.any(np.isinf(aux)):
                    aux = np.where(np.isnan(aux) | np.isinf(aux), 1.0, aux)
                    aux = np.where(
                        aux < 1e-10, 1e-10, aux
                    )  # Prevent zero probabilities

                # Normalize probabilities
                probas = aux / np.sum(aux)

                # Ensure probabilities are valid
                probas = np.clip(probas, 1e-10, 1.0)
                probas = probas / np.sum(probas)  # Renormalize after clipping

                # Calculate entropy with numerical stability
                log_probas = np.log(
                    probas + 1e-10
                )  # Add small constant to prevent log(0)
                entropy = -np.sum(probas * log_probas)

                # Bound entropy to prevent extreme values
                entropy = np.clip(entropy, 0.0, np.log(len(probas)))

                if not self.in_test:
                    # Update running average of entropy with bounds checking
                    if not np.isnan(entropy) and not np.isinf(entropy):
                        self.running_entropy = (
                            1 - self.entropy_lr
                        ) * self.running_entropy + self.entropy_lr * entropy

                    # Adjust epsilon based on entropy difference
                    if self.running_entropy < np.log(self.target_entropy):
                        self.epsilon = min(
                            1.0, self.epsilon + self.epsilon_lr
                        )  # Increase exploration
                    elif self.running_entropy > np.log(self.target_entropy):
                        self.epsilon = max(
                            self.min_epsilon, self.epsilon - self.epsilon_lr
                        )  # Decrease exploration

                action = np.random.choice([x for x in range(len(probas))], p=probas)
                if self.in_test:
                    self.agent.log_data(
                        "picked_proba_test", probas[action], accumulate=False
                    )
                else:
                    self.agent.log_data("picked_proba", probas[action])
                    self.agent.tensorboard.add_scalars(
                        "action_probas",
                        {
                            self.agent.action_label_mapper(i): probas[i]
                            for i in range(len(probas))
                        },
                        self.agent.training_steps,
                    )
                    self.agent.log_data("entropy", entropy)
                    self.agent.log_data("running_entropy", self.running_entropy)
            except Exception as e:
                print(f"Error in softmax sampling: {e}")
                print(f"epsilon: {epsilon}, values: {values}")
                self.greedy_policy.agent = self.agent
                action = self.greedy_policy(state, prev_action)
                if not self.in_test:
                    self.agent.log_data("picked_proba", 1)
                    self.agent.log_data("entropy", 0)
        else:
            self.greedy_policy.agent = self.agent
            action = self.greedy_policy(state, prev_action, epsilon=epsilon)
            if not self.in_test:
                self.agent.log_data("picked_proba", 1)
                self.agent.log_data("entropy", 0)

        if self.in_test:
            self.agent.log_data(
                "predicted_value_test", values[action], accumulate=False
            )
        else:
            self.agent.log_data("predicted_value", values[action])
        self.stats.update(self.greedy_policy.stats)

        return action

    def call_batch(self, state_batch, epsilon=None):
        raise NotImplementedError
