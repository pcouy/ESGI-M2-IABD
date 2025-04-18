import numpy as np
from .greedy import EGreedyPolicy
import torch
import traceback


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
        final_target_entropy=2 / 3,
        running_action_probas_lr=1,
        **kwargs,
    ):
        biases = kwargs.pop("biases", None)
        self.biases_decay = kwargs.pop("biases_decay", 0.9999)
        self.final_target_entropy = final_target_entropy
        self.target_entropy_decay = target_entropy_decay
        self.entropy_lr = entropy_lr
        self.epsilon_lr = epsilon_lr
        self.min_epsilon = min_epsilon
        self.running_action_probas_lr = running_action_probas_lr
        # Override parent's epsilon parameters to prevent automatic decay
        kwargs["epsilon_decay"] = 0
        kwargs["epsilon_min"] = min_epsilon

        super().__init__(*args, **kwargs)
        self.target_entropy = target_entropy if target_entropy is not None else 1
        self.initial_target_entropy = self.target_entropy
        self.running_entropy = self.target_entropy

        self.n_actions = self.value_function.action_space.n

        self.running_action_probas = np.ones((self.n_actions,)) / self.n_actions
        self.running_greedy_action_probas = np.ones((self.n_actions,)) / self.n_actions
        self.sampling_count = 0
        self.noisynet_strength = 1
        self._save_attr("final_target_entropy")
        self._save_attr("running_action_probas_lr")
        self._save_attr("entropy_lr")
        self._save_attr("epsilon_lr")
        self._save_attr("noisynet_strength")

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
        self.accumulated_action_probas = {}

    def log_action_probas(self, key, values, step):
        if key not in self.accumulated_action_probas:
            self.accumulated_action_probas[key] = {}
        for label, value in values.items():
            if label not in self.accumulated_action_probas[key]:
                self.accumulated_action_probas[key][label] = []
            self.accumulated_action_probas[key][label].append(value)

        if len(self.accumulated_action_probas[key][next(iter(values))]) > 1000:
            self.agent.tensorboard.add_scalars(
                key,
                {k: np.mean(v) for k, v in self.accumulated_action_probas[key].items()},
                step,
            )
            self.accumulated_action_probas[key] = {}

    def update_epsilon(self):
        """Override parent's epsilon update to prevent automatic decay"""
        pass

    def update(self):
        """Update target entropy and log stats"""
        # Decay target entropy
        if not self.agent.test:
            self.target_entropy = max(
                self.final_target_entropy,
                self.target_entropy - self.target_entropy_decay,
            )
            self.agent.log_data("target_entropy", self.target_entropy)
        super().update()

    def __call__(
        self, state, prev_action=None, epsilon=None, update_epsilon=True, **kwargs
    ):
        if epsilon is None:
            epsilon = self.epsilon

        # Logging variables
        tag_suffix = "_test" if self.in_test else ""
        accumulate = not self.in_test
        step = self.agent.testing_steps if self.in_test else self.agent.training_steps

        with torch.no_grad():
            if callable(getattr(self.value_function, "reset_noise", None)):
                if self.agent is not None and self.agent.test:
                    strength = None
                else:
                    strength = self.noisynet_strength
                self.value_function.reset_noise(strength)
            values = self.value_function.from_state(state, prev_action)

        if epsilon > 0:
            try:
                # Add numerical stability by scaling values
                scaled_values = self.value_unscaler(values)
                if isinstance(scaled_values, torch.Tensor):
                    scaled_values = scaled_values.cpu().numpy()
                scaled_values += self.biases - np.max(scaled_values + self.biases)
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

                action = np.random.choice([x for x in range(len(probas))], p=probas)

                greedy_probas = np.zeros_like(probas)
                greedy_probas[probas.argmax()] = 1
            except Exception:
                print(traceback.format_exc())
                print(f"epsilon: {epsilon}, values: {values}")
                self.greedy_policy.agent = self.agent
                action = self.greedy_policy(state, prev_action)
                probas = np.zeros((self.n_actions,))
                probas[action] = 1
                greedy_probas = probas
        else:
            self.greedy_policy.agent = self.agent
            action = self.greedy_policy(state, prev_action, epsilon=epsilon)
            probas = np.zeros((self.n_actions,))
            probas[action] = 1
            greedy_probas = probas

        self.running_greedy_action_probas = (
            (1 - self.running_action_probas_lr) * self.running_greedy_action_probas
            + self.running_action_probas_lr * greedy_probas
        )
        log_greedy_probas = np.log(self.running_greedy_action_probas + 1e-10) / np.log(
            self.n_actions
        )
        greedy_entropy = -np.sum(self.running_greedy_action_probas * log_greedy_probas)

        self.running_action_probas = (
            1 - self.running_action_probas_lr
        ) * self.running_action_probas + self.running_action_probas_lr * probas

        self.log_action_probas(
            f"running_action_probas{tag_suffix}",
            {
                self.agent.action_label_mapper(i): self.running_action_probas[i]
                for i in range(len(probas))
            },
            step,
        )

        log_probas = np.log(self.running_action_probas + 1e-10) / np.log(self.n_actions)

        entropy = -np.sum(self.running_action_probas * log_probas)

        # Bound entropy to prevent extreme values
        entropy = np.clip(entropy, 0.0, 1.0)

        if not self.in_test and update_epsilon:
            # Update running average of entropy with bounds checking
            if not np.isnan(entropy) and not np.isinf(entropy):
                self.running_entropy = (
                    1 - self.entropy_lr
                ) * self.running_entropy + self.entropy_lr * entropy

            # Adjust epsilon based on entropy difference
            if self.running_entropy < self.target_entropy:
                self.epsilon = min(
                    1.0, self.epsilon + self.epsilon_lr
                )  # Increase exploration
            elif self.running_entropy > self.target_entropy:
                self.epsilon = max(
                    self.min_epsilon, self.epsilon - self.epsilon_lr
                )  # Decrease exploration

        if not self.in_test:
            self.agent.log_data("entropy", entropy)
            self.agent.log_data("entropy_greedy", greedy_entropy)
            self.agent.log_data("running_entropy", self.running_entropy)

        self.agent.log_data(
            f"predicted_value{tag_suffix}",
            self.value_unscaler(values[action]),
            accumulate=accumulate,
        )
        self.agent.log_data(
            f"picked_proba{tag_suffix}", probas[action], accumulate=accumulate
        )
        self.agent.log_data(
            f"picked_surprise{tag_suffix}",
            -np.log(probas[action]) / np.log(self.n_actions),
            accumulate=accumulate,
        )
        self.log_action_probas(
            f"action_probas{tag_suffix}",
            {self.agent.action_label_mapper(i): probas[i] for i in range(len(probas))},
            step,
        )
        self.stats.update(self.greedy_policy.stats)
        self.sampling_count += 1
        if self.sampling_count % 1000 == 0:
            self._load_attr("final_target_entropy")
            self._load_attr("running_action_probas_lr")
            self._load_attr("entropy_lr")
            self._load_attr("epsilon_lr")
            self._load_attr("noisynet_strength")
        return action

    def _save_attr(self, attr_name):
        with open(f"{attr_name}.txt", "w") as f:
            f.write(str(getattr(self, attr_name)))

    def _load_attr(self, attr_name):
        try:
            with open(f"{attr_name}.txt", "r") as f:
                setattr(self, attr_name, float(f.read()))
        except FileNotFoundError:
            pass
        except ValueError:
            with open(f"{attr_name}.txt", "r") as f:
                print(f"Error loading {attr_name} from file :{f.read()=}")

    def call_batch(self, state_batch, epsilon=None):
        raise NotImplementedError
