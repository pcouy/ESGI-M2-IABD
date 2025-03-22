import numpy as np
import matplotlib.pyplot as plt
from .base import DiscreteQFunction
import torch
from torch import nn
from einops import rearrange
from .neural_nets.convolutional import ConvolutionalNN
from ..agents.target_value import TargetValueAgent


class ConvolutionalQFunction(DiscreteQFunction):
    """
    Implémente une approximation de la fonction de valeur basée sur un réseau de neurones
    convolutionnel en PyTorch (voir `code_tp.value_functions.neural_nets` pour plus
    de détails)
    """

    def to_tensor(self, x, dtype=torch.float32):
        """Convert input to tensor and move to correct device"""
        if isinstance(x, torch.Tensor):
            if x.device == self.device:
                return x.clone()
            return x.clone().detach().to(self.device)
        return torch.tensor(x, dtype=dtype, device=self.device)

    def __init__(
        self,
        env,
        nn_args,
        *args,
        nn_class=ConvolutionalNN,
        gradient_clipping=None,
        use_prev_action=False,
        prev_action_embedding_dim=16,  # Default embedding dimension when use_prev_action is True
        loss_fn=nn.functional.mse_loss,
        hist_log_interval=5000,
        test_noise=0,
        device=None,
        compile_nn=True,
        **kwargs,
    ):
        if use_prev_action:
            nn_args = nn_args.copy()
            nn_args["embedding_dim"] = prev_action_embedding_dim
            nn_args["embedding_size"] = env.action_space.n

        super().__init__(env, *args, **kwargs)
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_shape = env.observation_space.shape
        self.compile_nn = compile_nn
        self.nn = nn_class(
            img_shape=self.img_shape,
            n_actions=env.action_space.n,
            device=self.device,
            **nn_args,
        )
        if self.compile_nn:
            self.nn = torch.compile(self.nn)
        self.nn.share_memory()
        print(self.nn)
        self.optim = torch.optim.Adam(
            self.nn.parameters(),
            lr=torch.tensor(self.lr),
            eps=1.5e-4,
        )
        self.loss_fn = loss_fn
        self.scaler = torch.amp.GradScaler()

        self.has_time_dim = len(env.observation_space.shape) == 4
        self.use_prev_action = use_prev_action

        if gradient_clipping is not None:
            nn.utils.clip_grad_norm_(self.nn.parameters(), gradient_clipping)

        self.backward_warmup()
        for param_name, param in self.nn.named_parameters():
            if param.grad is not None:
                torch._dynamo.decorators.mark_static_address(param.grad)
            else:
                print(
                    f'Skipped `mark_static_address` for param "{param_name}" because grad is None'
                )

        self._test_noise = test_noise

        self.stats.update({"nn_loss": {"x_label": "step", "data": []}})
        self.init_args = locals()
        print(self.device)
        self.last_tensorboard_log = 0
        self.training_steps = 0
        self.hist_log_interval = hist_log_interval

    def backward_warmup(self):
        dummy_input = torch.randn((1, *self.img_shape))
        if self.use_prev_action:
            dummy_action = torch.tensor([0])
        else:
            dummy_action = None
        output = self.nn(dummy_input, dummy_action).mean()
        output.backward()
        self.optim.zero_grad(set_to_none=False)

    def add_batch_dim(self, tensor):
        if not self.has_time_dim:
            return rearrange(tensor, "a b c -> 1 a b c")
        else:
            return rearrange(tensor, "a b c d -> 1 a b c d")

    def from_state(self, state, prev_action=None):
        return self.from_state_batch(
            self.add_batch_dim(state), None if prev_action is None else [prev_action]
        )[0]

    def from_state_batch(self, states, prev_actions=None):
        states = self.to_tensor(states)

        if self.use_prev_action and prev_actions is not None:
            # Ensure prev_actions are long (int64) for embedding layer
            prev_actions = self.to_tensor(prev_actions, dtype=torch.long)
        else:
            prev_actions = None

        self.last_result = self.nn(states, prev_actions)
        return self.last_result

    def __call__(self, state, action, prev_action=None):
        return self.call_batch(
            [state], [action], None if prev_action is None else [prev_action]
        )[0]

    def call_batch(self, states, actions, prev_actions=None):
        states = self.to_tensor(states)
        # Use long for action indices
        actions = self.to_tensor(actions, dtype=torch.long)

        if prev_actions is not None:
            # Ensure prev_actions are long for embedding layer
            prev_actions = self.to_tensor(prev_actions, dtype=torch.long)

        values = self.nn(states, prev_actions)

        self.log_call_batch(values)
        return values.gather(-1, actions.reshape((len(actions), 1)))

    @torch.compiler.disable(recursive=True)
    def log_call_batch(self, values):
        if self.agent is not None:
            entropies = {}
            for i in range(values.shape[1]):
                self.agent.log_data(
                    f"action/{self.agent.action_label_mapper(i)}",
                    values[:, i],
                    test=False,
                    log_type="histogram",
                    accumulate=self.hist_log_interval,
                )

                # Compute standard deviation for each action across the batch
                std_dev = torch.std(values[:, i] - values.mean(dim=-1)).item()
                entropies[self.agent.action_label_mapper(i)] = std_dev

            if (
                self.agent.training_steps - self.last_tensorboard_log
                >= self.hist_log_interval
            ):
                # Rename the metric name to reflect std dev instead of entropy
                self.agent.tensorboard.add_scalars(
                    "action_std_dev",
                    entropies,  # We're reusing the dictionary but storing std dev values
                    self.agent.training_steps,
                )

    def update(self, state, action, target_value, prev_action=None, is_weight=None):
        return self.update_batch(
            [state],
            [action],
            [target_value],
            None if prev_action is None else [prev_action],
        )[0]

    @torch.compile
    def update_batch(
        self, states, actions, target_values, prev_actions=None, is_weights=None
    ):
        self.training_steps += 1
        if is_weights is None:
            is_weights = torch.ones(
                (states.shape[0],), dtype=torch.float32, device=self.device
            )
        else:
            is_weights = self.to_tensor(is_weights)
        target_values = self.to_tensor(target_values).detach()

        self.optim.zero_grad(set_to_none=False)

        pred_values = self.call_batch(states, actions, prev_actions)
        pred_error_indiv = torch.abs(pred_values[:, 0] - target_values)
        pred_error = (
            is_weights
            * self.loss_fn(pred_values[:, 0], target_values, reduction="none")
        ).mean()

        self.scaler.scale(pred_error).backward()
        self.scaler.step(self.optim)
        self.scaler.update()

        self.log_update_step(pred_error)
        return pred_error_indiv.detach().cpu().numpy()

    @torch.compiler.disable(recursive=True)
    def log_update_step(self, pred_error):
        if self.training_steps > 1000:
            self.agent.log_data("nn_loss", pred_error, test=False)
        if (
            self.agent.training_steps - self.last_tensorboard_log
            >= self.hist_log_interval
        ):
            self.nn.log_tensorboard(
                self.agent.tensorboard,
                self.agent.training_steps,
                action_mapper=self.agent.action_label_mapper,
            )
            self.last_tensorboard_log = self.agent.training_steps

    @property
    def _default_noise_strength(self):
        if getattr(self.agent, "test", False):
            return self._test_noise
        return 1

    @torch.compiler.disable(recursive=True)
    def reset_noise(self, strength=None):
        strength = self._default_noise_strength if strength is None else strength
        if self.agent is not None:
            self.agent.log_data("noise_strength", strength, test=False)
        if callable(getattr(self.nn, "reset_noise", None)):
            self.nn.reset_noise(strength)
            if isinstance(self.agent, TargetValueAgent):
                self.agent.target_value_function.reset_noise(strength)

    def best_action_value_from_state(self, state, prev_action=None):
        maxa, maxv = self.best_action_value_from_state_batch(
            torch.tensor([state]), None if prev_action is None else [prev_action]
        )
        return maxa[0], maxv[0]

    def best_action_value_from_state_batch(self, states, prev_actions=None):
        values = self.from_state_batch(states, prev_actions)
        m = values.max(axis=1)
        return m.indices, m.values

    def export_f(self):
        return self.nn.state_dict()

    def import_f(self, d):
        self.nn.load_state_dict(d)

    def mix_with(self, other, tau=0.001):
        for self_param, other_param in zip(self.nn.parameters(), other.nn.parameters()):
            self_param.data.copy_(tau * other_param.data + (1 - tau) * self_param.data)
