from .neural import ConvolutionalQFunction
import torch
import torch.nn.functional as F


class DistributionalQFunction(ConvolutionalQFunction):
    def __init__(self, env, *args, v_min=0.0, v_max=10.0, atom_size=51, **kwargs):
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)
        self.last_dist = None
        super().__init__(env, *args, **kwargs)
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(
            self.device
        )

    def make_nn(
        self,
        nn_class,
        img_shape,
        n_actions,
        device,
        **nn_args,
    ):
        return nn_class(
            img_shape=img_shape,
            n_actions=n_actions,
            n_atoms=self.atom_size,
            device=device,
            **nn_args,
        )

    def dist(self, states, prev_actions=None):
        states = self.to_tensor(states)

        if self.use_prev_action and prev_actions is not None:
            # Ensure prev_actions are long (int64) for embedding layer
            prev_actions = self.to_tensor(prev_actions, dtype=torch.long)
        else:
            prev_actions = None

        nn_out = super().from_state_batch(states, prev_actions)
        q_atoms = nn_out.view(-1, self.n_actions, self.atom_size)
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)
        self.last_dist = dist
        return dist

    def from_state_batch(self, states, prev_actions=None, log=False):
        dist = self.dist(states, prev_actions)
        if log:
            pass
            # self.log_call_batch(dist)
        return self.dist_to_value(dist)

    def dist_to_value(self, dist):
        return torch.sum(dist * self.support, dim=2)

    def best_action_dist_from_state(self, state, prev_action=None):
        maxa, maxv = self.best_action_dist_from_state_batch(
            torch.tensor([state]), None if prev_action is None else [prev_action]
        )
        return maxa[0], maxv[0]

    def best_action_dist_from_state_batch(self, states, prev_actions=None):
        dists = self.dist(states, prev_actions)
        values = self.dist_to_value(dists)
        actions = values.argmax(1)
        return actions, dists[range(actions.shape[0]), actions]

    def get_pred_error(
        self, states, actions, target_values, prev_actions=None, is_weights=None
    ):
        if is_weights is None:
            is_weights = torch.ones(
                (states.shape[0],), dtype=torch.float32, device=self.device
            )
        else:
            is_weights = self.to_tensor(is_weights)
        target_dists = self.to_tensor(target_values).detach()

        pred_dists = self.dist(states, prev_actions)
        pred_dists = pred_dists[range(actions.shape[0]), actions]
        log_probs = torch.log(pred_dists)

        pred_error_indiv = -(target_dists * log_probs).sum(1)
        pred_error = (is_weights * pred_error_indiv).mean()
        return pred_error, pred_error_indiv
