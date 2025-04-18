import gymnasium as gym
import numpy as np
import os
import gc
import json
import multiprocessing as mp
from einops import rearrange

# Set matplotlib backend before importing pyplot
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

import torch

from torch.utils.tensorboard import SummaryWriter
import subprocess


class AutoStartSummaryWriter(SummaryWriter):
    def serve(self, tb_args):
        print(f"{self.get_logdir()=}")
        self.tb_serve = subprocess.Popen(
            ["tensorboard", "--logdir", self.get_logdir(), *tb_args]
        )

    def close(self, *args, **kwargs):
        super().close(*args, **kwargs)
        try:
            self.tb_serve.kill()
        except:
            print("Error stopping TensorBoard")


class Agent:
    """
    Classe de base pour tous les agents.
    """

    def __init__(
        self,
        env: gym.Env,
        use_prev_action=False,
        save_dir="experiment",
        infos={},
        tensorboard_layout={},
        initial_prev_action=None,
        action_label_mapper=str,
        accumulate_stats=100,
        random_test_steps=0,
        random_train_steps=0,
        test_patience=0,
        rotate_tensorboard_every=1000,
    ):
        """
        * `env`: Environnement gym dans lequel l'agent va évoluer
        * `use_prev_action`: Si True, l'action précédente est utilisée comme entrée additionnelle
        * `save_dir`: Dossier dans lequel les résultats de l'expérience seront sauvegardés
        * `infos`: Dictionnaire qui sera sauvegardé au format JSON dans le dossier de
                    l'expérience, utile pour y écrire les paramètres.
        """
        self.training_steps = 0
        self.training_episodes = 0
        self.testing_steps = 0
        self.random_test_steps = random_test_steps
        self.random_train_steps = random_train_steps
        self.test_patience = test_patience
        self.env = env
        self.agent = self
        self.test = False
        self.should_stop = False
        self.use_prev_action = use_prev_action
        self.action_label_mapper = action_label_mapper
        self.initial_prev_action = initial_prev_action
        self.save_dir = save_dir
        self.stats = {
            "scores": {"x_label": "Episode", "data": []},
            "test_scores": {"data": []},
        }
        self.cumulative_stats = {}
        self.accumulate_stats = accumulate_stats
        self.last_stats_update = {}
        self.rotate_tensorboard_every = rotate_tensorboard_every
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "infos.json"), "w") as f:
            print(infos)
            json.dump(
                infos,
                f,
                indent=2,
                default=lambda x: x.__name__ if type(x) is type else str(x),
            )
        self.tensorboard = AutoStartSummaryWriter(
            os.path.join(self.save_dir, "tensorboard")
        )
        self.tensorboard.serve(
            [
                "--bind_all",
                "--port",
                "6006",
                "--samples_per_plugin",
                "scalars=50000,images=500",
            ]
        )
        self.tensorboard.add_custom_scalars(tensorboard_layout)
        self.episode_logger = AutoStartSummaryWriter(
            os.path.join("episodes", self.save_dir, "0-1000"),
        )
        self.episode_logger.serve(
            [
                "--bind_all",
                "--port",
                "6007",
                "--samples_per_plugin",
                "scalars=5000,images=5000",
            ]
        )
        self.episode_logger_range_start = 0

    @torch.compiler.disable(recursive=True)
    def log_data(self, key, value, accumulate=True, test=None, log_type="scalar"):
        if test is None:
            test = self.test
        if test:
            step = self.testing_steps
        else:
            step = self.training_steps
        if key not in self.stats:
            self.stats[key] = {"data": []}

        if isinstance(value, torch.Tensor):
            value = value.clone().detach().cpu()
            if log_type == "scalar":
                value = value.item()
        if not accumulate:
            if log_type == "scalar":
                self.stats[key]["data"].append(value)
                self.tensorboard.add_scalar(key, value, step)
            elif log_type == "histogram":
                self.tensorboard.add_histogram(key, value, step)
            else:
                raise ValueError(f"Unknown log type '{log_type}'")
            return

        if key not in self.last_stats_update:
            self.last_stats_update[key] = 0
        if key not in self.cumulative_stats:
            self.cumulative_stats[key] = {"data": np.array([])}
        self.cumulative_stats[key]["data"] = np.append(
            self.cumulative_stats[key]["data"], value
        )

        if isinstance(accumulate, bool):
            accumulate_stats_every = self.accumulate_stats
        else:
            accumulate_stats_every = accumulate

        if self.training_steps - self.last_stats_update[key] > accumulate_stats_every:
            self.last_stats_update[key] = self.training_steps
            if log_type == "scalar":
                self.stats[key]["data"].append(
                    self.cumulative_stats[key]["data"].mean()
                )
                self.tensorboard.add_scalar(
                    key, self.cumulative_stats[key]["data"].mean(), step
                )
            elif log_type == "histogram":
                self.tensorboard.add_histogram(
                    key, self.cumulative_stats[key]["data"], step
                )
            else:
                raise ValueError(f"Unknown log type '{log_type}'")
            self.cumulative_stats[key]["data"] = np.array([])

    def select_action(
        self, state: np.ndarray, prev_action=None, **kwargs
    ) -> np.ndarray:
        """
        Prend un état en entrée, renvoie une action
        * `prev_action`: Action précédente (optionnel)
        """
        raise NotImplementedError

    def train_with_transition(
        self, state, action, next_state, reward, done, infos, prev_action=None
    ):
        """
        Met à jour l'agent suite à une transition. Les noms des paramètres sont explicites.
        Le paramètre `infos`, rarement utilisé, fait partie des réponses standardisées des
        environnements gym, et est donc inclus
        """
        raise NotImplementedError

    def step(self, action, env=None):
        """
        Effectue une action. Dans cette classe de base, est utilisé uniquement pour appeler
        la méthode `step` de l'environnement.
        """
        if env is None:
            env = self.env
        return env.step(action)

    def log_step(self, episode_name, step_num, transition):
        self.episode_logger.add_scalar(
            f"{episode_name}/reward", transition[3], step_num
        )
        self.episode_reward_history.append(transition[3])
        # self.tensorboard.add_text(f"{episode_name}/action", str(transition[1]), step_num)
        if len(transition[0].shape) == 4:
            last_state = transition[0][:, :, -1, :]
        else:
            last_state = transition[0]
        img = rearrange(last_state, "w h c -> c w h")
        self.episode_logger.add_image(f"{episode_name}/states", img, step_num)

    def log_episode_end(self, *args, **kwargs):
        RANGE_LEN = self.rotate_tensorboard_every
        if self.training_episodes < self.episode_logger_range_start + RANGE_LEN:
            return
        self.episode_logger.close()
        self.episode_logger_range_start += RANGE_LEN
        self.episode_logger = AutoStartSummaryWriter(
            os.path.join(
                "episodes",
                self.save_dir,
                f"{self.episode_logger_range_start}-{self.episode_logger_range_start+RANGE_LEN}",
            ),
        )
        self.episode_logger.serve(
            [
                "--bind_all",
                "--port",
                "6007",
                "--samples_per_plugin",
                "scalars=5000,images=5000",
            ]
        )

    def run_episode(self, test=False):
        """
        Jouer un épisode dans l'enfironnement. Le paramètre `test` détermine si les transitions
        qui ont lieu durant l'épisode doivent mettre à jour l'agent.
        """
        self.test = test
        if not test:
            env = self.env
        else:
            os.makedirs(os.path.join(self.save_dir, "videos"), exist_ok=True)
            env = gym.wrappers.RecordVideo(
                self.env,
                os.path.join(
                    self.save_dir,
                    "videos",
                ),
                name_prefix="{:05d}".format(self.training_episodes),
                episode_trigger=lambda _: True,
                fps=self.env.metadata["render_fps"],
            )
            env.step_id = self.training_steps

        state, _ = env.reset()
        done = False
        # Initialize prev_action with a random action if the feature is enabled
        prev_action = (
            self.initial_prev_action
            if self.initial_prev_action is not None
            else (self.env.action_space.sample() if self.use_prev_action else None)
        )
        score = 0

        frames = []
        step_num = 1
        last_reward_step = 1
        train_test_transitions = False
        if test:
            self.episode_reward_history = []
            self.episode_value_history = []
        while not done:
            # frames.append(env.render('rgb_array'))
            if (test and step_num <= self.random_test_steps) or (
                not test and step_num <= self.random_train_steps
            ):
                action = self.env.action_space.sample()
            elif (
                test
                and self.test_patience > 0
                and step_num - last_reward_step > self.test_patience
            ):
                policy_epsilon = getattr(self.policy, "epsilon_test", None)
                if policy_epsilon is None:
                    action = self.env.action_space.sample()
                else:
                    new_epsilon = (
                        policy_epsilon
                        + 0.1
                        * (step_num - last_reward_step - self.test_patience)
                        / self.test_patience
                    )
                    new_epsilon = max(0, min(1, new_epsilon))
                    self.log_data(
                        "impatience_epsilon_test", new_epsilon, accumulate=False
                    )
                    with torch.no_grad():
                        action = self.select_action(
                            state,
                            prev_action,
                            epsilon=new_epsilon,
                            update_epsilon=False,
                        )
            else:
                with torch.no_grad():
                    action = self.select_action(state, prev_action)
            next_state, reward, terminated, truncated, infos = self.step(
                action, env=env
            )
            done = terminated or truncated
            transition = (state, action, next_state, reward, done, infos, prev_action)
            if not test:
                self.train_with_transition(*transition)
                self.training_steps += 1
            else:
                self.testing_steps += 1
                self.log_step(
                    f"test_episodes_{self.training_episodes}", step_num, transition
                )
            step_num += 1
            score += reward
            if reward > 0:
                last_reward_step = step_num
            state = next_state
            prev_action = (
                action if self.use_prev_action else None
            )  # Update prev_action if feature is enabled

        if not test:
            self.training_episodes += 1
        else:
            self.log_episode_end(f"test_episodes_{self.training_episodes}")

        self.episode_end(score)

        if test:
            video_tensor = rearrange(
                np.array(env.recorded_frames), "t w h c -> 1 t c w h"
            )
            self.tensorboard.add_video(
                "test_run",
                video_tensor,
                self.training_steps,
                fps=self.env.metadata.get("render_fps", 30),
            )
            env.stop_recording()
        self.test = False

        return score

    def run_n_episodes(self, n, test=False):
        for i in range(n):
            self.run_episode(test=test)

    def train(
        self,
        n_episodes=1000,
        test_interval=50,
        train_callbacks=[],
        test_callbacks=[],
        test_interval_type="episode",
    ):
        """
        Boucle d'entrainement.

        * `n_episodes`: nombre d'épisodes durant lesquels entrainer l'agent
        * `test_interval`: lance un épisode de test, et sauvegarde la vidéo, tous les `test_interval`
                            épisodes
        * `train_callbacks`: Liste de fonctions à appeler à la fin de chaque épisode d'entrainement
        * `test_callbacks`: Liste de fonctions à appeler à la fin de chaque épisode de test

        Les callbacks prennent comme unique argument le dossier de sauvegarde
        """
        if train_callbacks is None:
            train_callbacks = []
        if test_callbacks is None:
            test_callbacks = []
        if self.plot_stats not in test_callbacks:
            test_callbacks += [self.plot_stats]
        self.should_stop = False

        while self.training_episodes < n_episodes and not self.should_stop:
            self.run_episode(test=False)
            for cb in train_callbacks:
                cb(self.save_dir)
            if self.test_condition(
                self.training_episodes, test_interval, test_interval_type
            ):
                self.run_episode(test=True)
                for cb in test_callbacks:
                    cb(self.save_dir)
                gc.collect()

    def test_condition(self, i, test_interval, test_interval_type="episode"):
        if test_interval_type == "step":
            try:
                last_test_step = self.last_test_step
            except AttributeError:
                last_test_step = 0
            result = self.training_steps - last_test_step > test_interval
            if result:
                self.last_test_step = self.training_steps
            return result
        else:
            return (i + 1) % test_interval == 0

    def episode_end(self, score):
        """
        Méthode appelée par `run_episode` à la fin de chaque épisode, reçoit le
        score de l'épisode achevé comme argument.
        """
        print(
            "{} episode ({}) done with score = {:.2f}".format(
                "Testing" if self.test else "Training", self.training_episodes, score
            )
        )
        if not self.test:
            self.log_data("scores", score, accumulate=False)
        else:
            self.log_data("test_scores", score, accumulate=False)

    def plot_stats(self, save_dir=None):
        """
        Trace les courbes des tableaux contenus dans `self.stats`. Si `save_dir` vaut `None`,
        affiche directement les courbes, sinon les sauvegarde.
        """

        def plot_stat(name, x_label="", data=[]):
            if type(data) is torch.Tensor:
                data = data.detach().cpu().numpy()
            if len(data) and type(data[0]) is torch.Tensor:
                data = [item.detach() for item in data]
            fig, ax = plt.subplots()
            ax.plot(data)
            plt.xlabel(x_label)
            plt.ylabel(name)
            fig.tight_layout()
            if save_dir is not None:
                fig.savefig(
                    os.path.join(save_dir, "{}.png".format(name.replace("/", "_")))
                )
            else:
                plt.show()
            fig.clf()
            plt.clf()
            plt.close(fig)

        procs = []
        # "Hack" utilisant `multiprocessing` pour éviter une fuite de mémoire avec
        # matplotlib
        for stat in self.stats.keys():
            procs.append(
                mp.Process(target=plot_stat, args=(stat,), kwargs=self.stats[stat])
            )
            procs[-1].daemon = True
            procs[-1].start()
        for proc in procs:
            proc.join()

    def scale_target(self, target):
        return target

    def unscale_target(self, target):
        return target


class RandomAgent(Agent):
    """
    Agent effectuant une action aléatoire à tous les pas de temps
    """

    def select_action(self, *args, **kwargs):
        return self.env.action_space.sample()

    def train_with_transition(self, *args, **kwargs):
        print(args)


class QLearningAgent(Agent):
    """
    Implémente l'algorithme du *Q-Learning*
    """

    def __init__(
        self, env, value_function, policy, gamma=0.99, policy_on_cpu=False, **kwargs
    ):
        """
        * `env`: Environnement gym dans lequel l'agent évolue
        * `value_function`: Instance d'une fonction de valeur (voir `code_tp/value_functions`)
        * `policy`: Instance d'une politique (voir `code_tp/policies`)
        * `gamma`: Taux de discount de l'agent. Doit être compris entre 0 et 1
        """
        super().__init__(env, **kwargs)
        self.value_function = value_function
        self.cpu_value_function = self.value_function.clone()
        if hasattr(self.value_function, "nn"):
            self.cpu_value_function.nn.eval()
        self.distributional = callable(getattr(self.value_function, "dist", None))
        self.policy = policy
        self.policy_on_cpu = policy_on_cpu
        if self.policy_on_cpu:
            self.policy.value_function = self.cpu_value_function
        self.gamma = gamma
        self.value_function.agent = self
        self.policy.agent = self
        self.predicted_value_history = []

    def run_episode(self, test=False):
        if self.policy_on_cpu:
            self.cpu_value_function.import_f(self.value_function.export_f())
        if callable(getattr(self.policy.value_function, "reset_noise", None)):
            self.policy.value_function.reset_noise()
        return super().run_episode(test)

    @property
    def gamma(self):
        return self._gamma

    @gamma.setter
    def gamma(self, value):
        self._gamma = value

    def log_step(self, episode_name, step_num, transition):
        super().log_step(episode_name, step_num, transition)
        if self.distributional:
            if self.value_function.last_dist is None:
                return
            dists = self.value_function.last_dist
            action_values = self.value_function.dist_to_value(dists)
            if len(dists.shape) == 3:
                dists = dists[-1]
        else:
            if self.value_function.last_result is None:
                return
            action_values = self.value_function.last_result
        if len(action_values.shape) == 2:
            action_values = action_values[-1]
        value_scaling = getattr(self.policy, "value_scaling", 1)
        self.predicted_value_history.append(
            self.policy.value_unscaler(action_values.max())
        )
        action_mean = action_values.mean(dim=0)
        action_advantages = action_values - action_mean
        self.episode_logger.add_scalars(
            f"{episode_name}/action_advantages",
            {self.action_label_mapper(k): v for k, v in enumerate(action_advantages)},
            step_num,
        )

    def log_episode_end(self, episode_name, *args, **kwargs):
        actual_discounted_returns = [0]
        for reward in self.episode_reward_history[::-1]:
            actual_discounted_returns.insert(
                0,
                reward + self.gamma * actual_discounted_returns[0],
            )
        for i, (discounted_return, predicted_value) in enumerate(
            zip(actual_discounted_returns, self.predicted_value_history)
        ):
            self.episode_logger.add_scalars(
                f"{episode_name}/value_mean",
                {"actual": discounted_return, "predicted": predicted_value},
                i,
            )
        self.episode_reward_history = []
        self.predicted_value_history = []
        super().log_episode_end(episode_name, *args, **kwargs)

    def train_with_transition(
        self, state, action, next_state, reward, done, infos, prev_action=None
    ):
        if callable(getattr(self.value_function, "reset_noise", None)):
            self.value_function.reset_noise()
        target_value = self.target_value_from_state(
            next_state, reward, done, action
        )  # Pass current action as next prev_action
        self.value_function.update(state, action, target_value, prev_action)
        self.policy.value_unscaler = self.unscale_target
        if not self.test:
            self.policy.update()

    def eval_state(self, state, prev_action=None):
        if self.distributional:
            return self.value_function.best_action_dist_from_state(state, prev_action)
        return self.value_function.best_action_value_from_state(state, prev_action)

    def eval_state_batch(self, states, prev_actions=None):
        if self.distributional:
            return self.value_function.best_action_dist_from_state_batch(
                states, prev_actions
            )
        return self.value_function.best_action_value_from_state_batch(
            states, prev_actions
        )

    def target_value_from_state(
        self, next_state, reward, done, prev_action=None, n_step=1
    ):
        with torch.no_grad():
            _, next_value = self.eval_state(next_state, prev_action)
            if type(next_value) is torch.Tensor:
                next_value = next_value.detach().cpu().numpy()
            target = reward + self.gamma**n_step * self.unscale_target(next_value) * (
                1 - done
            )
        return self.scale_target(target)

    def target_value_from_state_batch(
        self, next_states, rewards, dones, actions, n_step=1
    ):
        """
        Calcule la valeur cible pour une batch de transitions
        """
        # Pass actions as prev_actions for next state evaluation
        if self.distributional:
            return self.target_dist_from_state_batch(
                next_states, rewards, dones, actions, n_step
            )
        with torch.no_grad():
            next_actions, next_values = self.eval_state_batch(
                next_states, actions if self.use_prev_action else None
            )
            # Convert boolean dones tensor to float for arithmetic operations
            dones = (
                dones.float()
                if isinstance(dones, torch.Tensor)
                else torch.tensor(dones, dtype=torch.float32)
            )
            targets = rewards + self.gamma**n_step * self.unscale_target(
                next_values
            ) * (1 - dones)
        return self.scale_target(targets)

    def target_dist_from_state_batch(
        self, next_states, rewards, dones, actions, n_step=1
    ):
        """
        Calcule la valeur cible pour une batch de transitions
        """
        # Pass actions as prev_actions for next state evaluation
        with torch.no_grad():
            batch_size = dones.shape[0]
            next_actions, next_dists = self.eval_state_batch(
                next_states, actions if self.use_prev_action else None
            )
            # Convert boolean dones tensor to float for arithmetic operations
            dones = (
                dones.float()
                if isinstance(dones, torch.Tensor)
                else torch.tensor(dones, dtype=torch.float32)
            )

            t_z = rewards.view(-1, 1) + (
                1 - dones.view(-1, 1)
            ) * self.gamma**n_step * self.value_function.support.expand(
                batch_size, self.value_function.atom_size
            )
            t_z = t_z.clamp(
                min=self.value_function.v_min, max=self.value_function.v_max
            )
            b = (t_z - self.value_function.v_min) / self.value_function.delta_z
            
            l = b.floor().long() # -1e-6 to avoid issues with integer values
            u = l + 1
            
            offset = (
                torch.linspace(
                    0, (batch_size - 1) * self.value_function.atom_size, batch_size
                )
                .long()
                .unsqueeze(1)
                .expand(batch_size, self.value_function.atom_size)
                .to(self.value_function.device)
            )

            proj_dist = torch.zeros(next_dists.size(), device=self.value_function.device)
            
            # Use the original b (without epsilon) for probability calculation
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dists * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u.clamp(max=self.value_function.atom_size - 1) + offset).view(-1), (next_dists * (b - l.float())).view(-1)
            )

            return proj_dist

    def select_action(self, state, prev_action=None, **kwargs):
        if not self.test:
            return self.policy(state, prev_action, **kwargs)
        else:
            return self.policy.test(state, prev_action, **kwargs)

    def plot_stats(self, save_dir):
        self.stats.update(self.value_function.stats)
        self.stats.update(self.policy.stats)
        super().plot_stats(save_dir)


class SARSAAgent(QLearningAgent):
    def train_with_transition(
        self, state, action, next_state, reward, done, infos, prev_action=None
    ):
        if not done:
            next_value = self.value_function(
                next_state, self.policy(next_state, prev_action)
            )
        else:
            next_value = 0
        target_value = reward + self.gamma * next_value
        self.value_function.update(state, action, target_value, prev_action)
        if not self.test:
            self.policy.update()
