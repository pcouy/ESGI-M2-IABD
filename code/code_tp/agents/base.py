import gymnasium as gym
import numpy as np
import os
import gc
import json
import multiprocessing as mp

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter

class Agent:
    """
    Classe de base pour tous les agents.
    """
    def __init__(self, env:gym.Env, use_prev_action=False, save_dir="experiment", infos={}, tensorboard_layout={}, initial_prev_action=None):
        """
        * `env`: Environnement gym dans lequel l'agent va évoluer
        * `use_prev_action`: Si True, l'action précédente est utilisée comme entrée additionnelle
        * `save_dir`: Dossier dans lequel les résultats de l'expérience seront sauvegardés
        * `infos`: Dictionnaire qui sera sauvegardé au format JSON dans le dossier de
                    l'expérience, utile pour y écrire les paramètres.
        """
        self.training_steps = 0
        self.training_episodes = 0
        self.env = env
        self.agent = self
        self.test = False
        self.should_stop = False
        self.use_prev_action = use_prev_action
        self.initial_prev_action = initial_prev_action
        self.save_dir = save_dir
        self.stats = {
            'scores': {
                'x_label': "Episode",
                'data': []
            },
            'test_scores': {
                'data': []
            }
        }
        os.makedirs(self.save_dir, exist_ok=True)
        with open(os.path.join(self.save_dir, "infos.json"), "w") as f:
            print(infos)
            json.dump(infos, f, indent=2, default=lambda x: x.__name__ if type(x) is type else str(x))
        self.tensorboard = SummaryWriter(os.path.join(self.save_dir, "tensorboard"))
        self.tensorboard.add_custom_scalars(tensorboard_layout)

    def log_data(self, key, value):
        if key not in self.stats:
            self.stats[key] = {'data':[]}
        self.stats[key]["data"].append(value)
        self.tensorboard.add_scalar(key, value, self.training_steps)

    def select_action(self, state: np.ndarray, prev_action=None) -> np.ndarray:
        """
        Prend un état en entrée, renvoie une action
        * `prev_action`: Action précédente (optionnel)
        """
        raise NotImplementedError

    def train_with_transition(self, state, action, next_state, reward, done, infos, prev_action=None):
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
            env = gym.wrappers.RecordVideo(self.env, os.path.join(
                self.save_dir,
                "videos",
                "{:05d}".format(self.training_episodes)
            ), episode_trigger=lambda _: True)

        state, _ = env.reset()
        done = False
        # Initialize prev_action with a random action if the feature is enabled
        prev_action = (self.initial_prev_action if self.initial_prev_action is not None 
                      else self.env.action_space.sample() if self.use_prev_action 
                      else None)
        score = 0

        frames = []
        while not done:
            #frames.append(env.render('rgb_array'))
            action = self.select_action(state, prev_action)
            next_state, reward, terminated, truncated, infos = self.step(action, env=env)
            done = terminated or truncated
            if not test:
                self.train_with_transition(state, action, next_state, reward, done, infos, prev_action)
                self.training_steps+= 1
            score+= reward
            state = next_state
            prev_action = action if self.use_prev_action else None  # Update prev_action if feature is enabled

        if not test:
            self.training_episodes+= 1

        self.episode_end(score)

        if test:
            env.stop_recording()
        self.test = False

        return score

    def run_n_episodes(self, n, test=False):
        for i in range(n):
            self.run_episode(test=test)

    def train(self, n_episodes=1000, test_interval=50, train_callbacks=[], test_callbacks=[], test_interval_type="episode"):
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
            test_callbacks+= [self.plot_stats]
        self.should_stop = False

        while self.training_episodes < n_episodes and not self.should_stop:
            self.run_episode(test=False)
            for cb in train_callbacks:
                cb(self.save_dir)
            if self.test_condition(self.training_episodes, test_interval, test_interval_type):
                with torch.no_grad():
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
            return (i+1)%test_interval == 0


    def episode_end(self, score):
        """
        Méthode appelée par `run_episode` à la fin de chaque épisode, reçoit le
        score de l'épisode achevé comme argument.
        """
        print("{} episode ({}) done with score = {:.2f}".format(
            "Testing" if self.test else "Training",
            self.training_episodes,
            score
        ))
        if not self.test:
            self.log_data("scores", score)
        else:
            self.log_data("test_scores", score)

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
                fig.savefig(os.path.join(
                    save_dir,
                    "{}.png".format(name)
                ))
            else:
                plt.show()
            fig.clf()
            plt.clf()
            plt.close(fig)
        procs = []
        # "Hack" utilisant `multiprocessing` pour éviter une fuite de mémoire avec
        # matplotlib
        for stat in self.stats.keys():
            procs.append(mp.Process(target=plot_stat, 
                                    args=(stat,), 
                                    kwargs=self.stats[stat]
                                   ))
            procs[-1].daemon = True
            procs[-1].start()
        for proc in procs:
            proc.join()

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
    def __init__(self, env, value_function, policy, gamma=0.99, **kwargs):
        """
        * `env`: Environnement gym dans lequel l'agent évolue
        * `value_function`: Instance d'une fonction de valeur (voir `code_tp/value_functions`)
        * `policy`: Instance d'une politique (voir `code_tp/policies`)
        * `gamma`: Taux de discount de l'agent. Doit être compris entre 0 et 1
        """
        super().__init__(env, **kwargs)
        self.value_function = value_function
        self.policy = policy
        self.gamma = gamma
        self.value_function.agent = self
        self.policy.agent = self

    def train_with_transition(self, state, action, next_state, reward, done, infos, prev_action=None):
        target_value = self.target_value_from_state(next_state, reward, done, action)  # Pass current action as next prev_action
        self.value_function.update(state, action, target_value, prev_action)
        self.policy.update()

    def eval_state(self, state, prev_action=None):
        return self.value_function.best_action_value_from_state(state, prev_action)

    def eval_state_batch(self, states, prev_actions=None):
        return self.value_function.best_action_value_from_state_batch(states, prev_actions)

    def target_value_from_state(self, next_state, reward, done, prev_action=None):
        _, next_value = self.eval_state(next_state, prev_action)
        if type(next_value) is torch.Tensor:
            next_value = next_value.detach().cpu().numpy()
        target = reward + self.gamma * next_value * (1-done)
        return target

    def target_value_from_state_batch(self, next_states, rewards, dones, actions):
        """
        Calcule la valeur cible pour une batch de transitions
        """
        # Pass actions as prev_actions for next state evaluation
        next_actions, next_values = self.eval_state_batch(next_states, actions if self.use_prev_action else None)
        # Convert boolean dones tensor to float for arithmetic operations
        dones = dones.float() if isinstance(dones, torch.Tensor) else torch.tensor(dones, dtype=torch.float32)
        targets = rewards + self.gamma * next_values * (1-dones)
        return targets

    def select_action(self, state, prev_action=None):
        if not self.test:
            return self.policy(state, prev_action)
        else:
            return self.policy.test(state, prev_action)

    def plot_stats(self, save_dir):
        self.stats.update(self.value_function.stats)
        self.stats.update(self.policy.stats)
        super().plot_stats(save_dir)

class SARSAAgent(QLearningAgent):
    def train_with_transition(self, state, action, next_state, reward, done, infos, prev_action=None):
        if not done:
            next_value = self.value_function(next_state, self.policy(next_state, prev_action))
        else:
            next_value = 0
        target_value = reward + self.gamma*next_value
        self.value_function.update(state, action, target_value, prev_action)
        self.policy.update()
    
