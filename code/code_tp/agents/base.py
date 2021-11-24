import gym
import numpy as np
import os
import json
import matplotlib.pyplot as plt


class Agent:
    """
    Classe de base pour tous les agents.
    """
    def __init__(self, env:gym.Env, save_dir="experiment", infos={}):
        """
        * `env`: Environnement gym dans lequel l'agent va évoluer
        * `save_dir`: Dossier dans lequel les résultats de l'expérience seront sauvegardés
        * `infos`: Dictionnaire qui sera sauvegardé au format JSON dans le dossier de
                    l'expérience, utile pour y écrire les paramètres.
        """
        self.training_steps = 0
        self.training_episodes = 0
        self.env = env
        self.test = False
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

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Prend un état en entrée, renvoie une action
        """
        raise NotImplementedError

    def train_with_transition(self, state, action, next_state, reward, done, infos):
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

        state = env.reset()
        done = False
        score = 0

        frames = []
        while not done:
            #frames.append(env.render('rgb_array'))
            action = self.select_action(state)
            next_state, reward, done, infos = self.step(action, env=env)
            if not test:
                self.train_with_transition(state, action, next_state, reward, done, infos)
                self.training_steps+= 1
            score+= reward
            state = next_state

        if not test:
            self.training_episodes+= 1

        self.episode_end(score)

        if test:
            env.close()
        self.test = False

        return score

    def run_n_episodes(self, n, test=False):
        for i in range(n):
            self.run_episode(test=test)

    def train(self, n_episodes=1000, test_interval=50, train_callbacks=[], test_callbacks=[]):
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
        for i in range(n_episodes):
            self.run_episode(test=False)
            for cb in train_callbacks:
                cb(self.save_dir)
            if (i+1)%test_interval == 0:
                self.run_episode(test=True)
                for cb in test_callbacks:
                    cb(self.save_dir)

    def episode_end(self, score):
        """
        Méthode appelée par `run_episode` à la fin de chaque épisode, reçoit le
        score de l'épisode achevé comme argument.
        """
        print("{} episode ({}) done with score = {}".format(
            "Testing" if self.test else "Training",
            self.training_episodes,
            score
        ))
        if not self.test:
            self.stats["scores"]["data"].append(score)
        else:
            self.stats["test_scores"]["data"].append(score)

    def plot_stats(self, save_dir=None):
        """
        Trace les courbes des tableaux contenus dans `self.stats`. Si `save_dir` vaut `None`,
        affiche directement les courbes, sinon les sauvegarde.
        """
        def plot_stat(name, x_label="", data=[]):
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
            plt.close(fig)
            plt.clf()

        for stat in self.stats.keys():
            plot_stat(stat, **self.stats[stat])

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
    def __init__(self, env, value_function, policy, gamma=0.99, *args, **kwargs):
        """
        * `env`: Environnement gym dans lequel l'agent évolue
        * `value_function`: Instance d'une fonction de valeur (voir `code_tp/value_functions`)
        * `policy`: Instance d'une politique (voir `code_tp/policies`)
        * `gamma`: Taux de discount de l'agent. Doit être compris entre 0 et 1
        """
        super().__init__(env, *args, **kwargs)
        self.value_function = value_function
        self.policy = policy
        self.gamma = gamma
        self.value_function.agent = self
        self.policy.agent = self

    def train_with_transition(self, state, action, next_state, reward, done, infos):
        if not done:
            values = self.value_function.from_state(next_state)
            if type(values) is not dict:
                values = {k:v for k,v in enumerate(values)}
            next_value, _ = max((v,a) for a,v in values.items())
        else:
            next_value = 0
        target_value = reward + self.gamma*next_value
        self.value_function.update(state, action, target_value)
        self.policy.update()

    def select_action(self, state):
        if not self.test:
            return self.policy(state)
        else:
            return self.policy.test(state)

    def plot_stats(self, save_dir):
        self.stats.update(self.value_function.stats)
        self.stats.update(self.policy.stats)
        super().plot_stats(save_dir)

class SARSAAgent(QLearningAgent):
    def train_with_transition(self, state, action, next_state, reward, done, infos):
        if not done:
            next_value = self.value_function(next_state, self.policy(next_state))
        else:
            next_value = 0
        target_value = reward + self.gamma*next_value
        self.value_function.update(state, action, target_value)
        self.policy.update()
    
