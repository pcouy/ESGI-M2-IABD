import gym
import numpy as np
import os
import json
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, env:gym.Env, save_dir="experiment", infos={}):
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
        raise NotImplementedError

    def train_with_transition(self, state, action, next_state, reward, done, infos):
        raise NotImplementedError

    def step(self, action, env=None):
        if env is None:
            env = self.env
        return env.step(action)

    def run_episode(self, test=False):
        self.test = test
        if not test:
            env = self.env
        else:
            os.makedirs(os.path.join(self.save_dir, "videos"), exist_ok=True)
            env = gym.wrappers.monitoring.video_recorder.VideoRecorder(self.env, os.path.join(
                self.save_dir,
                "videos",
                "{:5d}.mp4".format(self.training_episodes)
            ))
            env = self.env

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
        print("{} episode ({}) done with score = {}".format(
            "Testing" if self.test else "Training",
            self.training_episodes,
            score
        ))
        if not self.test:
            self.stats["scores"]["data"].append(score)
        else:
            self.stats["test_scores"]["data"].append(score)

    def plot_stats(self, save_dir):
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
    def select_action(self, *args, **kwargs):
        return self.env.action_space.sample()

    def train_with_transition(self, *args, **kwargs):
        print(args)

class QLearningAgent(Agent):
    def __init__(self, env, value_function, policy, gamma=0.99, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.value_function = value_function
        self.policy = policy
        self.gamma = gamma

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
    
