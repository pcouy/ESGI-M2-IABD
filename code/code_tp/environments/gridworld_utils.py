import numpy as np
from ..mazelab import BaseMaze
from ..mazelab import Object
from ..mazelab import DeepMindColor as color

from ..mazelab import BaseEnv
from ..mazelab import VonNeumannMotion

import gym
from gym.spaces import Box
from gym.spaces import Discrete

class Maze(BaseMaze):
    x = np.array([[1, 1, 1, 1, 1, 1], 
              [1, 2, 0, 0, 0, 1], 
              [1, 0, 0, 0, 0, 1], 
              [1, 0, 0, 0, 0, 1], 
              [1, 0, 0, 0, 3, 1], 
              [1, 1, 1, 1, 1, 1]])

    @property
    def size(self):
        return self.x.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(
            np.any((self.x == 0, self.x == 2), axis=0)
        ), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])
        goal = Object('goal', 3, color.goal, False, np.stack(np.where(self.x == 3), axis =1))
        lava = Object('lava', 4, color.lava, False, np.stack(np.where(self.x == 4), axis =1))
        return free, obstacle, agent, goal, lava

class GenericMaze(Maze):
    def __init__(self, x, objects=None, rewards_done=None):
        self.x = np.array(x)
        self.orig_x = self.x.copy()
        self.object_definitions = {
            'free': Object('free', 0, color.free, False, np.stack(np.where(
                np.any((self.x == 0, self.x == 2), axis=0)
            ), axis=1)),
            'obstacle': Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.x == 1), axis=1)),
            'agent': Object('agent', 2, color.agent, False, []),
            'goal': Object('goal', 3, color.goal, False, np.stack(np.where(self.x == 3), axis =1)),
            'lava': Object('lava', 4, color.lava, False, np.stack(np.where(self.x == 4), axis =1)),
        }
        if objects is not None:
            assert type(objects) is dict
            for k,v in objects.items():
                assert type(v) is Object
                self.object_definitions[k] = v


        self.rewards_done = {
            'obstacle': (-0.01,False),
            'goal': (10,True),
            'lava': (-2,True)
        }
        if rewards_done is not None:
            assert type(rewards_done) is dict
            for k,v in rewards_done.items():
                assert type(v) in [float, int]
                self.rewards_done[k] = v

        super().__init__()

    def make_objects(self):
        return tuple(self.object_definitions.values())


class Env(BaseEnv):
    maze = Maze()

    def __init__(self):
        super().__init__()

        self.motions = VonNeumannMotion()

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]

        reward, done = self.get_reward_and_done(new_position)

        return self.maze.objects.agent.positions[0], reward, done, {}

    def get_reward_and_done(self, new_position):
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +10
            done = True
        elif self._is_lava(new_position):
            reward = -10
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.01
            done = False

        return reward, done

    def reset(self):
        self.maze.objects.agent.positions = np.stack(np.where(self.maze.orig_x == 2), axis =1)
        self.maze.objects.goal.positions = np.stack(np.where(self.maze.orig_x == 3), axis =1)
        return self.maze.objects.agent.positions[0]

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def _is_lava(self, position):
        out = False
        for pos in self.maze.objects.lava.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()

class GenericEnv(Env):
    default_reward = -0.01

    def get_reward_and_done(self, new_position):
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        for o, (r,d) in self.maze.rewards_done.items():
            for pos in getattr(self.maze.objects, o).positions:
                if new_position[0] == pos[0] and new_position[1] == pos[1]:
                    return r, d

        return self.default_reward, False

def register_env_from_maze(this_maze, name):
    class GridEnv(GenericEnv):
        maze = this_maze

    gym.envs.register(id=name, entry_point=GridEnv, max_episode_steps=200)

class StochasticWrapper(gym.Wrapper):
    def __init__(self, env, stochasticity):
        super().__init__(env)
        self.stochasticity = stochasticity

    def step(self, action):
        if np.random.uniform(0,1) < self.stochasticity:
            action = self.env.action_space.sample()

        return self.env.step(action)
