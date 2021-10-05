import gym
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

class Bandits(gym.Env):
    metadata = {'render.modes': ['rgb_array']}
    K = 10
    VAR_BETWEEN_SLOTS = 10
    SLOT_VAR = 10
    AVG_REWARD = 10

    def __init__(self):
        self.observation_space = Discrete(1)
        self.action_space = Discrete(self.K)
        self.slots = [(np.random.normal(self.AVG_REWARD, np.sqrt(self.VAR_BETWEEN_SLOTS)), self.SLOT_VAR)
                       for _ in range(self.K)]

    def reset(self):
        self.reward_history = []
        self.last_render = 0
        self.last_image = []
        self.curr_step = 0
        return 0

    def step(self, action):
        slot = self.slots[action]
        r = np.random.normal(*slot)
        self.reward_history.append(r)
        self.curr_step+= 1
        return 1, r, False, {}

    def render(self, mode='rgb_array'):
        if self.last_render == 0 or self.last_render < self.curr_step - 100:
            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.gca()

            ax.plot(self.reward_history)

            canvas.draw()       # draw the canvas, cache the renderer
            width, height = fig.get_size_inches() * fig.get_dpi()

            image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height),int(width),3)
            self.last_image = image
            self.last_render = self.curr_step
        return self.last_image


    def close(self):
        pass

gym.envs.register(id="10ArmedBandits-v0", entry_point=Bandits, max_episode_steps=100000)

