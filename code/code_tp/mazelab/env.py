from abc import ABC
from abc import abstractmethod

import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from PIL import Image
import matplotlib.pyplot as plt


class BaseEnv(gym.Env, ABC):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 3}
    reward_range = (-float("inf"), float("inf"))

    def __init__(self):
        self.viewer = None
        self.seed()

    @abstractmethod
    def step(self, action):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_image(self):
        pass

    def render(self, mode="human", max_width=500):
        img = self.get_image()
        if mode == "rgb_array":
            img = np.asarray(img).astype(np.uint8)
            img = np.asarray(img)
            return img
        elif mode == "human":
            img_height, img_width = img.shape[:2]
            ratio = max_width / img_width
            img = Image.fromarray(img).resize(
                [int(ratio * img_width), int(ratio * img_height)]
            )
            img = np.asarray(img).astype(np.uint8)
            img = np.asarray(img)
            from gymnasium.envs.classic_control.rendering import SimpleImageViewer

            if self.viewer is None:
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(img)

            return self.viewer.isopen
        elif mode == "notebook":
            plt.imshow(img)
            plt.show()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
