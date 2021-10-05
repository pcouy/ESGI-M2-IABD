import matplotlib.pyplot as plt
import os
import numpy as np
from .base import QLearningAgent
from ..utils import isnotebook

class GridworldTabularValueAgent(QLearningAgent):
    def show_values(self, save_dir):
        img = self.env.render('rgb_array')
        fig, ax = plt.subplots(figsize=(10,10))
        im = ax.imshow(img)
        values = np.zeros(self.env.maze.size)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                try:
                    values[i,j], _ = max((v,a) for a,v in self.value_function.from_state([i,j]).items())
                except KeyError:
                    values[i,j] = 0
                ax.text(j,i, "{:.2f}".format(values[i,j]), ha='center', va='center')
        fig.tight_layout()
        if isnotebook() or save_dir is None:
            "Showing values :"
            plt.show()
        if save_dir is not None:
            os.makedirs(os.path.join(save_dir,"values"), exist_ok=True)
            fig.savefig(os.path.join(
                save_dir,
                "values",
                "values_{:05d}.png".format(self.training_episodes)
            ))
        plt.close(fig)

    def train(self, n_episodes=1000, test_interval=50, train_callbacks=None, test_callbacks=None):
        if train_callbacks is None:
            train_callbacks = []
        if test_callbacks is None:
            test_callbacks = []
        if self.show_values not in test_callbacks:
            test_callbacks+= [self.show_values]
        else:
            print("Trying to add 'show_values' callback twice")
        super().train(n_episodes, test_interval, train_callbacks, test_callbacks)
