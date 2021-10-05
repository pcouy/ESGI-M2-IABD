from . import gridworld_utils
import numpy as np

class LavaMaze(gridworld_utils.Maze):
    x = np.array([
            [1,1,1,1,1,1,1,1,1,1,1,1],
            [1,0,0,4,4,4,4,4,4,1,1,1],
            [1,0,2,0,0,0,0,0,3,0,0,1],
            [1,0,0,4,4,4,4,4,1,0,0,1],
            [1,0,0,1,0,0,0,0,0,0,0,1],
            [1,0,0,1,0,0,0,0,0,0,0,1],
            [1,0,0,1,0,0,0,0,0,4,4,1],
            [1,0,0,1,0,0,0,0,4,4,4,1],
            [1,0,0,4,0,0,0,4,4,4,4,1],
            [1,0,0,0,0,0,0,0,0,0,0,1],
            [1,0,0,4,0,0,0,0,0,0,0,1],
            [1,1,1,1,1,1,1,1,1,1,1,1]
        ])

class CliffMaze(gridworld_utils.Maze):
    x = np.array([
            [4,4,4,4,4,4,4,4,4,4],
            [1,2,0,0,0,0,0,0,3,1],
            [4,4,4,4,4,4,4,4,4,4]
        ])

gridworld_utils.register_env_from_maze(
    gridworld_utils.GenericMaze(
        x=LavaMaze.x,
    ), "GridLava-v0"
)

gridworld_utils.register_env_from_maze(gridworld_utils.GenericMaze(
    x=CliffMaze.x,
), "GridCliff-v0")
