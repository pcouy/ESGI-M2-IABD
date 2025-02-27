from ..wrappers.utils import TabularObservation, BoredomWrapper, LogScaleObs
import gymnasium as gym
import numpy as np


class AddDirectionToSnakeState(gym.ObservationWrapper):
    """Wrapper gym qui ajoute à l'observation du snake la direction de déplacement actuelle"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Tuple(
            (gym.spaces.Discrete(4), self.env.observation_space)
        )

    def observation(self, observation):
        return (observation, int(self.env.grid.snakes[0]._direction))


class SnakeFeatureObservation(gym.ObservationWrapper):
    """
    Wrapper gym qui permet de transformer l'observation brute du snake en un
    ensemble de *features expertes*
    """

    from gym_snake import constants

    OC = constants.ObjectColor

    def __init__(self, env, grid_size, food_direction="unit_vector", *args, **kwargs):
        """
        * `env`: Environnement à envelopper
        * `grid_size`: Taille de la grille de l'environnement
        * `food_direction`: prend les valeurs `"unit_vector"` ou `"angle"`, détermine si la direction
                    dans laquelle se trouve la nouriture doit être indiquée par un vecteur unitaire
                    ou un angle
        """
        env = AddDirectionToSnakeState(env)
        super().__init__(env, *args, **kwargs)
        self.food_direction = food_direction
        self.observation_space = gym.spaces.flatten_space(
            gym.spaces.Tuple(
                (
                    gym.spaces.Box(low=0, high=grid_size, shape=(9,)),
                    (
                        gym.spaces.Box(low=-1, high=1, shape=(2,))
                        if food_direction == "unit_vector"
                        else gym.spaces.Box(low=-np.pi, high=np.pi, shape=(1,))
                    ),
                )
            )
        )

    def observation(self, s):
        """
        Prend en entrée l'observation `s` et retourne l'observation dans le nouveau format
        """
        state, direction = s
        state = np.rot90(state, k=-int(direction))
        head_pos = np.stack(
            np.where(np.all(state == self.OC.own_head, axis=-1)), axis=-1
        )
        if len(head_pos) == 0:
            return ""
        elif len(head_pos) > 1:
            raise Exception
        else:
            head_pos = head_pos[0]
        w, h, _ = state.shape

        lidars = np.zeros((8,), dtype=float)
        lidar_idx = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                else:
                    scan_pos = head_pos.copy() + [dx, dy]
                    d = 1
                    while (
                        scan_pos[0] >= 0
                        and scan_pos[0] < w
                        and scan_pos[1] >= 0
                        and scan_pos[1] < h
                    ) and (
                        np.all(state[tuple(scan_pos)] == self.OC.empty)
                        or np.all(state[tuple(scan_pos)] == self.OC.apple)
                    ):
                        # print("Head:{} , Scan:{}, Color:{}".format(head_pos, scan_pos, state[tuple(scan_pos)]))
                        d += 1
                        scan_pos += [dx, dy]
                    lidars[lidar_idx] = d
                    lidar_idx += 1

        apple_pos = np.stack(np.where(np.all(state == self.OC.apple, axis=-1)), axis=-1)
        if len(apple_pos) > 0:
            apple_vec = (apple_pos - head_pos).astype(float).flatten()
        else:
            apple_vec = np.array([0, 0]).astype(float).flatten()
        apple_angle = np.arctan2(apple_vec[0], -apple_vec[1])
        apple_dist = np.linalg.norm(apple_vec)
        if apple_dist > 0:
            apple_unit = apple_vec / apple_dist
        else:
            apple_unit = apple_vec

        if self.food_direction == "unit_vector":
            return np.concatenate((lidars, [apple_dist], apple_unit))
        else:
            return np.concatenate((lidars, [apple_dist, apple_angle]))


def make_feature_snake_env(grid_size, log_scale=True, food_direction="unit_vector"):
    """
    Fonction utilitaire retournant un environnement snake équipé du wrapper `SnakeFeatureObservation`

    * `grid_size`: Taille de la grille (4 ; 8 ou 16)
    * `log_scale`: Si `True`, applique le wrapper `code_tp.wrappers.utils.LogScaleObs`
    * `food_direction`: Voir `SnakeFeatureObservation`
    """
    assert grid_size in [4, 8, 16]
    env = gym.make("Snake-{}x{}-v0".format(grid_size, grid_size))
    env = SnakeFeatureObservation(env, grid_size, food_direction)
    if log_scale:
        env = LogScaleObs(env)
    return env


def make_tabular_snake_env(grid_size, n_levels, log_scale=True):
    """
    Créé un environnement snake aux états discrets (en discretisant les *features* de
    `SnakeFeatureObservation`

    * `grid_size`: Taille de la grille (4 ; 8 ou 16)
    * `n_levels`: Tableau indiquant le nombre de valeurs possibles pour chaque *feature* discretisée
    * `log_scale`: Appliquer le wrapper `LogScaleObs` avant de discrétiser
    """
    env = make_feature_snake_env(grid_size, log_scale)
    env = TabularObservation(env, n_levels)
    return env


def manual_control(env):
    """
    Permet de controler manuellement le serpent.
    """
    import sys
    import gymnasium as gym
    import time
    from optparse import OptionParser

    import gym_snake
    from gym_snake.envs.constants import GridType, Action4, Action6
    from PyQt5.QtCore import Qt

    is_done = False

    def resetEnv():
        global is_done

        is_done = False
        env.reset()

    resetEnv()

    # Create a window to render into
    renderer = env.render("human")

    def keyDownCb(keyName):
        global is_done

        if keyName == Qt.Key_Escape:
            sys.exit(0)

        if keyName == Qt.Key_Backspace or is_done:
            resetEnv()
            return

        action = None
        if env.grid_type == GridType.square:
            if keyName == Qt.Key_Left or keyName == Qt.Key_A or keyName == Qt.Key_4:
                action = Action4.left
            elif keyName == Qt.Key_Right or keyName == Qt.Key_D or keyName == Qt.Key_6:
                action = Action4.right
            elif (
                keyName == Qt.Key_Up
                or keyName == Qt.Key_Space
                or keyName == Qt.Key_Return
                or keyName == Qt.Key_W
                or keyName == Qt.Key_8
            ):
                action = Action4.forward
            else:
                print("unknown key %s" % keyName)
                return

        elif env.grid_type == GridType.hex:
            if keyName == Qt.Key_Left or keyName == Qt.Key_Q or keyName == Qt.Key_7:
                action = Action6.left
            elif keyName == Qt.Key_Right or keyName == Qt.Key_E or keyName == Qt.Key_9:
                action = Action6.right
            elif (
                keyName == Qt.Key_Up
                or keyName == Qt.Key_Space
                or keyName == Qt.Key_Return
                or keyName == Qt.Key_W
                or keyName == Qt.Key_8
            ):
                action = Action6.forward
            elif keyName == Qt.Key_A or keyName == Qt.Key_4:
                action = Action6.left_left
            elif keyName == Qt.Key_D or keyName == Qt.Key_6:
                action = Action6.right_right

            else:
                print("unknown key %s" % keyName)
                return

        else:
            print("Unknown grid type: ", env.grid_type)

        if action is None:
            return

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print("step=%s, reward=%.2f" % (env.step_count, reward))

        if done:
            print("done!")
            is_done = True

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render("human")
        time.sleep(0.01)

        # If the window was closed
        if renderer.window is None:
            break
