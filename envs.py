import os

import gym
import numpy as np
from gym.spaces.box import Box

from baselines import bench
from gym.wrappers import TimeLimit

import gym_button.configured_env
import map_discrete_wrapper
import space_mapper

try:
    import dm_control2gym
except ImportError:
    pass

try:
    import roboschool
except ImportError:
    pass

try:
    import pybullet_envs
except ImportError:
    pass


def make_env(env_id, seed, rank, log_dir, add_timestep, map_discrete, time_limit):
    def _thunk():
        # env = base.make_env(env_id, process_idx=rank, outdir=logger.get_dir())
        # env.seed(seed + rank)
        # if logger.get_dir():
        #     env = bench.Monitor(env, os.path.join(logger.get_dir(), 'train-{}.monitor.json'.format(rank)))
        # return env

        # TODO use `env_id` so that people can use other environements
        env = gym_button.configured_env.get_configured_env()
        env.seed(seed + rank)

        if map_discrete:
            env = map_discrete_wrapper.MapDiscreteWrapper(env)



        obs_shape = env.observation_space.shape
        if add_timestep and len(
                obs_shape) == 1 and str(env).find('TimeLimit') > -1:
            env = AddTimestep(env)

        if time_limit is not None:
            env = TimeLimit(env, max_episode_steps=time_limit)


        if log_dir is not None:
            env = bench.Monitor(env, os.path.join(log_dir, str(rank)))


        return env

    return _thunk


class AddTimestep(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(AddTimestep, self).__init__(env)
        self.observation_space = Box(
            self.observation_space.low[0],
            self.observation_space.high[0],
            [self.observation_space.shape[0] + 1],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return np.concatenate((observation, [self.env._elapsed_steps]))


class WrapPyTorch(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(WrapPyTorch, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation.transpose(2, 0, 1)
