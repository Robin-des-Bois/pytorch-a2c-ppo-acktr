import gym
from gym import Wrapper

import space_mapper


class MapDiscreteWrapper(Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.obs_mapper = space_mapper.TorchDictSpaceMapper(env.observation_space)
        self.observation_space = self.obs_mapper.mapped_space


    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.obs_mapper.map_to_1d(obs)

    def step(self, action):
        (ob, rew, done, info) = self.env.step(action)
        d_ob = self.obs_mapper.map_to_1d(ob)
        return (d_ob, rew, done, info)