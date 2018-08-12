import numpy as np
import torch
from typing import Dict

from gym.spaces import Dict as DictSpace, Discrete, Box

from envs import make_env


def torch_dtype(x):
    if isinstance(x, torch.Tensor):
        x = x
    elif isinstance(x, np.ndarray):
        x = torch.tensor(x)
    else:
        raise NotImplementedError()
    return x.dtype

def numpy_dtype(x):
    if isinstance(x, torch.Tensor):
        x = x.numpy()
    elif isinstance(x, np.ndarray):
        x = x
    else:
        raise NotImplementedError()
    return x.dtype


class TorchDictSpaceMapper:

    def __init__(self, space_dict: DictSpace):
        n = 0
        self.dtype = None
        self.space_dict = space_dict
        self.low = None
        self.high = None
        self.lows = []
        self.highs = []
        for k in self.space_dict.spaces:
            space = self.space_dict.spaces[k]
            assert isinstance(space, Box)
            if self.dtype is None:
                self.dtype = space.dtype
                self.low = space.low.min()
                self.high = space.high.max()
            else:
                assert self.dtype == space.dtype
                self.low = min(self.low, space.low.min())
                self.high = max(self.high, space.high.max())

            self.lows.append(space.low)
            self.highs.append(space.high)
            # if isinstance(space, (Discrete,)):
            #     n += 1
            if isinstance(space, Box):
                n_subspace = 1
                for d in space.shape:
                    n_subspace *= d
                n += n_subspace
            else:
                raise NotImplementedError()

        self.torch_dtype = torch_dtype(np.array([0], dtype=self.dtype))
        self.mapped_space = Box(low=self.low, high=self.high, shape=(n,))
        self.mapped_space.dtype = self.dtype # TODO: is this bad?



    def map_to_1d(self, observation: Dict) -> torch.Tensor:
        # print("map to discrete")
        assert len(observation) == len(self.space_dict.spaces)
        arrays = []
        for k in self.space_dict.spaces:
            # print(k)
            space = self.space_dict.spaces[k]
            value = observation[k]
            # if isinstance(space, (Discrete,)):
            #     new_value = torch.tensor([value], dtype=self.dtype)
            #     assert value.shape == (1,)
            if isinstance(space, Box):
                assert len(space.shape) == len(value.shape)
                for d, _ in enumerate(space.shape):
                    assert space.shape[d] == value.shape[d]
                assert isinstance(value, torch.Tensor)
                assert value.dtype == self.torch_dtype
                new_value = value
            else:
                raise NotImplementedError()

            flat = new_value.view((-1,))
            arrays.append(flat)
        res = torch.cat(arrays, dim=0)
        return res

    def reverse_map(self, mapped: torch.Tensor, batch_size=None) -> Dict:
        # print("reverse map")
        # assert mapped.dtype == self.torch_dtype
        if batch_size is None:
            batch_size = 1
            mapped = mapped.view((-1,))

        if batch_size != 1:
            # raise NotImplementedError("Never tested with batch size > 1")
            pass # TODO check wether this works!!
        assert mapped.shape[1] == self.mapped_space.shape[0]
        res = {}
        offset = 0
        for k in self.space_dict.spaces:
            # print(k)
            space = self.space_dict.spaces[k]
            # if isinstance(space, (Discrete,)):
            #     res[k] = mapped[offset].item()
            #     offset += 1
            if isinstance(space, Box):
                n_subspace = 1
                for d in space.shape:
                    n_subspace *= d
                flat = mapped[0:batch_size,offset:offset+n_subspace]
                res[k] = flat.view((batch_size,) + space.shape)
                offset += n_subspace
            else:
                raise NotImplementedError()
        return res



def space_mapper_test():
    batch_size = 2
    debug_env = make_env("", 0, 0, None, False, False, None)()
    mapper = TorchDictSpaceMapper(debug_env.observation_space)
    initial_obs = debug_env.reset()
    debug_env.reset()
    obs = [debug_env.step(0)[0] for r in range(batch_size)]
    mapped_obs = torch.stack([mapper.map_to_1d(ob) for ob in obs])
    reversed = mapper.reverse_map(mapped_obs, batch_size=batch_size)
    assert len(reversed) == len(initial_obs)
    for k in reversed:
        # abs is not implemented for bytes... just compute the square
        initial_obs = torch.stack([ob[k] for ob in obs])
        diff = initial_obs - reversed[k]
        d2 = diff * diff
        assert d2.max().item() <= 0.0000001

