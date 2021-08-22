from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from collections import deque
from dataclasses import dataclass
import typing
import torch
import random
from collections.abc import Iterable

class Transition():

    def __init__(self, state, action, next_state, reward):
        if isinstance(reward, Iterable) and not isinstance(reward, torch.Tensor):
            state = torch.cat(state)
            action = torch.cat(action)
            reward = torch.cat(reward)
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward

    
def rl_collate_fn(batch):
        return Transition(*[x for x in zip(*[list(b.__dict__.values()) for b in batch])],)

class RLDataset(Dataset):
    def __init__(self, capacity):
        self._memory = deque([],maxlen=capacity)

    def push(self, instance: Transition):
        self._memory.append(instance)

    def __getitem__(self, idx: int):
        return self._memory[idx]

    def __len__(self):
        return len(self._memory)
    
    def sample(self, batch_size):
        return rl_collate_fn(random.sample(self._memory, batch_size))

# class RLDataLoaderIterProvider(_BaseDataLoaderIter):
#     def __init__(self, data_loader_iter: _BaseDataLoaderIter):
#         self = data_loader_iter

#     def _next_data(self):
#         data = super()._next_data()
#         return Transition(*zip(*data))

# class RLDataLoader(DataLoader):
#     def __init__(self, dataset: Dataset, batch_size: typing.Optional[int] = 1, shuffle: bool = False, num_workers: int = 0, pin_memory: bool = False):
#         DataLoader.__init__(self, dataset = dataset, batch_size = batch_size, shuffle = shuffle, num_workers = num_workers, pin_memory = pin_memory)
    
#     def __iter__(self) -> '_BaseDataLoaderIter':
#         iter = super().__iter__()
#         return iter
