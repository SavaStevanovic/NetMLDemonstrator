from torch.utils.data import Dataset
from collections import deque
import torch
import random
from collections.abc import Iterable
import pickle

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

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        for key, value in data.items():
            setattr(self, key, value)