from torch.utils.data import Dataset
from collections import deque
import torch
import random
from collections.abc import Iterable
import pickle


class Transition():
    def __init__(self, state, action, action_log_prob, next_state, reward, done):
        if isinstance(reward, Iterable) and not isinstance(reward, torch.Tensor):
            state = torch.cat(state)
            action = torch.cat(action)
            next_state = torch.cat(next_state)
            reward = torch.cat(reward)
            action_log_prob = torch.cat(action_log_prob)
            done = torch.cat(done)
        self.state = state
        self.action = action
        self.action_log_prob = action_log_prob
        self.next_state = next_state
        self.reward = reward
        self.done = done


def rl_collate_fn(batch):
    return Transition(*[x for x in zip(*[list(b.__dict__.values()) for b in batch])],)


class RLDataset(Dataset):
    def __init__(self, capacity):
        self._memory = deque([], maxlen=capacity)
        self._safe_loc = 0

    def reverse(self):
        self._memory.reverse()

    def clear(self):
        self._memory.clear()

    def push(self, instance: Transition):
        self._memory.append(instance)

    def __getitem__(self, idx: int):
        return self._memory[idx]

    def __len__(self):
        return len(self._memory)

    def as_batch(self):
        memory = rl_collate_fn(self._memory)
        self.clear()
        return memory

    def sample(self, batch_size):
        return rl_collate_fn(random.sample(self._memory, batch_size))

    def _get_path(self, path: str, index: int):
        path_parts = path.split("/")
        path_parts[-1] = str(index) + path_parts[-1]
        path = "/".join(path_parts)
        return path

    def save(self, path):
        self._safe_loc = 1 - self._safe_loc
        path = self._get_path(path, self._safe_loc)
        torch.save(self.__dict__, path)

    def load(self, path):
        for i in range(2):
            try:
                file_path = self._get_path(path, i)
                data = torch.load(file_path)
            except Exception as e:
                print(e)
        for key, value in data.items():
            setattr(self, key, value)
