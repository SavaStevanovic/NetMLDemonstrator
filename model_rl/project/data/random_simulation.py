from abc import abstractmethod
import pickle
import typing
import gym
from torch.utils.data import Dataset
from tqdm import tqdm

from data.step_data import StepDescriptor
from data.transforms import Standardizer, Transform

class DataFetchStrategy:
    @abstractmethod
    def generate_data(self) -> typing.List[StepDescriptor]:
        pass

class DoneDataFetch(DataFetchStrategy):
    def __init__(self, size: int, env: gym.Env) -> None:
        self._env = env
        self._size = size
        super().__init__()
    
    def generate_data(self) -> typing.List[StepDescriptor]:
        data = []
        cur_state = self._env.reset()
        for _ in  tqdm(range(self._size)):
            action = self._env.action_space.sample()  # sample random action
            next_state, reward, done, _ = self._env.step(action)
            data.append(StepDescriptor(cur_state, next_state, action, reward, done))
            cur_state = next_state
            if done:
                cur_state = self._env.reset()
        return data

class EpisodeLengthDataFetch(DataFetchStrategy):
    def __init__(self, episode_count: int, episode_length: int, env: gym.Env) -> None:
        self._episode_count = episode_count
        self._episode_length = episode_length
        self._env = env
        super().__init__()
    
    def generate_data(self) -> typing.List[StepDescriptor]:
        data = []
        cur_state = self._env.reset()
        for _ in  tqdm(range(self._episode_count)):
            for _ in range(self._episode_length):
                action = self._env.action_space.sample()  # sample random action
                next_state, reward, done, _ = self._env.step(action)
                data.append(StepDescriptor(cur_state, next_state, action, reward, done))
                cur_state = next_state
            cur_state = self._env.reset()

        return data


class RandomSymulation(Dataset):
    def __init__(self, data_fetcher: DataFetchStrategy, transoforms) -> None:
        super().__init__()
        self._data = data_fetcher.generate_data()
        self._transforms = transoforms

    @property
    def data(self) -> typing.List[StepDescriptor]:
        return self._data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx) -> StepDescriptor:
        sample = self._data[idx]
        for t in self._transforms:
            sample = t(sample)
        return sample
    
    def save(self, path: str):
        with open(path, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self._data, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        with open(path, 'rb') as inp:  # Overwrites any existing file.
            self._data = pickle.load(inp)
