import pickle
import gym
from torch.utils.data import Dataset
from tqdm import tqdm

from data.step_data import StepDescriptor

class RandomSymulation(Dataset):
    def __init__(self, size: int, env_name: str) -> None:
        super().__init__()
        self._data = []
        env = gym.make(env_name)
        for _ in  tqdm(range(size)):
            cur_state = env.reset()
            action = env.action_space.sample()  # sample random action
            next_state, reward, done, _ = env.step(action)
            self._data.append(StepDescriptor(cur_state, next_state, action, reward, done))
            cur_state = next_state
            if done:
                cur_state = env.reset()
        env.close()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        self._data[idx]
    
    def save(self, path: str):
        with open(path, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(self._data, outp, pickle.HIGHEST_PROTOCOL)

    def load(self, path: str):
        with open(path, 'rb') as inp:  # Overwrites any existing file.
            self._data = pickle.load(inp)
