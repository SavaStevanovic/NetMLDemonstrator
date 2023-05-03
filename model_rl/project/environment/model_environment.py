import gym
import numpy as np
import torch

class ModelEnv(gym.Env):
    def __init__(self, model, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        self.model = model.eval()
        self.observation_space = observation_space
        self.action_space = action_space
        self.state = None
    
    def reset(self):
        # reset the environment state
        self.state = np.zeros([x+1 if i==0 else x for i, x in enumerate(self.observation_space.shape)], dtype=np.float32)
        return self.state[1:]
    
    def step(self, action):
        # apply the action to the model to get the next state
        state_action =  torch.cat(
                    (torch.tensor(self.state[1:]), torch.tensor(action)), dim=-1)
        self.state = self.model(state_action).detach().numpy()
        
        # return the next state, reward, done, and info
        done = False  # in this example, the environment never terminates
        info = {}  # you can use this dictionary to provide additional information
        return self.state[1:], self.state[0], done, info
