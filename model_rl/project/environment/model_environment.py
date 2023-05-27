import typing
import gym
import numpy as np
import torch

from data.step_data import StepDescriptor

class StateActionRootReduction(gym.Env):
    def __init__(self, env: gym.Env, data: typing.List[StepDescriptor]):
        self._env = env
        self._data = data
        states = np.stack([x.current_state for x in self._data])
        rewards = np.stack([x.reward for x in self._data])
        actions = np.stack([x.action for x in self._data])
        self._action_ind = self._get_corelated_subset_ind(actions, rewards)
        self._state_ind = self._get_corelated_subset_ind(states, rewards)

    def get_subsampled_data(self, data: typing.List[StepDescriptor]):
        return [
            StepDescriptor(
                d.current_state[self._state_ind],
                d.next_state[self._state_ind], 
                d.action[self._action_ind], 
                d.reward, 
                d.done
            ) for d in data
        ]

    def _get_corelated_subset_ind(self, actions, rewards):
        act_cor = [abs(np.corrcoef(actions[:, i], rewards)[0, 1]) for i in range(len(actions[0]))]
        return np.argsort(act_cor)[-int(len(act_cor)**0.5):]

    def reset(self):
        return self._env.reset()[self._state_ind]
    
    def step(self, action):
        active_action = self._env.action_space.sample()
        active_action[self._action_ind] = action
        obs, reward, done, info = self._env.step(active_action)
        return obs[self._state_ind], reward, done, info

    @property
    def observation_space(self) -> gym.Space:
        space = self._env.observation_space
        return gym.spaces.Box(low=space.low[self._state_ind], high=space.high[self._state_ind])

    @property
    def action_space(self)  -> gym.Space:
        space = self._env.action_space
        return gym.spaces.Box(low=space.low[self._action_ind], high=space.high[self._action_ind])


class ModelEnv(gym.Env):
    def __init__(self, model, observation_space: gym.spaces.Box, action_space: gym.spaces.Box):
        self.model = model.eval()
        self.observation_space = observation_space
        self.action_space = action_space
        self.state = None
        self._ep_step = 0
    
    def reset(self):
        # reset the environment state
        self.state = np.random.normal(size=[x+1 if i==0 else x for i, x in enumerate(self.observation_space.shape)]).astype(np.float32)
        self._ep_step = 0
        return self.state[1:]
    
    def step(self, action):
        self._ep_step += 100
        action = action.clip(self.action_space.low, self.action_space.high)
        # apply the action to the model to get the next state
        state_action = torch.cat(
                    (torch.tensor(self.state[1:]), torch.tensor(action)), dim=-1)
        self.state = self.model(state_action).detach().numpy()
        
        # return the next state, reward, done, and info
        done = True if self._ep_step>=100 else False  # in this example, the environment never terminates
        info = {}  # you can use this dictionary to provide additional information
        return self.state[1:], self.state[0], done, info
