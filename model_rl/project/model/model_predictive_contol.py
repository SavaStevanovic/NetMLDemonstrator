import typing

import numpy as np
import torch
from model import action_space_generator
from gym.spaces import Box

class PolicyModel:
    def predict(self, state: np.ndarray) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
        pass

class RandomModel(PolicyModel):
    def __init__(
            self, 
            action_space: Box,
        ) -> None:
        self._action_space = action_space
        super().__init__()
    
    def predict(self, state: np.ndarray) -> typing.Tuple[np.ndarray, dict]:
        return self._action_space.sample(), {}



class MPC(PolicyModel):
    def __init__(
            self, 
            action_space_producer: action_space_generator.ActionSpaceProducer, 
            model: torch.nn.Module,
            action_space: Box,
            observation_space: Box,
            value_model: torch.nn.Module = None
        ) -> None:
        self._model = model
        self._action_space_producer = action_space_producer
        self._action_space = action_space
        self._num_samples = self._action_space_producer(action_space).shape[1]
        self._state_dim = observation_space.shape[0]
        self._value_model = value_model
        super().__init__()
    
    def predict(self, state: np.ndarray) -> typing.Tuple[np.ndarray, dict]:
        states = torch.tensor(
            np.zeros((self._action_space_producer.horizon+1, self._num_samples, self._state_dim))).float()
        rewards = torch.tensor(
            np.zeros((self._action_space_producer.horizon, self._num_samples))).float()
        value_rewards = torch.tensor(
            np.zeros((self._action_space_producer.horizon, self._num_samples))).float()

        action_seqs = torch.tensor(self._action_space_producer(self._action_space)).float()
        start_state = state
        states[0, :, :] = torch.tensor(start_state)
        for t in range(self._action_space_producer.horizon):
            if self._value_model is not None:
                value_rewards[t] = sum(self._value_model(states[t].float(), action_seqs[t].float())).squeeze().detach()
            state_action = torch.cat(
                (states[t], action_seqs[t]), dim=-1)
            next_state = self._model(state_action)
            rewards[t] = next_state[..., 0]
            next_state = next_state[..., 1:]
            next_state = next_state.detach()
            states[t+1, :, :] = next_state

        returns = ((0.95**torch.arange(rewards.shape[0])
                ).unsqueeze(1) * rewards).sum(0)
        value_returns = ((0.95**torch.arange(value_rewards.shape[0])
                ).unsqueeze(1) * value_rewards).sum(0)
        if self._value_model is not None:
            value_returns = returns.sum()/value_returns.sum()*value_returns
        best_index = np.argmax((returns + value_returns).detach().numpy())

        return action_seqs[0, best_index, :].detach().numpy(), {"reward": rewards[0, best_index].item()}
