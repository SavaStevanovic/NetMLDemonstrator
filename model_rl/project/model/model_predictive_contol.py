import typing

import numpy as np
import torch
from model import action_space_generator
from gym.spaces import Box

class PolicyModel:
    def predict(self, state: np.ndarray) -> typing.Tuple[np.ndarray, typing.Optional[np.ndarray]]:
        pass


class MPC(PolicyModel):
    def __init__(
            self, 
            action_space_producer: action_space_generator.ActionSpaceProducer, 
            model: torch.nn.Module,
            action_space: Box,
            observation_space: Box
        ) -> None:
        self._model = model
        self._action_space_producer = action_space_producer
        self._action_space = action_space
        self._num_samples = self._action_space_producer(action_space).shape[1]
        self._state_dim = observation_space.shape[0]
        super().__init__()
    
    def predict(self, state: np.ndarray) -> typing.Tuple[np.ndarray, dict]:
        states = torch.tensor(
            np.zeros((self._action_space_producer.horizon+1, self._num_samples, self._state_dim))).float()
        rewards = torch.tensor(
            np.zeros((self._action_space_producer.horizon, self._num_samples))).float()

        action_seqs = torch.tensor(self._action_space_producer(self._action_space)).float()
        start_state = state
        states[0, :, :] = torch.tensor(start_state)
        for t in range(self._action_space_producer.horizon):
            state_action = torch.cat(
                (states[t], action_seqs[t]), dim=-1)
            next_state = self._model(state_action)
            rewards[t] = next_state[..., 0]
            next_state = next_state[..., 1:]
            next_state = next_state.detach()
            states[t+1, :, :] = next_state
            # Compute the reward as the speed in the x-direction

        # Compute the returns for each action sequence
        returns = ((0.95**torch.arange(rewards.shape[0])
                ).unsqueeze(1) * rewards).sum(0)
        best_index = np.argmax(returns.detach().numpy())
        # print("Predicted return: ",
        #         (rewards[0, best_index]).item())
        # Choose the best action sequence and take the first action
        return action_seqs[0, best_index, :].detach().numpy(), {"reward": rewards[0, best_index].item()}
