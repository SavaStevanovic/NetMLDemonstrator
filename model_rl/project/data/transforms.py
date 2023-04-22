import typing

import numpy as np

from data.step_data import StepDescriptor


class Transform:
    def __init__(self, data: typing.List[StepDescriptor], noise_indices: typing.List[int], mean: float, std: float):
        self.noise_indices = noise_indices
        self.mean = mean
        self.std = std * np.std([d.current_state[self.noise_indices] for d in data], 0)

    def __call__(self, step: StepDescriptor) -> StepDescriptor:
        current_state = step.current_state.copy()
        next_state = step.next_state.copy()

        # Add noise to current state and next state at specified indices
        current_state[self.noise_indices] += np.random.normal(self.mean, self.std)
        next_state[self.noise_indices] += np.random.normal(self.mean, self.std)

        step.current_state = current_state
        step.next_state = next_state

        return step


class Standardizer:
    def __init__(self, data: typing.List[StepDescriptor]):
        self._means = np.mean([x.current_state for x in data], 0)
        self._std = np.std([x.current_state for x in data], 0)

    def __call__(self, step) -> StepDescriptor:
        return ((step-self._means)/self._std).float()

    def reverse(self, state):
        return (state * self._std + self._means).float()