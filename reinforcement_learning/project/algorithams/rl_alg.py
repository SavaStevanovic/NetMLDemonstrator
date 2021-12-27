from collections import deque
from statistics import mean
import typing
from torch.utils.tensorboard.writer import SummaryWriter
from data_loader.rldata import RLDataset, Transition
import os
import torch
from model_fitting.configuration import TrainingConfiguration
from torchsummary import summary
from cached_property import cached_property
import numpy as np
from model.utils import Identifier
from torch.distributions import Categorical, Normal, MultivariateNormal, distribution
from torch.distributions.beta import Beta
import abc
from gym.spaces import Box, Discrete, Space
import torch.nn.functional as F


class ReinforcmentAlgoritham(Identifier, abc.ABC):
    def __init__(self, capacity: int, input_space: typing.Union[Box, Discrete], output_shape: typing.Union[Box, Discrete]) -> None:
        super(ReinforcmentAlgoritham, self).__init__()
        self._memory = RLDataset(capacity)
        self._train_config = TrainingConfiguration()
        self._chp_dir = os.path.join('tmp/checkpoints', self.get_identifier())
        if isinstance(input_space, Box):
            self._input_size = input_space.shape[0]
        elif isinstance(input_space, Discrete):
            self._input_size = input_space.n

        if isinstance(output_shape, Box):
            self._output_size = output_shape.shape[0] * 2
            self._output_transformation = lambda x: x
            if output_shape.bounded_above.any() or output_shape.bounded_below.any():
                def multi_norm(x):
                    m, s = x.split(output_shape.shape[0], dim=-1)
                    return Beta(F.softplus(m.squeeze(-1))+1, F.softplus(s.squeeze(-1))+1)
            else:
                def multi_norm(x):
                    m, s = x.split(output_shape.shape[0], dim=-1)
                    return Normal(m.squeeze(-1), (0.1+F.softplus(s.squeeze(-1))))

            self._action_transformation = multi_norm
        if isinstance(output_shape, Discrete):
            self._output_size = output_shape.n
            self._output_transformation = lambda x: x.softmax(1)
            self._action_transformation = lambda x: Categorical(x)

    @cached_property
    def writer(self) -> SummaryWriter:
        summary_path = os.path.join(
            'tmp/logs',
            self.get_identifier()
        )
        return SummaryWriter(
            summary_path
        )

    @property
    def epoch(self) -> int:
        if self._train_config.epoch > self._train_config.EPOCHS:
            return None
        return self._train_config.epoch

    @property
    @abc.abstractmethod
    def optimizer(self) -> torch.optim.Optimizer:
        pass

    @abc.abstractmethod
    def load_last_state(self) -> None:
        pass

    @abc.abstractmethod
    def save_model_state(self) -> None:
        pass

    @abc.abstractmethod
    def preform_action(self, state: np.ndarray):
        pass

    def optimization_step(self, state: np.ndarray, action: np.ndarray, action_log_prob: np.ndarray, reward: np.ndarray, new_state: np.ndarray) -> None:
        self._train_config.steps_done += 1
        # Store the transition in memory
        self._memory.push(
            Transition(
                torch.Tensor([state]),
                torch.Tensor([action]),
                torch.Tensor([action_log_prob]),
                torch.Tensor([new_state]),
                torch.Tensor([reward])
            )
        )

    @abc.abstractmethod
    def _optimize_model(self, batch: Transition) -> torch.Tensor:
        pass

    def process_metric(self, episode_durations: deque) -> None:
        # Perform one step of the optimization (on the policy network)
        metric = mean(episode_durations)
        if self._train_config.best_metric < metric and len(episode_durations) >= episode_durations.maxlen/2:
            self._train_config.iteration_age = 0
            self._train_config.best_metric = metric
            print(
                f'Epoch {self._train_config.epoch}. Saving model with metric: {metric}'
            )
            torch.save(self._policy_net,
                       self._checkpoint_name_path.replace('.pth', '_final.pth')
                       )
        else:
            self._train_config.iteration_age += 1
            print('Epoch {} metric: {}'.format(
                self._train_config.epoch, metric))
        if self._train_config.iteration_age == self._train_config.LOWER_LEARNING_PERIOD:
            self._train_config.learning_rate *= 0.5
            self._train_config.iteration_age = 0
            self._optimizer = torch.optim.RMSprop(
                self._policy_net.parameters(), self._train_config.learning_rate)
            print("Learning rate lowered to {}".format(
                self._train_config.learning_rate))
        self._train_config.epoch += 1
