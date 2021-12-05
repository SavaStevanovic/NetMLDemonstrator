from collections import deque
from statistics import mean
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
from data_loader.rldata import RLDataset, Transition
from model import networks
from model import blocks
import os
import torch
import random
from model_fitting.configuration import TrainingConfiguration
import math
from torchsummary import summary
from cached_property import cached_property
import numpy as np
from model.utils import Identifier
from torch.distributions import Categorical


class PolicyGradient(Identifier):
    def __init__(self, inplanes, block_counts, input_size, output_size) -> None:
        self._inplanes = inplanes
        self._block_counts = block_counts
        self._input_size = input_size
        self._output_size = output_size
        self._backbone = networks.LinearResNetBackbone(
            inplanes=inplanes, block=blocks.BasicLinearBlock, block_counts=block_counts)

        self._policy_net = networks.LinearNet(
            backbone=[self._backbone],
            input_size=input_size,
            output_size=output_size
        ).cuda()

        model_dir_header = self.get_identifier()
        self._chp_dir = os.path.join('tmp/checkpoints', model_dir_header)
        self._checkpoint_name_path = os.path.join(
            self._chp_dir, 'checkpoints.pth'
        )
        self._checkpoint_conf_path = os.path.join(
            self._chp_dir, 'configuration.json'
        )
        self._train_config = TrainingConfiguration()

        self._optimizer = torch.optim.RMSprop(
            self._policy_net.parameters(),
            self._train_config.learning_rate
        )
        self._memory = RLDataset(400)
        summary(self._policy_net, torch.Size([self._input_size]))

    @property
    def inplanes(self):
        return self._inplanes

    @property
    def block_counts(self):
        return self._block_counts

    @cached_property
    def _criterion(self) -> SummaryWriter:
        return nn.SmoothL1Loss()

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
    def epoch(self):
        if self._train_config.epoch > self._train_config.EPOCHS:
            return None
        return self._train_config.epoch

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer

    def load_last_state(self) -> None:
        if not os.path.exists(self._checkpoint_conf_path):
            return
        checkpoint = torch.load(self._checkpoint_name_path)
        self._policy_net.load_state_dict(checkpoint["model_state"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        self._policy_net.train()
        self._train_config.load(self._checkpoint_conf_path)

    def save_model_state(self) -> None:
        os.makedirs((self._chp_dir), exist_ok=True)
        self._train_config.save(self._checkpoint_conf_path)
        checkpoint = {
            'model_state': self._policy_net.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'model': self._policy_net
        }
        torch.save(checkpoint, self._checkpoint_name_path)
        torch.save(
            self._policy_net.state_dict(),
            self._checkpoint_name_path.replace('.pth', '_final_state_dict.pth')
        )

    def select_action(self, state):
        state = torch.tensor(state).unsqueeze(0).cuda()
        assert state.shape[0] == 1, "Must run one action at the time"
        probs = self._policy_net(state).softmax(1)
        # sample an action from that set of probs
        action = Categorical(probs).sample()

        return action

    def optimization_step(self, state, action, reward, new_state):
        self._train_config.steps_done += 1
        # Store the transition in memory
        self._memory.push(
            Transition(
                torch.Tensor([state]),
                torch.Tensor([action]).long(),
                torch.Tensor([new_state]),
                torch.tensor([reward])
            )
        )

    def _optimize_model(self, batch):
        cum_reward = 0
        for i in range(1, len(batch.reward.flip(0)) + 1):
            cum_reward *= self._train_config.GAMMA
            reward = max(batch.reward[-i], 0)
            cum_reward += reward
            batch.reward[-i] = cum_reward

        reward_mean = batch.reward.mean()
        reward_std = batch.reward.std()
        for i in range(len(batch.reward)):
            batch.reward[i] = (batch.reward[i] - reward_mean) / reward_std

        probs = self._policy_net(batch.state.cuda()).softmax(1)
        state_action_values = probs.gather(
            1, batch.action.cuda().unsqueeze(-1)).squeeze(1)
        log_action_values = state_action_values.log()
        losses = batch.reward.cuda() * log_action_values

        return -losses.mean()

    def process_metric(self, episode_durations: deque):
        # Perform one step of the optimization (on the policy network)
        loss = self._optimize_model(self._memory.as_batch())
        self._memory.clear()
        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

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
