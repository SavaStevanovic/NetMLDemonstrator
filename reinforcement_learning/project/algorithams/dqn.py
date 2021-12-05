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
from model.utils import Identifier


class DQN(Identifier):
    def __init__(self, inplanes, block_counts, input_size, output_size) -> None:
        self._inplanes = inplanes
        self._block_counts = block_counts
        self._input_size = input_size
        self._output_size = output_size
        backbone = networks.LinearResNetBackbone(
            inplanes=inplanes, block=blocks.BasicLinearBlock, block_counts=block_counts)

        self._policy_net = networks.LinearNet(
            backbone=[backbone],
            input_size=input_size,
            output_size=output_size
        ).cuda()

        self._target_net = networks.LinearNet(
            backbone=[backbone],
            input_size=input_size,
            output_size=output_size
        ).cuda()

        self._target_net.load_state_dict(self._policy_net.state_dict())
        self._target_net.eval()

        model_dir_header = self._target_net.get_identifier()
        self._chp_dir = os.path.join('tmp/checkpoints', model_dir_header)
        self._checkpoint_memo_path = os.path.join(
            self._chp_dir, 'mem_checkpoints.pth'
        )
        self._checkpoint_name_path = os.path.join(
            self._chp_dir, 'checkpoints.pth'
        )
        self._checkpoint_conf_path = os.path.join(
            self._chp_dir, 'configuration.json'
        )
        self._train_config = TrainingConfiguration()

        self._memory = RLDataset(20000)
        self._optimizer = torch.optim.RMSprop(
            self._policy_net.parameters(),
            self._train_config.learning_rate
        )

        summary(self._target_net, torch.Size([self._input_size]))

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
            self._target_net.get_identifier()
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
        self._target_net.load_state_dict(checkpoint["model_state"])
        self._policy_net.load_state_dict(checkpoint["model_state"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        self._policy_net.train()
        self._target_net.eval()
        self._memory.load(self._checkpoint_memo_path)
        self._train_config.load(self._checkpoint_conf_path)

    def save_model_state(self) -> None:
        os.makedirs((self._chp_dir), exist_ok=True)
        self._train_config.save(self._checkpoint_conf_path)
        if self._train_config.epoch % 10 == 0:
            self._memory.save(self._checkpoint_memo_path)
        checkpoint = {
            'model_state': self._target_net.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'model': self._target_net
        }
        torch.save(checkpoint, self._checkpoint_name_path)
        torch.save(
            self._target_net.state_dict(),
            self._checkpoint_name_path.replace('.pth', '_final_state_dict.pth')
        )

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self._train_config.EPS_END + (self._train_config.EPS_START - self._train_config.EPS_END) * \
            math.exp(-1. * self._train_config.steps_done /
                     self._train_config.EPS_DECAY)
        if sample > eps_threshold:
            state = torch.tensor(state).unsqueeze(0).cuda()
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self._policy_net(state.cuda()).max(1)[1]
        else:
            # print(sample, eps_threshold)
            return torch.tensor(random.randrange(self._output_size))

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

        # Perform one step of the optimization (on the policy network)
        if len(self._memory) >= self._train_config.BATCH_SIZE:
            batch = self._memory.sample(self._train_config.BATCH_SIZE)
            loss = self._optimize_model(batch)
            # Optimize the model
            self._optimizer.zero_grad()
            loss.backward()
            for param in self._policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self._optimizer.step()

        if self._train_config.steps_done % self._train_config.TARGET_UPDATE == 0:
            # save_image((current_screen - last_screen+1)[0]/2, 'img1.png')
            self._target_net.load_state_dict(self._policy_net.state_dict())

    def _optimize_model(self, batch):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self._policy_net(
            batch.state.cuda()).gather(1, batch.action.cuda().unsqueeze(-1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = self._target_net(
            batch.next_state.cuda()).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self._train_config.GAMMA) + batch.reward.cuda()

        # Compute Huber loss
        loss = self._criterion(
            state_action_values,
            expected_state_action_values.unsqueeze(1)
        )
        # print(loss.item())
        return loss

    def process_metric(self, episode_durations: deque):
        metric = mean(episode_durations)
        if self._train_config.best_metric < metric and len(episode_durations) >= episode_durations.maxlen/2:
            self._train_config.iteration_age = 0
            self._train_config.best_metric = metric
            print(
                f'Epoch {self._train_config.epoch}. Saving model with metric: {metric}'
            )
            torch.save(self._target_net,
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
                self._target_net.parameters(), self._train_config.learning_rate)
            print("Learning rate lowered to {}".format(
                self._train_config.learning_rate))
        self._train_config.epoch += 1
