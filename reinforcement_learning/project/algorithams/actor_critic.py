from collections import deque
from algorithams.rl_alg import ReinforcmentAlgoritham
from model import networks
from model import blocks
import os
import torch
from torchsummary import summary
from torch.distributions import Categorical
from algorithams.policy_gradient import PolicyGradient
import torch.nn.functional as F


class A2C(PolicyGradient):
    def __init__(self, inplanes, block_counts, input_size, output_size) -> None:
        super().__init__(inplanes, block_counts, input_size, output_size)
        backbone = networks.LinearResNetBackbone(
            inplanes=inplanes, block=blocks.BasicLinearBlock, block_counts=block_counts)
        self._value_net = networks.LinearNet(
            backbone=[backbone],
            input_size=self._input_size,
            output_size=1
        ).cuda()

        self._value_optimizer = torch.optim.Adam(
            self._value_net.parameters(),
            self._train_config.learning_rate
        )

        self._batches = []
        summary(self._value_net, torch.Size([self._input_size]))

    def load_last_state(self) -> None:
        if not os.path.exists(self._checkpoint_conf_path):
            return
        checkpoint = torch.load(self._checkpoint_name_path)
        self._policy_net.load_state_dict(checkpoint["model_state"])
        self._value_net.load_state_dict(checkpoint["value_state"])
        self._policy_net.train()
        self._value_net.train()
        self._train_config.load(self._checkpoint_conf_path)

    def save_model_state(self) -> None:
        os.makedirs((self._chp_dir), exist_ok=True)
        self._train_config.save(self._checkpoint_conf_path)
        checkpoint = {
            'model_state': self._policy_net.state_dict(),
            'value_state': self._value_net.state_dict(),
            'optimizer': self._optimizer.state_dict(),
            'model': self._policy_net
        }
        torch.save(checkpoint, self._checkpoint_name_path)

    def _optimize_model(self, batch):
        rewards = self.acumulate_reward(batch)

        log_action_values = self._compute_log_probs(batch)
        cuda_reward = rewards.cuda()
        value = self._value_net(batch.state.cuda()).squeeze(-1)
        value_loss = F.smooth_l1_loss(value, cuda_reward)
        losses = self._compute_policy_loss(
            log_action_values, (cuda_reward - value), batch)

        return -losses.sum(), value_loss.sum()

    def _compute_policy_loss(self, log_action_values, advantage, batch):
        return advantage * log_action_values

    def process_metric(self, episode_durations: deque):
        self._batches.append(self._memory.as_batch())
        self._memory.clear()
        # Perform one step of the optimization (on the policy network)
        if (self._train_config.epoch % (self._train_config.BATCH_SIZE)) == 0:
            for _ in range(self._train_config.BATCH_SIZE):
                losses = 0
                value_losses = 0
                for batche in self._batches:
                    # Optimize the model
                    loss, value_loss = self._optimize_model(batche)
                    losses += loss
                    value_losses += value_loss
                self._optimizer.zero_grad()
                losses.backward(retain_graph=True)
                self._optimizer.step()

                self._value_optimizer.zero_grad()
                value_losses.backward()
                self._value_optimizer.step()
            print(losses.item(), value_losses.item())
            self._batches = []
        ReinforcmentAlgoritham.process_metric(self, episode_durations)
