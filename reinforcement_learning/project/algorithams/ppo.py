from collections import deque
from model import networks
from model import blocks
import os
import torch
from torchsummary import summary
from torch.distributions import Categorical
from algorithams.actor_critic import A2C
import torch.nn.functional as F
from copy import deepcopy


class PPO(A2C):
    def _optimize_model(self, batch):
        rewards = self.acumulate_reward(batch)

        probs = self._output_transformation(
            self._policy_net(batch.state.cuda()))
        log_action_values = self._action_transformation(
            probs).log_prob(batch.action.cuda())
        cuda_reward = rewards.cuda()
        value = self._value_net(batch.state.cuda()).squeeze(-1)
        adv = (cuda_reward - value.detach())

        ratios = (log_action_values - batch.action_log_prob.cuda()).exp()
        losses1 = adv * ratios
        losses2 = adv * torch.clamp(
            ratios, 1 - 0.2, 1 + 0.2)
        losses = torch.min(losses1, losses2)
        value_loss = F.mse_loss(value, cuda_reward)
        return -losses.sum(), value_loss

    def process_metric(self, episode_durations: deque):
        super().process_metric(episode_durations)
