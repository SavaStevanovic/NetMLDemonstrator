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
    def _compute_policy_loss(self, log_action_values, advantage, batch):
        ratios = (log_action_values - batch.action_log_prob.cuda()).exp()
        losses1 = advantage * ratios
        losses2 = advantage * torch.clamp(
            ratios, 1 - 0.2, 1 + 0.2)

        losses = torch.min(losses1, losses2)
        return losses
