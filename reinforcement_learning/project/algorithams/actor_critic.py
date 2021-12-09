from collections import deque
from model import networks
import os
import torch
from torchsummary import summary
from torch.distributions import Categorical
from algorithams.policy_gradient import PolicyGradient
import torch.nn.functional as F


class A2C(PolicyGradient):
    def __init__(self, inplanes, block_counts, input_size, output_size) -> None:
        super().__init__(inplanes, block_counts, input_size, output_size)

        self._value_net = networks.LinearNet(
            backbone=[self._backbone],
            input_size=input_size,
            output_size=1
        ).cuda()

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

    def preform_action(self, state):
        state = torch.tensor(state).unsqueeze(0).cuda()
        assert state.shape[0] == 1, "Must run one action at the time"
        probs = self._policy_net(state).softmax(1)
        # sample an action from that set of probss
        action = Categorical(probs).sample()

        return action

    def _optimize_model(self, batch):
        cum_reward = 0
        for i in range(1, len(batch.reward.flip(0)) + 1):
            cum_reward *= self._train_config.GAMMA
            reward = batch.reward[-i]
            cum_reward += reward
            batch.reward[-i] = cum_reward

        reward_mean = batch.reward.mean()
        reward_std = batch.reward.std()
        for i in range(len(batch.reward)):
            batch.reward[i] = (batch.reward[i] - reward_mean) / reward_std

        probs = self._policy_net(batch.state.cuda()).softmax(1)
        value = self._value_net(batch.state.cuda()).squeeze(-1)
        state_action_values = probs.gather(
            1, batch.action.cuda().unsqueeze(-1)).squeeze(1)
        log_action_values = state_action_values.log()
        cuda_reward = batch.reward.cuda()
        losses = (cuda_reward - value) * log_action_values
        value_loss = F.smooth_l1_loss(value, cuda_reward)

        return -losses.mean() + value_loss.mean()
