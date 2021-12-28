from collections import deque
from model import networks
from model import blocks
import os
import torch
from torchsummary import summary
from torch.distributions import Categorical
from algorithams.rl_alg import ReinforcmentAlgoritham


class PolicyGradient(ReinforcmentAlgoritham):
    def __init__(self, inplanes, block_counts, input_size, output_size) -> None:
        self._inplanes = inplanes
        self._block_counts = block_counts
        ReinforcmentAlgoritham.__init__(self, 1000, input_size, output_size)
        backbone = networks.LinearResNetBackbone(
            inplanes=inplanes, block=blocks.BasicLinearBlock, block_counts=block_counts)

        self._policy_net = networks.LinearNet(
            backbone=[backbone],
            input_size=self._input_size,
            output_size=self._output_size
        ).cuda()

        self._checkpoint_name_path = os.path.join(
            self._chp_dir, 'checkpoints.pth'
        )
        self._checkpoint_conf_path = os.path.join(
            self._chp_dir, 'configuration.json'
        )

        self._optimizer = torch.optim.Adam(
            self._policy_net.parameters(),
            self._train_config.learning_rate
        )
        summary(self._policy_net, torch.Size([self._input_size]))

    @property
    def inplanes(self):
        return self._inplanes

    @property
    def block_counts(self):
        return self._block_counts

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

    def preform_action(self, state):
        state = torch.tensor(state).unsqueeze(0).cuda()
        assert state.shape[0] == 1, "Must run one action at the time"
        dist = self._action_transformation(self._policy_net(state))
        action = dist.sample()
        return action.cpu().detach().numpy()[0], dist.log_prob(action).cpu().detach().sum().item()

    def _optimize_model(self, batch):
        rewards = self.acumulate_reward(batch)
        log_action_values = self._compute_log_probs(batch)
        losses = rewards.cuda() * log_action_values
        return -losses.sum()

    def _compute_log_probs(self, batch):
        dist = self._action_transformation(
            self._policy_net(batch.state.cuda()))
        log_action_values = dist.log_prob(batch.action.cuda())
        if len(log_action_values.shape) == 2:
            log_action_values = log_action_values.sum(-1)

        return log_action_values

    def acumulate_reward(self, batch):
        cum_reward = 0
        rewards = torch.zeros_like(batch.reward)
        for i in range(1, len(batch.reward) + 1):
            cum_reward *= self._train_config.GAMMA
            reward = batch.reward[-i]
            cum_reward += reward
            rewards[-i] = cum_reward
        return rewards

    def process_metric(self, episode_durations: deque):
        # Perform one step of the optimization (on the policy network)

        loss = self._optimize_model(self._memory.as_batch())
        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        self._memory.clear()
        super().process_metric(episode_durations)
