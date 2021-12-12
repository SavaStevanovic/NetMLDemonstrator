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
        self._backbone = networks.LinearResNetBackbone(
            inplanes=inplanes, block=blocks.BasicLinearBlock, block_counts=block_counts)

        self._policy_net = networks.LinearNet(
            backbone=[self._backbone],
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
        output = self._output_transformation(self._policy_net(state))

        return self._action_transformation(output).sample()

    def _optimize_model(self, batch):
        self.acumulate_reward(batch)

        probs = self._output_transformation(
            self._policy_net(batch.state.cuda()))
        log_action_values = self._action_transformation(
            probs).log_prob(batch.action.cuda())
        losses = batch.reward.cuda() * log_action_values
        return -losses.sum()

    def acumulate_reward(self, batch):
        cum_reward = 0
        for i in range(1, len(batch.reward) + 1):
            cum_reward *= self._train_config.GAMMA
            reward = batch.reward[-i]
            cum_reward += reward
            batch.reward[-i] = cum_reward

    def process_metric(self, episode_durations: deque):
        # Perform one step of the optimization (on the policy network)
        loss = self._optimize_model(self._memory.as_batch())
        self._memory.clear()
        # Optimize the model
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

        super().process_metric(episode_durations)
