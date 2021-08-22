import gym
from model import networks
from model import blocks
import matplotlib
from environment.visual_env import VisualEnv
matplotlib.use('TkAgg')
from model_fitting.train import fit

visual_env = VisualEnv(gym.make('CartPole-v0'))
visual_env.env.reset()

# Get number of actions from gym action space
backbone = networks.ResNetBackbone(inplanes = 64, block = blocks.BasicBlock, block_counts = [1, 1, 1])
net = networks.LinearNet(backbone = [backbone], output_size = visual_env.env.action_space.n)
policy_net = networks.LinearNet(backbone = [backbone], output_size = visual_env.env.action_space.n).cuda()
target_net = networks.LinearNet(backbone = [backbone], output_size = visual_env.env.action_space.n).cuda()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

fit(target_net, policy_net, visual_env)