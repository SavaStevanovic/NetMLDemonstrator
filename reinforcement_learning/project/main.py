from model_fitting.train import fit
import gym
from model import networks
from model import blocks
import matplotlib
import environment.environments as envs
matplotlib.use('TkAgg')

visual_env = envs.ParameterEnv(gym.make('CartPole-v0'), False)
visual_env.env.reset()

# Get number of actions from gym action space
backbone = networks.LinearResNetBackbone(
    inplanes=256, block=blocks.BasicLinearBlock, block_counts=[1])

policy_net = networks.LinearNet(
    backbone=[backbone],
    input_size=visual_env.env.observation_space.shape[0],
    output_size=visual_env.env.action_space.n
).cuda()

target_net = networks.LinearNet(
    backbone=[backbone],
    input_size=visual_env.env.observation_space.shape[0],
    output_size=visual_env.env.action_space.n
).cuda()

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

fit(target_net, policy_net, visual_env)
