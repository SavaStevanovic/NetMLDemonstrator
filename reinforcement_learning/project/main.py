from model_fitting.train import fit
import gym
import matplotlib
import model.algorithams as alg
import environment.environments as envs
matplotlib.use('TkAgg')

gym_env = envs.ParameterEnv(gym.make('CartPole-v0'), False)
gym_env.env.reset()

alg = alg.DQN(
    inplanes=256,
    block_counts=[1],
    input_size=gym_env.env.observation_space.shape[0],
    output_size=gym_env.env.action_space.n
)

fit(alg, gym_env)
