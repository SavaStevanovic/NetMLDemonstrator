from model_fitting.train import fit
import matplotlib
import algorithams
import os
import shutil
import environment.playgrounds as plg

matplotlib.use('TkAgg')
dirpath = "tmp"
if os.path.exists(dirpath) and os.path.isdir(dirpath):
    shutil.rmtree(dirpath)

gym_env = plg.CartPoleV0(False)
gym_env.env.reset()

alg = algorithams.actor_critic.A2C(
    inplanes=256,
    block_counts=[1],
    input_size=gym_env.env.observation_space.shape[0],
    output_size=gym_env.env.action_space.n
)

fit(alg, gym_env)
