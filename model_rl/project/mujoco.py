from collections import defaultdict
import json
import os
from time import sleep
import gym
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.sac import SAC
import torch
import typing
from data.random_simulation import DoneDataFetch, EpisodeLengthDataFetch, RandomSymulation
from data.step_data import StepDescriptor
from data.transforms import Standardizer, Transform
from environment.boptestGymEnv import BoptestGymEnv
from environment.model_environment import ModelEnv
from model import action_space_generator
from model.model_predictive_contol import MPC, PolicyModel
from model.networks import LinearNet
from model.trainer import fit, fit_epoch
from torch import nn
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from stable_baselines3.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.sac import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

def dict_mean(dict_list):
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
    return mean_dict

def infer(model: PolicyModel, env) -> dict:
    metrics = defaultdict(int)
    # Sample action sequences
    state = env.reset()
    done = False
    i=0
    reward_relative_error = []
    while (not done) and (i<1000):
        i+=1
        action, decription = model.predict(state)
        next_state, reward, done, info = env.step(action)
        # out = model(
        #     torch.cat((torch.tensor(state), torch.tensor(action)), dim=-1).float())
        # print("Predicted: ", out[0].item(), "Correct: ",
        #         reward, "Margin: ", out[0].item() - reward)
        state = next_state
        metrics["total_reward"] += reward
        # reward_relative_error.append(abs(reward - decription["reward"])/abs(reward + decription["reward"]))
    metrics["reward_relative_error"] = np.mean(reward_relative_error)
    print(metrics)
    return dict(metrics)


th_count = 24
batch_size = 32
env_name = 'citylearn_challenge_2022_phase_1'
environment = StableBaselines3Wrapper(NormalizedObservationWrapper(CityLearnEnv(env_name, central_agent=True, simulation_start_time_step = 0, simulation_end_time_step=1344)))
val_env = Monitor(StableBaselines3Wrapper(NormalizedObservationWrapper(CityLearnEnv(env_name, central_agent=True, simulation_start_time_step = 1344, simulation_end_time_step=2688, random_seed=42))))
# environment.get_summary()


data = RandomSymulation(DoneDataFetch(1334, environment), [])
val_data = RandomSymulation(DoneDataFetch(1334, val_env), [])
# data.save(f"{env_name}_data.pkl")
# val_data.save(f"{env_name}_val_data.pkl")
data.load(f"{env_name}_data.pkl")
val_data.load(f"{env_name}_val_data.pkl")

print(len(data))
sample = data[0]


def collate_fn(datas: typing.List[StepDescriptor]):
    return (
        torch.stack([torch.from_numpy(data.current_state)
                    for data in datas]).float(),
        torch.stack([torch.from_numpy(data.next_state)
                    for data in datas]).float(),
        torch.stack([torch.from_numpy(data.action) for data in datas]).float(),
        torch.stack([torch.tensor(data.reward) for data in datas]).float(),
        torch.stack([torch.tensor(data.done) for data in datas]).float(),
    )


dataloader = torch.utils.data.DataLoader(data, batch_size=(
    th_count > 1)*(batch_size-1)+1, shuffle=th_count > 1, num_workers=th_count, collate_fn=collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=(
    th_count > 1)*(batch_size-1)+1, shuffle=th_count > 1, num_workers=th_count, collate_fn=collate_fn)
model = LinearNet([len(sample.current_state) +
                   len(sample.action), 256, 256, len(sample.current_state) + 1])

model_dir_header = model.get_identifier()
writer = SummaryWriter(os.path.join('logs', model_dir_header))
fit(model, dataloader, val_dataloader, env_name, writer,
    epochs=1000, lower_learning_period=3)
# loss = fit_epoch(model, dataloader, 0.001, True, 1)
# val_loss = fit_epoch(model, val_dataloader, None, False, 1)
# print(loss, val_loss)
# model(torch.Tensor(np.concatenate((sample.current_state, sample.action))))

chp_dir = os.path.join('checkpoints', model_dir_header)
checkpoint_name_path = os.path.join(
    chp_dir, '{}_checkpoints_final.pth'.format(env_name))
model = torch.load(checkpoint_name_path)
fit(model, dataloader, val_dataloader, env_name, writer,
    epochs=1000, lower_learning_period=3)
horizon = 10


action_space_producer = action_space_generator.RandomSpaceProducer(horizon, 1000)
# action_space = action_space_generator.EvenlyspacedSpaceProducer(horizon, 2)
model.eval()
policy_model = MPC(action_space_producer, model, val_env.action_space, val_env.observation_space)
# torch.set_grad_enabled(True)
# train_env = ModelEnv(model, val_env.observation_space, val_env.action_space)
# policy_model = SAC('MlpPolicy', environment)
# eval_callback = EvalCallback(val_env, best_model_save_path='./best_model', log_path='./logs', eval_freq=1000, deterministic=True, render=False)
# checkpoint_callback = CheckpointCallback(save_freq=10000, save_path='./checkpoints')

# Train the model
# policy_model.learn(total_timesteps=100000, callback=[checkpoint_callback], progress_bar=True)

sleep(1)
infos = [infer(policy_model, val_env) for _ in range(2)]
info = dict_mean(infos)
writer.add_scalars(f'Plaining/{env_name}', info, 0)

info["env_name"] = env_name
with open("matrix.txt", "a") as f:
    json.dump(info, f)
    json.dump("\n", f)
#MF_RL(F) -> Policy


#Real environment -> F     F: SxA -> S*xR  F-real env citylearn 1344 sample
#Data(F)-> d
#Supervised model(d) -> f     f: SxA -> S*xR  f-
#MPC(f) -> reward(F) 
#------Model free fine tuning------
#MF_RL(f) -> Policy   