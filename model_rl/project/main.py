from collections import defaultdict
import itertools
import json
import os
from time import sleep
import numpy as np
from stable_baselines3.sac import SAC
import torch
import typing
from data.random_simulation import DoneDataFetch, RandomSymulation
from data.step_data import StepDescriptor
from environment.environment_factory import EnvironmentFactory
from environment.model_environment import ModelEnv
from model import action_space_generator
from model.model_factory import ModelFactory
from model.model_predictive_contol import MPC, PolicyModel, RandomModel
from model.networks import LinearNet
from stable_baselines3.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.sac import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

from model.trainer import SamplingModel

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
        state = next_state
        metrics["total_reward"] += reward
    metrics["reward_relative_error"] = np.mean(reward_relative_error)
    print(metrics)
    return dict(metrics)


DEBUG = False
training_steps = 1000 if DEBUG else 100000 
inference_episodes = 2 if DEBUG else 10 

env_names = [
    # "citylearn_challenge_2022_phase_1", 
    "HalfCheetah-v2"
]

model_strats = ["MPC_RL_real", "MPC_RL_modelenv", "MPC", "RL_real", "Random"]

env_model_pairs = itertools.product(env_names, model_strats)

env_factory = EnvironmentFactory()
for env_name, model_strat in itertools.product(env_names, model_strats):
    environment, base_valid_env = env_factory.create_environment(env_name)
    val_env = Monitor(base_valid_env)
    factory = ModelFactory(SamplingModel(environment, val_env, env_name, DEBUG), training_steps=training_steps)
    policy_model = factory.create_model(model_strat, environment)
            
    sleep(1)
    infos = [infer(policy_model, val_env) for _ in range(inference_episodes)]
    info = dict_mean(infos)
    info["env_name"] = env_name
    info["model_strat"] = model_strat
    with open("./checkpoints/matrix.txt", "a") as f:
        json.dump(info, f)
        f.write("\n")
#MF_RL(F) -> Policy


#Real environment -> F     F: SxA -> S*xR  F-real env citylearn 1344 sample
#Data(F)-> d
#Supervised model(d) -> f     f: SxA -> S*xR  f-
#MPC(f) -> reward(F) 
#------Model free fine tuning------
#MF_RL(f) -> Policy   
#gde je model u mpcu i sta je razlika sa rl-om