import os
from time import sleep
import gym
import numpy as np
import torch
import typing
from data.random_simulation import DoneDataFetch, EpisodeLengthDataFetch, RandomSymulation
from data.step_data import StepDescriptor
from data.transforms import Standardizer, Transform
from model.networks import LinearNet
from model.trainer import fit, fit_epoch
from torch import nn

def random_shooting_mpc(model, env, horizon, num_iterations, num_samples):
    model.eval()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(num_iterations):
        
        rewards = np.zeros((horizon, num_samples))

        # Sample action sequences
        

        # Simulate the system using the dynamics model
        state = env.reset()
        def get_next_action(state, env, model):
            states = torch.tensor(np.zeros((horizon+1, num_samples, state_dim))).float()
            action_seqs = torch.tensor(np.random.uniform(low=env.action_space.low, high=env.action_space.high, size=(horizon, num_samples, action_dim))).float()
            start_state = state
            states[0, :, :] = torch.tensor(start_state)
            for t in range(horizon):
                state_action = torch.cat((states[t][:, 1:], action_seqs[t]), dim=-1)
                next_state = model(state_action)
                next_state = next_state.detach()
                states[t+1, :, :] = next_state
                # Compute the reward as the speed in the x-direction

            # Compute the returns for each action sequence
            returns = -((0.95**torch.arange(states.shape[0])).unsqueeze(1)* states[:, :, 0]).sum(0)
            best_index = np.argmax(returns)
            print(returns[best_index], "Predicted return: ", (states[1, best_index, 0]).item())
            # Choose the best action sequence and take the first action
            return action_seqs[0, best_index, :].detach().numpy()
        done = False
        while True:
            action = get_next_action(state, env, model)
            next_state, reward, done, info = env.step(action)
            out = model(torch.cat((torch.tensor(state[1:]), torch.tensor(action)), dim=0).float())
            # torch.set_grad_enabled(True)
            # state_action = torch.cat((torch.tensor(state[1:]), torch.tensor(action)), dim=-1).float()
            # next_state_pred = model(state_action)
            # optimizer.zero_grad()
            # loss = nn.MSELoss()(next_state_pred, torch.tensor(next_state - state).float())
            # loss.backward()
            # print("Loss", loss.item())
            # optimizer.step()
            # torch.set_grad_enabled(False)
            print(info["x_position"], "Predicted: ", out[0].item(), "Correct distance: ", (next_state[0] - state[0]).item(), "Margin: ", out[0].item() - (next_state[0] - state[0]).item())
            state = next_state
            env.render()

        # Update the dynamics model using the best action sequence
        # state_actions = np.hstack((states[:-1, best_index, :], best_action_seq))
        # next_states = states[1:, best_index, :]
        # next_states = torch.tensor(next_states, dtype=torch.float32)
        # optimizer.zero_grad()
        # pred_next_states = model(torch.tensor(state_actions, dtype=torch.float32))
        # loss = nn.MSELoss()(pred_next_states, next_states)
        # loss.backward()
        # optimizer.step()

    return None

th_count = 24
batch_size = 512
env_name = "HalfCheetah-v3"
data = RandomSymulation(EpisodeLengthDataFetch(0, 1000, env_name), [])
val_data = RandomSymulation(EpisodeLengthDataFetch(0, 1000, env_name), [])
# data.save("data.pkl")
# val_data.save("val_data.pkl")
data.load("data.pkl")
val_data.load("val_data.pkl")

data._transforms = [Transform(data.data, [1,9,10,11,12,13,14,15,16,17], 0, 0.001)]
print(len(data))
# standardizer = Standardizer(data.data)
sample = data[0]
def collate_fn(datas: typing.List[StepDescriptor]):
    return (
        torch.stack([torch.from_numpy(data.current_state) for data in datas]).float(),
        torch.stack([torch.from_numpy(data.next_state) for data in datas]).float(),
        torch.stack([torch.from_numpy(data.action) for data in datas]).float(),
        torch.stack([torch.tensor(data.reward) for data in datas]).float(),
        torch.stack([torch.tensor(data.done) for data in datas]).float(),
    )

dataloader = torch.utils.data.DataLoader(data, batch_size=(th_count>1)*(batch_size-1)+1, shuffle=th_count>1, num_workers=th_count, collate_fn = collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_data, batch_size=(th_count>1)*(batch_size-1)+1, shuffle=th_count>1, num_workers=th_count, collate_fn = collate_fn)
model = LinearNet([len(sample.current_state) - 1 + len(sample.action), 292, 256, len(sample.current_state)])

# model.standardizer = standardizer
fit(model, dataloader, val_dataloader, env_name, epochs=1000, lower_learning_period=3)
# loss = fit_epoch(model, dataloader, 0.001, True, 1)
# val_loss = fit_epoch(model, val_dataloader, None, False, 1)
# print(loss, val_loss)
# model(torch.Tensor(np.concatenate((sample.current_state, sample.action))))
model_dir_header = model.get_identifier()
chp_dir = os.path.join('checkpoints', model_dir_header)
checkpoint_name_path = os.path.join(chp_dir, '{}_checkpoints_final.pth'.format(env_name))
model = torch.load(checkpoint_name_path)
fit(model, dataloader, val_dataloader, env_name, epochs=1000, lower_learning_period=3)
random_shooting_mpc(model, gym.make(env_name, exclude_current_positions_from_observation=False), horizon=10, num_samples = 1000, num_iterations=1)
