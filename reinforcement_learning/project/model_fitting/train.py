from collections import deque
import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from model_fitting.configuration import TrainingConfiguration
from itertools import count
import torch.nn as nn
from data_loader.rldata import RLDataset, Transition
import random
from model_fitting.train import TrainingConfiguration
import math
from statistics import mean
from torchsummary import summary

def select_action(policy_net, state, size, config: TrainingConfiguration):
    sample = random.random()
    eps_threshold = config.EPS_END + (config.EPS_START - config.EPS_END) * \
        math.exp(-1. * config.steps_done / config.EPS_DECAY)
    config.steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.cuda()).max(1)[1].view(1, 1)
    else:
        # print(sample, eps_threshold)
        return torch.tensor([[random.randrange(size)]], dtype=torch.long).cuda()

def optimize_model(target_net, policy_net, criterion, batch, config: TrainingConfiguration):

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(batch.state.cuda()).gather(1, batch.action.cuda())

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = target_net(batch.next_state.cuda()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * config.GAMMA) + batch.reward.cuda()

    # Compute Huber loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # print(loss.item())
    return loss

def fit(target_net, policy_net, visual_env):
    episode_durations = deque([],maxlen=100)
    model_dir_header = target_net.get_identifier()
    chp_dir = os.path.join('checkpoints', model_dir_header)
    checkpoint_memo_path = os.path.join(chp_dir, 'mem_checkpoints.pth')
    checkpoint_name_path = os.path.join(chp_dir, 'checkpoints.pth')
    checkpoint_conf_path = os.path.join(chp_dir, 'configuration.json')
    memory = RLDataset(2000)
    train_config = TrainingConfiguration()
    writer = SummaryWriter(os.path.join('logs', target_net.get_identifier()))
    screen, state = visual_env.get_screen()
    target_net(state.cuda())
    policy_net(state.cuda())
    optimizer = torch.optim.Adam(policy_net.parameters(), train_config.learning_rate)
    writer.add_image('Model view', screen)
    if os.path.exists(chp_dir):
        checkpoint = torch.load(checkpoint_name_path)
        target_net.load_state_dict(checkpoint["model_state"])
        policy_net.load_state_dict(checkpoint["model_state"])
        # optimizer.load_state_dict(checkpoint["optimizer"])
        policy_net.train()
        target_net.eval()
        memory.load(checkpoint_memo_path)
        train_config.load(checkpoint_conf_path)
    _, state = visual_env.get_screen()
    target_net(state.cuda())
    summary(target_net, tuple(state.shape[1:]))
    criterion = nn.SmoothL1Loss()
    for i_episode in tqdm(range(train_config.epoch, train_config.EPOCHS)):
        # Initialize the environment and state
        visual_env.env.reset()
        for tttttt in count():
            _, state = visual_env.get_screen()
            # Select and perform an action
            if train_config.steps_done == 0:
                target_net(state.cuda())
            action = select_action(policy_net, state, visual_env.env.action_space.n, train_config)
            new_state, reward, done, _ = visual_env.env.step(action.item())
            reward = torch.tensor([reward]).cuda()
            if done:
                reward = -torch.ones_like(reward)

            # Store the transition in memory
            memory.push(Transition(state, action.cpu(), torch.Tensor(new_state).unsqueeze(0), reward.cpu()))

            # Perform one step of the optimization (on the policy network)
            if len(memory) >= train_config.BATCH_SIZE:
                batch = memory.sample(train_config.BATCH_SIZE)
                loss = optimize_model(target_net, policy_net, criterion, batch, train_config)
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()
            if done:
                episode_durations.append(tttttt + 1)
                writer.add_scalars('Duration', {'current': episode_durations[-1], 'mean': mean(episode_durations)}, i_episode)
                break
        # Update the target network, copying all weights and biases in DQN
        if (i_episode + 1) % train_config.TARGET_UPDATE == 0:
            # save_image((current_screen - last_screen+1)[0]/2, 'img1.png')
            target_net.load_state_dict(policy_net.state_dict())

        metric = mean(episode_durations)
        os.makedirs((chp_dir), exist_ok=True)
        if train_config.best_metric < metric and len(episode_durations)>=episode_durations.maxlen/2:
            train_config.iteration_age = 0
            train_config.best_metric = metric
            print('Epoch {}. Saving model with metric: {}'.format(i_episode, metric))
            torch.save(target_net, checkpoint_name_path.replace('.pth', '_final.pth'))
        else:
            train_config.iteration_age+=1
            print('Epoch {} metric: {}'.format(i_episode, metric))
        if train_config.iteration_age==train_config.LOWER_LEARNING_PERIOD:
            train_config.learning_rate*=0.5
            train_config.iteration_age=0
            optimizer = torch.optim.RMSprop(target_net.parameters(), train_config.learning_rate)
            print("Learning rate lowered to {}".format(train_config.learning_rate))
        train_config.epoch = i_episode+1
        train_config.save(checkpoint_conf_path)
        if i_episode % 10 ==0:
            memory.save(checkpoint_memo_path)
        checkpoint = {
            'model_state': target_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'model': target_net
        }
        torch.save(checkpoint, checkpoint_name_path)
        torch.save(target_net.state_dict(), checkpoint_name_path.replace('.pth', '_final_state_dict.pth'))
    print('Complete')