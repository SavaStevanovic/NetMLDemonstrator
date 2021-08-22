
from collections import deque
import gym
from torch.utils.data import DataLoader
from model import networks
import torchvision.transforms as T
from data_loader.augmentation import ResizeTransform
from model import blocks
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from itertools import count
import random
import math
import matplotlib
import torch.nn as nn
from IPython import display
from tqdm import tqdm
from data_loader.rldata import RLDataset, Transition, rl_collate_fn
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from statistics import mean
import os
matplotlib.use('TkAgg')

env = gym.make('CartPole-v0')

resize = T.Compose(
    [
        T.ToPILImage(),
        T.ToTensor(),
        ResizeTransform((40, 30))
    ]
)


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.2)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen)


env.reset()

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

# Get screen size so that we can initialize layers correctly based on shape
# returned from AI gym. Typical dimensions at this point are close to 3x40x90
# which is the result of a clamped and down-scaled render buffer in get_screen()
init_screen = get_screen()
_, _, screen_height, screen_width = init_screen.shape

# Get number of actions from gym action space
backbone = networks.ResNetBackbone(inplanes = 64, block = blocks.BasicBlock, block_counts = [1, 1, 1])
net = networks.LinearNet(backbone = [backbone], output_size = env.action_space.n)
policy_net = networks.LinearNet(backbone = [backbone], output_size = env.action_space.n).cuda()
target_net = networks.LinearNet(backbone = [backbone], output_size = env.action_space.n).cuda()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = RLDataset(10000)
memory_dataloader = iter(DataLoader(memory, batch_size=BATCH_SIZE, shuffle=True, num_workers=1, pin_memory=False, collate_fn=rl_collate_fn))

steps_done = 0


def select_action(state):
    global steps_done
    if steps_done == 0:
        target_net(state.cuda())
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state.cuda()).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], dtype=torch.long).cuda()


episode_durations = deque([],maxlen=100)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    batch = memory.sample(BATCH_SIZE)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), dtype=torch.bool).cuda()
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(batch.state.cuda()).gather(1, batch.action.cuda())

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE).cuda()
    next_state_values[non_final_mask] = target_net(non_final_next_states.cuda()).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + batch.reward.cuda()

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

writer = SummaryWriter(os.path.join('logs', target_net.get_identifier()))
writer.add_image('Model view', get_screen().cpu().squeeze(0))

num_episodes = 5000
for i_episode in tqdm(range(num_episodes)):
    # Initialize the environment and state
    env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen
    for t in count():
        # Select and perform an action
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward]).cuda()

        # Observe new state
        last_screen = current_screen
        current_screen = get_screen()
        if not done:
            next_state = current_screen - last_screen
            # plt.imshow((next_state[0].permute(1, 2, 0)+1)/2)
            # plt.pause(0.001)  # pause a bit so that plots are updated
            # display.clear_output(wait=True)
            # display.display(plt.gcf())
        else:
            next_state = None

        # Store the transition in memory
        memory.push(Transition(state, action.cpu(), next_state, reward.cpu()))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            writer.add_scalars('Duration', {'current': episode_durations[-1], 'mean': mean(episode_durations)}, i_episode)
            break
    # Update the target network, copying all weights and biases in DQN
    if (i_episode + 1) % TARGET_UPDATE == 0:
        # save_image((current_screen - last_screen+1)[0]/2, 'img1.png')
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
env.close()
plt.ioff()
plt.show()