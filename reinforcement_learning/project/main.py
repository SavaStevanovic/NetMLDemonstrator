# import torch
# from model import blocks
# from model import networks
# from data_loader.unified_dataloader import UnifiedKeypointDataloader
# from model_fitting.train import fit
# import os


# th_count = 24

# dataloader = UnifiedKeypointDataloader(batch_size = 6, th_count=th_count)
# backbone = networks.VGGNetBackbone(inplanes = 64, block_counts = [2, 2, 4, 2])
# net = networks.OpenPoseNet([backbone], 4, 1, blocks.PoseCNNStage, 10, len(dataloader.trainloader.skeleton)*2, len(dataloader.trainloader.parts)+1, dataloader.trainloader.skeleton, dataloader.trainloader.parts)
# # net = networks.CocoPoseNet()

# fit(net, 
#     dataloader.trainloader, 
#     dataloader.validationloader, 
#     postprocessing = dataloader.postprocessing, 
#     epochs = 1000, 
#     lower_learning_period = 3
# )       

import gym
env = gym.make('CartPole-v0')
print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()