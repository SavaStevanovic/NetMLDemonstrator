from time import sleep
import gym

env = gym.make('Ant-v2')

obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()  # sample random action
    obs, reward, done, info = env.step(action)
    sleep(0.02)
    env.render()

env.close()