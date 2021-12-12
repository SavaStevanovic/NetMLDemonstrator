from environment.environments import ParameterEnv
import gym


class Playground(ParameterEnv):
    def __init__(self, visual):
        pass


class CartPole(ParameterEnv):
    def __init__(self, name, visual):
        env = gym.make(name)
        super().__init__(env, visual)

    def step(self, action):
        new_state, reward, done, d = super().step(action)
        if done:
            reward = -10
        return new_state, reward, done, d


class CartPoleV0(CartPole):
    def __init__(self, visual):
        super().__init__('CartPole-v0', visual=visual)


class CartPoleV1(CartPole):
    def __init__(self, visual):
        super().__init__('CartPole-v1', visual=visual)
