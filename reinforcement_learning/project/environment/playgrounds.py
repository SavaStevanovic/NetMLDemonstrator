from environment.environments import ParameterEnv
import gym
from numpy import sin, cos
import abc
import numpy as np


class Playground(ParameterEnv):
    def __init__(self, env, visual):
        super().__init__(env, visual)
        self._duration = 0

    @property
    @abc.abstractmethod
    def metric(self):
        pass

    @property
    @abc.abstractmethod
    def max_duration(self):
        pass

    def step(self, action):
        self._duration += 1
        return super().step(action)

    def reset(self):
        self._duration = 0
        return super().reset()

    @property
    def duration(self):
        return self._duration


class CartPole(Playground):
    def __init__(self, name, visual):
        env = gym.make(name)
        super().__init__(env, visual)

    def step(self, action):
        new_state, reward, done, d = super().step(action)
        if done:
            reward = 0
        return new_state, reward, done, d

    @property
    def metric(self):
        return self.duration


class CartPoleV0(CartPole):
    def __init__(self, visual):
        super().__init__('CartPole-v0', visual=visual)

    @property
    def max_duration(self):
        return 200


class CartPoleV1(CartPole):
    def __init__(self, visual):
        super().__init__('CartPole-v1', visual=visual)

    @property
    def max_duration(self):
        return 500


class Acrobot(Playground):
    def __init__(self, name, visual):
        env = gym.make(name)
        super().__init__(env, visual)

    def step(self, action):
        new_state, reward, done, d = super().step(action-1)
        return new_state, reward, done, d

    @property
    def metric(self):
        return self.max_duration - self.duration


class AcrobotV1(Acrobot):
    def __init__(self, visual):
        super().__init__("Acrobot-v1", visual=visual)

    @property
    def max_duration(self):
        return 500


class Pendulum(Playground):
    def __init__(self, name, visual):
        env = gym.make(name)
        super().__init__(env, visual)
        self._value = 0

    def step(self, action):

        action = np.array([action])
        if self._duration == 0:
            self._value = 0
        new_state, reward, done, d = super().step(action)
        self._value += reward
        return new_state, float(reward), done, d

    @property
    def metric(self):
        return self._value


class PendulumV1(Pendulum):
    def __init__(self, visual):
        super().__init__("Pendulum-v1", visual=visual)

    @property
    def max_duration(self):
        return 200


class MountainCar(Playground):
    def __init__(self, name, visual):
        env = gym.make(name)
        super().__init__(env, visual)
        self._value = 0

    def step(self, action):
        if self._duration == 0:
            self._value = 0
        new_state, reward, done, d = super().step(action)
        if done:
            print(self._value)
        reward += float(abs(new_state[1])*13)
        self._value += reward
        return new_state, reward, done, d

    @property
    def metric(self):
        return self.max_duration - self.duration


class MountainCarV0(MountainCar):
    def __init__(self, visual):
        super().__init__("MountainCar-v0", visual=visual)

    @property
    def max_duration(self):
        return 200
