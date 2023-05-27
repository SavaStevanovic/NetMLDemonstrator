import typing
import gym
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper

class EnvironmentFactory:    
    def create_environment(self, env_name: str) -> typing.Tuple[gym.Env, gym.Env]:
        if "citylearn_" in env_name:
            base_train_env = StableBaselines3Wrapper(NormalizedObservationWrapper(CityLearnEnv(env_name, central_agent=True, simulation_start_time_step = 0, simulation_end_time_step=1344)))
            base_valid_env = StableBaselines3Wrapper(NormalizedObservationWrapper(CityLearnEnv(env_name, central_agent=True, simulation_start_time_step = 1344, simulation_end_time_step=2688, random_seed=42)))
        else:
            base_train_env = gym.make(env_name)
            base_valid_env = gym.make(env_name)
        return base_train_env, base_valid_env