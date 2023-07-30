import gym
from stable_baselines3.sac import SAC
from environment.model_environment import ModelEnv
from model.action_space_generator import RandomSpaceProducer
from model.model_predictive_contol import MPC, PolicyModel, RandomModel
from stable_baselines3.sac import SAC
from model.trainer import SamplingModel

class ModelFactory:   
    def __init__(
            self, 
            environment_model: SamplingModel, 
            horizon = 10, 
            sample_count=1000, 
            training_steps = 100_000
        ) -> None:
        self._training_steps = training_steps
        self._environment_model = environment_model
        self._horizon = horizon
        self._sample_count = sample_count

    def create_model(self, model_strat: str,
            environment: gym.Env) -> PolicyModel:
        policy_model = None
        if "RL_real" in model_strat:
            policy_model = SAC('MlpPolicy', environment)
            policy_model.learn(total_timesteps=self._training_steps, callback=[], progress_bar=True)
        if "RL_modelenv" in model_strat:
            train_env = ModelEnv(self._environment_model.produce_model().eval(), environment.observation_space, environment.action_space)
            policy_model = SAC('MlpPolicy', train_env)
            policy_model.learn(total_timesteps=self._training_steps, callback=[], progress_bar=True)
        if model_strat == "Random":
            policy_model = RandomModel(environment.action_space)     
        if "MPC" in model_strat:
            critic = None if policy_model is None else policy_model.critic.cpu().eval()
            print("Starting MPC with critic", str(critic))
            policy_model = MPC(RandomSpaceProducer(self._horizon, self._sample_count), self._environment_model.produce_model(), environment.action_space, environment.observation_space, critic)
        return policy_model