import os
from stable_baselines3 import PPO, DQN


class AgentFactory:
    @staticmethod
    def create(agent_name: str, env, save_path: str = "./models"):
        os.makedirs(save_path, exist_ok=True)

        if agent_name.lower() == "ppo":
            return PPO("MlpPolicy", env, verbose=1)
        elif agent_name.lower() == "dqn":
            return DQN("MlpPolicy", env, verbose=1)
        else:
            raise ValueError(f"Unsupported agent: {agent_name}")
