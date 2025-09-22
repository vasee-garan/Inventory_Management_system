import numpy as np
from stable_baselines3 import PPO, DQN
from environment import InventoryEnv
from src.data_processing import prepare_m5_data



def evaluate(model_path, agent="ppo", episodes=5):
    DATA_DIRECTORY = "../data/raw/m5-forecasting-accuracy"
    df = prepare_m5_data(DATA_DIRECTORY, item_id="FOODS_1_001", store_id="CA_1")

    env = InventoryEnv(df)

    if agent == "ppo":
        model = PPO.load(model_path, env=env)
    else:
        model = DQN.load(model_path, env=env)

    rewards = []
    service_levels = []

    for ep in range(episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        served, demand = 0, 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
            served += min(obs[1], env.inventory_level + action)
            demand += obs[1]

        rewards.append(total_reward)
        service_levels.append(served / (demand + 1e-6))

    print("Evaluation Results:")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Average Service Level: {np.mean(service_levels):.2f}")
