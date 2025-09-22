import os
import argparse
from environment import InventoryEnv
from agent import AgentFactory
from src.data_processing import prepare_m5_data



def train(agent_name="dqn", timesteps=10000, save_path="../models/model.zip"):
    DATA_DIRECTORY = "../data/raw/m5-forecasting-accuracy"
    df = prepare_m5_data(DATA_DIRECTORY, item_id="FOODS_1_001", store_id="CA_1")

    env = InventoryEnv(df)
    model = AgentFactory.create(agent_name, env)
    model.learn(total_timesteps=timesteps)
    model.save(save_path)
    print(f"Model saved at {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=str, default="dqn", help="RL agent: ppo or dqn")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    train(agent_name=args.agent, timesteps=args.timesteps)
