import os
import argparse
import json
import numpy as np
from src.environment import InventoryEnv
from src.agent import AgentFactory
from src.data_processing import prepare_m5_data
from src.visualize import plot_inventory_dynamics
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# -------------------------
# Dataset path finder
# -------------------------
CANDIDATE_PATHS = [
    "./data/raw/m5-forecasting-accuracy",                      
    r"D:\Trail\Project_root\data\raw\m5-forecasting-accuracy", 
    r"D:\Datasets\m5-forecasting-accuracy",                    
]

def find_data_directory():
    """Find the first path that contains the required M5 dataset files."""
    required_files = ["sales_train_validation.csv", "calendar.csv", "sell_prices.csv"]

    for path in CANDIDATE_PATHS:
        if all(os.path.exists(os.path.join(path, f)) for f in required_files):
            print(f"📂 Using dataset from: {path}")
            return path
    print("❌ Could not find M5 dataset in any candidate path.")
    print("Please make sure the files exist in one of these locations:")
    for path in CANDIDATE_PATHS:
        print("   -", path)
    return None


def run_experiment(agent="ppo", timesteps=50000, episodes=3, item_id="FOODS_1_001", store_id="CA_1"):
    # -------------------------
    # Step 1: Prepare Data
    # -------------------------
    import random
    import torch
    
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    
    DATA_DIRECTORY = find_data_directory()
    if DATA_DIRECTORY is None:
        return  

    df = prepare_m5_data(DATA_DIRECTORY, item_id=item_id, store_id=store_id)

    if df is None:
        print("❌ No data available. Check item_id/store_id filter.")
        return

    # -------------------------
    # Step 2: Setup Environment
    # -------------------------
    env = InventoryEnv(df)
    vec_env = DummyVecEnv([lambda: InventoryEnv(df)])  

    # -------------------------
    # Step 3: Train Agent
    # -------------------------
    model = AgentFactory.create(agent, vec_env)
    print(f"🚀 Training {agent.upper()} for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)

    os.makedirs("./models", exist_ok=True)
    model_path = f"./models/{agent}_inventory_model"
    model.save(model_path)
    print(f"✅ Model saved at {model_path}.zip")

    # -------------------------
    # Step 4: Evaluate Agent
    # -------------------------
    print(f"📊 Evaluating {agent.upper()} model for {episodes} episodes...")
    if agent == "ppo":
        trained_model = PPO.load(model_path, env=vec_env)
    else:
        trained_model = DQN.load(model_path, env=vec_env)

    rewards = []
    service_levels = []
    profits = []   # ➕ store profits per episode
    history = []

    for ep in range(episodes):
        obs = vec_env.reset()
        done = False
        total_reward, served, demand = 0, 0, 0

        while not done:
            action, _ = trained_model.predict(obs)
            obs, reward, done, info = vec_env.step(action)

            reward = reward[0]  # vec_env returns arrays
            demand_step = sum(obs[0][1:8])
            inv_level = vec_env.envs[0].inventory_level

            total_reward += reward
            served += min(demand_step, inv_level + action[0])
            demand += demand_step

            history.append({
                "episode": ep,
                "step": vec_env.envs[0].current_step,
                "inventory": inv_level,
                "demand": demand_step,
                "reward": reward
            })

        rewards.append(total_reward)
        service_levels.append(served / (demand + 1e-6))
        profits.append(total_reward)   # ➕ profit == accumulated reward

    avg_reward = float(np.mean(rewards))
    avg_service = float(np.mean(service_levels))
    avg_profit = float(np.mean(profits))  # ➕ average profit

    print("🔎 Evaluation Results:")
    print(f"   ➤ Average Reward: {avg_reward:.2f}")
    print(f"   ➤ Average Service Level: {avg_service:.2f}")
    print(f"   ➤ Average Profit: {avg_profit:.2f}")   # ➕ display profit

    # -------------------------
    # Step 5: Save Metrics
    # -------------------------
    os.makedirs("./reports", exist_ok=True)
    metrics_path = "./reports/metrics.json"

    results = {
        "agent": agent,
        "timesteps": int(timesteps),
        "episodes": int(episodes),
        "item_id": item_id,
        "store_id": store_id,
        "avg_reward": float(avg_reward),
        "avg_service_level": float(avg_service),
        "avg_profit": float(avg_profit),          # ➕ save profit
        "rewards": [float(r) for r in rewards],
        "profits": [float(p) for p in profits],   # ➕ per-episode profit
        "service_levels": [float(s) for s in service_levels]
    }

    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"📂 Metrics saved to {metrics_path}")

    # -------------------------
    # Step 6: Visualization
    # -------------------------
    print("📈 Plotting inventory dynamics...")
    plot_inventory_dynamics(history)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL experiment for Inventory Management")
    parser.add_argument("--agent", type=str, default="dqn", choices=["ppo", "dqn"], help="RL agent type")
    parser.add_argument("--timesteps", type=int, default=5000, help="Training timesteps")
    parser.add_argument("--episodes", type=int, default=3, help="Evaluation episodes")
    parser.add_argument("--item_id", type=str, default="FOODS_1_001", help="Filter by item_id")
    parser.add_argument("--store_id", type=str, default="CA_1", help="Filter by store_id")
    args = parser.parse_args()

    run_experiment(
        agent=args.agent,
        timesteps=args.timesteps,
        episodes=args.episodes,
        item_id=args.item_id,
        store_id=args.store_id
    )
