import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class InventoryEnv(gym.Env):
    """
    Inventory Management Environment using Gymnasium API.

    Observation: [inventory_level, demand, day_of_week, promo_flag]
    Action: discrete order quantity in [0, max_order]
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, data: pd.DataFrame, max_inventory: int | None = None, max_order: int | None = None,
                 holding_cost: float = 1.0, stockout_penalty: float = 5.0):
        super().__init__()
        self.data = data.reset_index(drop=True)
        if self.data.empty:
            raise ValueError("Input data is empty. Provide processed data with a 'demand' column.")

        # dynamic max inventory/order
        self.max_inventory = max(10, int(self.data["demand"].mean() * 3)) if max_inventory is None else max_inventory
        self.max_order = max(1, int(self.data["demand"].quantile(0.95))) if max_order is None else max_order

        self.holding_cost = holding_cost
        self.stockout_penalty = stockout_penalty

        # Action space
        self.action_space = spaces.Discrete(self.max_order + 1)

        # Observation space (10-dimensional after 7-day demand window)
        obs_dim = 1 + 7 + 1 + 1
        high = np.array([self.max_inventory] + [np.finfo(np.float32).max] * 7 + [6, 1], dtype=np.float32)
        low = np.array([0] * obs_dim, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.current_step = 0
        self.inventory_level = 0
        self.total_reward = 0.0
        self.last_order_qty = 0  # for smoothness penalty

    def _get_state(self):
        idx = min(self.current_step, len(self.data) - 1)
        start_idx = max(0, idx - 6)  # last 7 days
        recent_demand = self.data.iloc[start_idx:idx + 1]["demand"].tolist()
        recent_demand = [0.0] * (7 - len(recent_demand)) + recent_demand  # pad if fewer than 7

        row = self.data.iloc[idx]
        promo_flag = 1 if "event_name_1" in row and pd.notna(row["event_name_1"]) else 0
        day_of_week = int(row["day_of_week"]) if "day_of_week" in row else 0

        obs = np.array([self.inventory_level] + recent_demand + [day_of_week, promo_flag], dtype=np.float32)
        return obs

    def step(self, action):
        """
        Returns: obs, reward, terminated, truncated, info
        """
        # clamp action
        order_qty = int(action)
        order_qty = max(0, min(order_qty, self.max_order))

        # current row corresponds to the demand we are serving this step
        row = self.data.iloc[self.current_step]
        demand = float(row["demand"]) if "demand" in row.index else 0.0
        sell_price = float(row.get("sell_price", 1.0)) if "sell_price" in row.index else 1.0

        # Receive order (instant delivery for simplicity)
        self.inventory_level = min(self.inventory_level + order_qty, self.max_inventory)

        # Fulfill demand
        sales = min(demand, self.inventory_level)
        self.inventory_level -= sales

        # --- Hybrid Reward ---
        alpha_service = 1.0      # reward weight for serving demand
        beta_hold = 0.05         # small penalty per unit inventory held
        gamma_stockout = 1.0     # penalty per unit unmet demand
        scale = max(1.0, self.data["demand"].mean())  # normalization factor

        served_units = sales
        unmet_units = max(0.0, demand - sales)

        base_reward = (alpha_service * (served_units * sell_price)
                       - beta_hold * self.inventory_level
                       - gamma_stockout * unmet_units)

        reward = base_reward / scale

        # smooth actions: penalty for drastic order changes
        reward -= 0.01 * abs(order_qty - self.last_order_qty)
        self.last_order_qty = order_qty

        # squash to keep values stable
        reward = float(np.tanh(reward) * 5.0)

        self.total_reward += reward

        # advance time
        self.current_step += 1

        # terminated if we've exhausted the dataset
        terminated = self.current_step >= len(self.data)
        truncated = False

        # next observation
        obs = self._get_state()
        info = {"inventory": self.inventory_level, "step": self.current_step}
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """
        Gymnasium-style reset accepting seed and options.
        Returns: observation, info
        """
        if seed is not None:
            np.random.seed(seed)

        self.current_step = 0
        self.inventory_level = 0
        self.total_reward = 0.0
        self.last_order_qty = 0  # reset smoothness tracker

        obs = self._get_state()
        info = {}
        return obs, info

    def render(self, mode="human"):
        print(f"Step={self.current_step}, Inventory={self.inventory_level}, TotalReward={self.total_reward:.2f}")
