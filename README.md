# Inventory_Management_system
Got it ✅
Here’s a professional **README.md** you can use for your GitHub repository:

---

# 🏪 Inventory Management with Reinforcement Learning

This project implements an **Inventory Management System** using **Reinforcement Learning (RL)**. It is designed around the **M5 Forecasting Dataset**, applying RL algorithms to optimize order decisions while balancing service levels, holding costs, and stockout penalties.

---

## 📌 Project Structure

```
Project_root/
│── data/                     # Raw & processed datasets (M5 dataset)
│── models/                   # Saved RL models (PPO, DQN)
│── reports/                  # Training & evaluation metrics, plots, reports
│── src/                      # Source code
│   ├── environment.py        # Custom InventoryEnv (Gym environment)
│   ├── agent.py              # AgentFactory for PPO/DQN
│   ├── data_processing.py    # Data preparation (M5 dataset)
│   ├── visualize.py          # Visualization utilities
│── run_experiment.py         # Main entry point for training/evaluation
│── requirements.txt          # Dependencies
│── README.md                 # Project documentation
```

---

## ⚙️ Installation

1. Clone this repository:

```bash
git clone https://github.com/your-username/inventory-rl.git
cd inventory-rl
```

2. Create & activate virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

The project uses the **M5 Forecasting - Accuracy** dataset. Download it from [Kaggle](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data) and place it in one of the following paths:

* `./data/raw/m5-forecasting-accuracy/`
* `D:\Trail\Project_root\data\raw\m5-forecasting-accuracy`
* `D:\Datasets\m5-forecasting-accuracy`

Required files:

* `sales_train_validation.csv`
* `calendar.csv`
* `sell_prices.csv`

---

## 🚀 Usage

Run experiments using PPO or DQN agents:

```bash
# Train & evaluate DQN
python run_experiment.py --agent dqn --timesteps 5000 --episodes 3

# Train & evaluate PPO
python run_experiment.py --agent ppo --timesteps 10000 --episodes 5
```

---

## 📈 Outputs

* **Metrics**: Saved to `./reports/metrics.json`
* **Models**: Saved to `./models/{agent}_inventory_model.zip`
* **Visualizations**: Time-series plots of inventory, demand, and rewards

Example plot:

* Inventory levels with reorder thresholds
* Smoothed demand & reward curves
* Reorder event markers

---

## 🧠 Reinforcement Learning Setup

* **State**: `[inventory_level, demand, day_of_week, promo_flag]`
* **Action**: Discrete order quantity (0 to max\_order)
* **Reward**: `revenue - holding_cost - stockout_penalty`
* **Algorithms**:

  * PPO (Proximal Policy Optimization)
  * DQN (Deep Q-Networks)

---

## ⚡ Challenges

* Balancing **stockouts vs. overstocking**
* Handling **non-stationary demand** patterns
* Ensuring **training stability** for PPO/DQN
* Tuning hyperparameters for reward shaping

---

## 🛠 Tech Stack

* **Python 3.9+**
* **Gymnasium** (custom RL environment)
* **Stable-Baselines3** (PPO, DQN)
* **Pandas, NumPy** (data processing)
* **Matplotlib** (visualization)

---

## 👨‍💻 Author

Developed by **VASEEGARAN**
Feel free to ⭐ this repo if you find it useful!

