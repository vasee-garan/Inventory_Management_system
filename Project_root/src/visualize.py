import matplotlib.pyplot as plt
import pandas as pd

def plot_inventory_dynamics(history, window=20):
    """
    Plots inventory, demand, reward, and reorder markers.
    Opens two figure windows:
      1. Full dataset (3 plots)
      2. Fixed zoom: July 2012 – Jan 2013 (3 plots)
    """
    df = pd.DataFrame(history)

    # Ensure date column exists
    if "date" not in df.columns:
        start_date = pd.to_datetime("2011-01-29")
        df["date"] = start_date + pd.to_timedelta(df["step"], unit="D")

    # Rolling smoothing
    #df["demand"] = df["demand"].fillna(0)
    df["demand"] = df["demand"].fillna(method="ffill").fillna(0)
    df["reward"] = df["reward"].fillna(method="ffill").fillna(0)


    df["demand_smooth"] = df["demand"].rolling(window=window, min_periods=1).mean()
    df["reward_smooth"] = df["reward"].rolling(window=window, min_periods=1).mean()

    # Reorder threshold & flag
    reorder_threshold = 2
    df["reorder_flag"] = (df["inventory"] <= reorder_threshold).astype(int)

    plt.style.use("seaborn-v0_8-whitegrid")

    # -----------------------
    # Window 1: Full dataset
    # -----------------------
    fig1, axs1 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Inventory
    axs1[0].step(df["date"], df["inventory"], where="post",
                 color="royalblue", linewidth=1.5, label="Inventory")
    axs1[0].axhline(y=reorder_threshold, color="red", linestyle="--", linewidth=1, label="Reorder Threshold")
    axs1[0].scatter(df.loc[df["reorder_flag"] == 1, "date"],
                    df.loc[df["reorder_flag"] == 1, "inventory"],
                    color="red", marker="v", s=60, label="Reorder Trigger")
    axs1[0].set_ylabel("Inventory")
    axs1[0].legend()

    # Demand
    axs1[1].plot(df["date"], df["demand"], color="darkorange", alpha=0.3, linewidth=0.8, label="Demand (raw)")
    axs1[1].plot(df["date"], df["demand_smooth"], color="darkorange", linestyle="--", linewidth=2,
                 label=f"Smoothed ({window})")
    axs1[1].set_ylabel("Demand")
    axs1[1].legend()

    # Reward
    axs1[2].plot(df["date"], df["reward"], color="forestgreen", alpha=0.3, linewidth=0.8, label="Reward (raw)")
    axs1[2].plot(df["date"], df["reward_smooth"], color="forestgreen", linestyle="--", linewidth=2,
                 label=f"Smoothed ({window})")
    axs1[2].set_ylabel("Reward")
    axs1[2].legend()
    axs1[2].set_xlabel("Date")

    fig1.suptitle("Full Dataset", fontsize=14)
    plt.tight_layout()
    plt.show(block=False)

    # -----------------------
    # Window 2: July 2012 – Jan 2013
    # -----------------------
    zoom_start = pd.to_datetime("2012-07-01")
    zoom_end = pd.to_datetime("2013-01-31")
    df_zoom = df[(df["date"] >= zoom_start) & (df["date"] <= zoom_end)]

    fig2, axs2 = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    # Inventory
    axs2[0].step(df_zoom["date"], df_zoom["inventory"], where="post",
                 color="royalblue", linewidth=1.5, label="Inventory (Jul 2012–Jan 2013)")
    axs2[0].axhline(y=reorder_threshold, color="red", linestyle="--", linewidth=1)
    axs2[0].scatter(df_zoom.loc[df_zoom["reorder_flag"] == 1, "date"],
                    df_zoom.loc[df_zoom["reorder_flag"] == 1, "inventory"],
                    color="red", marker="v", s=60)
    axs2[0].set_ylabel("Inventory")
    axs2[0].legend()

    # Demand
    axs2[1].plot(df_zoom["date"], df_zoom["demand"], color="darkorange", alpha=0.3, linewidth=0.8, label="Demand (raw)")
    axs2[1].plot(df_zoom["date"], df_zoom["demand_smooth"], color="darkorange", linestyle="--", linewidth=2,
                 label=f"Smoothed ({window})")
    axs2[1].set_ylabel("Demand")
    axs2[1].legend()

    # Reward
    axs2[2].plot(df_zoom["date"], df_zoom["reward"], color="forestgreen", alpha=0.3, linewidth=0.8, label="Reward (raw)")
    axs2[2].plot(df_zoom["date"], df_zoom["reward_smooth"], color="forestgreen", linestyle="--", linewidth=2,
                 label=f"Smoothed ({window})")
    axs2[2].set_ylabel("Reward")
    axs2[2].legend()
    axs2[2].set_xlabel("Date")

    fig2.suptitle("Plotting Details from July 2012 to Jan 2013", fontsize=14)
    plt.tight_layout()
    plt.show()
