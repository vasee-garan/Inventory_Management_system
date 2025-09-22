import pandas as pd
import numpy as np

def prepare_m5_data(data_path: str, item_id: str = None, store_id: str = None) -> pd.DataFrame:
    """
    Loads, cleans, and prepares the M5 Forecasting Accuracy dataset for RL environment simulation.
    This version includes an optional filter to work with a subset of the data.

    Args:
        data_path (str): The local file path to the raw M5 dataset.
        item_id (str): Optional. The item_id to filter the sales data by.
        store_id (str): Optional. The store_id to filter the sales data by.

    Returns:
        pd.DataFrame: A preprocessed DataFrame ready for use in the simulation environment.
    """
    print("Step 1: Data Preparation - Beginning the preprocessing pipeline.")

    # Data loading
    try:
        sales_df = pd.read_csv(f"{data_path}/sales_train_validation.csv")
        calendar_df = pd.read_csv(f"{data_path}/calendar.csv")
        sell_prices_df = pd.read_csv(f"{data_path}/sell_prices.csv")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure the M5 dataset files are in the specified directory.")
        return None

    # Filter the data if an item_id or store_id is provided.
    # This is the key change to handle the memory issue with the full dataset.
    if item_id:
        sales_df = sales_df[sales_df['item_id'] == item_id]
        sell_prices_df = sell_prices_df[sell_prices_df['item_id'] == item_id]
    if store_id:
        sales_df = sales_df[sales_df['store_id'] == store_id]
        sell_prices_df = sell_prices_df[sell_prices_df['store_id'] == store_id]
    
    if sales_df.empty:
        print("Warning: The filter resulted in an empty dataset. Please check your item_id or store_id.")
        return None

    print("Data loaded successfully. Merging datasets...")

    # Merge sales, calendar, and price data into a single DataFrame
    sales_melt = sales_df.melt(id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],
                                var_name='d', value_name='demand')
    
    merged_df = sales_melt.merge(calendar_df, on='d', how='left')
    merged_df = merged_df.merge(sell_prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    print("Datasets merged. Now proceeding with cleaning and feature engineering...")

    # Data cleaning and handling of missing values
    merged_df['sell_price'] = merged_df['sell_price'].fillna(0)

    # Feature Engineering
    merged_df['date'] = pd.to_datetime(merged_df['date'])
    merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['year'] = merged_df['date'].dt.year

    # Handling categorical data
    merged_df = pd.get_dummies(merged_df, columns=['state_id', 'store_id', 'cat_id', 'dept_id'])

    # Optional: Aggregating features to provide a rolling context to the agent
    merged_df['demand_rolling_7d'] = merged_df.groupby('id')['demand'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    merged_df['price_rolling_30d'] = merged_df.groupby('id')['sell_price'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())

    print("Feature engineering complete. Data is ready for environment simulation.")
    
    return merged_df

# Example Usage:
if __name__ == "__main__":
    # The absolute path to your data directory.
    # THIS IS THE LINE THAT WAS UPDATED to match your file structure.
    DATA_DIRECTORY = "D:/Trail/Project_root/data/raw/m5-forecasting-accuracy"

    # To run the script without memory issues, filter by a specific item or store.
    sample_item_id = 'FOODS_1_001'
    sample_store_id = 'CA_1'

    processed_data = prepare_m5_data(DATA_DIRECTORY, item_id=sample_item_id, store_id=sample_store_id)
    
    if processed_data is not None:
        print("\nProcessed Data Head:")
        print(processed_data.head())
        print("\nProcessed Data Info:")
        processed_data.info()