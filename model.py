import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, MODEL, CG_API_KEY


binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")


def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files")
    return files

def download_data_coingecko(token, training_days):
    files = download_coingecko_data(token, training_days, coingecko_data_path, CG_API_KEY)
    print(f"Downloaded {len(files)} new files")
    return files


def download_data(token, training_days, region, data_provider):
    if data_provider == "coingecko":
        return download_data_coingecko(token, int(training_days))
    elif data_provider == "binance":
        return download_data_binance(token, training_days, region)
    else:
        raise ValueError("Unsupported data provider")
    
def format_data(files, data_provider):
    if not files:
        print("Already up to date")
        return
    
    if data_provider == "binance":
        files = sorted([x for x in os.listdir(binance_data_path) if x.startswith(f"{TOKEN}USDT")])
    elif data_provider == "coingecko":
        files = sorted([x for x in os.listdir(coingecko_data_path) if x.endswith(".json")])

    # No files to process
    if len(files) == 0:
        return

    price_df = pd.DataFrame()
    if data_provider == "binance":
        for file in files:
            zip_file_path = os.path.join(binance_data_path, file)

            if not zip_file_path.endswith(".zip"):
                continue

            myzip = ZipFile(zip_file_path)
            with myzip.open(myzip.filelist[0]) as f:
                line = f.readline()
                header = 0 if line.decode("utf-8").startswith("open_time") else None
            df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
            df.columns = [
                "start_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "end_time",
                "volume_usd",
                "n_trades",
                "taker_volume",
                "taker_volume_usd",
            ]
            df.index = [pd.Timestamp(x + 1, unit="ms").to_datetime64() for x in df["end_time"]]
            df.index.name = "date"
            price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)
    elif data_provider == "coingecko":
        for file in files:
            with open(os.path.join(coingecko_data_path, file), "r") as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                df.columns = [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close"
                ]
                df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.drop(columns=["timestamp"], inplace=True)
                df.set_index("date", inplace=True)
                price_df = pd.concat([price_df, df])

            price_df.sort_index().to_csv(training_price_data_path)

def load_frame(frame, timeframe):
    frame = frame.copy()

    # Ensure 'date' is a string before conversion
    frame['date'] = frame['date'].astype(str)

    # Function to safely parse dates and filter invalid years
    def safe_parse_date(x):
        try:
            dt = pd.to_datetime(x, errors='coerce')  # Convert, setting invalid values to NaT
            if dt.year < 2000 or dt.year > 2100:  # Keep only reasonable years
                return pd.NaT  # Mark invalid dates as NaT
            return dt
        except Exception:
            return pd.NaT  # Mark non-parsable values as NaT

    # Apply date parsing safely
    frame['date'] = frame['date'].apply(safe_parse_date)

    # Print out problematic date values if any
    bad_dates = frame[frame['date'].isna()]
    if not bad_dates.empty:
        print("⚠️  Warning: Found invalid date values in dataset. These will be removed:")
        print(bad_dates)

    # Remove invalid date rows
    frame = frame.dropna(subset=['date'])

    return frame

def train_model(timeframe):
    # Load the price data
    price_data = pd.read_csv(training_price_data_path)
    df = load_frame(price_data, timeframe)

    print(df.tail())

    # Define target (next period's close price)
    y_train = df['close'].shift(-1).dropna().values
    
    # Define features (only numeric OHLC columns, exclude index)
    X_train = df[['open', 'high', 'low', 'close']].iloc[:-1].values

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")

    # Define the model
    if MODEL == "LinearRegression":
        model = LinearRegression()
    elif MODEL == "SVR":
        model = SVR()
    elif MODEL == "KernelRidge":
        model = KernelRidge()
    elif MODEL == "BayesianRidge":
        model = BayesianRidge()
    else:
        raise ValueError("Unsupported model")
    
    # Train the model
    model.fit(X_train, y_train)

    # Create the model's parent directory if it doesn't exist
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

    # Save the trained model to a file
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Trained model saved to {model_file_path}")

def get_inference(token, timeframe, region, data_provider):
    """Load model and predict current price."""
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)

    # Get current price
    if data_provider == "coingecko":
        X_new = load_frame(download_coingecko_current_day_data(token, CG_API_KEY), timeframe)
    else:
        X_new = load_frame(download_binance_current_day_data(f"{token}USDT", region), timeframe)
    
    print(X_new.tail())
    print(X_new.shape)

    # Convert to numeric array, excluding the index
    X_new_numeric = X_new[['open', 'high', 'low', 'close']].values
    
    current_price_pred = loaded_model.predict(X_new_numeric)
    return current_price_pred[0]
