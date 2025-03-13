import json
import os
import pickle
from zipfile import ZipFile
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, make_scorer
from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from updater import download_binance_daily_data, download_binance_current_day_data, download_coingecko_data, download_coingecko_current_day_data
from config import data_base_path, model_file_path, TOKEN, TIMEFRAME, TRAINING_DAYS, REGION, DATA_PROVIDER, MODEL

binance_data_path = os.path.join(data_base_path, "binance")
coingecko_data_path = os.path.join(data_base_path, "coingecko")
training_price_data_path = os.path.join(data_base_path, "price_data.csv")
scaler_file_path = os.path.join(data_base_path, "scaler.pkl")

def download_data_binance(token, training_days, region):
    files = download_binance_daily_data(f"{token}USDT", training_days, region, binance_data_path)
    print(f"Downloaded {len(files)} new files for {token}USDT")
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

def format_data(files_btc, files_eth, data_provider):
    print(f"Raw files for BTCUSDT: {files_btc[:5]}")
    print(f"Raw files for ETHUSDT: {files_eth[:5]}")
    print(f"Files for BTCUSDT: {len(files_btc)}, Files for ETHUSDT: {len(files_eth)}")
    if not files_btc or not files_eth:
        print("No files provided for BTCUSDT or ETHUSDT, exiting format_data")
        return
    
    if data_provider == "binance":
        files_btc = sorted([f for f in files_btc if "BTCUSDT" in os.path.basename(f) and f.endswith(".zip")])
        files_eth = sorted([f for f in files_eth if "ETHUSDT" in os.path.basename(f) and f.endswith(".zip")])
        print(f"Filtered BTCUSDT files: {files_btc[:5]}")
        print(f"Filtered ETHUSDT files: {files_eth[:5]}")
    elif data_provider == "coingecko":
        files_btc = sorted([x for x in files_btc if x.endswith(".json")])
        files_eth = sorted([x for x in files_eth if x.endswith(".json")])

    if len(files_btc) == 0 or len(files_eth) == 0:
        print("No valid files to process for BTCUSDT or ETHUSDT after filtering")
        return

    price_df_btc = pd.DataFrame()
    price_df_eth = pd.DataFrame()

    if data_provider == "binance":
        for file in files_btc:
            zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
            if not os.path.exists(zip_file_path):
                print(f"File not found: {zip_file_path}")
                continue
            try:
                myzip = ZipFile(zip_file_path)
                with myzip.open(myzip.filelist[0]) as f:
                    line = f.readline()
                    header = 0 if line.decode("utf-8").startswith("open_time") else None
                df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
                df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"]
                max_time = df["end_time"].max()
                print(f"Sample end_time values from {file}: {df['end_time'].head().tolist()}")
                if max_time > 1e15:  # Nanoseconds
                    df["date"] = pd.to_datetime(df["end_time"], unit="ns")
                elif max_time > 1e12:  # Microseconds
                    df["date"] = pd.to_datetime(df["end_time"], unit="us")
                else:  # Milliseconds
                    df["date"] = pd.to_datetime(df["end_time"], unit="ms")
                df.set_index("date", inplace=True)
                print(f"Processed {file} with {len(df)} rows")
                price_df_btc = pd.concat([price_df_btc, df])
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue

        for file in files_eth:
            zip_file_path = os.path.join(binance_data_path, os.path.basename(file))
            if not os.path.exists(zip_file_path):
                print(f"File not found: {zip_file_path}")
                continue
            try:
                myzip = ZipFile(zip_file_path)
                with myzip.open(myzip.filelist[0]) as f:
                    line = f.readline()
                    header = 0 if line.decode("utf-8").startswith("open_time") else None
                df = pd.read_csv(myzip.open(myzip.filelist[0]), header=header).iloc[:, :11]
                df.columns = ["start_time", "open", "high", "low", "close", "volume", "end_time", "volume_usd", "n_trades", "taker_volume", "taker_volume_usd"]
                max_time = df["end_time"].max()
                print(f"Sample end_time values from {file}: {df['end_time'].head().tolist()}")
                if max_time > 1e15:  # Nanoseconds
                    df["date"] = pd.to_datetime(df["end_time"], unit="ns")
                elif max_time > 1e12:  # Microseconds
                    df["date"] = pd.to_datetime(df["end_time"], unit="us")
                else:  # Milliseconds
                    df["date"] = pd.to_datetime(df["end_time"], unit="ms")
                df.set_index("date", inplace=True)
                print(f"Processed {file} with {len(df)} rows")
                price_df_eth = pd.concat([price_df_eth, df])
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue

    if price_df_btc.empty or price_df_eth.empty:
        print("No data processed for BTCUSDT or ETHUSDT")
        return

    price_df_btc = price_df_btc.rename(columns=lambda x: f"{x}_BTCUSDT")
    price_df_eth = price_df_eth.rename(columns=lambda x: f"{x}_ETHUSDT")
    price_df = pd.concat([price_df_btc, price_df_eth], axis=1)

    # Feature engineering for volatility prediction
    for pair in ["ETHUSDT", "BTCUSDT"]:
        price_df[f"log_return_{pair}"] = np.log(price_df[f"close_{pair}"].shift(-1) / price_df[f"close_{pair}"])
        price_df[f"volatility_6h_{pair}"] = price_df[f"log_return_{pair}"].rolling(window=360).std() * np.sqrt(360)
        for metric in ["close", "volume", "log_return"]:
            for lag in range(1, 11):
                price_df[f"{metric}_{pair}_lag{lag}"] = price_df[f"{metric}_{pair}"].shift(lag)
        price_df[f"volatility_6h_{pair}_lag1"] = price_df[f"volatility_6h_{pair}"].shift(1)

    price_df["hour_of_day"] = price_df.index.hour
    price_df["target_BTCUSDT"] = price_df["volatility_6h_BTCUSDT"]

    price_df = price_df.dropna()
    print(f"Total rows in price_df after preprocessing: {len(price_df)}")
    print(f"First few dates in price_df: {price_df.index[:5].tolist()}")

    price_df.to_csv(training_price_data_path, date_format='%Y-%m-%d %H:%M:%S')
    print(f"Data saved to {training_price_data_path}")

def load_frame(file_path, timeframe):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, index_col='date', parse_dates=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    features = [
        f"{metric}_{pair}_lag{lag}" 
        for pair in ["ETHUSDT", "BTCUSDT"]
        for metric in ["close", "volume", "log_return"]
        for lag in range(1, 11)
    ] + [f"volatility_6h_{pair}_lag1" for pair in ["ETHUSDT", "BTCUSDT"]] + ["hour_of_day"]
    
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in data: {missing_features}")
    
    X = df[features]
    y = df["target_BTCUSDT"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]  # Fixed syntax error
    
    print(f"Loaded {len(df)} rows, resampled to {timeframe}")
    return X_train, X_test, y_train, y_test, scaler

def preprocess_live_data(df_btc, df_eth):
    if "date" in df_btc.columns:
        df_btc.set_index("date", inplace=True)
    if "date" in df_eth.columns:
        df_eth.set_index("date", inplace=True)
    
    df_btc = df_btc.rename(columns=lambda x: f"{x}_BTCUSDT" if x != "date" else x)
    df_eth = df_eth.rename(columns=lambda x: f"{x}_ETHUSDT" if x != "date" else x)
    
    df = pd.concat([df_btc, df_eth], axis=1)
    print(f"Live data sample (raw):\n{df.tail()}")
    
    for pair in ["ETHUSDT", "BTCUSDT"]:
        df[f"log_return_{pair}"] = np.log(df[f"close_{pair}"].shift(-1) / df[f"close_{pair}"])
        df[f"volatility_6h_{pair}"] = df[f"log_return_{pair}"].rolling(window=360).std() * np.sqrt(360)
        for metric in ["close", "volume", "log_return"]:
            for lag in range(1, 11):
                df[f"{metric}_{pair}_lag{lag}"] = df[f"{metric}_{pair}"].shift(lag)
        df[f"volatility_6h_{pair}_lag1"] = df[f"volatility_6h_{pair}"].shift(1)

    df["hour_of_day"] = df.index.hour
    
    df = df.dropna()
    print(f"Live data after preprocessing:\n{df.tail()}")
    
    features = [
        f"{metric}_{pair}_lag{lag}" 
        for pair in ["ETHUSDT", "BTCUSDT"]
        for metric in ["close", "volume", "log_return"]
        for lag in range(1, 11)
    ] + [f"volatility_6h_{pair}_lag1" for pair in ["ETHUSDT", "BTCUSDT"]] + ["hour_of_day"]
    
    X = df[features]
    
    with open(scaler_file_path, "rb") as f:
        scaler = pickle.load(f)
    X_scaled = scaler.transform(X)
    
    return X_scaled

def train_model(timeframe, file_path=training_price_data_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found at {file_path}. Ensure data is downloaded and formatted.")
    
    X_train, X_test, y_train, y_test, scaler = load_frame(file_path, timeframe)
    print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    if MODEL == "KNN":
        print("\n🚀 Training kNN Model with Grid Search...")
        param_grid = {
            "n_neighbors": [25, 50, 100, 200],  # Adjusted range
            "weights": ["uniform", "distance"],
            "metric": ["minkowski", "manhattan"]
        }
        model = KNeighborsRegressor()
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"\n✅ Best k: {model.n_neighbors}, Metric: {model.metric}, Weighting: {model.weights}")
    elif MODEL == "LinearRegression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        print("\n✅ Trained LinearRegression model")
    elif MODEL == "SVR":
        print("\n🚀 Training SVR Model with Grid Search...")
        param_grid = {
            "C": [0.1, 1, 10],
            "epsilon": [0.01, 0.1, 1],
            "kernel": ["rbf", "linear"]
        }
        model = SVR()
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=tscv,
            scoring=make_scorer(mean_absolute_error, greater_is_better=False),
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_
        print(f"\n✅ Best C: {model.C}, Epsilon: {model.epsilon}, Kernel: {model.kernel}")
    elif MODEL == "KernelRidge":
        model = KernelRidge()
        model.fit(X_train, y_train)
        print("\n✅ Trained KernelRidge model")
    elif MODEL == "BayesianRidge":
        model = BayesianRidge()
        model.fit(X_train, y_train)
        print("\n✅ Trained BayesianRidge model")
    else:
        raise ValueError(f"Unsupported model: {MODEL}")
    
    train_pred = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    print(f"Training MAE: {train_mae:.6f}")
    print(f"Training RMSE: {train_rmse:.6f}")
    print(f"Training R²: {train_r2:.6f}")

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    print(f"Test MAE: {mae:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test R²: {r2:.6f}")
    
    os.makedirs(os.path.dirname(model_file_path), exist_ok=True)
    with open(model_file_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_file_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Trained model saved to {model_file_path}")
    print(f"Scaler saved to {scaler_file_path}")
    
    return model, scaler

def get_inference(token, timeframe, region, data_provider):
    with open(model_file_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    if data_provider == "coingecko":
        df_btc = download_coingecko_current_day_data("BTC", CG_API_KEY)
        df_eth = download_coingecko_current_day_data("ETH", CG_API_KEY)
    else:
        df_btc = download_binance_current_day_data("BTCUSDT", region)
        df_eth = download_binance_current_day_data("ETHUSDT", region)
    
    X_new = preprocess_live_data(df_btc, df_eth)
    print("Inference input data shape:", X_new.shape)
    volatility_pred = loaded_model.predict(X_new)[0]
    print(f"Predicted 6h BTC/USD Volatility: {volatility_pred:.6f}")
    return volatility_pred
