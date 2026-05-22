import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
import sys

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import indicators

def lorentzian_distance(x, y):
    """
    Custom Lorentzian distance metric.
    Calculates logarithmic warping of coordinates to dampen outlier impacts.
    """
    return np.sum(np.log(1 + np.abs(x - y)))

def prepare_features_and_targets(df, target_horizon=1, threshold=0.0002):
    """
    Calculate technical indicators and create feature vectors/targets.
    Target labels predict price direction over the next target_horizon bars:
    -  1 (Up): close price increases by more than the threshold percentage
    - -1 (Down): close price decreases by more than the threshold percentage
    -  0 (Neutral): close price remains flat / within threshold
    """
    df = df.copy()
    
    # 1. Feature Engineering
    df["f1_rsi"] = indicators.rsi(df["close"], period=14)
    wt1, wt2 = indicators.wavetrend(df["high"], df["low"], df["close"], n1=10, n2=11)
    df["f2_wt"] = wt1
    df["f3_cci"] = indicators.cci(df["high"], df["low"], df["close"], period=20)
    df["f4_adx"] = indicators.adx(df["high"], df["low"], df["close"], period=14)
    df["f5_rsi_short"] = indicators.rsi(df["close"], period=9)
    
    # Custom non-parametric kernel filter for trend confirmation
    df["kernel_reg"] = indicators.nadaraya_watson_rational_quadratic(df["close"], h=8, r=8.0, lookback=25)
    df["kernel_slope"] = df["kernel_reg"].diff()  # Positive is bullish, negative is bearish
    
    # 2. Target Labeling (Next 5-minute price direction)
    # The return over the next target_horizon bars
    future_change = (df["close"].shift(-target_horizon) - df["close"]) / df["close"]
    
    # For a clean binary classification problem, we assign 1 to positive changes and -1 to negative or zero changes
    targets = np.where(future_change > threshold, 1, -1)
    df["target"] = targets
    
    # Drop rows containing NaNs arising from lagging window operations and the future shifted targets
    df_clean = df.dropna().copy()
    
    # Align features (X) and labels (y)
    feature_cols = ["f1_rsi", "f2_wt", "f3_cci", "f4_adx", "f5_rsi_short"]
    X = df_clean[feature_cols].values
    y = df_clean["target"].values
    
    return df_clean, X, y

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train and compare the custom Lorentzian KNN against standard baseline classifiers.
    """
    # Scaling is crucial for distance-based ML models. RobustScaler is used because it
    # relies on IQR to resist technical indicator outliers.
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 1. Baseline: Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)
    
    # 2. Baseline: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    
    # 3. Baseline: standard Euclidean KNN
    knn_euclidean = KNeighborsClassifier(n_neighbors=8, metric="euclidean")
    knn_euclidean.fit(X_train_scaled, y_train)
    
    # 4. Custom: Physics-Inspired Lorentzian KNN using a custom callable distance metric
    knn_lorentzian = KNeighborsClassifier(n_neighbors=8, metric=lorentzian_distance)
    knn_lorentzian.fit(X_train_scaled, y_train)
    
    models = {
        "Logistic Regression": (lr, X_test_scaled),
        "Random Forest": (rf, X_test_scaled),
        "Euclidean KNN": (knn_euclidean, X_test_scaled),
        "Lorentzian KNN": (knn_lorentzian, X_test_scaled)
    }
    
    results = {}
    for name, (model, X_t) in models.items():
        preds = model.predict(X_t)
        acc = np.mean(preds == y_test)
        results[name] = {
            "model": model,
            "predictions": preds,
            "accuracy": acc
        }
        
    return results, scaler

if __name__ == "__main__":
    # Test block
    print("Testing ML Model module...")
    dataset_path = os.path.join("dataset", "btc_5m_historical.csv")
    if os.path.exists(dataset_path):
        df = pd.read_csv(dataset_path)
        df_clean, X, y = prepare_features_and_targets(df)
        print(f"Features shape: {X.shape}, Target labels shape: {y.shape}")
        
        # Chronological train-test split (80-20) to prevent temporal data leakage
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print("Training models...")
        results, scaler = train_and_evaluate_models(X_train, X_test, y_train, y_test)
        
        for name, res in results.items():
            print(f"- {name} Accuracy: {res['accuracy']:.4f}")
    else:
        print("BTC 5m CSV file not found. Run binance_klines_fetcher.py first.")
