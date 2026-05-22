import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import RobustScaler
import sys

# Ensure local imports work correctly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import model as ml_model

def run_pipeline():
    """
    Main pipeline to load the Binance dataset, build features, train the classifiers,
    generate performance evaluation metrics/plots, and backtest the trading strategy.
    """
    print("Starting ML time-series forecasting pipeline...")
    os.makedirs("results", exist_ok=True)
    os.makedirs("model", exist_ok=True)
    
    # 1. Load the historical Binance dataset
    csv_path = os.path.join("dataset", "btc_5m_historical.csv")
    if not os.path.exists(csv_path):
        print(f"Error: Dataset {csv_path} not found. Please run binance_klines_fetcher.py first.")
        return
    
    df = pd.read_csv(csv_path)
    
    # 2. Preprocess data
    # threshold=0.0 sets up a clean 2-class problem (Up vs Down/Flat) for simplified evaluation
    print("Engineering features and target labels...")
    df_clean, X, y = ml_model.prepare_features_and_targets(df, target_horizon=1, threshold=0.0)
    
    # Output class distributions
    unique, counts = np.unique(y, return_counts=True)
    class_balance = dict(zip(unique, counts))
    print(f"Class balance: {class_balance} (-1.0: Down/Flat, 1.0: Up)")
    
    # 3. Train-Test Split (80-20 chronological split to avoid data leakage)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    test_df = df_clean.iloc[split_idx:].copy()
    
    # 4. Train and evaluate models
    print("Training models and computing performance...")
    results, scaler = ml_model.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # Save the scaler and the best model (Lorentzian KNN) for modular packaging
    best_model_name = "Lorentzian KNN"
    best_model = results[best_model_name]["model"]
    
    with open(os.path.join("model", "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join("model", "lorentzian_knn_model.pkl"), "wb") as f:
        pickle.dump(best_model, f)
    print(f"Successfully saved scaler and best model ({best_model_name}) to model/ directory!")
    
    # 5. Output Accuracies and classification report
    print("\n" + "="*30)
    print("       MODEL ACCURACIES")
    print("="*30)
    for name, res in results.items():
        print(f"{name:<20} Accuracy: {res['accuracy']:.4f}")
    print("="*30)
        
    print("\nClassification Report for Lorentzian KNN:")
    lorentzian_preds = results["Lorentzian KNN"]["predictions"]
    print(classification_report(y_test, lorentzian_preds, target_names=["Down/Flat", "Up"]))
    
    # Generate Confusion Matrix
    print("Generating Confusion Matrix plot...")
    cm = confusion_matrix(y_test, lorentzian_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="GnBu", 
                xticklabels=["Down/Flat", "Up"], yticklabels=["Down/Flat", "Up"], 
                annot_kws={"size": 14})
    plt.title("Confusion Matrix: Lorentzian KNN Classifier", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Predicted Label", fontsize=11, labelpad=10)
    plt.ylabel("True Label", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "confusion_matrix.png"), dpi=300)
    plt.close()
    
    # 6. Generate ROC-AUC Curve Comparisons
    print("Generating ROC-AUC Curve plot...")
    plt.figure(figsize=(8, 6))
    
    X_test_scaled = scaler.transform(X_test)
    
    for name, res in results.items():
        model = res["model"]
        if hasattr(model, "predict_proba"):
            # Probability for class index 1 (representing label 1.0 / 'Up')
            probs = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs, pos_label=1.0)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})", linewidth=2)
            
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess (AUC = 0.5000)", linewidth=1.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=11, labelpad=10)
    plt.ylabel("True Positive Rate", fontsize=11, labelpad=10)
    plt.title("ROC-AUC Curves: Model Performance Comparison", fontsize=13, fontweight="bold", pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "roc_auc_curve.png"), dpi=300)
    plt.close()
    
    # 7. Simulated trading backtest
    print("Running simulated trading backtest on test data...")
    
    # Calculate simple close-to-close returns on the test dataframe
    test_df["returns"] = test_df["close"].pct_change()
    
    # Set our position signals: 1 for Long, -1 for Short
    test_df["signal"] = lorentzian_preds
    
    # The return in period t+1 is the signal in period t multiplied by close return in period t+1
    test_df["strategy_returns"] = test_df["signal"].shift(1) * test_df["returns"]
    
    # Standard transaction fee for Binance trades (0.05% per trade)
    # We pay a fee whenever we change our position (signal shifts)
    position_changes = test_df["signal"].diff().abs()
    transaction_fee = 0.0005  # 0.05%
    test_df["strategy_returns_net"] = test_df["strategy_returns"] - (position_changes * transaction_fee).fillna(0)
    
    # Calculate cumulative percentage returns
    test_df["cum_returns_bh"] = (1 + test_df["returns"].fillna(0)).cumprod() - 1
    test_df["cum_returns_strategy_gross"] = (1 + test_df["strategy_returns"].fillna(0)).cumprod() - 1
    test_df["cum_returns_strategy_net"] = (1 + test_df["strategy_returns_net"].fillna(0)).cumprod() - 1
    
    # Plotting Equity Curves
    plt.figure(figsize=(10, 6))
    time_series = test_df["close_time"]
    
    plt.plot(time_series, test_df["cum_returns_strategy_net"] * 100, 
             label="Lorentzian KNN Strategy (Net of Fees)", color="#009988", linewidth=2.2)
    plt.plot(time_series, test_df["cum_returns_strategy_gross"] * 100, 
             label="Lorentzian KNN Strategy (Gross)", color="#009988", linestyle="--", alpha=0.6, linewidth=1.2)
    plt.plot(time_series, test_df["cum_returns_bh"] * 100, 
             label="Buy & Hold BTC Benchmark", color="#CC3311", linewidth=1.8)
    
    plt.title("Backtest Equity Curves: Lorentzian KNN Strategy vs. BTC Benchmark", fontsize=13, fontweight="bold", pad=15)
    plt.xlabel("Date / Time", fontsize=11, labelpad=10)
    plt.ylabel("Cumulative Returns (%)", fontsize=11, labelpad=10)
    plt.legend(loc="upper left", fontsize=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(os.path.join("results", "equity_curve.png"), dpi=300)
    plt.close()
    
    # Save the detailed test table
    test_df.to_csv(os.path.join("results", "backtest_results.csv"), index=False)
    print("Pipeline successfully completed! All results and premium plots saved in 'results/' and 'model/' directories.")

if __name__ == "__main__":
    run_pipeline()
