import os
import json

def build_notebook():
    print("Building master self-contained Jupyter notebook: code/main.ipynb...")
    
    # Read indicators.py and adjust for notebook inline usage
    with open(os.path.join("code", "indicators.py"), "r") as f:
        indicators_code = f.read()
        
    # Read dataset/binance_klines_fetcher.py and extract the main fetch functions
    with open(os.path.join("dataset", "binance_klines_fetcher.py"), "r") as f:
        fetcher_code = f.read()
        # Remove the main testing execution block at the end
        if "if __name__ == \"__main__\":" in fetcher_code:
            fetcher_code = fetcher_code.split("if __name__ == \"__main__\":")[0].strip()
            
    # Read code/model.py and adjust to call indicators directly instead of "indicators.method"
    with open(os.path.join("code", "model.py"), "r") as f:
        model_code = f.read()
        # Remove imports of indicators and test blocks
        model_code = model_code.replace("import indicators", "")
        model_code = model_code.replace("indicators.rsi", "rsi")
        model_code = model_code.replace("indicators.wavetrend", "wavetrend")
        model_code = model_code.replace("indicators.cci", "cci")
        model_code = model_code.replace("indicators.adx", "adx")
        model_code = model_code.replace("indicators.nadaraya_watson_rational_quadratic", "nadaraya_watson_rational_quadratic")
        if "if __name__ == \"__main__\":" in model_code:
            model_code = model_code.split("if __name__ == \"__main__\":")[0].strip()
            
    # Read code/backtest_and_plots.py and adjust for notebook
    with open(os.path.join("code", "backtest_and_plots.py"), "r") as f:
        backtest_code = f.read()
        backtest_code = backtest_code.replace("import model as ml_model", "")
        # Replace ml_model call prefixes with direct function calls
        backtest_code = backtest_code.replace("ml_model.prepare_features_and_targets", "prepare_features_and_targets")
        backtest_code = backtest_code.replace("ml_model.train_and_evaluate_models", "train_and_evaluate_models")
        if "if __name__ == \"__main__\":" in backtest_code:
            backtest_code = backtest_code.split("if __name__ == \"__main__\":")[0].strip()

    # Read code/generate_pdf_deliverables.py and extract PDF classes and creators
    with open(os.path.join("code", "generate_pdf_deliverables.py"), "r") as f:
        pdf_code = f.read()
        if "if __name__ == \"__main__\":" in pdf_code:
            pdf_code = pdf_code.split("if __name__ == \"__main__\":")[0].strip()
            
    # Read code/package_submission.py and extract zip packaging function
    with open(os.path.join("code", "package_submission.py"), "r") as f:
        package_code = f.read()
        if "if __name__ == \"__main__\":" in package_code:
            package_code = package_code.split("if __name__ == \"__main__\":")[0].strip()

    # Construct the Jupyter notebook structure
    cells = []
    
    # 1. Introduction Header Markdown Cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# CSC-350 Artificial Intelligence Semester Project (Section E)\n",
            "## Advanced Time-Series Forecasting using Lorentzian Distance Classification & Kernel Regression\n\n",
            "**Instructor:** Engr. Muhammad Irfan Younas Mughal (eMIYM)  \n",
            "**Group Members:**\n",
            "1. **Tayyab Mangi** (CMS: 023-24-0118) - Coded model training, custom Lorentzian Scikit-Learn metric, backtesting, results report, and slides.\n",
            "2. **Asif Ali Rattar** (CMS: 023-24-0158) - Coded Binance Klines API data fetcher, indicator engineering, plotting scripts, introduction, and literature review.\n\n",
            "---"
        ]
    })
    
    # 2. Setup cell with standard dependencies
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# === Google Colab Setup / Library check ===\n",
            "import os\n",
            "import sys\n",
            "import requests\n",
            "import time\n",
            "import pickle\n",
            "import zipfile\n",
            "import shutil\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import seaborn as sns\n",
            "from sklearn.neighbors import KNeighborsClassifier\n",
            "from sklearn.ensemble import RandomForestClassifier\n",
            "from sklearn.linear_model import LogisticRegression\n",
            "from sklearn.preprocessing import RobustScaler\n",
            "from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc\n",
            "\n",
            "# Attempt to install fpdf if missing (useful for Google Colab/clean environments)\n",
            "try:\n",
            "    from fpdf import FPDF\n",
            "except ImportError:\n",
            "    print(\"fpdf library not found. Installing fpdf...\")\n",
            "    import subprocess\n",
            "    subprocess.check_call([sys.executable, \"-m\", \"pip\", \"install\", \"fpdf2\"])\n",
            "    from fpdf import FPDF\n",
            "    print(\"fpdf2 successfully installed!\")"
        ]
    })
    
    # 3. Data Acquisition Fetcher Code Cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Part 1: Binance Candlestick (Klines) Data Fetching\n",
            "We fetch 5,000 historical 5-minute candles of the BTC/USDT spot market from the Binance API. The data is paginated backwards in time using the `endTime` parameter."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            fetcher_code
        ]
    })
    
    # 4. Technical Indicators Code Cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Part 2: Feature Engineering & Technical Indicators\n",
            "We calculate standard indicators (RSI, WaveTrend, CCI, ADX) and a **causal, non-repainting Nadaraya-Watson Rational Quadratic Kernel Regression** filter to smooth out macro price movements."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            indicators_code
        ]
    })
    
    # 5. Lorentzian KNN model estimators
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Part 3: Custom Lorentzian Classifier & Feature Preparation\n",
            "Here we implement the custom **Lorentzian Distance formula** and build the training/testing classification vectors with binary targets (1: price Up, -1: price Down/Flat)."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            model_code
        ]
    })
    
    # 6. Training, Evaluation, and Backtester
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Part 4: Classification Pipeline, Model Training, & Backtesting\n",
            "We execute model training comparing Lorentzian KNN to Euclidean KNN, Random Forest, and Logistic Regression. We run a trading simulation (strategy vs. Buy & Hold BTC) adjusted for 0.05% transaction fees."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            backtest_code
        ]
    })
    
    # 7. University PDF Compilers
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Part 5: Academic Deliverables Programmatic PDF Compiler\n",
            "We write the programmatic FPDF engine to generate the 6-page IEEE conference paper report, the 12-slide landscape presentation slides, the contribution statement, and the plagiarism originality declaration."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            pdf_code
        ]
    })
    
    # 8. Project Zipping Package Creator
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Part 6: Project Submission ZIP Packager\n",
            "We define the zipping manager to bundle all required directories into the standard required submission format."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            package_code
        ]
    })
    
    # 9. Main interactive orchestrator
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Part 7: Master Pipeline Execution\n",
            "Let's execute all components in sequence! We fetch raw candlestick data, preprocess, train the classifiers, generate high-resolution evaluation charts, compile academic PDFs, and bundle everything into a ZIP file ready for E-Learning upload."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create output directories\n",
            "os.makedirs(\"dataset\", exist_ok=True)\n",
            "os.makedirs(\"results\", exist_ok=True)\n",
            "os.makedirs(\"model\", exist_ok=True)\n",
            "os.makedirs(\"report\", exist_ok=True)\n",
            "os.makedirs(\"slides\", exist_ok=True)\n",
            "os.makedirs(\"plagiarism\", exist_ok=True)\n",
            "os.makedirs(\"contribution\", exist_ok=True)\n",
            "os.makedirs(\"video\", exist_ok=True)\n",
            "\n",
            "# 1. Fetch 5000 candles from Binance\n",
            "df_klines = fetch_historical_dataset(symbol=\"BTCUSDT\", interval=\"5m\", total_candles=5000)\n",
            "df_klines.to_csv(\"dataset/btc_5m_historical.csv\", index=False)\n",
            "print(\"✓ Step 1: Binance dataset fetched and saved!\\n\")\n",
            "\n",
            "# 2. Run training, evaluation & trading backtest\n",
            "run_pipeline()\n",
            "print(\"✓ Step 2: Training & backtesting pipeline completed successfully!\\n\")\n",
            "\n",
            "# 3. Programmatically compile academic PDFs\n",
            "print(\"Compiling PDF deliverables...\")\n",
            "create_report()\n",
            "create_slides()\n",
            "create_statement()\n",
            "create_declaration()\n",
            "\n",
            "# Ensure plagiarism declaration copy exists in contribution folder\n",
            "shutil.copy(\"plagiarism/plagiarism_declaration.pdf\", \"contribution/plagiarism_declaration.pdf\")\n",
            "print(\"✓ Step 3: University PDF deliverables generated successfully!\\n\")\n",
            "\n",
            "# 4. Package into standard ZIP archive\n",
            "package_project()\n",
            "print(\"✓ Step 4: Submission package prepared and ready for upload!\")"
        ]
    })
    
    # 10. Display plots inline
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Part 8: Display Generated Plot Graphics Inline\n",
            "We load and render the high-resolution charts generated during the model evaluations and strategy backtests."
        ]
    })
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from IPython.display import Image, display\n",
            "\n",
            "print(\"--- Lorentzian KNN Confusion Matrix ---\")\n",
            "display(Image(filename=\"results/confusion_matrix.png\"))\n",
            "\n",
            "print(\"\\n--- Multi-Model ROC-AUC Comparison ---\")\n",
            "display(Image(filename=\"results/roc_auc_curve.png\"))\n",
            "\n",
            "print(\"\\n--- Cumulative Strategy Returns vs. Benchmark BTC Buy & Hold ---\")\n",
            "display(Image(filename=\"results/equity_curve.png\"))"
        ]
    })

    # Prepare final notebook structure
    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.12"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    
    # Save code/main.ipynb
    notebook_path = os.path.join("code", "main.ipynb")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook_dict, f, indent=1, ensure_ascii=False)
        
    print(f"Successfully generated master self-contained Jupyter notebook at: {notebook_path}")

if __name__ == "__main__":
    build_notebook()
