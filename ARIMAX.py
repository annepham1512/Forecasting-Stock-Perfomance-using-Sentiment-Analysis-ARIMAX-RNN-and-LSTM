import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# Define paths
SENTIMENT_FILE = './data/sentiment/company_sentiment_20250422_004923.csv'
QUANTITATIVE_DIR = './data/quantitative/'
SECTORS = ['ev', 'fin_bank', 'tech']

# Load sentiment data
sentiment_df = pd.read_csv(SENTIMENT_FILE)
sentiment_df.set_index('ticker', inplace=True)

# Function to load stock returns (handles headerless CSVs and cleans non-numeric rows)
def load_stock_returns(sector):
    returns_dict = {}
    sector_dir = os.path.join(QUANTITATIVE_DIR, sector)
    for filename in os.listdir(sector_dir):
        ticker = filename.split('.')[0]
        # Read CSV without headers and assign column names
        df = pd.read_csv(
            os.path.join(sector_dir, filename),
            header=None,  # No header row in the file
            names=['Date', 'Adj Close', 'Close', 'High', 'Low', 'Open', 'Volume'],  # Assign column names
            parse_dates=['Date'],
            index_col='Date',
            encoding='latin1',
            thousands=','  # Handle commas in numeric values
        )
        # Drop rows where 'Adj Close' is not numeric (e.g., header rows)
        df = df[pd.to_numeric(df['Adj Close'], errors='coerce').notnull()]
        
        # Convert 'Adj Close' to numeric
        df['Adj Close'] = pd.to_numeric(df['Adj Close'], errors='coerce')
        
        # Compute returns from 'Adj Close'
        df['Returns'] = df['Adj Close'].pct_change()
        returns_dict[ticker] = df['Returns']
    return pd.DataFrame(returns_dict)

# Load returns for all sectors
ev_returns = load_stock_returns('ev')
finbank_returns = load_stock_returns('fin_bank')
tech_returns = load_stock_returns('tech')

# Combine all returns
all_returns = pd.concat([ev_returns, finbank_returns, tech_returns], axis=1)

# Drop NaNs (first row for each ticker)
all_returns = all_returns.dropna()

# Align with sentiment data
combined_data = all_returns.copy()
for ticker in combined_data.columns:
    if ticker in sentiment_df.index:
        combined_data[f'{ticker}_general_sentiment'] = sentiment_df.loc[ticker, 'general_sentiment']
        combined_data[f'{ticker}_ai_sentiment'] = sentiment_df.loc[ticker, 'ai_sentiment']

# Function to fit and evaluate ARIMAX
def evaluate_arimax(ticker, endog, exog, order=(1, 1, 1)):
    # Split data
    train_size = int(0.8 * len(endog))
    endog_train, endog_test = endog.iloc[:train_size], endog.iloc[train_size:]
    exog_train, exog_test = exog.iloc[:train_size], exog.iloc[train_size:]
    
    # Fit model
    model = SARIMAX(endog_train, exog=exog_train, order=order, enforce_stationarity=False, enforce_invertibility=False)
    result = model.fit(disp=False)
    
    # Forecast
    forecast = result.forecast(steps=len(endog_test), exog=exog_test)
    forecast.index = endog_test.index
    
    # Metrics
    mae = mean_absolute_error(endog_test, forecast)
    mse = mean_squared_error(endog_test, forecast)
    rmse = np.sqrt(mse)
    aic = result.aic
    
    # Plot
    plt.figure(figsize=(14, 7))
    plt.plot(endog_test, label='Actual', color='blue')
    plt.plot(forecast, label='Forecast', linestyle='--', color='orange')
    plt.title(f"{ticker} - ARIMAX Forecast vs Actual")
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./graphs_ARIMAX/{ticker}_forecast.png')
    plt.close()
    
    return {
        'Ticker': ticker,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'AIC': aic
    }

# Process each ticker
results = []
for ticker in all_returns.columns:
    try:
        # Extract endogenous and exogenous variables
        endog = all_returns[ticker]
        exog = combined_data[[f'{ticker}_general_sentiment', f'{ticker}_ai_sentiment']].dropna()
        
        # Align indices
        endog, exog = endog.align(exog, join='inner')
        
        # Evaluate
        res = evaluate_arimax(ticker, endog, exog)
        results.append(res)
        
        print(f"Processed {ticker}: MAE={res['MAE']:.4f}, RMSE={res['RMSE']:.4f}")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Save results
pd.DataFrame(results).to_csv('arimax_results.csv', index=False)