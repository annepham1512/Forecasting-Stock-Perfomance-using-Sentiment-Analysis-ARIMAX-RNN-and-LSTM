# Forecasting-Stock-Perfomance-using-Sentiment-Analysis-ARIMAX-RNN-and-LSTM

Welcome to the repository for our Stock Performance Forecasting Project, conducted by Anne Pham, Robin Nguyen, and Emmanuel Arung Bate. This project analyzes stock performance of 90 listed companies across electric vehicle (EV), financial banking, and technology sectors, leveraging time series forecasting, fundamental analysis, and AI-driven sentiment analysis to explore AI’s impact on returns.

## Project Overview
The financial markets are shaped by systematic factors and behavioral drivers like investor sentiment and AI innovation. We forecast stock returns for 90 companies—EV (e.g., TSLA, NIO), financial banking (e.g., JPM, BAC), and technology (e.g., AAPL, MSFT)—using **ARIMAX**, **simple RNN**, and **LSTM** models. We integrate **FinBERT** sentiment scores from Google News (general and AI-specific) and verify findings with fundamental metrics (e.g., EPS, revenue growth) from Yahoo Finance. Our goal is to assess forecasting accuracy and AI’s performance impact, particularly in EV and tech sectors.

### Team Roles
- **Anne Pham**: Primary PIC for Sentiment and Neural Networks – Scrapes Google News, implements FinBERT sentiment analysis, trains RNN and LSTM models.
- **Robin Nguyen**: Primary PIC for ARIMAX – Develops ARIMAX models with sentiment as exogenous variables.
- **Emmanuel Arung Bate**: Primary PIC for Time Series and Fundamentals – Collects/preprocesses Yahoo Finance data, coordinates fundamental analysis.

## Objectives
- Forecast stock returns using ARIMAX, RNN, and LSTM with sentiment inputs.
- Compute general and AI-specific sentiment scores from Google News using FinBERT. After that, Seperate them into their own sectors : EV, Financial Bank, and Technology. They are then put into categorical terms : High Positive AI Sentiment, Neutral AI Sentiment, Negative/Low AI Sentiment.
- Construct sentiment-based portfolios to compare performance.
- Verify results with :
+ ARIMAX: Mean Absolute Error, Mean Squared Error, Root Mean Squared Error
+ Recurrent Neural Network (RNN) and Long Short-Term Memory (LSTM):  Mean Absolute Error, Mean Squared Error, Root Mean Squared Error.
- Evaluate AI’s impact on returns, expecting stronger effects in EV and technology.

## Data
- **Stock Returns**:
  - Source: Yahoo Finance via `yfinance` (https://finance.yahoo.com/)
  - Frequency: Daily, 2020–2025 (adjustable)
  - Variables: Adjusted closing prices for 90 tickers, converted to returns
- **Fundamental Metrics**:
  - Source: Yahoo Finance (`stock.info`)
  - Frequency: Quarterly/annual (latest)
  - Variables: EPS, P/E ratio, ROE, debt-to-equity, revenue growth (R&D/AI proxy)
- **Sentiment Data**:
  - Source: Google News (https://news.google.com/), scraped for general (e.g., “TSLA stock”) and AI-specific (e.g., “TSLA artificial intelligence”) articles
  - Frequency: Daily, March 2025 (adjustable)
  - Variables: Headlines/snippets, processed by FinBERT for sentiment scores (-1 to +1)

*Note*: Google News scraping uses rate-limited requests (10-second delays) for ethical compliance, with manual collection as a backup.

## Methods
- **ARIMAX**: Models returns with sentiment as exogenous variables: *Rt = c + φ1Rt−1 + … + θ1εt−1 + … + β1GeneralSentimentt + β2AISentimentt + εt*. Selects (p,d,q) via ACF/PACF, AIC; ensures stationarity with ADF tests.
- **Simple RNN**: Trains RNN for temporal patterns. Inputs: lagged returns, sentiment. Architecture: one RNN layer (50 units), dense output.
- **LSTM**: Uses LSTM for long-term dependencies. Inputs: returns, sentiment, fundamentals. Architecture: one/two LSTM layers (50–100 units), dense output.
- **Sentiment Analysis**: Scrapes Google News, applies FinBERT for daily general/AI sentiment scores, aggregated by ticker.
- **Portfolio Analysis**: Constructs portfolios via sentiment quantiles (High Positive, Neutral, High Negative).
- **Fundamental Analysis**: Correlates sentiment with fundamentals (e.g., revenue growth, P/E) to verify performance drivers.

## Requirements
- Python 3.8+
- Libraries: `pandas`, `yfinance`, `requests`, `beautifulsoup4`, `transformers`, `statsmodels`, `tensorflow`, `keras`, `sklearn`, `matplotlib`
- Install via:
  ```bash
  pip install -r requirements.txt
