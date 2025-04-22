import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from pygooglenews import GoogleNews
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import warnings
import logging
import os
warnings.filterwarnings('ignore')

# Set up logging
os.makedirs('./logs', exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logging.basicConfig(
    filename=f'./logs/sentiment_analysis_{timestamp}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)

# Load FinBERT model for financial sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

# Define company tickers and names for each sector
# EV Sector Dictionary
ev_companies = {
    "TSLA": "Tesla",
    "BYDDY": "BYD Company",
    "LI": "Li Auto",
    "NIO": "NIO",
    "RIVN": "Rivian",
    "LCID": "Lucid Group",
    "XPEV": "XPeng",
    "NKLA": "Nikola",
    "PSNY": "Polestar",
    "GM": "General Motors",
    "F": "Ford",
    "VWAGY": "Volkswagen",
    "BAMXF": "BMW",
    "HYMTF": "Hyundai",
    "KIMTF": "Kia",
    "POAHY": "Porsche",
    "MBGYY": "Mercedes-Benz",
    "STLA": "Stellantis",
    "GELYF": "Geely",
    "GWLLY": "Great Wall Motors",
    "SAIC": "SAIC Motor",
    "HYLN": "Hyliion",
    "GNZUF": "GAC Group",
    "TATAMOTORS.NS": "Tata Motors",
    "MAHMF": "Mahindra",
    "RNLSY": "Renault",
    "NSANY": "Nissan",
    "MMTOF": "Mitsubishi Motors"
}

# FinBank Sector Dictionary
finbank_companies = {
    "JPM": "JPMorgan Chase",
    "BAC": "Bank of America",
    "WFC": "Wells Fargo",
    "C": "Citigroup",
    "GS": "Goldman Sachs",
    "MS": "Morgan Stanley",
    "USB": "U.S. Bancorp",
    "PNC": "PNC Financial",
    "TFC": "Truist Financial",
    "COF": "Capital One",
    "TD": "TD Bank",
    "SCHW": "Charles Schwab",
    "BK": "Bank of New York Mellon",
    "STT": "State Street",
    "AXP": "American Express",
    "HSBC": "HSBC",
    "CFG": "Citizens Financial",
    "FITB": "Fifth Third Bank",
    "MTB": "M&T Bank",
    "HBAN": "Huntington Bancshares",
    "ALLY": "Ally Financial",
    "KEY": "KeyCorp",
    "RY": "Royal Bank of Canada",
    "SAN": "Santander",
    "NTRS": "Northern Trust",
    "RF": "Regions Financial",
    "SYF": "Synchrony Financial",
    "NBHC": "National Bank Holdings",
    "ZION": "Zions Bancorporation",
    "FHN": "First Horizon"
}

# Tech Sector Dictionary
tech_companies = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "GOOG": "Google",
    "AMZN": "Amazon",
    "META": "Meta",
    "NVDA": "NVIDIA",
    "TSM": "TSMC",
    "ADBE": "Adobe",
    "INTC": "Intel",
    "CSCO": "Cisco",
    "ORCL": "Oracle",
    "IBM": "IBM",
    "CRM": "Salesforce",
    "QCOM": "Qualcomm",
    "AVGO": "Broadcom",
    "TXN": "Texas Instruments",
    "AMD": "AMD",
    "AMAT": "Applied Materials",
    "MU": "Micron",
    "NET": "Cloudflare",
    "NOW": "ServiceNow",
    "SNOW": "Snowflake",
    "DOCU": "DocuSign",
    "SHOP": "Shopify",
    "UBER": "Uber",
    "LYFT": "Lyft",
    "SNAP": "Snap",
    "HRB": "H&R Block",
    "DDOG": "Datadog"
}

# Function to get FinBERT sentiment score 
def get_finbert_sentiment(text):
    """Analyze sentiment using FinBERT"""
    if pd.isna(text) or text == "":
        return {"score": 0, "label": "neutral"}
    
    # Truncate if text is too long for BERT (max 512 tokens)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
    # FinBERT returns sentiment as positive, negative, or neutral
    labels = ['negative', 'neutral', 'positive']
    label_id = torch.argmax(predictions, dim=1).item()
    label = labels[label_id]
    score = predictions[0][label_id].item()  # Confidence score
    
    # Convert to score between -1 and 1 (negative = -1, neutral = 0, positive = 1) 
    # weighted by confidence
    if label == 'negative':
        sentiment_score = -score
    elif label == 'positive':
        sentiment_score = score
    else:
        sentiment_score = 0
        
    return {"score": sentiment_score, "label": label}

# Function to scrape news from Google News
def get_company_news(ticker, company_name, search_term="", days_back=7, lang='en', country='US'):
    """
    Scrape Google News for a company
    ticker: company ticker symbol
    company_name: company name
    search_term: additional search term (e.g., "AI" for AI-related news)
    days_back: how many days to look back
    """
    all_news = []
    gn = GoogleNews(lang=lang, country=country)
    
    # Create search queries with both ticker and company name
    queries = []
    if search_term:
        queries.append(f"{ticker} stock {search_term}")
        queries.append(f"{company_name} {search_term}")
    else:
        queries.append(f"{ticker} stock")
        queries.append(f"{company_name}")
    
    # Define date range (today - days_back)
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    total_articles = 0
    for query in queries:
        try:
            # Search for news within date range
            search = gn.search(query, from_=start_date.strftime('%Y-%m-%d'), to_=end_date.strftime('%Y-%m-%d'))
            articles = search['entries']
            total_articles += len(articles)
            
            for article in articles:
                # Extract article data
                news_item = {
                    'ticker': ticker,
                    'company_name': company_name,
                    'search_term': search_term,
                    'title': article.title,
                    'link': article.link,
                    'published': article.published,
                    'summary': article.get('summary', '')
                }
                
                # Combine title and summary for sentiment analysis
                news_item['text'] = f"{article.title}. {article.get('summary', '')}"
                
                # Get sentiment analysis
                sentiment = get_finbert_sentiment(news_item['text'])
                news_item['sentiment_score'] = sentiment['score']
                news_item['sentiment_label'] = sentiment['label']
                
                all_news.append(news_item)
            
            logging.info(f"Found {len(articles)} articles for {query}")
            
        except Exception as e:
            logging.error(f"Error scraping {query}: {e}")
        
        # Be nice to the API with a delay
        time.sleep(3)
    
    logging.info(f"Total articles found for {ticker} ({company_name}) {search_term}: {total_articles}")
    return all_news

# Main function to scrape news for all companies
def scrape_all_company_news(companies_dict, search_terms=["", "AI"]):
    """
    Scrape news for all companies in the companies dictionary
    companies_dict: dictionary of company tickers and names
    search_terms: list of additional search terms (empty string for general news)
    """
    all_company_news = []
    total_articles = 0
    
    for ticker, company_name in companies_dict.items():
        for term in search_terms:
            logging.info(f"Scraping news for {ticker} ({company_name}) {term}")
            news = get_company_news(ticker, company_name, term)
            all_company_news.extend(news)
            total_articles += len(news)
    
    logging.info(f"Total articles scraped across all companies: {total_articles}")
    return all_company_news

# Function to create a dataframe from the scraped news
def create_news_dataframe(news_list):
    """Create a dataframe from the scraped news"""
    df = pd.DataFrame(news_list)
    
    # Convert published date to datetime
    df['published_date'] = pd.to_datetime(df['published']).dt.date
    
    # Sort by published date
    df = df.sort_values(by='published_date', ascending=False)
    
    return df

# Function to calculate aggregate sentiment scores for each company
def calculate_company_sentiment(news_df):
    """Calculate aggregate sentiment scores for each company"""
    # For general sentiment (no search term)
    general_sentiment = news_df[news_df['search_term'] == ''].groupby('ticker')['sentiment_score'].mean().reset_index()
    general_sentiment.rename(columns={'sentiment_score': 'general_sentiment'}, inplace=True)
    
    # For AI-specific sentiment
    ai_sentiment = news_df[news_df['search_term'] == 'AI'].groupby('ticker')['sentiment_score'].mean().reset_index()
    ai_sentiment.rename(columns={'sentiment_score': 'ai_sentiment'}, inplace=True)
    
    # Merge the two sentiment dataframes
    company_sentiment = pd.merge(general_sentiment, ai_sentiment, on='ticker', how='outer')
    
    # Fill NaN values with 0 (no sentiment)
    company_sentiment.fillna(0, inplace=True)
    
    # Calculate combined sentiment (average of general and AI sentiment)
    company_sentiment['combined_sentiment'] = (company_sentiment['general_sentiment'] + company_sentiment['ai_sentiment']) / 2
    
    return company_sentiment

# Function to create portfolios based on AI sentiment
def create_sentiment_portfolios(company_sentiment, num_portfolios=3):
    """
    Create portfolios based on AI sentiment
    company_sentiment: dataframe with company sentiment scores
    num_portfolios: number of portfolios to create
    """
    # Sort companies by AI sentiment score
    sorted_companies = company_sentiment.sort_values(by='ai_sentiment', ascending=False)
    
    # Calculate portfolio size
    portfolio_size = len(sorted_companies) // num_portfolios
    
    # Create portfolios
    portfolios = []
    
    for i in range(num_portfolios):
        start_idx = i * portfolio_size
        end_idx = start_idx + portfolio_size if i < num_portfolios - 1 else len(sorted_companies)
        
        portfolio = {
            'portfolio_id': i + 1,
            'portfolio_name': f"Portfolio {i + 1}",
            'description': "",
            'tickers': sorted_companies.iloc[start_idx:end_idx]['ticker'].tolist(),
            'avg_ai_sentiment': sorted_companies.iloc[start_idx:end_idx]['ai_sentiment'].mean(),
            'avg_general_sentiment': sorted_companies.iloc[start_idx:end_idx]['general_sentiment'].mean(),
            'avg_combined_sentiment': sorted_companies.iloc[start_idx:end_idx]['combined_sentiment'].mean()
        }
        
        # Add portfolio description based on sentiment
        if i == 0:
            portfolio['description'] = "High Positive AI Sentiment"
        elif i == num_portfolios - 1:
            portfolio['description'] = "Negative/Low AI Sentiment"
        else:
            portfolio['description'] = "Neutral AI Sentiment"
        
        portfolios.append(portfolio)
    
    return pd.DataFrame(portfolios)

# Main execution function
def main():
    logging.info("Starting sentiment analysis process")
    
    # Combine all companies
    all_companies = {**ev_companies, **finbank_companies, **tech_companies}
    logging.info(f"Total companies to analyze: {len(all_companies)}")
    
    # Define sector mapping for each ticker
    sector_mapping = {}
    for ticker in ev_companies.keys():
        sector_mapping[ticker] = "EV"
    for ticker in finbank_companies.keys():
        sector_mapping[ticker] = "Financial"
    for ticker in tech_companies.keys():
        sector_mapping[ticker] = "Technology"
    
    # 1. Scrape news for all companies (general and AI-related)
    logging.info("Starting news scraping...")
    all_news = scrape_all_company_news(all_companies)
    
    # 2. Create news dataframe
    news_df = create_news_dataframe(all_news)
    logging.info(f"Created news dataframe with {len(news_df)} articles")
    
    # 3. Add sector information
    news_df['sector'] = news_df['ticker'].map(sector_mapping)
    
    # 4. Calculate company sentiment
    company_sentiment = calculate_company_sentiment(news_df)
    company_sentiment['sector'] = company_sentiment['ticker'].map(sector_mapping)
    
    # 5. Create sentiment portfolios (3 portfolios based on AI sentiment)
    portfolios = create_sentiment_portfolios(company_sentiment, num_portfolios=3)
    
    # 6. Create sector-specific portfolios
    ev_sentiment = company_sentiment[company_sentiment['sector'] == "EV"]
    fin_sentiment = company_sentiment[company_sentiment['sector'] == "Financial"]
    tech_sentiment = company_sentiment[company_sentiment['sector'] == "Technology"]
    
    ev_portfolios = create_sentiment_portfolios(ev_sentiment, num_portfolios=3)
    fin_portfolios = create_sentiment_portfolios(fin_sentiment, num_portfolios=3)
    tech_portfolios = create_sentiment_portfolios(tech_sentiment, num_portfolios=3)
    
    # 7. Create data directory if it doesn't exist
    os.makedirs('./data/sentiment', exist_ok=True)
    
    # Save all results to CSV files in the sentiment folder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    news_df.to_csv(f'./data/sentiment/company_news_{timestamp}.csv', index=False)
    company_sentiment.to_csv(f'./data/sentiment/company_sentiment_{timestamp}.csv', index=False)
    portfolios.to_csv(f'./data/sentiment/sentiment_portfolios_{timestamp}.csv', index=False)
    
    # Save sector-specific portfolios
    ev_portfolios.to_csv(f'./data/sentiment/ev_portfolios_{timestamp}.csv', index=False)
    fin_portfolios.to_csv(f'./data/sentiment/fin_portfolios_{timestamp}.csv', index=False)
    tech_portfolios.to_csv(f'./data/sentiment/tech_portfolios_{timestamp}.csv', index=False)
    
    logging.info("Analysis complete. Files saved.")
    
    # 8. Print summary statistics
    logging.info("\nSummary Statistics:")
    logging.info(f"Total news articles: {len(news_df)}")
    logging.info(f"Articles per sector: {news_df.groupby('sector')['ticker'].count()}")
    logging.info("\nAverage Sentiment by Sector:")
    logging.info(company_sentiment.groupby('sector')[['general_sentiment', 'ai_sentiment', 'combined_sentiment']].mean())
    
    logging.info("\nPortfolio Summary:")
    logging.info(portfolios[['portfolio_id', 'description', 'avg_ai_sentiment', 'avg_general_sentiment']])
    
    return news_df, company_sentiment, portfolios

# Execute the main function
if __name__ == "__main__":
    news_df, company_sentiment, portfolios = main()