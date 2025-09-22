# Early initialization of OpenMP
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Must be first

import torch  # Import torch early to manage OpenMP
_ = torch.zeros(1)  # Dummy operation to initialize

print("OpenMP warning should be gone:", os.environ.get('KMP_DUPLICATE_LIB_OK', 'Not set'))

import re
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from tqdm import tqdm
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import yfinance as yf
from datetime import datetime

import scipy.stats as stats
from scipy.signal import correlate
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox

import traceback

import warnings
warnings.filterwarnings('ignore')

from streamlit_extras.metric_cards import style_metric_cards

# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(layout="wide", page_title="From Tweets to Trades: AI-Powered Bitcoin Sentiment & Market Timing", page_icon="₿")

# =============================================================================
# CSS STYLING
# =============================================================================

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Professional Navy Blue Main Container */
    .stApp {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
        color: #f8fafc;
    }
    
    /* Main content area styling */
    .main .block-container {
        background: transparent !important;
        padding-top: 2rem;
    }
    
    /* Status message styling with professional accents */
    .status-message {
        padding: 12px 16px;
        border-radius: 10px;
        margin: 8px 0;
        font-size: 14px;
        display: inline-block;
        font-weight: 500;
        border: 1px solid transparent;
        backdrop-filter: blur(8px);
        animation: fadeIn 0.3s ease-out;
    }
    
    /* Status variants */
    .status-info {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(147, 197, 253, 0.15) 100%);
        border-left: 4px solid #3b82f6;
        color: #e0f2fe !important;
    }
    .status-success {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.2) 0%, rgba(134, 239, 172, 0.2) 100%);
        border-left: 4px solid #22c55e;
        color: #f0fdf4 !important;
    }
    .status-warning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(251, 191, 36, 0.15) 100%);
        border-left: 4px solid #f59e0b;
        color: #fffbeb !important;
    }
    .status-error {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(248, 113, 113, 0.15) 100%);
        border-left: 4px solid #ef4444;
        color: #fef2f2 !important;
    }
    
    /* ULTRA HIGH PRIORITY: Metric cards with maximum specificity */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.98) 0%, rgba(248, 250, 252, 0.95) 100%) !important;
        border: 2px solid rgba(59, 130, 246, 0.5) !important;
        border-radius: 12px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px rgba(30, 58, 138, 0.4) !important;
        backdrop-filter: blur(15px) !important;
        position: relative !important;
        z-index: 1000 !important;
    }
    
    /* MAXIMUM PRIORITY: Metric values - multiple selectors for redundancy */
    div[data-testid="metric-container"] [data-testid="stMetricValue"],
    div[data-testid="metric-container"] [data-testid="stMetricValue"] > div,
    div[data-testid="metric-container"] div[data-testid="stMetricValue"],
    .stMetric [data-testid="stMetricValue"] {
        color: #0f172a !important;
        font-size: 2.5rem !important;
        font-weight: 800 !important;
        line-height: 1.1 !important;
        text-shadow: none !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
    }
    
    /* MAXIMUM PRIORITY: Metric labels - multiple selectors */
    div[data-testid="metric-container"] [data-testid="stMetricLabel"],
    div[data-testid="metric-container"] [data-testid="stMetricLabel"] > div,
    div[data-testid="metric-container"] label,
    div[data-testid="metric-container"] .metric-label,
    .stMetric label,
    .stMetric [data-testid="stMetricLabel"] {
        color: #1e293b !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        margin-bottom: 8px !important;
        text-shadow: none !important;
    }
    
    /* MAXIMUM PRIORITY: Metric delta text */
    div[data-testid="metric-container"] [data-testid="stMetricDelta"],
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] > div,
    .stMetric [data-testid="stMetricDelta"] {
        color: #374151 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        opacity: 1 !important;
        visibility: visible !important;
        display: block !important;
    }
    
    /* Force visibility on all metric text */
    div[data-testid="metric-container"] * {
        color: inherit !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    
    /* Override any Streamlit defaults that might interfere */
    .stMetric {
        background: transparent !important;
    }
    
    .stMetric > div {
        background: transparent !important;
        color: inherit !important;
    }
    
    /* Enhanced buttons */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        color: #ffffff !important;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.6);
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
    }
    
    h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
    }
    
    /* Data tables styling */
    .stDataFrame {
        border-radius: 12px;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        background: rgba(30, 58, 138, 0.1) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(59, 130, 246, 0.1);
        padding: 8px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 500;
        color: #e0e7ff !important;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%) !important;
        color: #ffffff !important;
        font-weight: 600;
    }
    
    /* Input fields styling */
    .stTextInput input, .stTextArea textarea, .stNumberInput input {
        background: rgba(30, 58, 138, 0.2) !important;
        color: white !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 8px !important;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def status_message(message, message_type="info"):
    """Custom status message display"""
    # Clean HTML tags for fallback
    import re
    clean_message = re.sub(r'<[^>]+>', '', message)
    
    if message_type == "success":
        st.success(clean_message)
    elif message_type == "warning":
        st.warning(clean_message)
    elif message_type == "error":
        st.error(clean_message)
    else:  # info or default
        st.info(clean_message)

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
@st.cache_resource
def load_sentiment_model():
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    
    # Enhanced GPU detection
    print("=== GPU DIAGNOSTIC ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version in PyTorch: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        device = 0
        status_message("🚀 GPU detected! Using CUDA for acceleration", "success")
        torch.cuda.empty_cache()
    else:
        device = -1
        if torch.version.cuda is None:
            status_message("❌ PyTorch installed without CUDA support. Install CUDA version for GPU acceleration", "error")
        else:
            status_message("⚠️ CUDA not available. Check NVIDIA drivers and CUDA installation", "warning")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Move model to GPU if available
    if device == 0:
        model = model.cuda()
    
    nlp = pipeline(
        "sentiment-analysis", 
        model=model, 
        tokenizer=tokenizer,
        device=device,
        batch_size=128,  # Process 128 tweets at once
        max_length=512,
        truncation=True,
        padding=True
    )
    
    return nlp

@st.cache_data
def load_data(uploaded_file=None):
    tweets_bitcoin = None
    
    if uploaded_file is not None:
        # Reset file pointer
        uploaded_file.seek(0)
        
        # Strategy 1: Try automatic delimiter detection
        try:
            # Read a small sample to detect format
            sample = uploaded_file.read(2048).decode('utf-8', errors='ignore')
            uploaded_file.seek(0)
            
            # Count potential delimiters
            delimiters = [',', ';', '\t', '|']
            delimiter_counts = {d: sample.count(d) for d in delimiters}
            best_delimiter = max(delimiter_counts, key=delimiter_counts.get)
            
            tweets_bitcoin = pd.read_csv(
                uploaded_file,
                sep=best_delimiter,
                encoding='utf-8',
                on_bad_lines='skip',
                na_values=["?", "", "NULL", "null", "NaN", "nan"],
                dtype=str  # Read all as strings initially
            )
            
        except Exception as e1:
            uploaded_file.seek(0)
            
            # Strategy 2: Try common separators explicitly
            separators = [',', ';', '\t', '|']
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for sep in separators:
                for enc in encodings:
                    try:
                        tweets_bitcoin = pd.read_csv(
                            uploaded_file,
                            sep=sep,
                            encoding=enc,
                            on_bad_lines='skip',
                            na_values=["?", "", "NULL", "null", "NaN", "nan"],
                            dtype=str,
                            engine='python'  # More flexible parser
                        )
                        
                        if len(tweets_bitcoin.columns) > 1:  # Valid CSV should have multiple columns
                            status_message(f"✅ Successfully read with separator: '{sep}', encoding: '{enc}'", "success")
                            break
                        
                    except Exception:
                        continue
                    finally:
                        uploaded_file.seek(0)
                
                if tweets_bitcoin is not None and len(tweets_bitcoin.columns) > 1:
                    break
    else:
        try:
            tweets_bitcoin = pd.read_csv(
                'uncleaned_tweets.csv', 
                encoding='utf-8',
                on_bad_lines='skip',
                dtype=str
            )
        except FileNotFoundError:
            status_message("❌ No file uploaded and 'uncleaned_tweets.csv' not found", "error")
            return None, None
        except Exception as e:
            status_message(f"Error reading local CSV file: {e}", "error")
            return None, None

    if tweets_bitcoin is None or len(tweets_bitcoin) == 0:
        status_message("No valid data could be loaded from the CSV file", "error")
        return None, None

    # Display CSV info
    status_message(f"✅ Loaded CSV with {len(tweets_bitcoin)} rows and {len(tweets_bitcoin.columns)} columns", "success")
    status_message(f"📋 Column names: {', '.join(tweets_bitcoin.columns)}", "info")
    
    # Check for duplicate columns
    duplicate_columns_found = []
    if 'Datetime' in tweets_bitcoin.columns and 'user_created' in tweets_bitcoin.columns:
        # Check if values are identical
        identical_mask = tweets_bitcoin['Datetime'].fillna('') == tweets_bitcoin['user_created'].fillna('')
        num_identical = identical_mask.sum()
        total_rows = len(tweets_bitcoin)
        identical_percentage = (num_identical / total_rows) * 100 if total_rows > 0 else 0
        
        if identical_percentage > 95:  # Consider 95%+ as identical
            duplicate_columns_found.append(('Datetime', 'user_created'))
            status_message(f"🚨 CRITICAL: 'Datetime' and 'user_created' are {identical_percentage:.1f}% identical!", "error")
            
            # Drop both problematic columns
            columns_to_drop = ['Datetime', 'user_created']
            existing_drops = [col for col in columns_to_drop if col in tweets_bitcoin.columns]
            if existing_drops:
                tweets_bitcoin = tweets_bitcoin.drop(columns=existing_drops)
                status_message(f"✅ Dropped duplicate columns: {', '.join(existing_drops)}", "success")

    # Check for common column name variations
    text_columns = [col for col in tweets_bitcoin.columns if 'text' in col.lower() or 'tweet' in col.lower() or 'content' in col.lower()]
    id_columns = [col for col in tweets_bitcoin.columns if 'id' in col.lower()]
    
    # Prioritize 'date' column for temporal analysis
    date_columns = []
    if 'date' in tweets_bitcoin.columns:
        date_columns = ['date']
        status_message("🎯 Using 'date' column for temporal and market correlation analysis!", "success")
    else:
        alternative_date_cols = [col for col in tweets_bitcoin.columns if any(word in col.lower() for word in ['date', 'time', 'created'])]
        if alternative_date_cols:
            date_columns = alternative_date_cols
            status_message(f"⚠️ No 'date' column found. Alternative: {date_columns}", "warning")
        else:
            status_message("❌ No date-related columns found", "error")
    
    # Standardize column names
    if text_columns and not 'Text' in tweets_bitcoin.columns:
        tweets_bitcoin['Text'] = tweets_bitcoin[text_columns[0]]
        status_message(f"📝 Using '{text_columns[0]}' as Text column", "info")
    
    if id_columns and not 'Tweet Id' in tweets_bitcoin.columns:
        tweets_bitcoin['Tweet Id'] = tweets_bitcoin[id_columns[0]]
        status_message(f"🆔 Using '{id_columns[0]}' as Tweet Id column", "info")
    
    # Handle date columns
    if date_columns and 'date' in date_columns:
        status_message(f"📅 Using 'date' column for market correlation analysis", "info")
    elif date_columns and not 'date' in tweets_bitcoin.columns:
        tweets_bitcoin['date'] = tweets_bitcoin[date_columns[0]]
        status_message(f"📅 Renamed '{date_columns[0]}' to 'date'", "info")
    
    # Final validation
    if 'Text' not in tweets_bitcoin.columns:
        status_message("❌ No text content column found", "error")
        return None, None
    
    if 'Tweet Id' not in tweets_bitcoin.columns:
        tweets_bitcoin['Tweet Id'] = range(len(tweets_bitcoin))
        status_message("ℹ️ No ID column found. Created sequential IDs", "info")
    
    # Clean up the data
    tweets_bitcoin = tweets_bitcoin.dropna(subset=['Text'])
    tweets_bitcoin = tweets_bitcoin[tweets_bitcoin['Text'].str.strip() != '']
    
    status_message(f"📊 After basic cleaning: {len(tweets_bitcoin)} tweets remaining", "info")

    # Load market data - DYNAMICALLY MATCH TWEET DATE RANGE
    market_data = None
    
    # First, determine date range from tweets
    if 'date' in tweets_bitcoin.columns:
        try:
            # Parse tweet dates to find range
            tweet_dates = pd.to_datetime(tweets_bitcoin['date'], errors='coerce')
            valid_dates = tweet_dates.dropna()
            
            if len(valid_dates) > 0:
                start_date = valid_dates.min().strftime('%Y-%m-%d')
                end_date = valid_dates.max().strftime('%Y-%m-%d')
                status_message(f"📅 Tweet date range: {start_date} to {end_date}", "info")
            else:
                # Fallback if no valid dates
                start_date = '2020-08-01'
                end_date = '2024-08-01'
                status_message("⚠️ No valid dates found, using default range", "warning")
        except Exception as e:
            status_message(f"Error parsing tweet dates: {e}", "warning")
            start_date = '2020-08-01'
            end_date = '2024-08-01'
    else:
        status_message("No date column found, using default market data range", "info")
        start_date = '2020-08-01'
        end_date = '2024-08-01'
    
    try:
        status_message(f"📈 Fetching Bitcoin market data from {start_date} to {end_date}...", "info")
        
        market_data_raw = yf.download("BTC-USD", start=start_date, end=end_date, interval='1d', progress=False)
        
        if market_data_raw is None or market_data_raw.empty:
            ticker = yf.Ticker("BTC-USD")
            market_data_raw = ticker.history(start=start_date, end=end_date)
        
        if market_data_raw is not None and not market_data_raw.empty:
            status_message(f"✅ Successfully fetched {len(market_data_raw)} days of market data", "success")
            
            market_data = market_data_raw.copy()
            
            # Handle multi-level columns
            if hasattr(market_data.columns, 'levels'):
                status_message("🔧 Flattening multi-level columns from yfinance", "info")
                market_data.columns = market_data.columns.get_level_values(0)
            
            # Reset index to convert Date
            market_data = market_data.reset_index()
            
            if 'Date' not in market_data.columns and market_data.index.name == 'Date':
                market_data = market_data.reset_index()
            
            # Convert Date to datetime and then to date
            market_data['Date'] = pd.to_datetime(market_data['Date']).dt.date
            market_data = market_data.set_index('Date')
            
            # Add growth column
            try:
                market_data['growth'] = np.where(market_data["Open"] > market_data["Close"], "negative", "positive")
            except Exception as e:
                status_message(f"⚠️ Could not calculate growth column: {e}", "warning")
            
            status_message("🎯 Market data processed successfully", "success")
            
        else:
            status_message("❌ Unable to fetch market data", "error")
            market_data = None
            
    except Exception as e:
        status_message(f"❌ Error loading market data: {str(e)}", "error")
        market_data = None

    if market_data is not None:
        status_message(f"📊 Final market data: {len(market_data)} days loaded", "success")
    else:
        status_message("⚠️ Continuing without market data", "warning")

    return tweets_bitcoin, market_data

# =============================================================================
# DATA PREPROCESSING FUNCTIONS
# =============================================================================

def Drop_Unused_Columns(data):
    """Function to drop unnecessary columns from the dataset"""
    columns_to_drop = [
        'user_name', 'user_location', 'user_description', 'user_created',
        'user_followers', 'user_friends', 'user_favourites', 'user_verified',
        'hashtags', 'source', 'is_retweet'
    ]
    
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    
    if existing_columns_to_drop:
        status_message(f"🗑️ Dropping unused columns: {', '.join(existing_columns_to_drop)}", "info")
        return data.drop(columns=existing_columns_to_drop)
    else:
        status_message("ℹ️ No unused columns found to drop", "info")
        return data

def Preprocess_Tweets(data):
    """Comprehensive preprocessing with detailed tracking"""
    data = data.copy()

    # Check for Text column
    if 'Text' in data.columns:
        col = 'Text'
    elif 'text' in data.columns:
        data = data.rename(columns={'text': 'Text'})
        col = 'Text'
    else:
        raise KeyError("The required column 'Text' is not found in the uploaded data.")
    
    # Initialize tracking columns
    data['Removed_Hashtags'] = ""
    data['Removed_Mentions'] = ""
    data['Removed_Links'] = ""
    data['Removed_Emojis'] = ""
    data['Original_Text'] = data[col]
    
    # Process each tweet
    for idx, row in data.iterrows():
        text = str(row[col])
        removed_hashtags = []
        removed_mentions = []
        removed_links = []
        removed_emojis = []
        
        # Extract and remove hashtags
        hashtags = re.findall(r'#\w+', text)
        removed_hashtags.extend(hashtags)
        text = re.sub(r'#\w+', ' [HASHTAG] ', text)
        
        # Extract and remove mentions
        mentions = re.findall(r'@\w+', text)
        removed_mentions.extend(mentions)
        text = re.sub(r'@\w+', ' [MENTION] ', text)
        
        # Extract and remove links
        links = re.findall(r'https?://[^\s<>"{}|\\^`\[\]]*|www\.[^\s<>"{}|\\^`\[\]]*', text)
        removed_links.extend(links)
        text = re.sub(r'https?://[^\s<>"{}|\\^`\[\]]*|www\.[^\s<>"{}|\\^`\[\]]*', ' [LINK] ', text)
        
        # Extract and remove emojis
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"  # Dingbats
            u"\U000024C2-\U0001F251"  # Enclosed characters
            "]+", flags=re.UNICODE)
        emojis = emoji_pattern.findall(text)
        removed_emojis.extend(emojis)
        text = emoji_pattern.sub(' [EMOJI] ', text)
        
        # Remove emoticons
        emoticons_pattern = r'(:\)|:-\)|:\(|:-\(|:d|:-d|:p|:-p|;\)|;-\)|:\||:-\||<3|:\*|:\^\)|:\'\()'
        text = re.sub(emoticons_pattern, ' [EMOTICON] ', text, flags=re.IGNORECASE)
        
        # Remove standalone 'http' or 'https'
        text = re.sub(r'\bhttps?\b', ' [LINK] ', text)
        
        # Remove extra numbers
        text = re.sub(r'\b\d+\b', ' [NUMBER] ', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[.!?]{2,}', ' [PUNCTUATION] ', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Store results
        data.at[idx, 'Removed_Hashtags'] = ', '.join(removed_hashtags) if removed_hashtags else ""
        data.at[idx, 'Removed_Mentions'] = ', '.join(removed_mentions) if removed_mentions else ""
        data.at[idx, 'Removed_Links'] = ', '.join(removed_links) if removed_links else ""
        data.at[idx, 'Removed_Emojis'] = ', '.join(removed_emojis) if removed_emojis else ""
        data.at[idx, 'Text_Cleaned'] = text.lower()
    
    # Remove very short tweets
    data['word_count'] = data['Text_Cleaned'].str.split().str.len()
    data = data[data['word_count'] >= 3]
    
    return data

# =============================================================================
# SENTIMENT ANALYSIS FUNCTIONS
# =============================================================================

def analyze_sentiment(tweets, nlp, sample_size=1000):
    """Fixed sentiment analysis function"""
    # Generate new random seed each time
    random_seed = np.random.randint(0, 100000)
    sample_tweets = tweets.sample(min(sample_size, len(tweets)), random_state=random_seed)

    sent_results = {}
    
    for i, d in tqdm(sample_tweets.iterrows(), total=len(sample_tweets)):
        try:
            sent = nlp(d["Text_Cleaned"][:512])  # Truncate to 512 tokens
            sent_results[d["Tweet Id"]] = sent
        except:
            continue
    
    sent_df = pd.DataFrame(sent_results).T
    sent_df["label"] = sent_df[0].apply(lambda x: x["label"])
    sent_df["score"] = sent_df[0].apply(lambda x: x["score"])
    sent_df = sent_df.merge(tweets.set_index("Tweet Id"), left_index=True, right_index=True)
    
    # Enhanced date processing - prioritize 'date' column
    date_column = None
    if 'date' in sent_df.columns:
        date_column = 'date'
        status_message("🎯 Using 'date' column for time-series analysis", "success")
    elif 'Datetime' in sent_df.columns:
        date_column = 'Datetime'
        status_message("📅 Using 'Datetime' column for time-series analysis", "info")
    
    if date_column:
        try:
            # Try multiple datetime formats
            sent_df['Date'] = pd.to_datetime(sent_df[date_column], format='mixed', errors='coerce')
            
            if sent_df['Date'].isna().all():
                # Try common formats
                formats_to_try = [
                    '%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%m/%d/%Y %H:%M:%S',
                    '%m/%d/%Y %H:%M', '%d/%m/%Y %H:%M:%S', '%d/%m/%Y %H:%M',
                    '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y'
                ]
                
                for fmt in formats_to_try:
                    try:
                        sent_df['Date'] = pd.to_datetime(sent_df[date_column], format=fmt, errors='coerce')
                        if not sent_df['Date'].isna().all():
                            status_message(f"Successfully parsed dates with format: {fmt}", "info")
                            break
                    except:
                        continue
            
            # Convert to date only
            if not sent_df['Date'].isna().all():
                sent_df['Date'] = sent_df["Date"].dt.date
                status_message(f"✅ Date parsing successful. Found {sent_df['Date'].notna().sum()} valid dates", "success")
            else:
                status_message("Could not parse datetime column. Time-series analysis will be skipped", "warning")
                if 'Date' in sent_df.columns:
                    sent_df = sent_df.drop('Date', axis=1)
                    
        except Exception as e:
            status_message(f"Date parsing failed: {e}. Time-series analysis will be skipped", "warning")
            if 'Date' in sent_df.columns:
                sent_df = sent_df.drop('Date', axis=1)
    else:
        status_message("No datetime column found. Time-series analysis will be skipped", "info")

    return sent_df

# =============================================================================
# MARKET CORRELATION FUNCTIONS
# =============================================================================
def create_advanced_market_correlation_analysis(sent_df, market_data):
    """Enhanced correlation analysis with advanced methods for quant finance"""
    try:
        if 'Date' not in sent_df.columns or sent_df['Date'].isna().all():
            st.warning("⚠️ No valid dates in sentiment data - skipping correlation analysis")
            return None
            
        # Clean and prepare sentiment data
        sent_df_clean = sent_df.dropna(subset=['Date']).copy()
        sent_df_clean['Date'] = pd.to_datetime(sent_df_clean['Date']).dt.date
        
        # Aggregate sentiment by date with additional metrics
        sentiment_daily = sent_df_clean.groupby('Date').agg({
            'score': ['mean', 'std', 'count'],
            'label': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'Neutral'
        }).reset_index()
        
        sentiment_daily.columns = ['Date', 'avg_confidence', 'confidence_std', 'tweet_count', 'dominant_sentiment']
        
        # Clean and prepare market data
        market_df = market_data.copy()
        if market_df.index.name == 'Date' or 'Date' in str(market_df.index.names):
            market_df = market_df.reset_index()
        
        if hasattr(market_df.columns, 'levels'):
            market_df.columns = market_df.columns.get_level_values(0)
        
        if 'Date' in market_df.columns:
            market_df['Date'] = pd.to_datetime(market_df['Date']).dt.date
        else:
            st.error("❌ No Date column found in market data")
            return None
        
        # Merge datasets
        combined_data = pd.merge(sentiment_daily, market_df, on='Date', how='inner')
        if len(combined_data) == 0:
            st.warning("⚠️ No overlapping dates found between sentiment and market data")
            return None
        
        # Sort by date for time series analysis
        combined_data = combined_data.sort_values('Date').reset_index(drop=True)
        
        # Map sentiment to numeric polarity
        sentiment_map = {'Positive': 1, 'Negative': -1, 'Neutral': 0}
        combined_data['sentiment_numeric'] = combined_data['dominant_sentiment'].map(sentiment_map)
        
        # Calculate returns and volatility
        combined_data['price_return'] = combined_data['Close'].pct_change()
        combined_data['log_return'] = np.log(combined_data['Close'] / combined_data['Close'].shift(1))
        combined_data['volatility'] = combined_data['price_return'].rolling(window=5).std()
        
        # Remove initial NaN values
        analysis_data = combined_data.dropna().copy()
        
        if len(analysis_data) < 20:
            st.warning("⚠️ Insufficient data points for advanced analysis")
            return combined_data
        
        st.markdown("#### 🎯 Advanced Quantitative Correlation Analysis")
        
        # 1. CROSS-CORRELATION FUNCTION ANALYSIS
        st.markdown("##### 📊 Cross-Correlation Function Analysis")
        
        def cross_correlation_analysis(x, y, max_lags=10):
            """Advanced cross-correlation with significance testing"""
            x_norm = (x - np.mean(x)) / np.std(x)
            y_norm = (y - np.mean(y)) / np.std(y)
            
            cross_corr = correlate(x_norm, y_norm, mode='full')
            cross_corr = cross_corr / len(x_norm)
            
            lags = np.arange(-len(x_norm)+1, len(x_norm))
            
            # Focus on reasonable lags
            center = len(cross_corr) // 2
            lag_range = min(max_lags, len(x_norm)//4)
            start_idx = center - lag_range
            end_idx = center + lag_range + 1
            
            selected_corr = cross_corr[start_idx:end_idx]
            selected_lags = lags[start_idx:end_idx]
            
            # Calculate confidence intervals (95%)
            n = len(x_norm)
            confidence_bound = 1.96 / np.sqrt(n)
            
            return selected_lags, selected_corr, confidence_bound
        
        sentiment_series = analysis_data['sentiment_numeric'].values
        price_series = analysis_data['log_return'].values
        
        lags, cross_corr, conf_bound = cross_correlation_analysis(sentiment_series, price_series)
        
        # Find significant correlations
        significant_corr = np.abs(cross_corr) > conf_bound
        best_lag_idx = np.argmax(np.abs(cross_corr))
        best_lag = lags[best_lag_idx]
        best_corr = cross_corr[best_lag_idx]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cross-correlation plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=lags, y=cross_corr,
                mode='lines+markers',
                name='Cross-correlation',
                line=dict(color='#3b82f6', width=2)
            ))
            
            # Add confidence bounds
            fig.add_hline(y=conf_bound, line_dash="dash", line_color="red", 
                         annotation_text="95% Confidence")
            fig.add_hline(y=-conf_bound, line_dash="dash", line_color="red")
            
            # Highlight significant correlations
            significant_lags = lags[significant_corr]
            significant_values = cross_corr[significant_corr]
            
            fig.add_trace(go.Scatter(
                x=significant_lags, y=significant_values,
                mode='markers',
                name='Significant',
                marker=dict(color='red', size=8, symbol='diamond')
            ))
            
            fig.update_layout(
                title="Cross-Correlation Function: Sentiment vs Returns",
                xaxis_title="Lag (days)",
                yaxis_title="Correlation Coefficient",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("🎯 Optimal Lag", f"{best_lag} days", 
                     help="Lag with strongest cross-correlation")
            st.metric("📈 Max Cross-Correlation", f"{best_corr:.4f}",
                     help="Strongest correlation coefficient found")
            st.metric("✅ Significant Lags", f"{np.sum(significant_corr)}",
                     help="Number of statistically significant correlations")
            st.metric("📊 Confidence Bound", f"±{conf_bound:.4f}",
                     help="95% statistical significance threshold")
        
        # 2. GRANGER CAUSALITY TESTS
        st.markdown("##### 🔄 Granger Causality Analysis")
        
        def perform_granger_causality(data, max_lags=5):
            """Granger causality tests with multiple lags"""
            
            # Prepare data for Granger test (remove any remaining NaN)
            test_data = data[['sentiment_numeric', 'log_return']].dropna()
            
            if len(test_data) < max_lags * 3:  # Need sufficient data
                return None, None
            
            results = {}
            causality_direction = {}
            
            try:
                # Test if sentiment Granger-causes returns
                test_result = grangercausalitytests(
                    test_data[['log_return', 'sentiment_numeric']], 
                    maxlag=max_lags, verbose=False
                )
                
                results['sentiment_to_returns'] = test_result
                
                # Extract p-values for different lags
                p_values = []
                f_stats = []
                for lag in range(1, max_lags + 1):
                    if lag in test_result:
                        p_val = test_result[lag][0]['ssr_ftest'][1]  # F-test p-value
                        f_stat = test_result[lag][0]['ssr_ftest'][0]  # F-statistic
                        p_values.append(p_val)
                        f_stats.append(f_stat)
                
                # Find the most significant result
                if p_values:
                    min_p_idx = np.argmin(p_values)
                    causality_direction['sentiment_to_returns'] = {
                        'optimal_lag': min_p_idx + 1,
                        'p_value': p_values[min_p_idx],
                        'f_statistic': f_stats[min_p_idx],
                        'is_significant': p_values[min_p_idx] < 0.05
                    }
                
            except Exception as e:
                st.warning(f"Could not perform sentiment→returns Granger test: {str(e)}")
                causality_direction['sentiment_to_returns'] = None
            
            try:
                # Test if returns Granger-cause sentiment
                test_result = grangercausalitytests(
                    test_data[['sentiment_numeric', 'log_return']], 
                    maxlag=max_lags, verbose=False
                )
                
                results['returns_to_sentiment'] = test_result
                
                p_values = []
                f_stats = []
                for lag in range(1, max_lags + 1):
                    if lag in test_result:
                        p_val = test_result[lag][0]['ssr_ftest'][1]
                        f_stat = test_result[lag][0]['ssr_ftest'][0]
                        p_values.append(p_val)
                        f_stats.append(f_stat)
                
                if p_values:
                    min_p_idx = np.argmin(p_values)
                    causality_direction['returns_to_sentiment'] = {
                        'optimal_lag': min_p_idx + 1,
                        'p_value': p_values[min_p_idx],
                        'f_statistic': f_stats[min_p_idx],
                        'is_significant': p_values[min_p_idx] < 0.05
                    }
                    
            except Exception as e:
                st.warning(f"Could not perform returns→sentiment Granger test: {str(e)}")
                causality_direction['returns_to_sentiment'] = None
            
            return results, causality_direction
        
        granger_results, causality = perform_granger_causality(analysis_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔄 Sentiment → Returns Causality**")
            if causality.get('sentiment_to_returns'):
                result = causality['sentiment_to_returns']
                significance = "✅ Significant" if result['is_significant'] else "❌ Not Significant"
                st.metric("Statistical Significance", significance)
                st.metric("Optimal Lag", f"{result['optimal_lag']} days")
                st.metric("P-value", f"{result['p_value']:.6f}")
                st.metric("F-statistic", f"{result['f_statistic']:.3f}")
            else:
                st.info("Could not calculate sentiment→returns causality")
        
        with col2:
            st.markdown("**🔄 Returns → Sentiment Causality**")
            if causality.get('returns_to_sentiment'):
                result = causality['returns_to_sentiment']
                significance = "✅ Significant" if result['is_significant'] else "❌ Not Significant"
                st.metric("Statistical Significance", significance)
                st.metric("Optimal Lag", f"{result['optimal_lag']} days")
                st.metric("P-value", f"{result['p_value']:.6f}")
                st.metric("F-statistic", f"{result['f_statistic']:.3f}")
            else:
                st.info("Could not calculate returns→sentiment causality")
        
        # 3. VAR MODEL ANALYSIS
        st.markdown("##### 📈 Vector Autoregression (VAR) Model Analysis")
        
        def var_model_analysis(data, max_lags=5):
            """Fit VAR model and analyze impulse responses"""
            
            # Prepare stationary data
            model_data = data[['sentiment_numeric', 'log_return']].dropna()
            
            if len(model_data) < max_lags * 4:
                return None, None
            
            try:
                # Test for stationarity
                def check_stationarity(series, name):
                    result = adfuller(series.dropna())
                    return {
                        'name': name,
                        'adf_stat': result[0],
                        'p_value': result[1],
                        'is_stationary': result[1] < 0.05
                    }
                
                stationarity_results = {}
                stationarity_results['sentiment'] = check_stationarity(model_data['sentiment_numeric'], 'Sentiment')
                stationarity_results['returns'] = check_stationarity(model_data['log_return'], 'Returns')
                
                # Fit VAR model
                var_model = VAR(model_data)
                
                # Select optimal lag using information criteria
                lag_selection = var_model.select_order(maxlags=max_lags)
                optimal_lag = lag_selection.aic  # Use AIC for selection
                
                # Fit model with optimal lag
                var_fitted = var_model.fit(optimal_lag)
                
                # Generate impulse response functions
                irf = var_fitted.irf(periods=10)
                
                return var_fitted, {
                    'lag_selection': lag_selection,
                    'optimal_lag': optimal_lag,
                    'irf': irf,
                    'stationarity': stationarity_results
                }
                
            except Exception as e:
                st.warning(f"VAR model fitting error: {str(e)}")
                return None, None
        
        var_model, var_results = var_model_analysis(analysis_data)
        
        if var_model and var_results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🎯 Optimal VAR Lag", f"{var_results['optimal_lag']}")
                st.metric("📊 AIC Score", f"{var_results['lag_selection'].aic}")
            
            with col2:
                # Stationarity test results
                sent_stat = var_results['stationarity']['sentiment']
                returns_stat = var_results['stationarity']['returns']
                
                sent_status = "✅ Stationary" if sent_stat['is_stationary'] else "❌ Non-stationary"
                returns_status = "✅ Stationary" if returns_stat['is_stationary'] else "❌ Non-stationary"
                
                st.metric("Sentiment Stationarity", sent_status)
                st.metric("Returns Stationarity", returns_status)
            
            with col3:
                # Model fit statistics
                try:
                    st.metric("📈 Model R²", f"{var_model.rsquared['sentiment_numeric']:.4f}")
                    st.metric("🎯 Log Likelihood", f"{var_model.llf:.2f}")
                except:
                    st.info("Model statistics available in summary")
            
            # VAR Model Summary
            with st.expander("📋 VAR Model Detailed Results", expanded=False):
                st.text(str(var_model.summary()))
                
                # Impulse Response Analysis
                if var_results.get('irf'):
                    st.markdown("**📈 Impulse Response Functions**")
                    
                    try:
                        # Plot IRF
                        irf_data = var_results['irf']
                        periods = range(10)
                        
                        # Sentiment shock to Returns
                        sentiment_to_returns = irf_data.irfs[1, 0, :]  # Returns response to sentiment shock
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=list(periods), 
                            y=sentiment_to_returns,
                            mode='lines+markers',
                            name='Returns Response to Sentiment Shock',
                            line=dict(color='#22c55e', width=3)
                        ))
                        
                        fig.update_layout(
                            title="Impulse Response: Sentiment Shock → Returns Response",
                            xaxis_title="Periods",
                            yaxis_title="Response Magnitude",
                            height=300
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as irf_error:
                        st.info("Impulse response visualization not available")
        
        else:
            st.info("VAR model analysis could not be completed with available data")
        
        # 4. ADVANCED STATISTICAL SUMMARY
        st.markdown("##### 📊 Advanced Statistical Summary")
        
        # Calculate additional advanced metrics
        advanced_metrics = {}
        
        # Rolling correlation
        rolling_window = min(30, len(analysis_data)//3)
        if rolling_window >= 5:
            rolling_corr = analysis_data['sentiment_numeric'].rolling(rolling_window).corr(
                analysis_data['log_return']
            )
            advanced_metrics['rolling_corr_mean'] = rolling_corr.mean()
            advanced_metrics['rolling_corr_std'] = rolling_corr.std()
        
        # Correlation with volatility
        if 'volatility' in analysis_data.columns:
            vol_corr = analysis_data['sentiment_numeric'].corr(analysis_data['volatility'])
            advanced_metrics['sentiment_volatility_corr'] = vol_corr
        
        # Display advanced metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if 'rolling_corr_mean' in advanced_metrics:
                st.metric("📈 Rolling Correlation (Mean)", 
                         f"{advanced_metrics['rolling_corr_mean']:.4f}")
        
        with col2:
            if 'rolling_corr_std' in advanced_metrics:
                st.metric("📊 Rolling Correlation (Std)", 
                         f"{advanced_metrics['rolling_corr_std']:.4f}")
        
        with col3:
            if 'sentiment_volatility_corr' in advanced_metrics:
                st.metric("💥 Sentiment-Volatility Corr", 
                         f"{advanced_metrics['sentiment_volatility_corr']:.4f}")
        
        with col4:
            # Information ratio (if applicable)
            if len(analysis_data) > 0:
                info_ratio = analysis_data['sentiment_numeric'].std() / (analysis_data['log_return'].std() + 1e-8)
                st.metric("📊 Information Ratio", f"{info_ratio:.4f}")
        
        # Professional interpretation
        st.markdown("##### 🎯 Quantitative Analysis Interpretation")
        
        interpretation = []
        
        if best_corr != 0:
            strength = "strong" if abs(best_corr) > 0.3 else "moderate" if abs(best_corr) > 0.1 else "weak"
            direction = "positive" if best_corr > 0 else "negative"
            interpretation.append(
                f"📊 **Cross-correlation analysis** reveals a {strength} {direction} "
                f"relationship (r={best_corr:.4f}) at {best_lag}-day lag."
            )
        
        if causality:
            if causality.get('sentiment_to_returns', {}).get('is_significant'):
                lag = causality['sentiment_to_returns']['optimal_lag']
                p_val = causality['sentiment_to_returns']['p_value']
                interpretation.append(
                    f"🔄 **Granger causality** confirms sentiment predicts returns "
                    f"with {lag}-day lag (p={p_val:.4f})."
                )
            
            if causality.get('returns_to_sentiment', {}).get('is_significant'):
                lag = causality['returns_to_sentiment']['optimal_lag']
                p_val = causality['returns_to_sentiment']['p_value']
                interpretation.append(
                    f"🔄 **Reverse causality** detected: returns influence sentiment "
                    f"with {lag}-day lag (p={p_val:.4f})."
                )
        
        if var_results and var_results.get('optimal_lag'):
            interpretation.append(
                f"📈 **VAR model** suggests optimal {var_results['optimal_lag']}-lag "
                f"specification for joint sentiment-returns dynamics."
            )
        
        for item in interpretation:
            st.info(item)
        
        return combined_data
            
    except Exception as e:
        st.error(f"❌ Error in advanced correlation analysis: {str(e)}")
        import traceback
        st.text("Detailed error information:")
        st.text(traceback.format_exc())
        return None

def create_enhanced_time_series_plots(combined_data):
    """Enhanced time series visualizations"""
    
    st.markdown("##### 📊 Enhanced Time Series Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Price and sentiment with secondary axis
        fig = go.Figure()
        
        # Bitcoin price
        fig.add_trace(go.Scatter(
            x=combined_data['Date'],
            y=combined_data['Close'],
            mode='lines',
            name='BTC Price (USD)',
            line=dict(color='#f59e0b', width=3),
            yaxis='y'
        ))
        
        # Add price volatility if available
        if 'volatility' in combined_data.columns:
            fig.add_trace(go.Scatter(
                x=combined_data['Date'],
                y=combined_data['volatility'] * 100000,  # Scale for visibility
                mode='lines',
                name='Price Volatility (scaled)',
                line=dict(color='#ef4444', width=2, dash='dot'),
                yaxis='y'
            ))
        
        fig.update_layout(
            title="Bitcoin Price and Volatility Over Time",
            xaxis_title="Date",
            yaxis=dict(title="Price (USD)", side="left"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sentiment metrics over time
        fig = go.Figure()
        
        # Average sentiment confidence
        fig.add_trace(go.Scatter(
            x=combined_data['Date'],
            y=combined_data['avg_confidence'],
            mode='lines+markers',
            name='Avg Sentiment Confidence',
            line=dict(color='#3b82f6', width=2),
            marker=dict(size=4)
        ))
        
        # Tweet volume
        if 'tweet_count' in combined_data.columns:
            fig.add_trace(go.Scatter(
                x=combined_data['Date'],
                y=combined_data['tweet_count'],
                mode='lines',
                name='Tweet Volume',
                line=dict(color='#06b6d4', width=2),
                yaxis='y2'
            ))
            
            fig.update_layout(
                yaxis2=dict(title="Tweet Count", side="right", overlaying="y")
            )
        
        fig.update_layout(
            title="Sentiment Metrics Over Time",
            xaxis_title="Date",
            yaxis=dict(title="Sentiment Confidence"),
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def create_enhanced_scatter_analysis(combined_data):
    """Enhanced scatter plot analysis"""
    
    st.markdown("##### 🎯 Advanced Scatter Plot Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment vs Price with size = volume
        fig = px.scatter(
            combined_data,
            x='avg_confidence',
            y='Close',
            color='dominant_sentiment',
            size='tweet_count' if 'tweet_count' in combined_data.columns else None,
            hover_data=['Date'],
            title="Sentiment vs Price (Size = Tweet Volume)",
            labels={
                'avg_confidence': 'Avg Sentiment Confidence',
                'Close': 'BTC Price (USD)',
                'dominant_sentiment': 'Sentiment'
            },
            color_discrete_map={
                'Positive': '#22c55e',
                'Negative': '#ef4444',
                'Neutral': '#06b6d4'
            }
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Returns vs Sentiment scatter
        if 'log_return' in combined_data.columns:
            fig = px.scatter(
                combined_data,
                x='sentiment_numeric',
                y='log_return',
                color='tweet_count' if 'tweet_count' in combined_data.columns else 'avg_confidence',
                title="Returns vs Sentiment Polarity",
                labels={
                    'sentiment_numeric': 'Sentiment Polarity',
                    'log_return': 'Log Returns'
                }
            )
            
            # Add trend line
            fig.add_traces(px.scatter(
                combined_data, 
                x='sentiment_numeric', 
                y='log_return',
                trendline="ols"
            ).data)
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def create_market_movement_analysis(combined_data):
    """Enhanced market movement analysis"""
    
    st.markdown("##### 📈 Market Movement vs Sentiment Analysis")
    
    if 'price_change' not in combined_data.columns:
        combined_data['price_change'] = combined_data['Close'].pct_change() * 100
        combined_data['price_direction'] = combined_data['price_change'].apply(
            lambda x: 'Up' if x > 1 else 'Down' if x < -1 else 'Stable'
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sentiment vs price movement heatmap
        pivot_data = combined_data.groupby(['dominant_sentiment', 'price_direction']).size().unstack(fill_value=0)
        
        fig = px.imshow(
            pivot_data.values,
            labels=dict(x="Price Direction", y="Sentiment", color="Count"),
            x=pivot_data.columns,
            y=pivot_data.index,
            title="Sentiment vs Price Movement Heatmap",
            color_continuous_scale="Blues"
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution of price changes by sentiment
        fig = px.box(
            combined_data,
            x='dominant_sentiment',
            y='price_change',
            title="Price Change Distribution by Sentiment",
            labels={'price_change': 'Price Change (%)', 'dominant_sentiment': 'Sentiment'},
            color='dominant_sentiment',
            color_discrete_map={
                'Positive': '#22c55e',
                'Negative': '#ef4444',
                'Neutral': '#06b6d4'
            }
        )
        
        fig.add_hline(y=0, line_dash="dot", line_color="gray")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

def create_rolling_correlation_visualizations(combined_data):
    """Create rolling correlation analysis"""
    
    if len(combined_data) < 30:
        st.info("Insufficient data for rolling correlation analysis")
        return
    
    st.markdown("##### 📊 Rolling Correlation Analysis")
    
    # Calculate rolling correlations with different windows
    windows = [7, 14, 30]
    
    fig = go.Figure()
    
    for window in windows:
        if len(combined_data) >= window:
            rolling_corr = combined_data['sentiment_numeric'].rolling(window).corr(
                combined_data['Close']
            ).dropna()
            
            fig.add_trace(go.Scatter(
                x=combined_data['Date'].iloc[window-1:],
                y=rolling_corr,
                mode='lines',
                name=f'{window}-day Rolling Correlation',
                line=dict(width=2)
            ))
    
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.add_hline(y=0.3, line_dash="dash", line_color="green", 
                  annotation_text="Moderate Correlation")
    fig.add_hline(y=-0.3, line_dash="dash", line_color="red")
    
    fig.update_layout(
        title="Rolling Correlation: Sentiment vs Bitcoin Price",
        xaxis_title="Date",
        yaxis_title="Correlation Coefficient",
        height=400,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.title("₿ From Tweets to Trades: AI-Powered Bitcoin Sentiment & Market Timing")

    # Enhanced project description
    st.markdown("""
    <div style="text-align: center; margin: 30px 0; padding: 30px; 
                background: linear-gradient(135deg, #1e40af 0%, #3b82f6 25%, #60a5fa 75%, #93c5fd 100%);
                border-radius: 20px; color: white; box-shadow: 0 15px 35px rgba(30, 64, 175, 0.3);">
        <h3 style="color: white; font-size: 2rem; margin-bottom: 1rem;">🎯 Advanced AI-Powered Crypto Analysis</h3>
        <p style="font-size: 18px; line-height: 1.8; margin: 24px; font-weight: 400; opacity: 0.95;">
            This application leverages state-of-the-art Transformer models to analyze Social media sentiment 
            about Bitcoin and correlates it with real-time market data. Using deep learning techniques 
            including pre-trained RoBERTa models, we decode the complex relationship between social media 
            sentiment and cryptocurrency market volatility.
        </p>
        <div style="display: flex; justify-content: center; gap: 30px; margin-top: 20px; flex-wrap: wrap;">
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">🤖</div>
                <div style="font-weight: 600;">AI-Powered</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">📊</div>
                <div style="font-weight: 600;">Real-Time Data</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 2rem; margin-bottom: 8px;">🔗</div>
                <div style="font-weight: 600;">Market Correlation</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # File upload section
    st.markdown("---")
    st.markdown("### 📤 Data Upload & Configuration")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your tweets CSV file", 
            type=["csv"],
            help="Upload a CSV file containing tweet data with columns: Text, Tweet Id, date"
        )
    
    with col2:
        # CSV inspector
        if uploaded_file is not None:
            if st.button("🔍 Inspect CSV", help="Preview raw CSV content"):
                try:
                    uploaded_file.seek(0)
                    raw_content = uploaded_file.read(2000).decode('utf-8', errors='ignore')
                    uploaded_file.seek(0)
                    
                    st.text_area("Raw file content:", raw_content, height=200)
                    
                    lines = raw_content.split('\n')[:10]
                    st.markdown("**📊 File Analysis:**")
                    
                    separators = {',': 'Comma', ';': 'Semicolon', '\t': 'Tab', '|': 'Pipe'}
                    separator_counts = {}
                    
                    if lines:
                        first_line = lines[0]
                        for sep, name in separators.items():
                            count = first_line.count(sep)
                            if count > 0:
                                separator_counts[name] = count
                        
                        if separator_counts:
                            st.info(f"🎯 Detected separators: {separator_counts}")
                        else:
                            st.warning("No common separators detected")
                        
                        st.markdown("**📋 First few lines:**")
                        for i, line in enumerate(lines[:5]):
                            st.code(f"Line {i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")
                
                except Exception as e:
                    status_message(f"Error inspecting file: {e}", "error")
        
        # File format helper
        with st.expander("📋 Expected CSV Format", expanded=False):
            st.markdown("""
            **Your CSV should have:**
            - **Text column**: Tweet content (required) 📝
            - **ID column**: Tweet IDs (optional) 🆔  
            - **date column**: Timestamps (recommended) 📅
            
            **Supported formats:**
            - Comma separated (,) ✅
            - Semicolon separated (;) ✅
            - Tab separated ✅
            - Pipe separated (|) ✅
            """)
            
            sample_data = {
                'Text': ['Bitcoin is going to the moon! 🚀', 'Crypto market is volatile', 'HODL forever'],
                'Tweet Id': ['123456789', '987654321', '555666777'],
                'date': ['2022-01-01', '2022-01-02', '2022-01-03']
            }
            st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

    # Load data
    tweets_bitcoin, market_data = load_data(uploaded_file)

    if tweets_bitcoin is None:
        st.markdown("""
        <div style="text-align: center; padding: 40px; 
                    background: linear-gradient(135deg, rgba(239, 68, 68, 0.3) 0%, rgba(248, 113, 113, 0.2) 100%);
                    border-radius: 15px; border: 1px solid rgba(239, 68, 68, 0.2);">
            <h3 style="color: #dc2626; margin-bottom: 16px;">⚠️ No Data Available</h3>
            <p style="color: #f8fafc; font-size: 18px;">Please upload a CSV file with tweets to begin analysis</p>
            <p style="color: #f8fafc; font-size: 16px;">💡 Expected format: Text column, Tweet Id column, date column</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Dataset Overview
    st.markdown("### 📊 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Tweets", f"{len(tweets_bitcoin):,}", help="Total number of tweets")
    with col2:
        if 'Text' in tweets_bitcoin.columns:
            avg_length = tweets_bitcoin['Text'].str.len().mean()
            st.metric("📝 Avg Tweet Length", f"{avg_length:.0f} chars", help="Average character count")
    with col3:
        if market_data is not None:
            st.metric("📈 Market Data Days", len(market_data), help="Days of Bitcoin price data")
        else:
            st.metric("📈 Market Data", "Unavailable", help="Bitcoin price data unavailable")
    with col4:
        st.metric("🗂️ Data Columns", len(tweets_bitcoin.columns), help="Number of data columns")
    
    # Data Preview
    st.markdown("### 🔍 Data Preview & Structure")
    with st.expander("🔍 View raw tweet data", expanded=False):
        st.markdown("**📋 Raw Dataset Sample**")
        st.dataframe(tweets_bitcoin.head(10), use_container_width=True, height=350)
        
        st.markdown("**📊 Dataset Structure Analysis:**")
        cols_info = pd.DataFrame({
            'Column': tweets_bitcoin.columns,
            'Data Type': tweets_bitcoin.dtypes,
            'Non-Null Count': tweets_bitcoin.count(),
            'Null Count': tweets_bitcoin.isnull().sum(),
            'Null Percentage': (tweets_bitcoin.isnull().sum() / len(tweets_bitcoin) * 100).round(2)
        })
        st.dataframe(cols_info, use_container_width=True)
    
    # Data Preprocessing
    st.markdown("---")
    st.markdown("### 🧹 Data Preprocessing & Cleaning")
    
    # Sample size configuration
    max_sample_size = min(100000, len(tweets_bitcoin))
    sample_size = st.slider(
        "Sample size for preprocessing and analysis", 
        1000, 
        max_sample_size, 
        min(10000, max_sample_size),
        step=1000,
        help="Select number of tweets to process"
    )
    
    with st.spinner("🔄 Cleaning and preprocessing tweets..."):
        # Initialize Text_Cleaned column
        tweets_bitcoin["Text_Cleaned"] = tweets_bitcoin["Text"]
        
        # Show initial sample
        with st.expander("🔄 Initial Data Sample", expanded=False):
            st.dataframe(tweets_bitcoin.head(5), use_container_width=True)
        
        # Apply preprocessing
        cleaned_tweets = Preprocess_Tweets(tweets_bitcoin)
        cleaned_tweets['len'] = cleaned_tweets['Text_Cleaned'].astype(str).str.len()
        st.success("✅ Text preprocessing completed successfully")
        
        # Enhanced cleaning insights
        with st.expander("🔍 See what was removed during cleaning", expanded=False):
            st.markdown("**🧹 Cleaning Process Details**")
            
            # Show examples of removed content
            sample_for_cleaning = cleaned_tweets.head(5)
            for idx, row in sample_for_cleaning.iterrows():
                st.markdown(f"**Tweet {idx + 1}:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area("Original:", row['Original_Text'], height=100, key=f"orig_{idx}")
                with col2:
                    st.text_area("Cleaned:", row['Text_Cleaned'], height=100, key=f"clean_{idx}")
                
                if row['Removed_Hashtags'] or row['Removed_Mentions'] or row['Removed_Links']:
                    removed_items = []
                    if row['Removed_Hashtags']:
                        removed_items.append(f"Hashtags: {row['Removed_Hashtags']}")
                    if row['Removed_Mentions']:
                        removed_items.append(f"Mentions: {row['Removed_Mentions']}")
                    if row['Removed_Links']:
                        removed_items.append(f"Links: {row['Removed_Links']}")
                    st.info("Removed: " + " | ".join(removed_items))
                st.markdown("---")
        
        # Drop unused columns
        cleaned_tweets = Drop_Unused_Columns(cleaned_tweets)
        
        # Convert to lowercase
        cleaned_tweets['Text_Cleaned'] = cleaned_tweets['Text_Cleaned'].str.lower()
        
        # Show cleaned sample
        with st.expander("✨ Cleaned Data Sample", expanded=False):
            st.dataframe(cleaned_tweets.head(15), use_container_width=True)
        
        # FIXED: Calculate tweet lengths from cleaned data
        tweets_bitcoin['len'] = cleaned_tweets['len']  # Use cleaned lengths
        
        # Length distribution analysis
        st.markdown("#### 📝 Tweet Length Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                fig = px.histogram(
                    tweets_bitcoin,
                    x='len',
                    nbins=30,
                    title="📊 Original Tweet Length Distribution",
                    labels={'len': 'Tweet Length (characters)', 'count': 'Number of Tweets'},
                    color_discrete_sequence=["#530FD1"]
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#1e40af', size=11),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as hist_error:
                st.error(f"Error creating histogram: {hist_error}")
                st.info(f"Average length: {tweets_bitcoin['len'].mean():.1f} characters")
        
        # Filter tweets by length
        tweets_btc_filtrd = cleaned_tweets[cleaned_tweets['len'] <= 500]
        status_message(f"📊 Filtered to tweets ≤500 chars: {tweets_btc_filtrd.shape[0]:,} remaining", "info")
        
        # Sample for analysis
        BTC_Tweets = tweets_btc_filtrd.sample(
            min(sample_size, len(tweets_btc_filtrd)), 
            random_state=42
        )
        
        # Find longest tweet
        if len(tweets_btc_filtrd) > 0:
            max_length_tweet = tweets_btc_filtrd.loc[tweets_btc_filtrd['len'].idxmax()]
            status_message(f"📝 Longest tweet after filtering: {max_length_tweet['len']} characters", "info")
        
        with col2:
            try:
                fig = px.histogram(
                    BTC_Tweets,
                    x='len',
                    nbins=30,
                    title=f"📊 Filtered Sample Distribution (n={len(BTC_Tweets):,})",
                    labels={'len': 'Tweet Length (characters)', 'count': 'Number of Tweets'},
                    color_discrete_sequence=['#60a5fa']
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#1e40af', size=11),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as hist2_error:
                st.error(f"Error creating filtered histogram: {hist2_error}")
        
        # Show final sample
        with st.expander("🎯 Final Processed Sample", expanded=False):
            st.dataframe(BTC_Tweets.tail(15), use_container_width=True)
    
    # Before/After comparison
    st.markdown("#### 🔄 Before vs After Cleaning Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔄 Original Tweets (Raw)**")
        if 'Text' in tweets_bitcoin.columns:
            sample_original = tweets_bitcoin['Text'].dropna().head(3)
            for i, tweet in enumerate(sample_original):
                display_text = str(tweet)[:300] + "..." if len(str(tweet)) > 300 else str(tweet)
                st.text_area(f"Original Tweet {i+1}", display_text, height=120, key=f"original_tweet_{i}")
        else:
            st.info("Original text not available")
        
    with col2:
        st.markdown("**✨ Cleaned Tweets (Processed)**")
        sample_cleaned = BTC_Tweets['Text_Cleaned'].head(3)
        for i, tweet in enumerate(sample_cleaned):
            display_text = str(tweet)[:300] + "..." if len(str(tweet)) > 300 else str(tweet)
            st.text_area(f"Cleaned Tweet {i+1}", display_text, height=120, key=f"cleaned_tweet_{i}")
    
    # Processing statistics
    st.markdown("#### 📊 Processing Statistics Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        original_count = len(tweets_bitcoin)
        st.metric("🔄 Original Dataset", f"{original_count:,}", help="Total tweets before processing")
    with col2:
        filtered_count = len(tweets_btc_filtrd)
        reduction = original_count - filtered_count
        st.metric(
            "📝 After Length Filter", 
            f"{filtered_count:,}",
            delta=f"-{reduction:,}" if reduction > 0 else "0",
            help="Tweets after removing those >500 characters"
        )
    with col3:
        final_count = len(BTC_Tweets)
        st.metric("🎯 Final Sample Size", f"{final_count:,}", help="Tweets selected for sentiment analysis")
    with col4:
        avg_length = BTC_Tweets['len'].mean()
        st.metric("📝 Avg Length", f"{avg_length:.0f} chars", help="Average character count in final sample")
    
    # Apply styling to metric cards
    try:
        style_metric_cards(
            background_color="rgba(248, 250, 252, 0.95)",
            border_left_color="#3b82f6",
            border_color="rgba(59, 130, 246, 0.2)",
            border_radius_px=12,
            box_shadow=True
        )
    except Exception:
        pass  # Ignore styling errors

    # Advanced Data Analysis
    st.markdown("---")
    st.markdown("### 📈 Advanced Data Analysis & Visualizations")
    
    st.markdown("#### 📝 Comprehensive Length Distribution Analysis")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        try:
            fig = px.histogram(
                BTC_Tweets,
                x='len',
                nbins=40,
                title="📊 Distribution of Tweet Lengths After Cleaning",
                labels={'len': 'Tweet Length (characters)', 'count': 'Frequency'},
                color_discrete_sequence=["#184ca0"]
            )
            
            # Add statistical lines
            mean_val = BTC_Tweets['len'].mean()
            median_val = BTC_Tweets['len'].median()
            
            fig.add_vline(x=mean_val, line_dash="dash", line_color="#0f32a4", line_width=2)
            fig.add_vline(x=median_val, line_dash="dot", line_color="#5ea1f4", line_width=2)
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#071441", size=11),
                height=450
            )
            
            # Add annotations
            try:
                max_count = max(BTC_Tweets['len'].value_counts()) if len(BTC_Tweets) > 0 else 10
                fig.add_annotation(
                    x=mean_val, y=max_count * 0.8,
                    text=f"Mean: {mean_val:.0f}",
                    showarrow=True, arrowhead=2,
                    bgcolor="rgba(15, 50, 164, 0.8)", bordercolor="white",
                    font=dict(color="white", size=10)
                )
            except Exception:
                st.info(f"Mean={mean_val:.1f}, Median={median_val:.1f}")
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as length_dist_error:
            st.error(f"Error creating length distribution: {length_dist_error}")
            st.info(f"Mean length: {BTC_Tweets['len'].mean():.1f} characters")
    
    with col2:
        st.markdown("**📊 Length Statistics:**")
        try:
            stats_df = pd.DataFrame({
                'Statistic': ['Minimum', 'Maximum', 'Mean', 'Median', 'Std Dev', '25th %ile', '75th %ile'],
                'Value': [
                    f"{BTC_Tweets['len'].min():.0f}",
                    f"{BTC_Tweets['len'].max():.0f}",
                    f"{BTC_Tweets['len'].mean():.1f}",
                    f"{BTC_Tweets['len'].median():.0f}",
                    f"{BTC_Tweets['len'].std():.1f}",
                    f"{BTC_Tweets['len'].quantile(0.25):.0f}",
                    f"{BTC_Tweets['len'].quantile(0.75):.0f}"
                ]
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
        except Exception as stats_error:
            st.error(f"Error calculating statistics: {stats_error}")
    
    # Load sentiment model
    try:
        with st.spinner("🤖 Loading advanced sentiment analysis model..."):
            nlp = load_sentiment_model()
        status_message("✅ RoBERTa sentiment model loaded successfully!", "success")
    except Exception as e:
        status_message(f"❌ Error loading sentiment model: {e}", "error")
        st.stop()
    
    # Sentiment Analysis
    st.markdown("---")
    st.markdown("### 🎭 Advanced Sentiment Analysis")

    # Enhanced label mapping and colors
    label_mapping = {
        'LABEL_0': 'Negative',
        'LABEL_1': 'Neutral', 
        'LABEL_2': 'Positive',
        'negative': 'Negative',
        'neutral': 'Neutral',
        'positive': 'Positive'
    }    

    sentiment_colors = {
        'Negative': '#ef4444',
        'Neutral': '#06b6d4', 
        'Positive': '#22c55e'
    }  

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("**🔧 Configure Analysis Parameters:**")
        analysis_sample_size = st.slider(
            "Sample size for sentiment analysis", 
            100, 
            min(55000, len(BTC_Tweets)), 
            min(1000, len(BTC_Tweets)),
            help="Larger samples provide more comprehensive results"
        )
        
        # Add confidence threshold slider
        confidence_threshold = st.slider(
            "Minimum confidence threshold",
            0.5,
            1.0,
            0.7,
            0.05,
            help="Only include predictions above this confidence level"
        )
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(147, 197, 253, 0.1) 100%); 
                    padding: 20px; border-radius: 15px; border: 1px solid rgba(59, 130, 246, 0.2);">
            <strong>🤖 Model Information</strong><br><br>
            <strong>🤗 Architecture:</strong> HuggingFace Twitter RoBERTa<br>
            <strong>🎯 Labels:</strong> Positive, Negative, Neutral<br>
            <strong>📈 Confidence:</strong> Score 0-1<br>
            <strong>🔧 Max Tokens:</strong> 512 per tweet<br>
            <strong>⚡ Processing:</strong> GPU-accelerated
        </div>
        """, unsafe_allow_html=True)
    
    # Run sentiment analysis button
    if st.button("🚀 Analyze Sentiment", type="primary"):
        progress_bar = st.progress(0)
        status_container = st.empty()
        
        with st.spinner(f"🔄 Analyzing {analysis_sample_size:,} tweets with AI model..."):
            try:
                status_container.markdown("**Status:** Initializing sentiment analysis...")
                progress_bar.progress(10)
                
                # FIXED: Single function call without invalid parameter
                sent_df = analyze_sentiment(BTC_Tweets, nlp, sample_size=analysis_sample_size)
                progress_bar.progress(70)
                
                if len(sent_df) == 0:
                    status_message("❌ No tweets could be analyzed. Check data format", "error")
                    st.stop()
                
                # Map the labels to human-readable format
                sent_df['label'] = sent_df['label'].map(label_mapping)
                progress_bar.progress(80)
                
                # Filter by confidence threshold
                high_confidence_df = sent_df[sent_df['score'] >= confidence_threshold].copy()
                progress_bar.progress(90)
                
                status_message(f"✅ Successfully analyzed {len(sent_df):,} tweets!", "success")
                status_message(f"🎯 High confidence predictions: {len(high_confidence_df):,} ({len(high_confidence_df)/len(sent_df)*100:.1f}%)", "info")
                
                progress_bar.progress(100)
                status_container.empty()
                progress_bar.empty()
                
            except Exception as sentiment_error:
                st.error(f"Error in sentiment analysis: {sentiment_error}")
                st.stop()
            
            # Sentiment results overview
            st.markdown("#### 📋 Sentiment Analysis Results Dashboard")
            
            # Use high confidence data for metrics if available
            display_df = high_confidence_df if len(high_confidence_df) > 50 else sent_df
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            sentiment_counts = display_df['label'].value_counts()
            total_analyzed = len(display_df)
            
            with col1:
                positive_count = sentiment_counts.get('Positive', 0)
                positive_pct = (positive_count / total_analyzed) * 100 if total_analyzed > 0 else 0
                st.metric(
                    "😊 Positive Sentiment", 
                    f"{positive_pct:.1f}%",
                    delta=f"{positive_count:,} tweets",
                    help="Tweets expressing positive sentiment about Bitcoin"
                )
            with col2:
                negative_count = sentiment_counts.get('Negative', 0)
                negative_pct = (negative_count / total_analyzed) * 100 if total_analyzed > 0 else 0
                st.metric(
                    "😢 Negative Sentiment", 
                    f"{negative_pct:.1f}%",
                    delta=f"{negative_count:,} tweets",
                    help="Tweets expressing negative sentiment about Bitcoin"
                )
            with col3:
                neutral_count = sentiment_counts.get('Neutral', 0)
                neutral_pct = (neutral_count / total_analyzed) * 100 if total_analyzed > 0 else 0
                st.metric(
                    "😐 Neutral Sentiment", 
                    f"{neutral_pct:.1f}%",
                    delta=f"{neutral_count:,} tweets",
                    help="Tweets with neutral sentiment about Bitcoin"
                )
            with col4:
                avg_confidence = display_df['score'].mean() if len(display_df) > 0 else 0
                st.metric(
                    "🎯 Avg Confidence", 
                    f"{avg_confidence:.3f}",
                    delta=f"Min: {confidence_threshold:.2f}",
                    help="Average model confidence in predictions"
                )
            
            # Apply styling
            try:
                style_metric_cards(
                    background_color="rgba(248, 250, 252, 0.95)",
                    border_left_color="#3b82f6",
                    border_color="rgba(59, 130, 246, 0.2)",
                    border_radius_px=12,
                    box_shadow=True
                )
            except Exception:
                pass
            
            # Sample results
            with st.expander("🔍 Sample Sentiment Results with Confidence Scores", expanded=False):
                try:
                    display_sample = display_df[['Text_Cleaned', 'label', 'score']].head(20).copy()
                    display_sample.columns = ['Tweet Text', 'Sentiment', 'Confidence']
                    display_sample['Confidence'] = display_sample['Confidence'].round(3)
                    display_sample['Tweet Text'] = display_sample['Tweet Text'].apply(
                        lambda x: x[:100] + '...' if len(x) > 100 else x
                    )
                    st.dataframe(display_sample, use_container_width=True, hide_index=True)
                except Exception as sample_error:
                    st.error(f"Error displaying sample results: {sample_error}")
            
            # Interactive visualizations
            st.markdown("#### 📊 Interactive Sentiment Visualizations")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sentiment distribution pie chart
                try:
                    fig = px.pie(
                        sentiment_counts.reset_index(),
                        values='count',
                        names='label',
                        title="🎭 Sentiment Distribution",
                        color_discrete_map=sentiment_colors,
                        hole=0.4
                    )
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    fig.update_layout(
                        height=400,
                        showlegend=True,
                        font=dict(size=12)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as pie_error:
                    st.error(f"Error creating pie chart: {pie_error}")
            
            with col2:
                # Confidence score distribution
                try:
                    fig = px.histogram(
                        display_df,
                        x='score',
                        nbins=30,
                        title="🎯 Confidence Score Distribution",
                        labels={'score': 'Confidence Score', 'count': 'Number of Tweets'},
                        color='label',
                        color_discrete_map=sentiment_colors
                    )
                    fig.add_vline(x=confidence_threshold, line_dash="dash", 
                                line_color="red", line_width=2,
                                annotation_text=f"Threshold: {confidence_threshold}")
                    fig.update_layout(
                        height=400,
                        barmode='overlay',
                        bargap=0.1
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as hist_error:
                    st.error(f"Error creating histogram: {hist_error}")
            
            # Sentiment by confidence level
            st.markdown("#### 📈 Sentiment Analysis by Confidence Level")
            
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    # Box plot of confidence by sentiment
                    fig = px.box(
                        display_df,
                        x='label',
                        y='score',
                        title="📊 Confidence Distribution by Sentiment",
                        labels={'label': 'Sentiment', 'score': 'Confidence Score'},
                        color='label',
                        color_discrete_map=sentiment_colors
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as box_error:
                    st.error(f"Error creating box plot: {box_error}")
            
            with col2:
                try:
                    # Create confidence brackets
                    display_df['confidence_bracket'] = pd.cut(
                        display_df['score'],
                        bins=[0, 0.6, 0.7, 0.8, 0.9, 1.0],
                        labels=['<60%', '60-70%', '70-80%', '80-90%', '90-100%']
                    )
                    
                    confidence_sentiment = display_df.groupby(['confidence_bracket', 'label']).size().unstack(fill_value=0)
                    
                    fig = px.bar(
                        confidence_sentiment.T,
                        title="📊 Sentiment Distribution Across Confidence Brackets",
                        labels={'value': 'Count', 'index': 'Sentiment'},
                        color_discrete_map=sentiment_colors,
                        barmode='group'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as bracket_error:
                    st.error(f"Error creating confidence bracket chart: {bracket_error}")
            
            # Time series analysis if dates are available
            if 'Date' in display_df.columns and not display_df['Date'].isna().all():
                st.markdown("#### 📅 Temporal Sentiment Analysis")
                
                try:
                    # Clean dates first - remove NaN values
                    df_with_dates = display_df.dropna(subset=['Date']).copy()
                    
                    if len(df_with_dates) > 0:
                        # Aggregate sentiment by date
                        daily_sentiment = df_with_dates.groupby(['Date', 'label']).size().unstack(fill_value=0)
                    
                    # Calculate sentiment ratio
                    if 'Positive' in daily_sentiment.columns and 'Negative' in daily_sentiment.columns:
                        daily_sentiment['sentiment_ratio'] = (
                            daily_sentiment['Positive'] - daily_sentiment['Negative']
                        ) / (daily_sentiment['Positive'] + daily_sentiment['Negative'] + 1)  # Add 1 to avoid division by zero
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Sentiment over time
                        fig = go.Figure()
                        for sentiment in ['Positive', 'Negative', 'Neutral']:
                            if sentiment in daily_sentiment.columns:
                                fig.add_trace(go.Scatter(
                                    x=daily_sentiment.index,
                                    y=daily_sentiment[sentiment],
                                    mode='lines+markers',
                                    name=sentiment,
                                    line=dict(color=sentiment_colors[sentiment], width=2)
                                ))
                        
                        fig.update_layout(
                            title="📈 Daily Sentiment Trends",
                            xaxis_title="Date",
                            yaxis_title="Tweet Count",
                            height=400,
                            hovermode='x unified'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        if 'sentiment_ratio' in daily_sentiment.columns:
                            # Sentiment ratio over time
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=daily_sentiment.index,
                                y=daily_sentiment['sentiment_ratio'],
                                mode='lines',
                                fill='tozeroy',
                                name='Sentiment Ratio',
                                line=dict(color='#3b82f6', width=3)
                            ))
                            
                            fig.add_hline(y=0, line_dash="dash", line_color="gray")
                            fig.update_layout(
                                title="📊 Sentiment Ratio (Positive vs Negative)",
                                xaxis_title="Date",
                                yaxis_title="Ratio",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                
                except Exception as time_error:
                    st.error(f"Error in temporal analysis: {time_error}")
            
            # Word frequency analysis by sentiment
            st.markdown("#### 🔤 Word Frequency Analysis by Sentiment")
            
            from collections import Counter
            import string
            
            try:
                # Common words to exclude
                stop_words = set(['the', 'to', 'and', 'a', 'in', 'is', 'it', 'you', 'that', 
                                'of', 'for', 'on', 'are', 'as', 'with', 'was', 'at', 'be',
                                'this', 'have', 'from', 'or', 'had', 'by', 'but', 'not',
                                'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said',
                                'there', 'use', 'an', 'each', 'which', 'she', 'do', 'how',
                                'their', 'if', 'will', 'up', 'other', 'about', 'out', 'many',
                                'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make',
                                'like', 'him', 'into', 'time', 'has', 'look', 'two', 'more',
                                'write', 'go', 'see', 'number', 'no', 'way', 'could', 'people',
                                'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now',
                                'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made',
                                'may', 'part', 'i', 'me', 'just', 'link', 'mention', 'hashtag',
                                'emoji', 'emoticon', 'number', 'punctuation'])
                
                word_freq_by_sentiment = {}
                
                for sentiment in ['Positive', 'Negative', 'Neutral']:
                    sentiment_tweets = display_df[display_df['label'] == sentiment]['Text_Cleaned']
                    
                    if len(sentiment_tweets) > 0:
                        # Tokenize and count words
                        words = []
                        for tweet in sentiment_tweets:
                            # Remove punctuation and convert to lowercase
                            tweet_clean = tweet.translate(str.maketrans('', '', string.punctuation))
                            words.extend([w.lower() for w in tweet_clean.split() if w.lower() not in stop_words and len(w) > 2])
                        
                        # Get top 10 words
                        word_freq = Counter(words).most_common(10)
                        word_freq_by_sentiment[sentiment] = word_freq
                
                # Display word frequencies
                cols = st.columns(3)
                
                for idx, (sentiment, color) in enumerate(sentiment_colors.items()):
                    with cols[idx]:
                        st.markdown(f"**{sentiment} Sentiment Top Words**")
                        if sentiment in word_freq_by_sentiment:
                            words_df = pd.DataFrame(word_freq_by_sentiment[sentiment], columns=['Word', 'Frequency'])
                            
                            fig = px.bar(
                                words_df,
                                x='Frequency',
                                y='Word',
                                orientation='h',
                                color_discrete_sequence=[color],
                                title=f"Top 10 Words - {sentiment}"
                            )
                            fig.update_layout(height=350, showlegend=False)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info(f"No {sentiment} tweets found")
                
            except Exception as word_freq_error:
                st.error(f"Error in word frequency analysis: {word_freq_error}")
            
            # Market correlation analysis (if market data available)
            if market_data is not None:
                st.markdown("---")
                st.markdown("### 📈 Market Correlation Analysis")
                
                combined_data = create_advanced_market_correlation_analysis(display_df, market_data)
                
                if combined_data is not None and len(combined_data) > 0:
                    # Create additional visualizations
                    create_enhanced_time_series_plots(combined_data)
                    create_enhanced_scatter_analysis(combined_data)
                    create_market_movement_analysis(combined_data)
                    create_rolling_correlation_visualizations(combined_data)
                    
                    # Trading signals based on sentiment
                    st.markdown("---")
                    st.markdown("### 🎯 Trading Signal Generator")
                    
                    if 'sentiment_numeric' in combined_data.columns and 'Close' in combined_data.columns:
                        # Generate simple trading signals
                        combined_data['sentiment_signal'] = combined_data['sentiment_numeric'].rolling(3).mean()
                        combined_data['price_sma_5'] = combined_data['Close'].rolling(5).mean()
                        combined_data['price_sma_20'] = combined_data['Close'].rolling(20).mean()
                        
                        # Signal generation
                        combined_data['signal'] = 0
                        combined_data.loc[
                            (combined_data['sentiment_signal'] > 0.3) & 
                            (combined_data['price_sma_5'] > combined_data['price_sma_20']), 
                            'signal'
                        ] = 1  # Buy signal
                        
                        combined_data.loc[
                            (combined_data['sentiment_signal'] < -0.3) & 
                            (combined_data['price_sma_5'] < combined_data['price_sma_20']), 
                            'signal'
                        ] = -1  # Sell signal
                        
                        # Visualize signals
                        fig = go.Figure()
                        
                        # Price line
                        fig.add_trace(go.Scatter(
                            x=combined_data['Date'],
                            y=combined_data['Close'],
                            mode='lines',
                            name='BTC Price',
                            line=dict(color='#3b82f6', width=2)
                        ))
                        
                        # Buy signals
                        buy_signals = combined_data[combined_data['signal'] == 1]
                        if len(buy_signals) > 0:
                            fig.add_trace(go.Scatter(
                                x=buy_signals['Date'],
                                y=buy_signals['Close'],
                                mode='markers',
                                name='Buy Signal',
                                marker=dict(color='#22c55e', size=12, symbol='triangle-up')
                            ))
                        
                        # Sell signals
                        sell_signals = combined_data[combined_data['signal'] == -1]
                        if len(sell_signals) > 0:
                            fig.add_trace(go.Scatter(
                                x=sell_signals['Date'],
                                y=sell_signals['Close'],
                                mode='markers',
                                name='Sell Signal',
                                marker=dict(color='#ef4444', size=12, symbol='triangle-down')
                            ))
                        
                        fig.update_layout(
                            title="🎯 Trading Signals Based on Sentiment Analysis",
                            xaxis_title="Date",
                            yaxis_title="BTC Price (USD)",
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Signal statistics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            buy_count = len(buy_signals)
                            st.metric("📈 Buy Signals", buy_count, help="Number of buy signals generated")
                        
                        with col2:
                            sell_count = len(sell_signals)
                            st.metric("📉 Sell Signals", sell_count, help="Number of sell signals generated")
                        
                        with col3:
                            total_signals = buy_count + sell_count
                            signal_ratio = buy_count / total_signals if total_signals > 0 else 0
                            st.metric("⚖️ Buy/Total Ratio", f"{signal_ratio:.2%}", help="Proportion of buy signals")
                        
                        # Backtesting results
                        st.markdown("#### 📊 Simple Backtesting Results")
                        
                        # Calculate simple returns
                        initial_capital = 10000
                        position = 0
                        capital = initial_capital
                        trades = []
                        
                        for idx in range(1, len(combined_data)):
                            if combined_data.iloc[idx]['signal'] == 1 and position == 0:
                                # Buy
                                position = capital / combined_data.iloc[idx]['Close']
                                trades.append({
                                    'Date': combined_data.iloc[idx]['Date'],
                                    'Action': 'Buy',
                                    'Price': combined_data.iloc[idx]['Close'],
                                    'Capital': capital
                                })
                                capital = 0
                                
                            elif combined_data.iloc[idx]['signal'] == -1 and position > 0:
                                # Sell
                                capital = position * combined_data.iloc[idx]['Close']
                                trades.append({
                                    'Date': combined_data.iloc[idx]['Date'],
                                    'Action': 'Sell',
                                    'Price': combined_data.iloc[idx]['Close'],
                                    'Capital': capital
                                })
                                position = 0
                        
                        # Final position
                        if position > 0:
                            capital = position * combined_data.iloc[-1]['Close']
                        
                        final_value = capital if capital > 0 else position * combined_data.iloc[-1]['Close']
                        total_return = ((final_value - initial_capital) / initial_capital) * 100
                        
                        # Buy and hold comparison
                        buy_hold_shares = initial_capital / combined_data.iloc[0]['Close']
                        buy_hold_value = buy_hold_shares * combined_data.iloc[-1]['Close']
                        buy_hold_return = ((buy_hold_value - initial_capital) / initial_capital) * 100
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric(
                                "💰 Strategy Return",
                                f"{total_return:.2f}%",
                                help="Return using sentiment-based strategy"
                            )
                        
                        with col2:
                            st.metric(
                                "📊 Buy & Hold Return",
                                f"{buy_hold_return:.2f}%",
                                help="Return using buy and hold strategy"
                            )
                        
                        with col3:
                            outperformance = total_return - buy_hold_return
                            st.metric(
                                "🎯 Outperformance",
                                f"{outperformance:.2f}%",
                                help="Strategy return minus buy & hold return"
                            )
                        
                        # Trade history
                        if trades:
                            with st.expander("📋 Trade History", expanded=False):
                                trades_df = pd.DataFrame(trades)
                                st.dataframe(trades_df, use_container_width=True, hide_index=True)
                        
                        # Risk metrics
                        st.markdown("#### 📊 Risk Metrics")
                        
                        if 'log_return' in combined_data.columns:
                            returns = combined_data['log_return'].dropna()
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
                                st.metric("📈 Sharpe Ratio", f"{sharpe_ratio:.3f}", help="Risk-adjusted return metric")
                            
                            with col2:
                                max_dd = (combined_data['Close'] / combined_data['Close'].expanding().max() - 1).min() * 100
                                st.metric("📉 Max Drawdown", f"{max_dd:.2f}%", help="Maximum peak-to-trough decline")
                            
                            with col3:
                                volatility = returns.std() * np.sqrt(252) * 100  # Annualized
                                st.metric("💥 Annual Volatility", f"{volatility:.2f}%", help="Annualized standard deviation")
                            
                            with col4:
                                win_rate = (returns > 0).sum() / len(returns) * 100
                                st.metric("✅ Win Rate", f"{win_rate:.1f}%", help="Percentage of positive returns")
            
            # Export functionality
            st.markdown("---")
            st.markdown("### 💾 Export Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Export sentiment results
                csv_sentiment = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Sentiment Results (CSV)",
                    data=csv_sentiment,
                    file_name=f"bitcoin_sentiment_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download the complete sentiment analysis results"
                )
            
            with col2:
                if combined_data is not None and len(combined_data) > 0:
                    # Export combined data
                    csv_combined = combined_data.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Combined Data (CSV)",
                        data=csv_combined,
                        file_name=f"bitcoin_sentiment_market_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        help="Download sentiment and market data combined"
                    )
            
            with col3:
                # Export summary report
                # Handle Date range safely
                date_range_str = 'N/A'
                if 'Date' in display_df.columns:
                    valid_dates = display_df['Date'].dropna()
                    if len(valid_dates) > 0:
                        date_range_str = f"{valid_dates.min()} to {valid_dates.max()}"
                
                summary = f"""
Bitcoin Sentiment Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY
===============
Total Tweets Analyzed: {len(display_df):,}
Date Range: {date_range_str}

SENTIMENT DISTRIBUTION
======================
Positive: {sentiment_counts.get('Positive', 0):,} ({positive_pct:.1f}%)
Negative: {sentiment_counts.get('Negative', 0):,} ({negative_pct:.1f}%)
Neutral: {sentiment_counts.get('Neutral', 0):,} ({neutral_pct:.1f}%)

Average Confidence Score: {avg_confidence:.3f}
Confidence Threshold Used: {confidence_threshold:.2f}

MARKET CORRELATION
==================
Data Available: {'Yes' if combined_data is not None else 'No'}
                """
                
                if combined_data is not None and 'sentiment_numeric' in combined_data.columns and 'Close' in combined_data.columns:
                    correlation = combined_data['sentiment_numeric'].corr(combined_data['Close'])
                    summary += f"Sentiment-Price Correlation: {correlation:.4f}\n"
                
                if trades:
                    summary += f"""
TRADING PERFORMANCE
==================
Total Trades: {len(trades)}
Strategy Return: {total_return:.2f}%
Buy & Hold Return: {buy_hold_return:.2f}%
Outperformance: {outperformance:.2f}%
"""
                
                st.download_button(
                    label="📥 Download Summary Report (TXT)",
                    data=summary,
                    file_name=f"bitcoin_sentiment_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                    help="Download a text summary of the analysis"
                )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 20px; color: #f1f5f9;">
        <p>🤖 Powered by Transformer AI Models | 📊 Real-time Market Data | 🎯 Advanced Analytics</p>
        <p style="font-size: 14px; opacity: 0.8;">
            Built with Streamlit, PyTorch, HuggingFace Transformers, and YFinance
        </p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
                   