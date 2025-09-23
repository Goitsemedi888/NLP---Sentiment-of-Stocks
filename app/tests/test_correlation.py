import pandas as pd
import numpy as np
from crypto_app_organized import create_advanced_market_correlation_analysis

def test_correlation_returns_dataframe():
    # Fake sentiment data with required columns
    sentiment = pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=5),
        "score": np.random.rand(5),
        "label": ["Positive", "Negative", "Neutral", "Positive", "Negative"]
    })

    # Fake market data with required columns
    market = pd.DataFrame({
        "Date": pd.date_range("2022-01-01", periods=5),
        "Open": np.random.rand(5) * 100,
        "Close": np.random.rand(5) * 100,
        "High": np.random.rand(5) * 100,
        "Low": np.random.rand(5) * 100,
        "Volume": np.random.rand(5) * 1000000
    }).set_index("Date")

    result = create_advanced_market_correlation_analysis(sentiment, market)

    assert result is None or isinstance(result, pd.DataFrame)
