import pandas as pd

from unittest.mock import MagicMock

from crypto_app_organized import analyze_sentiment

def test_analyze_sentiment_positive():
    df = pd.DataFrame({"Text_Cleaned": ["I love Bitcoin, it's going to the moon! 🚀"]})
    result = analyze_sentiment(df)

    assert "Sentiment" in result.columns
    assert "Sentiment_Score" in result.columns
    assert result.iloc[0]["Sentiment"] in ["Positive", "Neutral", "Negative"]
    assert 0.0 <= result.iloc[0]["Sentiment_Score"] <= 1.0

def test_analyze_sentiment_positive():
    # Create mock NLP pipeline
    mock_nlp = MagicMock()
    mock_nlp.return_value = [{'label': 'LABEL_2', 'score': 0.9}]  # Positive
    
    df = pd.DataFrame({
        "Text_Cleaned": ["I love Bitcoin, it's going to the moon!"],
        "Tweet Id": [1]
    })
    
    result = analyze_sentiment(df, mock_nlp)  # Pass the mock
    
    assert len(result) > 0
    assert "label" in result.columns
    assert "score" in result.columns

def test_analyze_sentiment_negative():
    mock_nlp = MagicMock()
    mock_nlp.return_value = [{'label': 'LABEL_0', 'score': 0.8}]  # Negative
    
    df = pd.DataFrame({
        "Text_Cleaned": ["Bitcoin is a scam and worthless."],
        "Tweet Id": [1]
    })
    
    result = analyze_sentiment(df, mock_nlp)
    
    assert len(result) > 0
    assert "label" in result.columns

def test_analyze_sentiment_empty_text():
    mock_nlp = MagicMock()
    mock_nlp.return_value = [{'label': 'LABEL_1', 'score': 0.5}]  # Neutral
    
    df = pd.DataFrame({
        "Text_Cleaned": [""],
        "Tweet Id": [1]
    })
    
    result = analyze_sentiment(df, mock_nlp)
    
    assert len(result) > 0    

def test_analyze_sentiment_negative():
    df = pd.DataFrame({"Text_Cleaned": ["Bitcoin is a scam and worthless."]})
    result = analyze_sentiment(df)

    assert result.iloc[0]["Sentiment"] in ["Positive", "Neutral", "Negative"]
    assert 0.0 <= result.iloc[0]["Sentiment_Score"] <= 1.0

def test_analyze_sentiment_empty_text():
    df = pd.DataFrame({"Text_Cleaned": [""]})
    result = analyze_sentiment(df)

    # Should handle gracefully (Neutral or default sentiment)
    assert result.iloc[0]["Sentiment"] in ["Neutral", "Positive", "Negative"]
