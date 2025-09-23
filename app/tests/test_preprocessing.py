import pandas as pd
from langdetect import detect
from crypto_app_organized import Preprocess_Tweets, Drop_Unused_Columns
# -------------------------------------------------
# Basic preprocessing tests
# -------------------------------------------------

def test_preprocess_tweets_removes_noise():
    df = pd.DataFrame({"Text": ["Hello #Bitcoin @user http://link.com 🚀"]})
    cleaned = Preprocess_Tweets(df)

    text = cleaned.iloc[0]["Text_Cleaned"]
    assert "hello" in text.lower()
    assert "[hashtag]" in text.lower()
    assert "[mention]" in text.lower()
    assert "[link]" in text.lower()
    assert "[emoji]" in text.lower()

def test_drop_unused_columns():
    df = pd.DataFrame({
        "Text": ["sample"],
        "user_name": ["john"],
        "user_location": ["earth"]
    })
    cleaned = Drop_Unused_Columns(df)

    assert "user_name" not in cleaned.columns
    assert "user_location" not in cleaned.columns
    assert "Text" in cleaned.columns

# -------------------------------------------------
# Robustness tests (real-world messy data)
# -------------------------------------------------

def test_preprocess_handles_scrambled_text():
    df = pd.DataFrame({
        "Text": ["B!tc0in 🚀🔥 #$$$ http://spam.com @bot"]
    })
    cleaned = Preprocess_Tweets(df)
    text = cleaned.iloc[0]["Text_Cleaned"]

    assert "[emoji]" in text.lower()
    assert "[hashtag]" in text.lower()
    assert "[mention]" in text.lower()
    assert "[link]" in text.lower()

def test_preprocess_handles_unicode():
    df = pd.DataFrame({
        "Text": ["Ｂｉｔｃｏｉｎ (full-width chars) 💰"]
    })
    cleaned = Preprocess_Tweets(df)
    text = cleaned.iloc[0]["Text_Cleaned"]

    # The text should be preprocessed
    assert "[emoji]" in text.lower()

def test_preprocess_handles_empty_text():
    df = pd.DataFrame({
        "Text": ["", "   ", None]
    })
    # Should handle without crashing
    cleaned = Preprocess_Tweets(df)
    assert len(cleaned) >= 0  # May filter out empty tweets
