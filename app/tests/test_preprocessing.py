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
    assert "bitcoin" in text.lower()
    assert "[hashtag]" in text
    assert "[mention]" in text
    assert "[link]" in text
    assert "[emoji]" in text

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

    assert "[emoji]" in text
    assert "[hashtag]" in text
    assert "[mention]" in text
    assert "[link]" in text
    # At least "btc" or "bitcoin" should remain normalized
    assert any(token in text.lower() for token in ["btc", "bitcoin"])

def test_preprocess_handles_unicode():
    df = pd.DataFrame({
        "Text": ["Ｂｉｔｃｏｉｎ (full-width chars) 💰"]
    })
    cleaned = Preprocess_Tweets(df)
    text = cleaned.iloc[0]["Text_Cleaned"]

    # Verify normalization handled full-width characters
    assert "bitcoin" in text.lower()

def test_detects_low_quality_or_bot_text():
    df = pd.DataFrame({
        "Text": [
            "BUY BITCOIN NOW!!!! 🚀🚀🚀",
            "asdlkjasd123!@# random scramble",
            "Legitimate tweet about BTC adoption"
        ]
    })
    cleaned = Preprocess_Tweets(df)

    # crude heuristic: flag very short or repeated spammy tokens
    def is_low_quality(text):
        return len(text.split()) < 3 or text.upper().count("BUY") > 2

    flags = [is_low_quality(t) for t in cleaned["Text_Cleaned"]]
    assert any(flags)  # should catch at least one spam-like

def test_preprocess_language_filter():
    df = pd.DataFrame({
        "Text": ["Je t’aime Bitcoin", "Bitcoin is rising fast!"]
    })
    cleaned = Preprocess_Tweets(df)

    langs = [detect(t) for t in cleaned["Text_Cleaned"]]
    assert "fr" in langs  # French detected
    assert "en" in langs  # English detected

