import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import clean_text, preprocess
import pandas as pd


def test_clean_text_lowercase():
    assert clean_text("Hello World!") == "hello world"


def test_clean_text_removes_punctuation():
    assert clean_text("it's great, isn't it?") == "its great isnt it"


def test_preprocess_adds_clean_column():
    df = pd.DataFrame({"headline": ["Hello World"], "is_sarcastic": [0]})
    result = preprocess(df)
    assert "clean" in result.columns
    assert result["clean"].iloc[0] == "hello world"


def test_preprocess_preserves_rows():
    df = pd.DataFrame({"headline": ["a", "b", "c"], "is_sarcastic": [0, 1, 0]})
    result = preprocess(df)
    assert len(result) == 3
