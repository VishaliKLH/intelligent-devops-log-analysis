"""
Basic tests for the log anomaly detection pipeline.
Run with: pytest tests/
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import preprocess, parse_line
from train_model import build_features, train_and_detect


def test_parse_line_valid():
    line = "Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; rhost=1.2.3.4"
    result = parse_line(line)
    assert result is not None
    assert result["host"] == "combo"
    assert "authentication failure" in result["message"]


def test_parse_line_invalid():
    result = parse_line("this is not a valid log line")
    assert result is None


def test_preprocess_produces_dataframe():
    df = preprocess()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "reference_label" in df.columns


def test_features_created():
    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "output", "cleaned_logs.csv")
    )
    df_feat = build_features(df)
    expected_cols = [
        "message_length",
        "component_encoded",
        "hour_of_day",
        "is_night",
        "errors_last_5min",
        "same_message_freq",
    ]
    for col in expected_cols:
        assert col in df_feat.columns


def test_model_detects_some_anomalies():
    df = train_and_detect()
    assert "predicted_label" in df.columns
    assert df["predicted_label"].sum() > 0
    assert df["predicted_label"].sum() < len(df)  # not everything flagged
