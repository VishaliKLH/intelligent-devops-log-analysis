"""
Step 2: Feature Engineering + Step 3: ML Model (Isolation Forest)
-------------------------------------------------------------------
Loads the cleaned logs, builds a richer feature set than the original
project (which only used log-level score + message length), trains an
Isolation Forest anomaly detector, and saves the results.

Features used:
  1. message_length        - length of the log message text
  2. component_encoded     - numeric code for which system component logged it
  3. hour_of_day            - time-based feature (0-23)
  4. is_night                - 1 if log happened between 12am-5am (higher risk window)
  5. errors_last_5min       - rolling count of "error-like" messages in the
                              preceding 5-minute window (captures bursts/spikes)
  6. same_message_freq      - how many times this exact message text appears
                              in the whole dataset (rare/unique messages are
                              often more suspicious than common ones)
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import joblib

CLEAN_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "cleaned_logs.csv")
FEATURES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "features.csv")
ANOMALIES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "anomalies_report.csv")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "isolation_forest_model.pkl")
BASELINE_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "baseline.pkl")

FEATURE_COLS = [
    "message_length",
    "component_encoded",
    "hour_of_day",
    "is_night",
    "errors_last_5min",
    "same_message_freq",
]

RISK_KEYWORDS = ["fail", "error", "abnormal", "invalid", "refused", "denied"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 1. Message length
    df["message_length"] = df["message"].apply(len)

    # 2. Component encoded as a number
    le = LabelEncoder()
    df["component_encoded"] = le.fit_transform(df["component"].astype(str))

    # 3. Hour of day
    df["hour_of_day"] = df["timestamp"].dt.hour

    # 4. Night-time flag (12am - 5am)
    df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if 0 <= h < 5 else 0)

    # 5. Rolling count of risky-looking messages in the preceding 5 minutes
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["is_risky_word"] = df["message"].str.lower().apply(
        lambda m: int(any(k in m for k in RISK_KEYWORDS))
    )
    df = df.set_index("timestamp")
    df["errors_last_5min"] = df["is_risky_word"].rolling("5min").sum()
    df = df.reset_index()

    # 6. Frequency of this exact message across the whole dataset
    #    (rare, one-off messages are often more anomalous than very common ones)
    freq_map = df["message"].value_counts()
    df["same_message_freq"] = df["message"].map(freq_map)

    return df


def save_baseline(df: pd.DataFrame) -> dict:
    """
    Remember what 'normal' looked like during training, so NEW log lines
    typed into the dashboard can be turned into the same 6 features and
    judged by the same trained model.

    - component_map : component name -> number (same numbering as LabelEncoder)
    - message_freq  : how many times each exact message appeared in training
    """
    baseline = {
        "component_map": {
            c: i for i, c in enumerate(sorted(df["component"].astype(str).unique()))
        },
        "message_freq": df["message"].value_counts().to_dict(),
    }
    joblib.dump(baseline, BASELINE_PATH)
    return baseline


def train_and_detect(contamination: float = 0.1):
    df = pd.read_csv(CLEAN_CSV_PATH)
    df = build_features(df)

    feature_cols = FEATURE_COLS

    X = df[feature_cols].fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    df["anomaly_raw"] = model.fit_predict(X)
    df["predicted_label"] = df["anomaly_raw"].apply(lambda x: 1 if x == -1 else 0)
    df["status"] = df["predicted_label"].apply(lambda x: "Anomaly" if x == 1 else "Normal")

    os.makedirs(os.path.dirname(FEATURES_CSV_PATH), exist_ok=True)
    df.to_csv(FEATURES_CSV_PATH, index=False)
    joblib.dump(model, MODEL_PATH)
    save_baseline(df)

    anomalies = df[df["predicted_label"] == 1][
        ["timestamp", "component", "message", "reference_label", "status"]
    ]
    anomalies.to_csv(ANOMALIES_CSV_PATH, index=False)

    total = len(df)
    n_anomalies = df["predicted_label"].sum()

    print("Model training complete.")
    print(f"  Total logs analyzed : {total}")
    print(f"  Anomalies detected  : {n_anomalies} ({n_anomalies / total * 100:.2f}%)")
    print(f"  Model saved to      : {MODEL_PATH}")
    print(f"  Full features saved : {FEATURES_CSV_PATH}")
    print(f"  Anomaly report saved: {ANOMALIES_CSV_PATH}")

    return df


if __name__ == "__main__":
    train_and_detect()
