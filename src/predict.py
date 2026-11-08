"""
Live Log Check
---------------
Takes NEW log lines (typed or pasted) and judges them with the already-trained
Isolation Forest model. It builds the same 6 features used in training, using
the 'baseline' saved by train_model.py (component numbers + message counts).

Simple idea for explaining to mam:
  train once on the 2,000 logs  ->  save the model + what 'normal' looked like
  new line typed in             ->  same 6 features  ->  model says Normal / Anomaly
"""

import os
import joblib
import pandas as pd

from preprocess import parse_line
from train_model import FEATURE_COLS, RISK_KEYWORDS, MODEL_PATH, BASELINE_PATH


WATCH_SCORE = 0.03  # normal-but-borderline cut-off (training normals median ~0.12)


def artifacts_ready() -> bool:
    return os.path.exists(MODEL_PATH) and os.path.exists(BASELINE_PATH)


def load_artifacts():
    return joblib.load(MODEL_PATH), joblib.load(BASELINE_PATH)


def _reason(row, baseline) -> str:
    reasons = []
    if row["message_freq_raw"] == 0:
        reasons.append("message never seen in training logs")
    elif row["same_message_freq"] <= 2:
        reasons.append("very rare message")
    if row["component_encoded"] == -1:
        reasons.append("unknown component")
    if row["is_night"] == 1:
        reasons.append("night-time activity (12am-5am)")
    if row["errors_last_5min"] >= 3:
        reasons.append(f"burst of {int(row['errors_last_5min'])} risky messages within 5 min")
    elif row["is_risky_word"]:
        reasons.append("contains risk keyword (fail/error/invalid...)")
    return "; ".join(reasons) if reasons else "looks like normal training behaviour"


def analyze_lines(text: str, model=None, baseline=None) -> pd.DataFrame:
    """Return a DataFrame with one row per typed line: status, score, reason."""
    if model is None or baseline is None:
        model, baseline = load_artifacts()

    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parsed = parse_line(line)
        if parsed is None:
            rows.append({"log_line": line, "status": "Invalid format", "score": None,
                         "reason": "does not match syslog format "
                                   "'Mon DD HH:MM:SS host component[pid]: message'"})
            continue
        parsed["log_line"] = line
        rows.append(parsed)

    good = [r for r in rows if "timestamp" in r]
    if good:
        df = pd.DataFrame(good)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp", kind="stable")

        df["message_length"] = df["message"].apply(len)
        df["component_encoded"] = df["component"].astype(str).map(
            baseline["component_map"]).fillna(-1).astype(int)
        df["hour_of_day"] = df["timestamp"].dt.hour
        df["is_night"] = df["hour_of_day"].apply(lambda h: 1 if 0 <= h < 5 else 0)
        df["is_risky_word"] = df["message"].str.lower().apply(
            lambda m: int(any(k in m for k in RISK_KEYWORDS)))
        df["errors_last_5min"] = (
            df.set_index("timestamp")["is_risky_word"].rolling("5min").sum().values
        )
        df["message_freq_raw"] = df["message"].map(baseline["message_freq"]).fillna(0)
        df["same_message_freq"] = df["message_freq_raw"]

        X = df[FEATURE_COLS].fillna(0)
        pred = model.predict(X)
        score = model.decision_function(X)
        # Three levels: the model's own verdict, plus a "Watch" level for lines the
        # model calls normal but that sit very close to its decision boundary.
        df["status"] = [
            "Anomaly" if p == -1 else ("Watch" if sc < WATCH_SCORE else "Normal")
            for p, sc in zip(pred, score)
        ]
        df["score"] = score.round(3)
        df["reason"] = df.apply(lambda r: _reason(r, baseline), axis=1)
        by_line = {}
        for _, r in df.iterrows():
            by_line.setdefault(r["log_line"], []).append(r)
        out = []
        for r in rows:
            if "timestamp" in r:
                out.append(by_line[r["log_line"]].pop(0))
            else:
                out.append(pd.Series(r))
        final = pd.DataFrame(out)
    else:
        final = pd.DataFrame(rows)

    if final.empty:
        return pd.DataFrame(columns=["log_line", "status", "score", "reason"])
    return final[["log_line", "status", "score", "reason"]].reset_index(drop=True)
