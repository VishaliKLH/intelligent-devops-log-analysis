"""
Step 1: Data Preprocessing
---------------------------
Reads the raw Linux system log file, cleans it, and converts it into a
structured pandas DataFrame that later scripts (feature engineering,
model training, dashboard) can use.

Dataset: Linux_2k.log (from Loghub - a public collection of system log
datasets for AI-driven log analytics research: https://github.com/logpai/loghub)
- 2,000 real log lines collected from a Linux server
- Contains SSH authentication attempts, session open/close events,
  cron/logrotate alerts, and other system activity
"""

import pandas as pd
import re
import os

RAW_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "linux_system_logs.txt")
CLEAN_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "cleaned_logs.csv")

# Linux syslog format looks like:
# Jun 14 15:16:01 combo sshd(pam_unix)[19939]: authentication failure; ...
# <month> <day> <time> <host> <component>[<pid>]: <message>
LOG_PATTERN = re.compile(
    r"^(?P<month>\w{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<component>[^:\[]+)(?:\[(?P<pid>\d+)\])?:\s*(?P<message>.*)$"
)

# We don't have hand-labeled ground truth for this raw log file, so we
# derive simple *reference* labels using known risk keywords. This lets us
# compute Precision/Recall/F1 later to evaluate how well the unsupervised
# Isolation Forest model agrees with an obvious rule-based definition of
# "suspicious" activity. This is documented clearly in the report as a
# proxy/reference label, not ground truth from a human annotator.
ANOMALY_KEYWORDS = [
    "authentication failure",
    "check pass; user unknown",
    "exited abnormally",
    "failed",
    "error",
    "invalid user",
    "refused connect",
]


def load_raw_logs(path: str) -> list[str]:
    with open(path, "r", errors="ignore") as f:
        lines = f.readlines()
    return [line.strip() for line in lines if line.strip()]


def parse_line(line: str, year: int = 2026) -> dict | None:
    match = LOG_PATTERN.match(line)
    if not match:
        return None
    parts = match.groupdict()
    timestamp_str = f"{year} {parts['month']} {parts['day']} {parts['time']}"
    try:
        timestamp = pd.to_datetime(timestamp_str, format="%Y %b %d %H:%M:%S")
    except ValueError:
        return None

    component = parts["component"].strip()
    message = parts["message"].strip()

    return {
        "timestamp": timestamp,
        "host": parts["host"],
        "component": component,
        "pid": parts["pid"] if parts["pid"] else None,
        "message": message,
    }


def label_reference_anomaly(message: str) -> int:
    """Returns 1 if the message matches a known 'suspicious activity' keyword, else 0."""
    msg_lower = message.lower()
    return int(any(keyword in msg_lower for keyword in ANOMALY_KEYWORDS))


def preprocess() -> pd.DataFrame:
    raw_lines = load_raw_logs(RAW_LOG_PATH)
    records = []
    skipped = 0

    for line in raw_lines:
        parsed = parse_line(line)
        if parsed is None:
            skipped += 1
            continue
        records.append(parsed)

    df = pd.DataFrame(records)

    # Drop only fully identical rows (same timestamp, component, pid AND message).
    # We intentionally include `pid` here: rapid-fire login attempts often share
    # the same timestamp and message text but come from distinct processes/attempts,
    # so they must NOT be collapsed into one row.
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp", "component", "pid", "message"])
    duplicates_removed = before - len(df)

    # Handle any missing messages (shouldn't happen, but just in case)
    df = df.dropna(subset=["message"])

    # Sort by time
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Reference label for evaluation later
    df["reference_label"] = df["message"].apply(label_reference_anomaly)

    os.makedirs(os.path.dirname(CLEAN_CSV_PATH), exist_ok=True)
    df.to_csv(CLEAN_CSV_PATH, index=False)

    print("Preprocessing complete.")
    print(f"  Raw lines read       : {len(raw_lines)}")
    print(f"  Lines skipped/malformed: {skipped}")
    print(f"  Duplicate rows removed: {duplicates_removed}")
    print(f"  Final clean records  : {len(df)}")
    print(f"  Reference anomalies  : {df['reference_label'].sum()} "
          f"({df['reference_label'].mean() * 100:.2f}%)")
    print(f"  Saved to: {CLEAN_CSV_PATH}")

    return df


if __name__ == "__main__":
    preprocess()
