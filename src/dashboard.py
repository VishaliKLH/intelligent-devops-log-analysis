"""
Step 5: Streamlit Dashboard
-----------------------------
A simple, demo-friendly web UI showing:
  - Total logs, normal count, anomaly count
  - Anomalies-over-time chart
  - Table of anomalous log records
  - Evaluation metrics (Precision/Recall/F1)

Run with: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

FEATURES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "features.csv")
METRICS_TXT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "evaluation_metrics.txt")

st.set_page_config(page_title="Intelligent DevOps Log Analyzer", layout="wide")

st.title("🖥️ Intelligent DevOps Log Analyzer")
st.caption("AI-based anomaly detection in system logs using Isolation Forest")

if not os.path.exists(FEATURES_CSV_PATH):
    st.error(
        "No results found yet. Please run the pipeline first:\n\n"
        "```\npython src/preprocess.py\npython src/train_model.py\npython src/evaluate.py\n```"
    )
    st.stop()

df = pd.read_csv(FEATURES_CSV_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

total_logs = len(df)
normal_count = int((df["predicted_label"] == 0).sum())
anomaly_count = int((df["predicted_label"] == 1).sum())

col1, col2, col3 = st.columns(3)
col1.metric("Total Logs", f"{total_logs:,}")
col2.metric("Normal", f"{normal_count:,}")
col3.metric("Anomalies", f"{anomaly_count:,}", delta=f"{anomaly_count/total_logs*100:.1f}% of total")

st.divider()

st.subheader("📈 Anomalies Over Time")
time_series = (
    df[df["predicted_label"] == 1]
    .set_index("timestamp")
    .resample("1D")
    .size()
    .rename("anomaly_count")
)
if len(time_series) > 0:
    st.line_chart(time_series)
else:
    st.info("No anomalies detected to plot.")

st.divider()

st.subheader("🚨 Anomalous Log Records")
anomalies_view = df[df["predicted_label"] == 1][
    ["timestamp", "component", "message", "message_length", "errors_last_5min"]
].sort_values("timestamp", ascending=False)
st.dataframe(anomalies_view, use_container_width=True, height=400)

st.divider()

st.subheader("📊 Model Evaluation")
if os.path.exists(METRICS_TXT_PATH):
    with open(METRICS_TXT_PATH) as f:
        st.code(f.read(), language="text")
else:
    st.info("Run `python src/evaluate.py` to generate evaluation metrics.")

st.divider()

st.subheader("🔎 Live Log Check")
st.caption(
    "Type or paste NEW syslog lines below. The already-trained Isolation Forest "
    "model judges them (Normal / Watch / Anomaly). Tip: several failed logins in a "
    "short time are flagged as a burst."
)

EXAMPLE_LINES = (
    "Jun 20 10:15:01 combo su(pam_unix)[21416]: session opened for user cyrus by (uid=0)\n"
    "Jun 20 03:10:01 combo sshd(pam_unix)[501]: authentication failure; logname= uid=0 euid=0 "
    "tty=NODEVssh ruser= rhost=203.0.113.5  user=root\n"
    "Jun 20 03:10:02 combo sshd(pam_unix)[502]: authentication failure; logname= uid=0 euid=0 "
    "tty=NODEVssh ruser= rhost=203.0.113.5  user=root\n"
    "Jun 20 03:10:03 combo sshd(pam_unix)[503]: Failed password for invalid user admin from 203.0.113.5"
)

live_text = st.text_area(
    "Log lines (one per line, format: Mon DD HH:MM:SS host component[pid]: message)",
    value=EXAMPLE_LINES,
    height=180,
)

if st.button("Analyze lines"):
    try:
        from predict import analyze_lines, artifacts_ready
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not load the live checker: {exc}")
    else:
        if not artifacts_ready():
            st.error("Model files missing. Run `python src/train_model.py` first.")
        else:
            result = analyze_lines(live_text)
            if result.empty:
                st.info("Please enter at least one log line.")
            else:
                st.dataframe(result, use_container_width=True)
                n_bad = int((result["status"] == "Anomaly").sum())
                n_watch = int((result["status"] == "Watch").sum())
                st.write(f"**{n_bad}** anomaly, **{n_watch}** watch, "
                         f"out of {len(result)} lines.")

st.divider()
st.caption(
    "Dataset: Linux_2k.log (Loghub — public system log dataset collection). "
    "Model: Isolation Forest (unsupervised anomaly detection)."
)
