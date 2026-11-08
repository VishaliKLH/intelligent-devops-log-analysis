# Intelligent DevOps Log Analysis and Anomaly Detection

An AIOps project that uses machine learning (Isolation Forest) to automatically
detect abnormal entries in system logs, wrapped with a full DevOps pipeline:
data preprocessing, evaluation, a visual dashboard, containerization, and
CI/CD automation via GitHub Actions.

> Inspired by the AIOps concept demonstrated in
> [iam-veeramalla/ai-assisted-devops](https://github.com/iam-veeramalla/ai-assisted-devops)
> (day-6: log analysis with Isolation Forest). This project extends that idea
> with a larger real-world dataset, proper preprocessing, richer features,
> quantitative evaluation, a dashboard, Docker packaging, and a CI pipeline.

## Problem Statement

DevOps engineers traditionally monitor system logs manually to catch failures
and security issues — this doesn't scale as systems and log volume grow.
This project automates that process using an unsupervised ML model that learns
what "normal" log behavior looks like and flags unusual entries automatically.

## Dataset

- **Source**: [Loghub](https://github.com/logpai/loghub) — a public collection
  of system log datasets for AI-driven log analytics research.
- **File used**: `Linux_2k.log`
- **Size**: 2,000 real log lines
- **Content**: SSH authentication attempts, session open/close events,
  cron/logrotate alerts, and other Linux system activity
- **Fields extracted**: timestamp, host, component, process ID, message

## Pipeline

```
Raw logs (Linux_2k.log)
       ↓
Preprocessing (parse, clean, deduplicate)  →  src/preprocess.py
       ↓
Feature Engineering (message length, component, time features,
                      rolling error frequency, message rarity)
       ↓
Isolation Forest Model (unsupervised anomaly detection)  →  src/train_model.py
       ↓
Evaluation (Precision / Recall / F1 vs rule-based reference labels)  →  src/evaluate.py
       ↓
Streamlit Dashboard (visual results)  →  src/dashboard.py
       ↓
Docker container + GitHub Actions CI/CD
```

## Features Used by the Model

| Feature | Description |
|---|---|
| `message_length` | Length of the log message text |
| `component_encoded` | Which system component generated the log (numeric code) |
| `hour_of_day` | Hour (0–23) the log occurred |
| `is_night` | 1 if the log happened between 12am–5am |
| `errors_last_5min` | Rolling count of risk-keyword messages in the preceding 5 minutes |
| `same_message_freq` | How often this exact message text occurs overall (rare messages are often more suspicious) |

## Evaluation

Since the raw dataset has no human-annotated ground truth, a transparent
**rule-based reference label** is derived during preprocessing (flagging
messages containing keywords like "failure", "error", "invalid user").
This is used only to sanity-check the unsupervised model — it is documented
clearly and is **not** claimed as ground truth.

Metrics reported: Precision, Recall, F1-score, and a confusion matrix
(see `output/evaluation_metrics.txt` after running the pipeline).

## How to Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline
python src/preprocess.py
python src/train_model.py
python src/evaluate.py

# 3. Launch the dashboard
streamlit run src/dashboard.py
```

## How to Run with Docker

```bash
docker build -t log-anomaly-detector .
docker run -p 8501:8501 log-anomaly-detector
```

Then open http://localhost:8501 in your browser.

## Running Tests

```bash
pytest tests/ -v
```

## CI/CD

Every push to `main` triggers `.github/workflows/ci.yml`, which:
1. Installs dependencies
2. Runs the full pipeline (preprocess → train → evaluate)
3. Runs the test suite
4. Uploads the anomaly report as a build artifact

## Project Structure

```
.
├── data/
│   └── linux_system_logs.txt      # raw dataset
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   └── dashboard.py
├── output/                        # generated at runtime (csv reports, model file)
├── tests/
│   └── test_pipeline.py
├── .github/workflows/ci.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Real-World Relevance

This pattern (AIOps) is used in production by companies to monitor server
fleets, detect security incidents, and catch failures before they cause
outages — replacing manual log review with automated, continuously-running
detection integrated into the DevOps pipeline.

## Live Log Check (dashboard)

The dashboard has a **Live Log Check** box: paste new syslog lines
(`Mon DD HH:MM:SS host component[pid]: message`) and the trained Isolation Forest
labels each line **Normal / Watch / Anomaly** with a score and a reason.
It reuses the model plus `output/baseline.pkl` (saved by `train_model.py`).
Limitation: the model was trained on one 2,000-line file, so it is a demo, not a production detector.
