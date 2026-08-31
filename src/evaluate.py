"""
Step 4: Evaluation
--------------------
Compares the Isolation Forest model's predictions against the
`reference_label` created during preprocessing (a rule-based proxy for
"suspicious" log lines, based on known risk keywords).

NOTE: This is not human-annotated ground truth (the raw Linux_2k dataset
does not ship with per-line labels). The reference label is a documented,
transparent approximation used purely to give a *quantitative* sense of how
well the unsupervised model agrees with an obvious rule-based definition of
risk. This is explained clearly in the project report.
"""

import pandas as pd
import os
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

FEATURES_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "features.csv")
METRICS_TXT_PATH = os.path.join(os.path.dirname(__file__), "..", "output", "evaluation_metrics.txt")


def evaluate():
    df = pd.read_csv(FEATURES_CSV_PATH)

    y_true = df["reference_label"]
    y_pred = df["predicted_label"]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    report_lines = [
        "Evaluation Report — Isolation Forest vs Reference (rule-based) Labels",
        "=" * 70,
        f"Total logs               : {len(df)}",
        f"Reference anomalies (rule-based)  : {int(y_true.sum())}",
        f"Model-predicted anomalies         : {int(y_pred.sum())}",
        "",
        "Confusion Matrix:",
        f"  True Negatives  (correctly Normal)  : {tn}",
        f"  False Positives (flagged, but rule says Normal) : {fp}",
        f"  False Negatives (missed by model)   : {fn}",
        f"  True Positives  (correctly flagged) : {tp}",
        "",
        f"Precision : {precision:.3f}",
        f"Recall    : {recall:.3f}",
        f"F1-score  : {f1:.3f}",
        "",
        "Note: 'Reference labels' are derived from risk keywords (e.g. 'failure',",
        "'error', 'invalid user') during preprocessing, not human-annotated ground",
        "truth. They provide a transparent, explainable baseline to sanity-check",
        "the unsupervised Isolation Forest model's behavior.",
    ]

    report_text = "\n".join(report_lines)
    print(report_text)

    with open(METRICS_TXT_PATH, "w") as f:
        f.write(report_text)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    evaluate()
