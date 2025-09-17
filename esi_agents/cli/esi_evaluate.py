"""Evaluation CLI."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..eval import compute_classification_metrics, metrics_to_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores")
    parser.add_argument("--scores", required=True, help="Path to parquet/csv scores file")
    parser.add_argument("--labels", required=False, help="Optional labels CSV with window_start,label")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if scores_path.suffix == ".parquet":
        scores = pd.read_parquet(scores_path)
    else:
        scores = pd.read_csv(scores_path)

    if args.labels:
        labels = pd.read_csv(args.labels)
        scores = scores.merge(labels, on="window_start", how="left", suffixes=('', '_label'))
        scores["label"] = scores.get("label_label", scores.get("label", 0)).fillna(0)

    if "label" not in scores.columns:
        raise ValueError("Labels are required to compute metrics")

    score_column = "anomaly_score" if "anomaly_score" in scores.columns else "score"
    metrics = compute_classification_metrics(scores[score_column].to_numpy(), scores["label"].to_numpy())
    df = metrics_to_frame(metrics)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
