"""CLI for evaluating saved scores."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ..eval import compute_classification_metrics, plot_pr_curve, plot_roc_curve


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate anomaly scores")
    parser.add_argument("--scores", required=True, help="Path to scores CSV/Parquet")
    parser.add_argument("--labels", required=True, help="CSV with ground truth labels")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if scores_path.suffix == ".parquet":
        scores_df = pd.read_parquet(scores_path)
    else:
        scores_df = pd.read_csv(scores_path)
    labels_df = pd.read_csv(args.labels)
    labels = labels_df[labels_df.columns[-1]].to_numpy()
    scores = scores_df["anomaly_score"].to_numpy()
    metrics = compute_classification_metrics(labels, scores)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(metrics.to_json(), encoding="utf-8")
    plot_roc_curve(labels, scores, out_dir / "roc_curve.png")
    plot_pr_curve(labels, scores, out_dir / "pr_curve.png")


if __name__ == "__main__":
    main()
