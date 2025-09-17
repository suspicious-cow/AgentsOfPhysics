"""CLI to assemble a markdown report from saved artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble anomaly detection report")
    parser.add_argument("--metrics", required=True, help="Path to metrics.json")
    parser.add_argument("--band-scan", required=False, help="Path to band_scan.json")
    parser.add_argument("--out", required=True, help="Output markdown file")
    args = parser.parse_args()

    metrics = json.loads(Path(args.metrics).read_text())
    bands = []
    if args.band_scan:
        band_path = Path(args.band_scan)
        if band_path.exists():
            bands = json.loads(band_path.read_text())
    lines = ["# ESI Evaluation Report", "", "## Metrics"]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value}")
    if bands:
        lines.append("")
        lines.append("## Band Scan Highlights")
        for band in bands[:5]:
            lines.append(
                f"- {band['band_start']:.2f}–{band['band_end']:.2f}: z={band['z_score']:.2f}, p={band['p_value']:.3g}"
            )
    Path(args.out).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
