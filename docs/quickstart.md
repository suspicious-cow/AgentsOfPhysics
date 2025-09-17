# Quickstart

This project provides a multi-agent anomaly detection stack for electrical, sensor, and industrial (ESI) data. The agents orchestrate ingestion, feature extraction, modeling, evaluation, and reporting for both batch and streaming use cases.

## Installation

```bash
python -m pip install -e .
```

## Batch pipeline

```bash
python -m cli.esi_batch --config esi_agents/configs/turbine_vibration.yaml --input data/turbine.csv --out artifacts/runs/turbine_demo
```

The command creates calibrated anomaly scores, evaluation plots, a Markdown report, and drift/band-scan diagnostics under `artifacts/runs/turbine_demo`.

## Evaluation

```bash
python -m cli.esi_evaluate --scores artifacts/runs/turbine_demo/scores.parquet --labels data/turbine_labels.csv
```

## Streaming demo

```bash
python -m cli.esi_stream --config esi_agents/configs/generator_esi.yaml
```

The streaming pipeline trains models, subscribes to a CSV-backed stream, and prints JSONL alerts when scores exceed the calibrated threshold.
