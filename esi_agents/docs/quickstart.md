# Quickstart

This quickstart demonstrates how to run the multi-agent anomaly detection platform on a synthetic turbine vibration dataset.

## Prerequisites

Install dependencies:

```bash
pip install -e .
```

## Batch scoring

```bash
python -m esi_agents.cli.esi_batch --config esi_agents/configs/turbine_vibration.yaml \
    --input data/turbine.csv \
    --out artifacts/runs/turbine_example
```

The orchestrator will ingest data, compute features, train multiple detectors, select the best model, perform evaluation and write a Markdown report under `artifacts/runs/turbine_example`.

## Evaluation

```bash
python -m esi_agents.cli.esi_evaluate --scores artifacts/runs/turbine_example/scores.parquet \
    --labels data/turbine_labels.csv \
    --out artifacts/runs/turbine_example/eval
```

## Streaming demo

```bash
python -m esi_agents.cli.esi_stream --config esi_agents/configs/generator_esi.yaml
```

The streaming pipeline trains a model from historical data and attaches to the configured stream adapter. Alerts are emitted as JSON lines to stdout.
