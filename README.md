# ESI Agents Platform

A modular, multi-agent anomaly detection platform for electrical/signature/sensor (ESI) data from rotating machinery such as turbines, compressors, gearboxes and generators. The architecture mirrors the researcher → coder → reviewer workflow described in [Agents of Discovery (arXiv:2509.08535)](https://arxiv.org/abs/2509.08535) with orchestration, logic review and code review agents gating each run.

## Features

- **Data adapters** for CSV, Parquet, SQLite demos and optional InfluxDB, TimescaleDB, MQTT and OPC-UA transports.
- **Feature engineering** library providing time, frequency, envelope/demodulation and order-tracking features on sliding windows.
- **Model zoo** with classical detectors (Isolation Forest, LOF, HBOS, One-Class SVM) and residual baselines (STL, ARIMA), plus an optional PyTorch autoencoder.
- **Evaluation toolkit** computing ROC/PR metrics, SIC surrogate scans and bump-hunt style band scans with 512×512 plots.
- **Agents** implementing ingest, feature extraction, training, selection, evaluation, drift monitoring, batch/stream scoring and report writing with logic/code review gates.
- **Workflows & CLIs** for batch pipelines, evaluation and streaming demos.
- **Documentation & tests** covering adapters, features, models and evaluators on synthetic fixtures.

## Installation

```bash
pip install -e .[dev]
```

Optional extras:

```bash
pip install -e .[stream]    # MQTT / OPC-UA transports
pip install -e .[databases] # InfluxDB / TimescaleDB adapters
pip install -e .[torch]     # PyTorch autoencoder
```

## Usage

### Batch pipeline

```bash
python -m esi_agents.cli.esi_batch --config esi_agents/configs/turbine_vibration.yaml \
    --input data/turbine.csv \
    --out artifacts/runs/turbine_demo \
    --labels data/turbine_labels.csv
```

Artifacts include calibrated scores (`scores.parquet`), plots, band-scan JSON and a Markdown report reviewed for consistency.

### Evaluate saved scores

```bash
python -m esi_agents.cli.esi_evaluate --scores artifacts/runs/turbine_demo/scores.parquet \
    --labels data/turbine_labels.csv \
    --out artifacts/runs/turbine_demo/eval
```

### Streaming demo

```bash
python -m esi_agents.cli.esi_stream --config esi_agents/configs/generator_esi.yaml
```

The streaming workflow trains from historical data then consumes the configured stream adapter, emitting JSONL alerts with calibrated anomaly scores.

## Repository layout

```
esi_agents/
  adapters/      # data sources and transports
  agents/        # orchestrator, reviewers and worker agents
  features/      # time/frequency/order feature library
  models/        # detector implementations
  eval/          # metrics, calibration and plotting
  workflows/     # batch & stream pipelines
  cli/           # command line interfaces
  configs/       # example YAML configurations
  docs/          # quickstart and component docs
  tests/         # pytest suite with synthetic fixtures
```

Generated artifacts (scores, plots, reports) are stored under `artifacts/` by default with timestamped subdirectories recommended for production deployments.

## Development

```bash
make setup
make test
```

The pytest suite covers adapter contracts, feature calculations, detector normalisation and evaluation artifact creation.

## License

MIT License © 2024 Contributors.
