# ESI Agents Platform

A modular, multi-agent anomaly detection platform for electrical, sensor, and industrial (ESI) data. The project adapts the researcher → coder → reviewer loop described in [Agents of Discovery (arXiv:2509.08535)](https://arxiv.org/abs/2509.08535) to rotating machinery and industrial assets. It provides batch and streaming pipelines, classical anomaly detectors, band-scan diagnostics, drift monitoring, and automated reporting.

## Architecture

```
Orchestrator
 ├── DataIngestor → schema + data quality checks
 ├── FeatureEngineer → windowed time/frequency/order features
 ├── ModelTrainer → trains IsolationForest/LOF/HBOS/OCSVM/STL/ARIMA
 ├── ModelSelector → selects by PR-AUC + SIC surrogate
 ├── Evaluator → metrics, ROC/PR plots, band scans, calibrated scores
 ├── DriftMonitor → PSI/KL drift statistics
 ├── BatchScorer / StreamScorer → batch scoring & JSONL streaming alerts
 ├── LogicReviewer → verifies report claims vs metrics
 └── CodeReviewer → asserts artifact presence & runnable outputs
```

Agents are stateless modules under `esi_agents/agents`. Workflows in `esi_agents/workflows` wire them together for batch (`run_batch_pipeline`) and streaming (`run_stream_pipeline`) execution.

## Repository layout

```
esi_agents/
  adapters/      # CSV, Parquet, InfluxDB, Timescale, MQTT, OPC-UA adapters
  agents/        # multi-agent components + orchestrator, reviewers
  eval/          # metrics, band scans, calibration, plotting
  features/      # time/frequency/envelope/order/window features
  models/        # anomaly detectors (IsolationForest, LOF, HBOS, OCSVM, STL, ARIMA)
  workflows/     # batch and stream pipelines
  cli/           # reusable CLI implementations
  configs/       # example YAML configurations
  tests/         # pytest suite with synthetic fixtures
cli/             # python -m friendly entry points
configs/         # (alias) example configs
data/            # synthetic turbine dataset + labels
artifacts/       # default artifact root (empty placeholder)
docs/            # quickstart and module docs
```

## Installation

```bash
python -m pip install -e .
```

## Example usage

### Batch detection

```bash
python -m cli.esi_batch \
    --config esi_agents/configs/turbine_vibration.yaml \
    --input data/turbine.csv \
    --out artifacts/runs/turbine_demo
```

Outputs include calibrated scores (`scores.parquet`), ROC/PR/band-scan plots (512×512), drift metrics, and a Markdown report validated by the logic/code reviewers.

### Evaluate saved scores

```bash
python -m cli.esi_evaluate --scores artifacts/runs/turbine_demo/scores.parquet --labels data/turbine_labels.csv
```

### Streaming alerts

```bash
python -m cli.esi_stream --config esi_agents/configs/generator_esi.yaml
```

The streaming pipeline trains the configured model, subscribes to the chosen adapter (CSV demo by default), maintains sliding windows per asset, and prints JSONL alerts when scores exceed the calibrated SIC threshold.

## Configuration

YAML files under `esi_agents/configs/` describe data adapters, window parameters, candidate models, and optional streaming sources. Example: `turbine_vibration.yaml` covers vibration accelerometer data with 5-second windows, 50% overlap, FFT features, and classical detectors.

## Testing

Run the unit/integration suite:

```bash
pytest -q
```

`tests/test_workflow.py` exercises the full batch pipeline on the synthetic turbine dataset and expects ROC-AUC and PR-AUC above 0.8.

## Documentation

Additional notes live in `docs/quickstart.md`, `docs/adapters.md`, and `docs/features.md`.
