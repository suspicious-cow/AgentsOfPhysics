.PHONY: setup test batch stream docs

setup:
pip install -e .[dev]

test:
pytest

batch:
python -m esi_agents.cli.esi_batch --config esi_agents/configs/turbine_vibration.yaml --out artifacts/runs/example

stream:
python -m esi_agents.cli.esi_stream --config esi_agents/configs/generator_esi.yaml

docs:
@echo "Documentation available under esi_agents/docs"
