.PHONY: setup test lint batch stream docs

setup:
python -m pip install -e .[stream,db] --quiet

lint:
python -m compileall esi_agents

test:
pytest -q

batch:
python -m cli.esi_batch --config configs/turbine_vibration.yaml --input data/turbine.csv --out artifacts/runs/turbine_example

stream:
python -m cli.esi_stream --config configs/generator_esi.yaml

docs:
python -m pip install -r docs/requirements.txt
echo "Docs dependencies installed"
