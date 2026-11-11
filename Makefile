PY ?= .venv/bin/python

.PHONY: install test lint reproduce quick sweep notebook clean

install:
	/opt/homebrew/bin/python3.12 -m venv .venv || python3 -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev,lstm]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .
	$(PY) -m mypy

# Regenerates every number and figure in report.md (about 10 minutes on a laptop CPU).
reproduce:
	$(PY) -m organoid_reservoir.run --task all --config configs/default.yaml
	$(PY) -m organoid_reservoir.sweep --config configs/sweep.yaml

quick:
	$(PY) -m organoid_reservoir.run --task all --config configs/quick.yaml

sweep:
	$(PY) -m organoid_reservoir.sweep --config configs/sweep.yaml

notebook:
	$(PY) scripts/build_notebook.py

clean:
	rm -rf results/quick .pytest_cache .mypy_cache .ruff_cache
