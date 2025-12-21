.PHONY: setup fmt lint type test check data check-data preprocess split pipeline train-baseline

setup:
	uv sync --dev

data:
	uv run python scripts/download_uci_heart_disease.py

check-data:
	uv run python scripts/check_data.py

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run pyright

test:
	uv run pytest

check: fmt lint type test

preprocess:
	PYTHONPATH=src uv run python -m mlproj.data.make_dataset

split:
	PYTHONPATH=src uv run python -m mlproj.data.split

# End-to-end (no model training yet)
pipeline: setup data check-data preprocess split check

train-baseline: preprocess split
	PYTHONPATH=src uv run python -m mlproj.models.train_baseline
