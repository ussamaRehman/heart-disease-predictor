.PHONY: setup fmt lint type test check data

setup:
	uv sync --dev

data:
	uv run python scripts/download_uci_heart_disease.py

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run pyright

test:
	uv run pytest

check: fmt lint type test
