.PHONY: setup fmt lint type test check

setup:
	uv sync --dev

fmt:
	uv run ruff format .

lint:
	uv run ruff check .

type:
	uv run pyright

test:
	uv run pytest

check: fmt lint type test
