# Makefile for Ask, Don't Guess (ADG)
# Per spec §5 and Phase 1 requirements

.PHONY: help setup lint format typecheck test test-slow clean demo demo-mock data experiment report

# Default target
help:
	@echo "Ask, Don't Guess (ADG) - Makefile targets:"
	@echo ""
	@echo "Setup and development:"
	@echo "  make setup          - Install dependencies and setup pre-commit hooks"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Format code with ruff"
	@echo "  make typecheck      - Run mypy type checker"
	@echo "  make test           - Run tests (fast, mock-only)"
	@echo "  make test-slow      - Run all tests including slow model tests"
	@echo "  make clean          - Remove generated files and caches"
	@echo ""
	@echo "Demo and experiments (to be implemented in later phases):"
	@echo "  make demo-mock      - Run demo with mock backends"
	@echo "  make demo           - Run demo with real models"
	@echo "  make data           - Build datasets"
	@echo "  make experiment     - Run experiments"
	@echo "  make report         - Generate reports"

# Setup development environment
setup:
	py -3.11 -m uv sync --all-extras
	py -3.11 -m uv run pre-commit install

# Linting
lint:
	py -3.11 -m uv run ruff check src/ tests/

# Formatting
format:
	py -3.11 -m uv run ruff format src/ tests/

# Type checking
typecheck:
	py -3.11 -m uv run mypy src/adg/core/

# Run fast tests only (no model loading)
test:
	py -3.11 -m uv run pytest tests/ -v -m "not slow"

# Run all tests including slow ones
test-slow:
	py -3.11 -m uv run pytest tests/ -v

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .mypy_cache/ .pytest_cache/ .ruff_cache/ htmlcov/ .coverage 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

# Demo targets
demo-mock:
	@echo "Starting Gradio demo with mock backends..."
	py -3.11 -m uv run python -m adg.demo.gradio_app

demo:
	@echo "Demo with real models requires model configuration."
	@echo "Use 'make demo-mock' for quick demo with mock backends."
	@exit 1

# Data pipeline (to be implemented)
data:
	@echo "Data pipeline - to be implemented in Phase 11"
	@exit 1

# Experiments (to be implemented)
experiment:
	@echo "Experiments - to be implemented in Phase 13"
	@exit 1

# Report generation (to be implemented)
report:
	@echo "Report generation - to be implemented in Phase 14"
	@exit 1
