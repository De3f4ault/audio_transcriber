.PHONY: install dev test lint format clean help
.PHONY: transcribe transcribe-srt history search info download
.PHONY: docs docs-serve docs-stop

VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
CLI := $(VENV)/bin/transcriber

# ── Help ─────────────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────

install: ## Install base dependencies
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

dev: ## Install with dev dependencies (editable)
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[all]"

# ── Transcription ────────────────────────────────────────────

transcribe: ## Transcribe a file (make transcribe FILE=audio.m4a)
	$(CLI) transcribe $(FILE)

transcribe-srt: ## Transcribe → SRT (make transcribe-srt FILE=audio.m4a)
	$(CLI) transcribe -f srt $(FILE)

history: ## View transcription history
	$(CLI) history

search: ## Search transcriptions (make search Q="keyword")
	$(CLI) search "$(Q)"

info: ## Show system info and settings
	$(CLI) info

download: ## Download a model (make download MODEL=large-v3-turbo)
	$(CLI) download $(or $(MODEL),large-v3-turbo)

# ── Development ──────────────────────────────────────────────

test: ## Run test suite with coverage
	$(PYTHON) -m pytest tests/ -v --cov=src/transcriber --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ -v

test-integration: ## Run integration tests
	$(PYTHON) -m pytest tests/integration/ -v

lint: ## Run linters (ruff + mypy)
	$(PYTHON) -m ruff check src/ cli/
	$(PYTHON) -m mypy src/ cli/

format: ## Auto-format code (black + ruff fix)
	$(PYTHON) -m black src/ cli/
	$(PYTHON) -m ruff check --fix src/ cli/

clean: ## Clean build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ── Documentation ────────────────────────────────────────────

docs: ## Build documentation
	$(VENV)/bin/zensical build

docs-serve: ## Serve documentation locally
	$(VENV)/bin/zensical serve

docs-stop: ## Stop the docs server
	@lsof -ti:8000 | xargs -r kill && echo "Docs server stopped" || echo "No server running"
