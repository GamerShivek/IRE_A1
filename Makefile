# Reproducible Data Pipeline — Makefile
# Usage:
#   make data           → full pipeline (both datasets)
#   make data-mind      → MIND only
#   make data-ebnerd    → EB-NeRD only
#   make test           → anti-leakage tests only
#   make clean          → remove interim + processed directories

PYTHON := python3
PIPELINE := $(PYTHON) build_pipeline.py

.PHONY: data data-mind data-ebnerd test clean help

## Default target
data:
	$(PIPELINE)

## MIND only
data-mind:
	$(PIPELINE) --mind-only

## EB-NeRD only
data-ebnerd:
	$(PIPELINE) --ebnerd-only

## Run tests only (assumes pipeline has already been run)
test:
	$(PYTHON) src/tests/test_no_leakage.py

## Remove generated data (raw files are kept)
clean:
	rm -rf data/interim data/processed
	@echo "Cleaned interim and processed directories."

## Show help
help:
	@echo "Available targets:"
	@echo "  make data          - Full pipeline (MIND + EB-NeRD)"
	@echo "  make data-mind     - MIND only"
	@echo "  make data-ebnerd   - EB-NeRD only"
	@echo "  make test          - Anti-leakage tests"
	@echo "  make clean         - Remove interim + processed data"
