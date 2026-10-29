# Makefile for QML-01 Delay-Risk-Aware Fleet Dispatch
.PHONY: help install test pipeline all clean zip

PYTHON ?= python
PYTEST ?= pytest

help:
	@echo "QML-01 Delay-Risk-Aware Fleet Dispatch"
	@echo ""
	@echo "Targets:"
	@echo "  make install   - Install requirements from requirements.txt"
	@echo "  make test      - Run comprehensive pytest unit test suite"
	@echo "  make pipeline  - Run full 7-stage master reproduction pipeline (run_all.py)"
	@echo "  make all       - Run master pipeline followed by test suite verification"
	@echo "  make zip       - Package distribution zip archive"
	@echo "  make clean     - Remove Python cache and pytest temporary artifacts"

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTEST) -v tests/

pipeline:
	$(PYTHON) run_all.py

all: pipeline test

zip:
	$(PYTHON) scripts/create_zip.py

clean:
	$(PYTHON) -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; shutil.rmtree('.pytest_cache', ignore_errors=True)"
