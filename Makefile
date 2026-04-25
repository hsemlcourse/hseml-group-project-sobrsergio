.PHONY: install lint test prepare eda train report cp1 clean

PYTHON ?= python
PYTEST ?= pytest
RUFF ?= ruff

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

lint:
	$(RUFF) check src tests

test:
	PYTHONPATH=src $(PYTEST) -q

prepare:
	PYTHONPATH=src $(PYTHON) -m moscow_housing.prepare_data

eda:
	PYTHONPATH=src $(PYTHON) -m moscow_housing.make_eda

train:
	PYTHONPATH=src $(PYTHON) -m moscow_housing.train

report:
	PYTHONPATH=src $(PYTHON) -m moscow_housing.make_cp1_report

cp1: prepare eda train report lint test

clean:
	rm -rf data/processed/* report/images/* report/metrics/* models/* report/report.md
	touch data/processed/.gitkeep models/.gitkeep
