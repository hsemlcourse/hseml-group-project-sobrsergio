.PHONY: install lint test prepare eda train train-cp2 api api-smoke cp1 cp2 cp3 clean

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

train-cp2:
	PYTHONPATH=src $(PYTHON) -m moscow_housing.train_cp2

api:
	PYTHONPATH=src uvicorn moscow_housing.api:app --host 0.0.0.0 --port 8000

api-smoke:
	PYTHONPATH=src $(PYTHON) -m moscow_housing.smoke_api

cp1: prepare eda train lint test

cp2: prepare eda train train-cp2 lint test

cp3: prepare eda train train-cp2 lint test

clean:
	rm -rf data/processed/* report/images/* report/metrics/* models/*
	touch data/processed/.gitkeep models/.gitkeep
