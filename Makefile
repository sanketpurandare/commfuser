.PHONY: default

PY_FILES := $(wildcard simulation/*py bucketing/*.py scheduling/*.py recomputation/*py graph_profiling/*py demo/*py)
PIP ?= python -m pip

format:
	isort $(PY_FILES)
	black $(PY_FILES)

lint:
	black --check --diff $(PY_FILES)
	isort --check --diff $(PY_FILES)

