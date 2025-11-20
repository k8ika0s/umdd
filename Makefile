PYTHON ?= python
PACKAGE ?= umdd

.PHONY: install lint format fmtcheck typecheck test check docker-build-test docker-test

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check src tests

format:
	ruff format src tests

fmtcheck:
	ruff format --check src tests

typecheck:
	mypy src

test:
	pytest

check: fmtcheck lint typecheck test

docker-build-test:
	docker build -f docker/Dockerfile.test -t umdd-test .

docker-test: docker-build-test
	docker run --rm -v $$PWD:/workspace -w /workspace umdd-test make check
