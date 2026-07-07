.PHONY: install test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/ -v --tb=short $(ARGS)

test-cov:
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	ruff check src tests experiments --quiet || ruff check src tests experiments

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf *.egg-info .pytest_cache
	@echo "Cleaned."
