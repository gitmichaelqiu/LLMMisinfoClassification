.PHONY: install test lint clean

install:
	pip install -r requirements.txt
	pip install -e .

test:
	python -m pytest tests/ -v --tb=short $(ARGS)

test-cov:
	python -m pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

lint:
	@echo "No linter configured yet. Install ruff or black for linting."
	@python -c "import py_compile; import glob; [py_compile.compile(f, doraise=True) for f in glob.glob('src/**/*.py', recursive=True) if '__pycache__' not in f]"
	@echo "Syntax check passed."

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf *.egg-info .pytest_cache
	@echo "Cleaned."
