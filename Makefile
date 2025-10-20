.PHONY: help setup install test lint format clean run-api run-app train generate-data docker-build docker-up docker-down

help:
	@echo "Available commands:"
	@echo "  make setup           - Setup development environment"
	@echo "  make install         - Install dependencies"
	@echo "  make test            - Run tests"
	@echo "  make lint            - Run linting"
	@echo "  make format          - Format code"
	@echo "  make generate-data   - Generate synthetic data"
	@echo "  make train           - Train models"
	@echo "  make run-api         - Run FastAPI server"
	@echo "  make run-app         - Run Streamlit app"
	@echo "  make docker-build    - Build Docker images"
	@echo "  make docker-up       - Start Docker containers"
	@echo "  make docker-down     - Stop Docker containers"
	@echo "  make clean           - Clean temporary files"

setup:
	python -m venv venv
	@echo "Virtual environment created. Activate with:"
	@echo "  source venv/bin/activate  (Linux/Mac)"
	@echo "  venv\\Scripts\\activate     (Windows)"

install:
	pip install --upgrade pip
	pip install -r requirements-dev.txt
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -x

lint:
	flake8 src/ tests/ app/ --max-line-length=120
	mypy src/ --ignore-missing-imports
	pylint src/ --max-line-length=120

format:
	black src/ tests/ app/
	isort src/ tests/ app/

generate-data:
	python scripts/generate_data.py

train:
	python scripts/train_model.py

evaluate:
	python scripts/evaluate_model.py

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-app:
	streamlit run app/streamlit_app.py

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-up:
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	docker-compose -f docker/docker-compose.yml down

docker-logs:
	docker-compose -f docker/docker-compose.yml logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf dist/
	rm -rf build/
	rm -f .coverage

all: format lint test