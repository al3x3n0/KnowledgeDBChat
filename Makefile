.PHONY: help setup build start stop restart logs test clean validate-env check-health doctor fmt lint typecheck-frontend test-backend-coverage

# Prefer legacy `docker-compose` if installed, otherwise use `docker compose`.
DC ?= $(shell command -v docker-compose >/dev/null 2>&1 && echo docker-compose || echo "docker compose")

# Default Ollama model to pull (override with `make pull-model MODEL=...`).
MODEL ?= llama3.2:1b

help: ## Show this help message
	@echo "Knowledge Database Chat - Makefile Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: ## Initial setup - create directories and copy env files
	@echo "🚀 Setting up Knowledge Database Chat..."
	@mkdir -p data/documents data/embeddings data/chroma_db data/logs data/postgres-init
	@if [ ! -f backend/.env ]; then \
		cp backend/env.example backend/.env; \
		echo "✅ Created backend/.env"; \
	fi
	@if [ ! -f frontend/.env ]; then \
		cp frontend/.env.example frontend/.env && \
		echo "✅ Created frontend/.env"; \
	fi
	@echo "✅ Setup complete!"

build: ## Build Docker containers
	$(DC) build

start: ## Start all services
	$(DC) up -d
	@echo "✅ Services started. Use 'make logs' to view logs."

stop: ## Stop all services
	$(DC) down

restart: ## Restart all services
	$(DC) restart

logs: ## View logs from all services
	$(DC) logs -f

logs-backend: ## View backend logs only
	$(DC) logs -f backend

logs-frontend: ## View frontend logs only
	$(DC) logs -f frontend

logs-celery: ## View Celery worker logs
	$(DC) logs -f celery

shell-backend: ## Open shell in backend container
	$(DC) exec backend /bin/bash

shell-frontend: ## Open shell in frontend container
	$(DC) exec frontend /bin/sh

db-migrate: ## Run database migrations
	$(DC) exec backend python -c "import asyncio; from app.core.database import create_tables; asyncio.run(create_tables())"

db-shell: ## Open PostgreSQL shell
	$(DC) exec postgres psql -U user -d knowledge_db

redis-shell: ## Open Redis CLI
	$(DC) exec redis redis-cli

test-backend: ## Run backend tests
	$(DC) exec backend pytest

test-rnd-evals: ## Run autonomous R&D evaluation and trajectory regression tests
	$(DC) exec -T backend pytest -q --no-cov tests/test_autonomous_rnd_eval_service.py tests/test_autonomous_rnd_eval_run_service.py tests/test_autonomous_rnd_evidence_verification_service.py tests/test_autonomous_rnd_verification_planner_service.py tests/test_autonomous_rnd_trajectory_service.py tests/test_autonomous_rnd_verification_audit_service.py tests/test_autonomous_rnd_eval_endpoints.py tests/test_agent_experiment_runner_service.py tests/test_benchmark_assets.py

test-external-agents: ## Run external-agent gateway and registry regression tests
	$(DC) exec -T backend pytest -q --no-cov tests/test_external_agent_gateway_service.py tests/test_external_agents_endpoints.py

test-backend-coverage: ## Run backend tests with CI-style coverage threshold
	$(DC) exec backend pytest --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=70

test-frontend: ## Run frontend tests (non-interactive)
	$(DC) exec frontend npm run test:ci

typecheck-frontend: ## Run frontend TypeScript typecheck
	$(DC) exec frontend sh -lc "./node_modules/.bin/tsc --noEmit -p tsconfig.json"

test-frontend-watch: ## Run frontend tests (watch mode)
	$(DC) exec frontend npm test

test: test-backend test-frontend ## Run all tests

pull-model: ## Pull default Ollama model
	$(DC) exec ollama ollama pull $(MODEL)

download-models: ## Download all necessary models (Ollama, embeddings, reranking)
	python scripts/download_models.py

validate-env: ## Validate backend environment variables
	python3 scripts/validate_env.py

check-health: ## Run local health checks (Docker + services)
	bash scripts/check_health.sh

doctor: validate-env check-health ## Validate env + health checks

fmt-backend: ## Format backend code (isort + black)
	$(DC) exec backend isort .
	$(DC) exec backend black .

lint-backend: ## Lint backend code (flake8)
	$(DC) exec backend flake8

fmt: fmt-backend ## Run formatters

lint: lint-backend ## Run linters

clean: ## Remove containers, volumes, and images
	@echo "⚠️  This will remove containers, volumes, and images (all data will be lost). Continue? [y/N]"
	@read -r confirm && [ "$$confirm" = "y" ] || exit 1
	$(DC) down -v --rmi all

clean-data: ## Remove only data volumes (keeps images)
	@echo "⚠️  This will remove containers and volumes (all data will be lost). Continue? [y/N]"
	@read -r confirm && [ "$$confirm" = "y" ] || exit 1
	$(DC) down -v

status: ## Show status of all services
	$(DC) ps

health: ## Check health of all services
	@echo "Checking service health..."
	@echo ""
	@echo "Backend API:"
	@curl -s http://localhost:8000/health || echo "❌ Backend not responding"
	@echo ""
	@echo "Nginx:"
	@curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000/health || echo "❌ Nginx not responding"
	@echo ""
	@echo "Frontend:"
	@curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000 || echo "❌ Frontend not responding"
	@echo ""
	@echo "PostgreSQL:"
	@$(DC) exec -T postgres pg_isready -U user || echo "❌ PostgreSQL not ready"
	@echo ""
	@echo "Redis:"
	@$(DC) exec -T redis redis-cli ping || echo "❌ Redis not responding"
	@echo ""
	@echo "MinIO:"
	@curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:9000/minio/health/live || echo "❌ MinIO not responding"
	@echo ""
	@echo "Video streamer:"
	@curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8080/health || echo "❌ Video streamer not responding"
	@echo ""
	@echo "Ollama:"
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ Ollama is running" || echo "❌ Ollama not responding"

dev-backend: ## Start backend in development mode (manual setup)
	cd backend && . venv/bin/activate && uvicorn main:app --reload

dev-frontend: ## Start frontend in development mode (manual setup)
	cd frontend && npm start

dev-celery: ## Start Celery worker in development mode (manual setup)
	cd backend && . venv/bin/activate && celery -A app.core.celery worker --loglevel=info
