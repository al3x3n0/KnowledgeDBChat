.PHONY: help setup build start stop restart logs test clean

help: ## Show this help message
	@echo "Knowledge Database Chat - Makefile Commands"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: ## Initial setup - create directories and copy env files
	@echo "🚀 Setting up Knowledge Database Chat..."
	@mkdir -p data/documents data/chroma_db data/logs data/postgres-init
	@if [ ! -f backend/.env ]; then \
		cp backend/env.example backend/.env; \
		echo "✅ Created backend/.env"; \
	fi
	@if [ ! -f frontend/.env ]; then \
		cp frontend/.env.example frontend/.env; \
		echo "✅ Created frontend/.env"; \
	fi
	@echo "✅ Setup complete!"

build: ## Build Docker containers
	docker-compose build

start: ## Start all services
	docker-compose up -d
	@echo "✅ Services started. Use 'make logs' to view logs."

stop: ## Stop all services
	docker-compose down

restart: ## Restart all services
	docker-compose restart

logs: ## View logs from all services
	docker-compose logs -f

logs-backend: ## View backend logs only
	docker-compose logs -f backend

logs-frontend: ## View frontend logs only
	docker-compose logs -f frontend

logs-celery: ## View Celery worker logs
	docker-compose logs -f celery

shell-backend: ## Open shell in backend container
	docker-compose exec backend /bin/bash

shell-frontend: ## Open shell in frontend container
	docker-compose exec frontend /bin/sh

db-migrate: ## Run database migrations
	docker-compose exec backend python -c "import asyncio; from app.core.database import create_tables; asyncio.run(create_tables())"

db-shell: ## Open PostgreSQL shell
	docker-compose exec postgres psql -U user -d knowledge_db

redis-shell: ## Open Redis CLI
	docker-compose exec redis redis-cli

test-backend: ## Run backend tests
	docker-compose exec backend pytest

test-frontend: ## Run frontend tests
	docker-compose exec frontend npm test

test: test-backend test-frontend ## Run all tests

pull-model: ## Pull default Ollama model
	docker-compose exec ollama ollama pull llama2

download-models: ## Download all necessary models (Ollama, embeddings, reranking)
	python scripts/download_models.py

clean: ## Remove containers, volumes, and images
	docker-compose down -v
	@echo "⚠️  All data will be lost. Continue? [y/N]"
	@read -r confirm && [ "$$confirm" = "y" ] || exit 1
	docker-compose down -v --rmi all

clean-data: ## Remove only data volumes (keeps images)
	docker-compose down -v
	@echo "⚠️  All data will be lost. Continue? [y/N]"
	@read -r confirm && [ "$$confirm" = "y" ] || exit 1

status: ## Show status of all services
	docker-compose ps

health: ## Check health of all services
	@echo "Checking service health..."
	@echo ""
	@echo "Backend API:"
	@curl -s http://localhost:8000/health || echo "❌ Backend not responding"
	@echo ""
	@echo "Frontend:"
	@curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:3000 || echo "❌ Frontend not responding"
	@echo ""
	@echo "PostgreSQL:"
	@docker-compose exec -T postgres pg_isready -U user || echo "❌ PostgreSQL not ready"
	@echo ""
	@echo "Redis:"
	@docker-compose exec -T redis redis-cli ping || echo "❌ Redis not responding"
	@echo ""
	@echo "Ollama:"
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "✅ Ollama is running" || echo "❌ Ollama not responding"

dev-backend: ## Start backend in development mode (manual setup)
	cd backend && source venv/bin/activate && uvicorn main:app --reload

dev-frontend: ## Start frontend in development mode (manual setup)
	cd frontend && npm start

dev-celery: ## Start Celery worker in development mode (manual setup)
	cd backend && source venv/bin/activate && celery -A app.core.celery worker --loglevel=info

