.PHONY: help setup build start stop restart logs test clean validate-env check-health doctor fmt lint typecheck-frontend test-backend-coverage \
	helm-lint helm-template helm-validate helm-smoke minikube-up minikube-reinstall minikube-down \
	k8s-status k8s-logs-backend k8s-logs-celery k8s-logs-migrate k8s-test k8s-shell-backend k8s-uninstall

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
# The transcription worker builds FROM the backend image and compose does not
# infer build order from a FROM, so the backend is built first by name. Without
# this a clean machine fails on "pull access denied" for an image that is about
# to be built two lines later.
	$(DC) build backend
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

db-migrate: ## Run database migrations (Alembic is the source of schema truth)
	$(DC) exec backend alembic upgrade head

db-revision: ## Autogenerate a migration from model changes (make db-revision M="add x")
	$(DC) exec backend alembic revision --autogenerate -m "$(M)"

db-current: ## Show the applied migration revision
	$(DC) exec backend alembic current

db-check-drift: ## Verify migrations still build the schema the models describe
	$(DC) exec backend python scripts/check_schema_drift.py

db-stamp-legacy: ## Bring a pre-Alembic database (built by create_all) under Alembic
	$(DC) exec backend python scripts/stamp_legacy_database.py

db-shell: ## Open PostgreSQL shell
	$(DC) exec postgres psql -U user -d knowledge_db

redis-shell: ## Open Redis CLI
	$(DC) exec redis redis-cli

test-backend: ## Run backend tests
	$(DC) exec backend pytest

test-rnd-evals: ## Run autonomous R&D evaluation and trajectory regression tests
	$(DC) exec -T backend pytest -q --no-cov tests/test_autonomous_rnd_eval_service.py tests/test_autonomous_rnd_eval_run_service.py tests/test_autonomous_rnd_eval_launch_service.py tests/test_autonomous_rnd_eval_tasks.py tests/test_autonomous_rnd_evidence_verification_service.py tests/test_autonomous_rnd_verification_planner_service.py tests/test_autonomous_rnd_trajectory_service.py tests/test_autonomous_rnd_verification_audit_service.py tests/test_autonomous_rnd_eval_endpoints.py tests/test_agent_experiment_runner_service.py tests/test_benchmark_assets.py

test-external-agents: ## Run external-agent gateway and registry regression tests
	$(DC) exec -T backend pytest -q --no-cov tests/test_external_agent_gateway_service.py tests/test_external_agents_endpoints.py

test-backend-coverage: ## Run backend tests with CI-style coverage threshold
# 48 is the measured floor; the suite reports 49.55%. Keep this equal to the
# --cov-fail-under in .github/workflows/ci.yml.
	$(DC) exec backend pytest --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=48

test-frontend: ## Run frontend tests (non-interactive)
	$(DC) exec frontend npm run test:ci

typecheck-frontend: ## Run frontend TypeScript typecheck
	$(DC) exec frontend sh -lc "./node_modules/.bin/tsc --noEmit -p tsconfig.json"

test-frontend-watch: ## Run frontend tests (watch mode)
	$(DC) exec frontend npm test

test: test-backend test-frontend ## Run all tests

pull-model: ## Pull a model into an Ollama you run yourself
	@echo "The stack no longer bundles Ollama; this project uses an external"
	@echo "LLM API. Point OLLAMA_BASE_URL at your own instance and pull there:"
	@echo "  ollama pull $(MODEL)"
	@exit 1

download-models: ## Download embedding and reranking models
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

dev-backend: ## Start backend in development mode (manual setup)
	cd backend && . venv/bin/activate && uvicorn main:app --reload

dev-frontend: ## Start frontend in development mode (manual setup)
	cd frontend && npm start

dev-celery: ## Start Celery worker in development mode (manual setup)
	cd backend && . venv/bin/activate && celery -A app.core.celery worker --loglevel=info

# --- Kubernetes / Helm -------------------------------------------------------
# The chart lives in deploy/helm/knowledgedbchat; deploy/README.md has the guide.
CHART ?= deploy/helm/knowledgedbchat
K8S_NAMESPACE ?= knowledgedbchat
K8S_RELEASE ?= kdbc

helm-lint: ## Lint the Helm chart against every values profile
	helm lint $(CHART)
	helm lint $(CHART) -f $(CHART)/values-minikube.yaml
	helm lint $(CHART) -f $(CHART)/values-prod.example.yaml

helm-template: ## Render the chart for every values profile
	@helm template $(K8S_RELEASE) $(CHART) > /dev/null && echo "✅ defaults render"
	@helm template $(K8S_RELEASE) $(CHART) -f $(CHART)/values-minikube.yaml > /dev/null && echo "✅ minikube profile renders"
	@helm template $(K8S_RELEASE) $(CHART) -f $(CHART)/values-prod.example.yaml > /dev/null && echo "✅ prod profile renders"

helm-validate: helm-lint ## Render the chart and validate it against the Kubernetes API schemas
	@command -v kubeconform >/dev/null 2>&1 || { echo "kubeconform not installed (brew install kubeconform)"; exit 1; }
	@helm template $(K8S_RELEASE) $(CHART) -f $(CHART)/values-minikube.yaml | kubeconform -strict -summary -kubernetes-version 1.31.0
	@helm template $(K8S_RELEASE) $(CHART) \
		--set ollama.enabled=true --set celeryLatex.enabled=true \
		--set celeryTranscription.enabled=true \
		--set networkPolicy.enabled=true --set ingress.enabled=true \
		--set backend.autoscaling.enabled=true --set celery.autoscaling.enabled=true \
		--set backend.podDisruptionBudget.enabled=true --set secrets.redisPassword=test \
		| kubeconform -strict -summary -kubernetes-version 1.31.0

minikube-up: ## Start minikube, build images into it, and install the chart
	./deploy/minikube/bootstrap.sh

minikube-reinstall: ## Reinstall the chart on the running minikube without rebuilding images
	SKIP_START=1 SKIP_BUILD=1 ./deploy/minikube/bootstrap.sh

minikube-down: ## Delete the minikube profile (and everything in it)
	minikube delete --profile=$${MINIKUBE_PROFILE:-knowledgedbchat}

k8s-status: ## Show pods of the Helm release
	kubectl -n $(K8S_NAMESPACE) get pods,svc -l app.kubernetes.io/instance=$(K8S_RELEASE)

k8s-logs-backend: ## Tail backend logs in Kubernetes
	kubectl -n $(K8S_NAMESPACE) logs -f deploy/$(K8S_RELEASE)-knowledgedbchat-backend

k8s-logs-celery: ## Tail Celery worker logs in Kubernetes
	kubectl -n $(K8S_NAMESPACE) logs -f deploy/$(K8S_RELEASE)-knowledgedbchat-celery

k8s-logs-migrate: ## Show the Alembic migration Job output
	kubectl -n $(K8S_NAMESPACE) logs job/$(K8S_RELEASE)-knowledgedbchat-migrate

k8s-test: ## Run the chart's in-cluster smoke test
	helm test $(K8S_RELEASE) -n $(K8S_NAMESPACE)

k8s-shell-backend: ## Open a shell in a backend pod
	kubectl -n $(K8S_NAMESPACE) exec -it deploy/$(K8S_RELEASE)-knowledgedbchat-backend -- /bin/bash

k8s-uninstall: ## Uninstall the Helm release (PVCs are kept)
	helm uninstall $(K8S_RELEASE) -n $(K8S_NAMESPACE)

helm-smoke: ## Install the chart on the current cluster and assert its wiring (needs a reachable cluster)
	./deploy/smoke-test.sh
