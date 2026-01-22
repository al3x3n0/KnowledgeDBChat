#!/bin/bash

# Health check script for Knowledge Database Chat
# Checks all services and dependencies

echo "🔍 Health Check for Knowledge Database Chat"
echo "=========================================="
echo ""

ERRORS=0

# Detect Docker Compose command
if command -v docker-compose &> /dev/null; then
    COMPOSE=(docker-compose)
elif docker compose version &> /dev/null; then
    COMPOSE=(docker compose)
else
    COMPOSE=()
fi

# Detect configured model (best-effort)
DEFAULT_MODEL="llama3.2:1b"
if [ -f backend/.env ]; then
    MODEL_FROM_ENV="$(
        grep -E '^DEFAULT_MODEL=' backend/.env | tail -n 1 | cut -d= -f2- \
            | sed -E 's/[[:space:]]+#.*$//' \
            | tr -d '"' \
            | tr -d "'"
    )"
    if [ -n "$MODEL_FROM_ENV" ]; then
        DEFAULT_MODEL="$MODEL_FROM_ENV"
    fi
fi

# Check Docker
echo "Checking Docker..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    if docker ps &> /dev/null; then
        echo "✅ Docker daemon is running"
    else
        echo "❌ Docker daemon is not running"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "❌ Docker is not installed"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check Docker Compose
echo "Checking Docker Compose..."
if [ "${#COMPOSE[@]}" -gt 0 ]; then
    echo "✅ Docker Compose is installed"
else
    echo "❌ Docker Compose is not installed"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check if services are running (via compose if available)
SERVICES_RUNNING=0
if [ "${#COMPOSE[@]}" -gt 0 ]; then
    if [ -n "$("${COMPOSE[@]}" ps -q 2>/dev/null)" ]; then
        SERVICES_RUNNING=1
    fi
fi

if [ "$SERVICES_RUNNING" -eq 1 ]; then
    echo "Checking running services (via Docker Compose)..."
    
    # Backend
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend API is responding"
    else
        echo "❌ Backend API is not responding"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Nginx (frontend reverse proxy)
    if curl -s http://localhost:3000/health > /dev/null 2>&1; then
        echo "✅ Nginx is responding"
    else
        echo "❌ Nginx is not responding"
        ERRORS=$((ERRORS + 1))
    fi

    # Frontend app (served via nginx)
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Frontend is responding"
    else
        echo "❌ Frontend is not responding"
        ERRORS=$((ERRORS + 1))
    fi
    
    # PostgreSQL
    if "${COMPOSE[@]}" exec -T postgres pg_isready -U user > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
    else
        echo "❌ PostgreSQL is not ready"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Redis
    if "${COMPOSE[@]}" exec -T redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is responding"
    else
        echo "❌ Redis is not responding"
        ERRORS=$((ERRORS + 1))
    fi

    # MinIO
    if curl -s http://localhost:9000/minio/health/live > /dev/null 2>&1; then
        echo "✅ MinIO is responding"
    else
        echo "❌ MinIO is not responding"
        ERRORS=$((ERRORS + 1))
    fi

    # Video streamer
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
        echo "✅ Video streamer is responding"
    else
        echo "❌ Video streamer is not responding"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is responding"
        
        # Check if model is available
        if curl -s http://localhost:11434/api/tags | grep -Fq "$DEFAULT_MODEL"; then
            echo "✅ Ollama model '$DEFAULT_MODEL' is available"
        else
            echo "⚠️  Ollama model '$DEFAULT_MODEL' not found. Run: make pull-model MODEL=$DEFAULT_MODEL"
        fi
    else
        echo "❌ Ollama is not responding"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "⚠️  Services are not running. Start with: make start"
fi
echo ""

# Check environment files
echo "Checking configuration files..."
if [ -f backend/.env ]; then
    echo "✅ backend/.env exists"
else
    echo "⚠️  backend/.env not found. Copy from backend/env.example"
fi

if [ -f frontend/.env ]; then
    echo "✅ frontend/.env exists"
else
    echo "⚠️  frontend/.env not found. Copy from frontend/.env.example"
fi
echo ""

# Check data directories
echo "Checking data directories..."
for dir in data/documents data/embeddings data/chroma_db data/logs data/postgres-init; do
    if [ -d "$dir" ]; then
        echo "✅ $dir exists"
    else
        echo "⚠️  $dir not found. Run: make setup"
    fi
done
echo ""

# Summary
echo "=========================================="
if [ $ERRORS -eq 0 ]; then
    echo "✅ Health check passed! All services are healthy."
    exit 0
else
    echo "❌ Health check found $ERRORS error(s). Please fix the issues above."
    exit 1
fi
