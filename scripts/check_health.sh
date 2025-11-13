#!/bin/bash

# Health check script for Knowledge Database Chat
# Checks all services and dependencies

echo "🔍 Health Check for Knowledge Database Chat"
echo "=========================================="
echo ""

ERRORS=0

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
if command -v docker-compose &> /dev/null || docker compose version &> /dev/null; then
    echo "✅ Docker Compose is installed"
else
    echo "❌ Docker Compose is not installed"
    ERRORS=$((ERRORS + 1))
fi
echo ""

# Check if services are running
if docker ps | grep -q knowledge_db; then
    echo "Checking running services..."
    
    # Backend
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Backend API is responding"
    else
        echo "❌ Backend API is not responding"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Frontend
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo "✅ Frontend is responding"
    else
        echo "❌ Frontend is not responding"
        ERRORS=$((ERRORS + 1))
    fi
    
    # PostgreSQL
    if docker exec knowledge_db_postgres pg_isready -U user > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
    else
        echo "❌ PostgreSQL is not ready"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Redis
    if docker exec knowledge_db_redis redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis is responding"
    else
        echo "❌ Redis is not responding"
        ERRORS=$((ERRORS + 1))
    fi
    
    # Ollama
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is responding"
        
        # Check if model is available
        if curl -s http://localhost:11434/api/tags | grep -q llama2; then
            echo "✅ Ollama model 'llama2' is available"
        else
            echo "⚠️  Ollama model 'llama2' not found. Run: make pull-model"
        fi
    else
        echo "❌ Ollama is not responding"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "⚠️  Services are not running. Start with: docker-compose up -d"
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
for dir in data/documents data/chroma_db data/logs; do
    if [ -d "$dir" ]; then
        echo "✅ $dir exists"
    else
        echo "⚠️  $dir not found. Creating..."
        mkdir -p "$dir"
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

