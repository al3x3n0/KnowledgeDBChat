# Quick Start - Launch Everything

This guide shows you how to launch all services for the Knowledge Database Chat application.

## 🚀 Fastest Way (Using Makefile)

```bash
# 1. Initial setup (one-time)
make setup

# 2. Start all services
make start

# 3. Check health
make health

# 4. View logs (optional)
make logs
```

That's it! The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

---

## 📋 Step-by-Step Launch Guide

### Option 1: Docker (Recommended)

#### First Time Setup

```bash
# 1. Navigate to project directory
cd /Users/al3x3n0/huawei/KnowledgeDBChat

# 2. Run initial setup
make setup
# This creates directories and copies .env files

# 3. Build Docker containers (first time only)
make build
# or: docker-compose build
```

#### Launch Services

```bash
# Start all services
make start
# or: docker-compose up -d

# Check status
docker-compose ps

# View logs
make logs
# or: docker-compose logs -f
```

#### Initialize Database

```bash
# Run database migrations
make db-migrate
# or: docker-compose exec backend python -c "import asyncio; from app.core.database import create_tables; asyncio.run(create_tables())"
```

#### Download Ollama Model (for chat functionality)

```bash
# Pull the LLM model
make pull-model
# or: docker-compose exec ollama ollama pull llama2
```

#### Verify Everything is Running

```bash
# Check health of all services
make health
# or: ./scripts/check_health.sh
```

---

### Option 2: Manual Setup (Without Docker)

#### Prerequisites
- PostgreSQL running on port 5432
- Redis running on port 6379
- Ollama running on port 11434

#### Launch Backend

**Terminal 1 - Backend API:**
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Celery Worker:**
```bash
cd backend
source venv/bin/activate
celery -A app.core.celery worker --loglevel=info
```

#### Launch Frontend

**Terminal 3 - Frontend:**
```bash
cd frontend
npm install  # First time only
npm start
```

---

## 🔍 Verify Services are Running

### Check Service Status

```bash
# Using Makefile
make status

# Or manually
docker-compose ps
```

### Check Health Endpoints

```bash
# Backend health
curl http://localhost:8000/health

# Frontend
curl http://localhost:3000

# Ollama
curl http://localhost:11434/api/tags
```

### View Logs

```bash
# All services
make logs

# Specific service
make logs-backend
make logs-frontend
make logs-celery

# Or manually
docker-compose logs -f [service_name]
```

---

## 🎯 Complete Launch Sequence

Here's the complete sequence for first-time launch:

```bash
# 1. Setup
make setup

# 2. Build (first time only)
make build

# 3. Start services
make start

# 4. Wait a few seconds for services to initialize
sleep 10

# 5. Initialize database
make db-migrate

# 6. Download Ollama model (optional, for chat)
make pull-model

# 7. Check health
make health

# 8. Access the application
# Frontend: http://localhost:3000
# API Docs: http://localhost:8000/docs
```

---

## 🛑 Stop Services

```bash
# Stop all services
make stop
# or: docker-compose down

# Stop and remove volumes (clean slate)
docker-compose down -v
```

---

## 🔧 Troubleshooting

### Services won't start

```bash
# Check Docker is running
docker ps

# Check ports are available
lsof -i :8000  # Backend
lsof -i :3000  # Frontend
lsof -i :5432  # PostgreSQL
lsof -i :6379  # Redis
lsof -i :11434 # Ollama
```

### View specific service logs

```bash
# Backend errors
make logs-backend

# Database connection issues
docker-compose logs postgres

# Redis issues
docker-compose logs redis
```

### Restart a specific service

```bash
docker-compose restart backend
docker-compose restart frontend
docker-compose restart celery
```

---

## 📝 Next Steps After Launch

1. **Create your first user**:
   - Open http://localhost:3000
   - Click "Register"
   - Create an account

2. **Upload documents**:
   - Go to Documents page
   - Upload a PDF, Word doc, or text file
   - Wait for processing to complete

3. **Start chatting**:
   - Go to Chat page
   - Ask questions about your documents
   - The RAG system will retrieve relevant information

4. **Check API documentation**:
   - Visit http://localhost:8000/docs
   - Explore available endpoints
   - Test API calls directly

---

## 🎉 Quick Reference

| Action | Command |
|--------|---------|
| Setup | `make setup` |
| Build | `make build` |
| Start | `make start` |
| Stop | `make stop` |
| Restart | `make restart` |
| Logs | `make logs` |
| Health | `make health` |
| Status | `make status` |
| Pull model | `make pull-model` |

For more details, see [BUILD_AND_RUN.md](BUILD_AND_RUN.md).

