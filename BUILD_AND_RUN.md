# Build and Run Guide

This guide provides step-by-step instructions for building and running the Knowledge Database Chat application.

## Prerequisites

### For Docker Setup (Recommended)
- Docker 20.10+
- Docker Compose 2.0+
- 8GB+ RAM (for running Ollama models)
- 20GB+ free disk space

### For Manual Setup
- Python 3.9+ (3.11 recommended)
- Node.js 18+ (20+ recommended)
- PostgreSQL 13+
- Redis 6+
- Ollama (for local LLM)

---

## Option 1: Docker Setup (Recommended)

### Quick Start

1. **Clone the repository** (if not already done):
   ```bash
   git clone <repository-url>
   cd KnowledgeDBChat
   ```

2. **Set up environment variables**:
   ```bash
   # Copy example environment file
   cp backend/env.example backend/.env
   
   # Edit with your settings (optional for development)
   # nano backend/.env
   ```

3. **Create data directories**:
   ```bash
   mkdir -p data/documents data/chroma_db data/logs data/postgres-init
   ```

4. **Start all services**:
   ```bash
   docker-compose up -d
   ```

5. **Wait for services to be ready** (especially Ollama):
   ```bash
   # Check service status
   docker-compose ps
   
   # View logs
   docker-compose logs -f
   ```

6. **Pull Ollama model** (in a new terminal):
   ```bash
   # Connect to Ollama container
   docker exec -it knowledge_db_ollama ollama pull llama2
   
   # Or use local Ollama if installed
   ollama pull llama2
   ```

7. **Run database migrations**:
   ```bash
   # Connect to backend container
   docker exec -it knowledge_db_backend python -c "
   import asyncio
   from app.core.database import create_tables
   asyncio.run(create_tables())
   "
   ```

8. **Access the application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Ollama: http://localhost:11434

### Docker Commands

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f [service_name]

# Restart a service
docker-compose restart [service_name]

# Rebuild after code changes
docker-compose build [service_name]
docker-compose up -d [service_name]

# Stop and remove volumes (clean slate)
docker-compose down -v
```

### Production Docker Setup

For production deployment:

```bash
# Use production compose file
docker-compose -f docker-compose.prod.yml up -d

# Or build first
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
```

---

## Option 2: Manual Setup

### Step 1: Install Prerequisites

#### Install PostgreSQL
```bash
# macOS
brew install postgresql@15
brew services start postgresql@15

# Ubuntu/Debian
sudo apt-get install postgresql-15
sudo systemctl start postgresql

# Create database
createdb knowledge_db
```

#### Install Redis
```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis
```

#### Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Start Ollama
ollama serve

# In another terminal, pull a model
ollama pull llama2
```

### Step 2: Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Copy environment file
cp env.example .env

# Edit .env with your settings
# nano .env  # or use your preferred editor
```

**Important environment variables to set in `.env`**:
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/knowledge_db
REDIS_URL=redis://localhost:6379/0
OLLAMA_BASE_URL=http://localhost:11434
SECRET_KEY=your-secret-key-here  # Generate a secure key!
```

**Generate a secure SECRET_KEY**:
```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Step 3: Database Setup

```bash
# Make sure you're in the backend directory with venv activated
cd backend
source venv/bin/activate  # if not already activated

# Create database tables
python -c "
import asyncio
from app.core.database import create_tables
asyncio.run(create_tables())
"

# Or using Alembic (if migrations exist)
alembic upgrade head
```

### Step 4: Start Backend Services

**Terminal 1 - FastAPI Backend**:
```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Celery Worker** (for background tasks):
```bash
cd backend
source venv/bin/activate
celery -A app.core.celery worker --loglevel=info
```

### Step 5: Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Create .env file (optional, defaults work for local dev)
echo "REACT_APP_API_URL=http://localhost:8000" > .env
echo "REACT_APP_WS_URL=ws://localhost:8000" >> .env

# Start development server
npm start
```

The frontend will open at http://localhost:3000

---

## Verification

### Check Backend Health

```bash
# Using curl
curl http://localhost:8000/health

# Or visit in browser
open http://localhost:8000/docs
```

### Check Services

```bash
# PostgreSQL
psql -U user -d knowledge_db -c "SELECT version();"

# Redis
redis-cli ping
# Should return: PONG

# Ollama
curl http://localhost:11434/api/tags
```

### Create First User

1. Open http://localhost:3000
2. Click "Register" or use the API:
   ```bash
   curl -X POST http://localhost:8000/api/v1/auth/register \
     -H "Content-Type: application/json" \
     -d '{
       "username": "admin",
       "email": "admin@example.com",
       "password": "SecurePassword123!",
       "full_name": "Admin User"
     }'
   ```

---

## Development Workflow

### Backend Development

```bash
cd backend
source venv/bin/activate

# Run with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run tests
pytest

# Run tests with coverage
pytest --cov=app --cov-report=html
```

### Frontend Development

```bash
cd frontend

# Start dev server
npm start

# Run tests
npm test

# Build for production
npm run build
```

### Database Migrations

```bash
cd backend
source venv/bin/activate

# Create a new migration
alembic revision --autogenerate -m "description of changes"

# Apply migrations
alembic upgrade head

# Rollback one migration
alembic downgrade -1
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Error
```bash
# Check PostgreSQL is running
pg_isready

# Verify connection string in .env
# Test connection
psql $DATABASE_URL
```

#### 2. Ollama Connection Failed
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Verify model is available
ollama list

# Check OLLAMA_BASE_URL in .env
```

#### 3. Redis Connection Error
```bash
# Check Redis is running
redis-cli ping

# Verify REDIS_URL in .env
```

#### 4. Port Already in Use
```bash
# Find process using port
lsof -i :8000  # Backend
lsof -i :3000  # Frontend
lsof -i :5432  # PostgreSQL

# Kill process (replace PID)
kill -9 <PID>
```

#### 5. Docker Issues
```bash
# Check container logs
docker-compose logs [service_name]

# Restart specific service
docker-compose restart [service_name]

# Rebuild containers
docker-compose build --no-cache
docker-compose up -d
```

#### 6. Vector Store Issues
```bash
# Check ChromaDB directory permissions
ls -la data/chroma_db

# Reset vector store (WARNING: deletes all embeddings)
rm -rf data/chroma_db/*
```

#### 7. Module Import Errors
```bash
# Reinstall dependencies
cd backend
source venv/bin/activate
pip install -r requirements.txt --force-reinstall

# Or for Docker
docker-compose build --no-cache backend
```

---

## Production Deployment

### Using Docker Compose (Production)

1. **Set production environment variables**:
   ```bash
   # Create production .env file
   cp backend/env.example backend/.env.prod
   # Edit with production values
   ```

2. **Build and start**:
   ```bash
   docker-compose -f docker-compose.prod.yml build
   docker-compose -f docker-compose.prod.yml up -d
   ```

3. **Set up reverse proxy** (nginx example):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:3000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /api {
           proxy_pass http://localhost:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /ws {
           proxy_pass http://localhost:8000;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
       }
   }
   ```

### Manual Production Setup

1. **Backend with Gunicorn**:
   ```bash
   cd backend
   source venv/bin/activate
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
   ```

2. **Frontend Build**:
   ```bash
   cd frontend
   npm run build
   # Serve with nginx or similar
   ```

3. **Set up systemd services** (Linux):
   ```ini
   # /etc/systemd/system/knowledge-db-backend.service
   [Unit]
   Description=Knowledge DB Backend
   After=network.target

   [Service]
   User=your-user
   WorkingDirectory=/path/to/KnowledgeDBChat/backend
   Environment="PATH=/path/to/venv/bin"
   ExecStart=/path/to/venv/bin/gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

   [Install]
   WantedBy=multi-user.target
   ```

---

## Quick Reference

### Service URLs (Development)
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Ollama: http://localhost:11434

### Important Directories
- Backend code: `backend/app/`
- Frontend code: `frontend/src/`
- Data storage: `data/`
- Logs: `data/logs/`
- Vector DB: `data/chroma_db/`
- Documents: `data/documents/`

### Key Commands

#### Using Makefile (Recommended)
```bash
make setup          # Initial setup
make build          # Build containers
make start          # Start all services
make stop           # Stop all services
make logs           # View logs
make health         # Check service health
make test           # Run all tests
make pull-model     # Pull Ollama model
```

#### Using Docker Compose
```bash
docker-compose up -d              # Start all services
docker-compose logs -f           # View logs
docker-compose restart backend    # Restart backend
```

#### Backend
```bash
uvicorn main:app --reload         # Run backend
pytest                            # Run tests
alembic upgrade head              # Run migrations
```

#### Frontend
```bash
npm start                         # Run dev server
npm run build                     # Build for production
npm test                          # Run tests
```

### Utility Scripts

```bash
# Health check
./scripts/check_health.sh

# Environment validation
python scripts/validate_env.py

# Download all models
python scripts/download_models.py
# or using Makefile
make download-models
```

---

## Next Steps

1. **Create your first user** via the registration page
2. **Upload documents** through the Documents page
3. **Start chatting** to test the RAG pipeline
4. **Configure data sources** (GitLab, Confluence) if needed
5. **Review logs** for any issues: `data/logs/app.log`

For more information, see the main [README.md](README.md).

