#!/bin/bash

# Knowledge Database Chat Setup Script
echo "🚀 Setting up Knowledge Database Chat Application..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Detect Docker Compose (legacy plugin or modern subcommand)
if command -v docker-compose &> /dev/null; then
    COMPOSE=(docker-compose)
elif docker compose version &> /dev/null; then
    COMPOSE=(docker compose)
else
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating data directories..."
mkdir -p data/{documents,embeddings,logs,chroma_db,postgres-init}

# Set permissions
chmod -R 755 data/

# Copy environment file
echo "⚙️  Setting up environment configuration..."
if [ ! -f backend/.env ]; then
    cp backend/env.example backend/.env
    echo "✅ Created backend/.env from template. Please update it with your settings."
fi

# Pull required Docker images
echo "🐳 Pulling Docker images..."
"${COMPOSE[@]}" pull

# Build the application
echo "🔨 Building application containers..."
"${COMPOSE[@]}" build

# Start the services
echo "🚀 Starting services..."
"${COMPOSE[@]}" up -d postgres redis minio ollama

wait_for() {
    local name="$1"
    local cmd="$2"
    local timeout_seconds="${3:-90}"
    local start_ts
    start_ts="$(date +%s)"

    echo "⏳ Waiting for $name..."
    while true; do
        if eval "$cmd" > /dev/null 2>&1; then
            echo "✅ $name is ready"
            return 0
        fi

        if [ $(( "$(date +%s)" - start_ts )) -ge "$timeout_seconds" ]; then
            echo "❌ Timed out waiting for $name (>${timeout_seconds}s)"
            return 1
        fi

        sleep 2
    done
}

# Wait for core dependencies
wait_for "PostgreSQL" "\"${COMPOSE[@]}\" exec -T postgres pg_isready -U user" 90 || exit 1
wait_for "Redis" "\"${COMPOSE[@]}\" exec -T redis redis-cli ping" 60 || exit 1
wait_for "MinIO" "curl -fsS http://localhost:9000/minio/health/live" 90 || exit 1
wait_for "Ollama API" "curl -fsS http://localhost:11434/api/tags" 90 || exit 1

# Initialize database
echo "🗄️  Initializing database..."
"${COMPOSE[@]}" run --rm backend python -c "
import asyncio
from app.core.database import create_tables
asyncio.run(create_tables())
print('Database tables created successfully')
"

# Initialize vector store
echo "🔍 Initializing vector store..."
"${COMPOSE[@]}" run --rm backend python -c "
import asyncio
from app.services.vector_store import VectorStoreService
async def init_vector_store():
    vs = VectorStoreService()
    await vs.initialize()
    print('Vector store initialized successfully')
asyncio.run(init_vector_store())
"

# Pull default LLM model
echo "🤖 Pulling default LLM model (this may take a while)..."
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
"${COMPOSE[@]}" exec -d ollama ollama pull "$DEFAULT_MODEL"

# Start Celery worker for background tasks
echo "⚙️  Starting background task worker..."
"${COMPOSE[@]}" up -d celery

echo "✅ Setup complete!"
echo ""
echo "🎉 Knowledge Database Chat is now running!"
echo ""
echo "📝 Next steps:"
echo "1. Update backend/.env with your configuration"
echo "2. Set up data sources:"
echo "   - cd backend && python scripts/setup_sources.py example"
echo "   - Edit sources_config.json with your settings"
echo "   - python scripts/setup_sources.py sources_config.json"
echo "3. Access the application:"
echo "   - Frontend: http://localhost:3000"
echo "   - Backend API: http://localhost:8000"
echo "   - API Documentation: http://localhost:8000/docs"
echo "   - Admin Panel: http://localhost:8000/api/v1/admin/health"
echo "4. Start all services: make start"
echo "5. View logs: make logs"
echo ""
echo "🛠️  Useful commands:"
echo "   - Stop services: make stop"
echo "   - View status: make status"
echo "   - Restart services: make restart"
echo "   - View backend logs: make logs-backend"
echo ""
echo "📚 For more information, see README.md"
