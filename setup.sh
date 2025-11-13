#!/bin/bash

# Knowledge Database Chat Setup Script
echo "🚀 Setting up Knowledge Database Chat Application..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
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
docker-compose pull

# Build the application
echo "🔨 Building application containers..."
docker-compose build

# Start the services
echo "🚀 Starting services..."
docker-compose up -d postgres redis ollama

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Initialize database
echo "🗄️  Initializing database..."
docker-compose run --rm backend python -c "
import asyncio
from app.core.database import create_tables
asyncio.run(create_tables())
print('Database tables created successfully')
"

# Initialize vector store
echo "🔍 Initializing vector store..."
docker-compose run --rm backend python -c "
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
docker-compose exec -d ollama ollama pull llama2

# Start Celery worker for background tasks
echo "⚙️  Starting background task worker..."
docker-compose up -d celery

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
echo "4. Start all services: docker-compose up -d"
echo "5. View logs: docker-compose logs -f"
echo ""
echo "🛠️  Useful commands:"
echo "   - Stop services: docker-compose down"
echo "   - View status: docker-compose ps"
echo "   - Restart services: docker-compose restart"
echo "   - View backend logs: docker-compose logs backend"
echo ""
echo "📚 For more information, see README.md"
