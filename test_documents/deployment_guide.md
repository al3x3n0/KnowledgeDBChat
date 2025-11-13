# Deployment Guide

## Prerequisites

Before deploying the Knowledge Database Chat application, ensure you have:

- Docker and Docker Compose installed
- PostgreSQL 15+ (or use Docker image)
- Redis 7+ (or use Docker image)
- Python 3.11+ (for local development)
- Node.js 18+ (for frontend development)

## Docker Deployment

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourorg/knowledge-db-chat.git
cd knowledge-db-chat
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Build and start services:
```bash
docker-compose up -d --build
```

4. Initialize the database:
```bash
docker-compose exec backend alembic upgrade head
```

5. Create admin user:
```bash
docker-compose exec backend python scripts/create_admin.py admin
```

### Service Configuration

#### Backend Service
- Port: 8000
- Environment: Development/Production
- Database: PostgreSQL
- Cache: Redis

#### Frontend Service
- Port: 3000
- Build: Production build served by Nginx
- API URL: Configured via environment variable

#### Celery Worker
- Processes background tasks
- Connects to Redis for task queue
- Handles document processing and indexing

## Environment Variables

### Backend (.env)
```
DATABASE_URL=postgresql://user:password@postgres:5432/knowledge_db
REDIS_URL=redis://redis:6379/0
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440
OLLAMA_BASE_URL=http://ollama:11434
CHROMA_PERSIST_DIRECTORY=/app/data/chroma_db
```

### Frontend (.env)
```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

## Health Checks

Verify all services are running:

```bash
# Check service status
docker-compose ps

# Check backend health
curl http://localhost:8000/health

# Check logs
docker-compose logs -f backend
```

## Troubleshooting

### Database Connection Issues
- Verify PostgreSQL is running: `docker-compose ps postgres`
- Check database credentials in .env
- Ensure database exists: `docker-compose exec postgres psql -U user -d knowledge_db`

### Redis Connection Issues
- Verify Redis is running: `docker-compose ps redis`
- Test connection: `docker-compose exec redis redis-cli ping`

### Frontend Build Issues
- Clear node_modules: `docker-compose exec frontend rm -rf node_modules`
- Rebuild: `docker-compose build frontend`

## Production Considerations

1. **Security**
   - Use strong SECRET_KEY
   - Enable HTTPS
   - Configure CORS properly
   - Use environment-specific settings

2. **Performance**
   - Enable database connection pooling
   - Configure Redis for caching
   - Set up CDN for static assets
   - Use load balancer for multiple instances

3. **Monitoring**
   - Set up logging aggregation
   - Monitor service health
   - Track API metrics
   - Set up alerts

4. **Backup**
   - Regular database backups
   - Backup vector store data
   - Document storage backups

## Scaling

### Horizontal Scaling
- Run multiple backend instances behind load balancer
- Use shared Redis for session storage
- Use shared PostgreSQL database
- Scale Celery workers based on load

### Vertical Scaling
- Increase container resources
- Optimize database queries
- Enable caching strategies
- Use faster storage for vector database


