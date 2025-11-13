# Knowledge Database Chat Application

A comprehensive knowledge management system with LLM-based chat interface for organizational data sources. This application enables organizations to build a searchable knowledge base from multiple sources and provides an intelligent chat interface powered by local LLMs.

## 🌟 Features

### Core Functionality
- **Multi-Source Data Ingestion**: Automatically index content from GitLab, Confluence, internal websites, and document files
- **Local LLM Integration**: Privacy-focused local deployment using Ollama for complete data control
- **Semantic Search**: Advanced vector-based document retrieval using ChromaDB
- **RAG Pipeline**: Retrieval-Augmented Generation for contextually accurate responses
- **Real-time Chat**: WebSocket-based chat interface with typing indicators
- **Document References**: Source attribution and links for all AI responses

### Data Sources Supported
- **GitLab**: Repository files, wikis, issues, merge requests
- **Confluence**: Pages, attachments, comments
- **Web Scraping**: Internal websites and documentation
- **File Upload**: PDF, Word, Text, Markdown, HTML files
- **Extensible**: Easy to add new data source connectors

### Security & Privacy
- **Local LLM Deployment**: No data sent to external services
- **User Authentication**: JWT-based authentication with role management
- **Access Control**: Document-level permissions and user roles
- **Audit Logging**: Complete audit trail of all interactions

## 🏗️ System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │    Frontend     │    │    Backend      │
│                 │    │                 │    │                 │
│ • GitLab        │    │ • React/TS      │    │ • FastAPI       │
│ • Confluence    │    │ • Real-time UI  │    │ • Python        │
│ • Web Content   │    │ • Chat Interface│    │ • LLM Interface │
│ • Documents     │    │ • Document View │    │ • RAG Pipeline  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Ingestion │    │   Web Server    │    │  Vector Store   │
│                 │    │                 │    │                 │
│ • Content Sync  │◄──►│ • API Routes    │◄──►│ • ChromaDB      │
│ • Text Extract  │    │ • WebSocket     │    │ • Embeddings    │
│ • Processing    │    │ • Authentication│    │ • Similarity    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   Local LLM     │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Documents     │    │ • Ollama        │    │ • Health Checks │
│ • Chat History  │    │ • Multiple      │    │ • Metrics       │
│ • User Data     │    │   Models        │    │ • Logging       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Option 1: Docker Setup (Recommended)

1. **Clone and Setup**
   ```bash
   git clone <repository-url>
   cd KnowledgeDBChat
   ./setup.sh
   ```

2. **Configure Environment**
   ```bash
   # Edit backend/.env with your settings
   nano backend/.env
   ```

3. **Start Services**
   ```bash
   docker-compose up -d
   ```

4. **Access Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

### Option 2: Manual Setup

#### Prerequisites
- Python 3.9+
- Node.js 18+
- PostgreSQL 13+
- Redis 6+
- Ollama (for LLM)

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env
# Edit .env with your database and service URLs
uvicorn main:app --reload
```

#### Frontend Setup
```bash
cd frontend
npm install
npm start
```

#### Database Setup
```bash
# Create PostgreSQL database
createdb knowledge_db

# Run database migrations
cd backend
python -c "
import asyncio
from app.core.database import create_tables
asyncio.run(create_tables())
"
```

#### LLM Setup
```bash
# Install and start Ollama
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve

# Pull a model (in another terminal)
ollama pull llama2
```

## 📊 Configuration

### Environment Variables

#### Backend Configuration (`backend/.env`)
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/knowledge_db
REDIS_URL=redis://localhost:6379/0

# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama2
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Data Sources
GITLAB_URL=https://gitlab.company.com
GITLAB_TOKEN=your-gitlab-token
CONFLUENCE_URL=https://company.atlassian.net
CONFLUENCE_USER=your-username
CONFLUENCE_API_TOKEN=your-api-token
```

### Data Source Configuration

#### GitLab Integration
1. Create a Personal Access Token in GitLab
2. Add to environment variables
3. Configure repositories in the admin panel

#### Confluence Integration
1. Create an API token in Atlassian
2. Add credentials to environment variables
3. Configure spaces in the admin panel

## 🔧 Development

### Project Structure
```
KnowledgeDBChat/
├── backend/                 # Python FastAPI backend
│   ├── app/
│   │   ├── api/            # API routes and endpoints
│   │   │   └── endpoints/  # Individual endpoint modules
│   │   ├── core/           # Core functionality (config, database)
│   │   ├── models/         # SQLAlchemy database models
│   │   ├── schemas/        # Pydantic request/response models
│   │   ├── services/       # Business logic services
│   │   └── utils/          # Utility functions
│   ├── requirements.txt    # Python dependencies
│   ├── Dockerfile         # Docker configuration
│   └── main.py            # Application entry point
├── frontend/               # React TypeScript frontend
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── chat/      # Chat-related components
│   │   │   ├── documents/ # Document management
│   │   │   └── common/    # Shared components
│   │   ├── contexts/       # React contexts
│   │   ├── hooks/          # Custom React hooks
│   │   ├── services/       # API service functions
│   │   ├── types/          # TypeScript type definitions
│   │   └── utils/          # Utility functions
│   ├── package.json       # Node.js dependencies
│   └── Dockerfile         # Docker configuration
├── data/                   # Data storage directories
│   ├── documents/         # Uploaded documents
│   ├── chroma_db/         # Vector database
│   └── logs/              # Application logs
├── docker-compose.yml      # Multi-service Docker setup
├── setup.sh               # Automated setup script
└── README.md
```

### API Endpoints

#### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `GET /api/v1/auth/me` - Get current user

#### Chat
- `POST /api/v1/chat/sessions` - Create chat session
- `GET /api/v1/chat/sessions` - List user sessions
- `POST /api/v1/chat/sessions/{id}/messages` - Send message
- `WS /api/v1/chat/sessions/{id}/ws` - WebSocket chat

#### Documents
- `GET /api/v1/documents/` - List documents
- `POST /api/v1/documents/upload` - Upload document
- `DELETE /api/v1/documents/{id}` - Delete document
- `POST /api/v1/documents/{id}/reprocess` - Reprocess document

#### Admin
- `GET /api/v1/documents/sources/` - List data sources
- `POST /api/v1/documents/sources/` - Create data source
- `POST /api/v1/documents/sources/{id}/sync` - Trigger sync

### Development Commands

```bash
# Backend development
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Frontend development
cd frontend
npm install
npm start

# Database operations
# Create migration
alembic revision --autogenerate -m "description"

# Run migrations
alembic upgrade head

# Reset database
python -c "
import asyncio
from app.core.database import drop_tables, create_tables
asyncio.run(drop_tables())
asyncio.run(create_tables())
"
```

## 🧪 Testing

```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test

# Integration tests
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 📦 Production Deployment

### Docker Production Setup
```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with environment-specific config
docker-compose -f docker-compose.prod.yml up -d
```

### Manual Production Setup
1. Set up production database (PostgreSQL)
2. Configure Redis instance
3. Set up Ollama with required models
4. Deploy backend with gunicorn
5. Build and serve frontend with nginx
6. Configure reverse proxy and SSL

## 🔍 Monitoring & Maintenance

### Health Checks
- `/health` - Application health status
- `/api/v1/health` - Detailed service health

### Logs
```bash
# View application logs
docker-compose logs -f backend

# View specific service logs
docker-compose logs -f ollama
```

### Backup
```bash
# Database backup
pg_dump knowledge_db > backup.sql

# Vector database backup
tar -czf chroma_backup.tar.gz data/chroma_db/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Commit: `git commit -am 'Add feature'`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support & Troubleshooting

### Common Issues

**Ollama Connection Failed**
- Ensure Ollama is running: `ollama serve`
- Check model is available: `ollama list`
- Verify OLLAMA_BASE_URL in configuration

**Vector Search Not Working**
- Check ChromaDB directory permissions
- Verify embedding model is downloaded
- Restart vector store service

**Database Connection Issues**
- Verify PostgreSQL is running
- Check DATABASE_URL configuration
- Ensure database exists and is accessible

### Getting Help
1. Check the [Documentation](docs/)
2. Search [Issues](../../issues)
3. Create a new issue with detailed information
4. Join our [Discord Community](discord-invite-link)

---

**Built with ❤️ for organizational knowledge management**
