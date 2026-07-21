# Knowledge Database Chat Application

A comprehensive knowledge management system with LLM-based chat interface for organizational data sources. This application enables organizations to build a searchable knowledge base from multiple sources and provides an intelligent chat interface powered by local LLMs.

## 🌟 Features

### Core Functionality
- **Multi-Source Data Ingestion**: Automatically index content from GitLab, Confluence, internal websites, and document files
- **Local LLM Integration**: Privacy-focused local deployment using Ollama for complete data control
- **Semantic Search**: Advanced vector-based document retrieval using Qdrant (default) or ChromaDB
- **RAG Pipeline**: Retrieval-Augmented Generation for contextually accurate responses
- **Real-time Chat**: WebSocket-based chat interface with typing indicators
- **Document References**: Source attribution and links for all AI responses
- **LaTeX Studio**: In-app LaTeX editor with KB-assisted copilot and optional server-side PDF compilation (see `docs/LATEX_STUDIO.md`)
- **Autonomous RnD Agents**: Domain research jobs, recurring research profiles, research fleets, checkpoint approvals, and bounded scientific validation (see `docs/AUTONOMOUS_RND_AGENTS.md`)

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

### MCP (Model Context Protocol)
- Exposes an MCP-compatible tool API for external agents (see `backend/app/mcp/server.py`)
- Tools include semantic search, document browsing, chat/Q&A, and `web_scrape` for extracting readable text/links from wiki/portal pages

## 📚 Key Docs

- `docs/ARCHITECTURE_ASCII.md` - Canonical ASCII system map for frontend, backend, autonomy runtime, and operator surface layout
- `docs/AUTONOMOUS_RND_AGENTS.md` - Operator runbook for autonomous research and engineering agents
- `docs/pilots/research_lab_pilot.md` - Pilot workflow for AI Hub and research-native training/eval loops
- `docs/INGESTION_GUIDE.md` - Source ingestion and document pipeline guidance
- `docs/KNOWLEDGE_GRAPH.md` - Knowledge graph concepts and admin workflow
- `docs/LATEX_STUDIO.md` - LaTeX editor, copilot, compile, and citation sync workflow

Autonomous RnD integrations should use the canonical autonomy contract: `automation_profile`, `automation_policy`, and `effective_policy`. Legacy fields such as `validation_policy` and `follow_up_autonomy` remain backend-compatible, but are compatibility mirrors rather than the preferred interface.

## 🏗️ System Architecture

The detailed current architecture lives in `docs/ARCHITECTURE_ASCII.md`. The diagram below is the high-level product view.

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
│ • Content Sync  │◄──►│ • API Routes    │◄──►│ • Qdrant        │
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
   make start
   # or: docker compose up -d
   ```

   Optional:
   - Enable Docker-based custom tools (unsafe): `docker compose -f docker-compose.yml -f docker-compose.docker-tools.yml up -d`

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
npm install --legacy-peer-deps
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
ollama pull llama3.2:1b
```

##### Optional: Use DeepSeek (External API)
- Set backend to use DeepSeek by editing `backend/.env`:
  - `LLM_PROVIDER=deepseek`
  - `DEEPSEEK_API_KEY=...` (required)
  - Optionally adjust `DEEPSEEK_MODEL` (e.g., `deepseek-chat`) and `DEEPSEEK_MAX_RESPONSE_TOKENS`.
- Note: This sends prompts and context to an external provider. Ensure compliance with your data policies.

###### Heavy Summarization & Chunking
- Large documents are summarized in chunks and then combined into a cohesive summary.
- Heavy jobs (based on `SUMMARIZATION_HEAVY_THRESHOLD_CHARS`) automatically prefer DeepSeek if a key is configured.
- Tuning variables in `backend/.env`:
  - `SUMMARIZATION_CHUNK_SIZE_CHARS` and `SUMMARIZATION_CHUNK_OVERLAP_CHARS` control chunking.
  - `SUMMARIZATION_HEAVY_THRESHOLD_CHARS` controls when external routing is preferred.

## 📊 Configuration

### Environment Variables

#### Backend Configuration (`backend/.env`)
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/knowledge_db
REDIS_URL=redis://localhost:6379/0

# LLM Configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
DEFAULT_MODEL=llama3.2:1b
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Vector store
VECTOR_STORE_PROVIDER=qdrant  # qdrant | chroma
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION_NAME=knowledge_base

# DeepSeek (only if LLM_PROVIDER=deepseek)
DEEPSEEK_API_BASE=https://api.deepseek.com/v1
DEEPSEEK_API_KEY=your-deepseek-api-key
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT_SECONDS=120
DEEPSEEK_MAX_RESPONSE_TOKENS=2000

# Summarization
SUMMARIZATION_HEAVY_THRESHOLD_CHARS=30000
SUMMARIZATION_CHUNK_SIZE_CHARS=12000
SUMMARIZATION_CHUNK_OVERLAP_CHARS=800

# Security
SECRET_KEY=your-secret-key-here
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Custom tools (safety)
CUSTOM_TOOL_DOCKER_ENABLED=false

# Agent governance (writes)
AGENT_KB_PATCH_APPLY_ENABLED=false

# Data Sources
GITLAB_URL=https://gitlab.company.com
GITLAB_TOKEN=your-gitlab-token
CONFLUENCE_URL=https://company.atlassian.net
CONFLUENCE_USER=your-username
CONFLUENCE_API_TOKEN=your-api-token
```

### Tool Policies (Allow-By-Default)

The platform supports a unified tool policy layer across agents, MCP, and workflows:
- Default behavior is allow-by-default; add explicit denies to block tools.
- Policies can also mark a tool as `require_approval`, which creates a pending approval in the tool audit log.

Conventions:
- MCP tool names are policy-addressable as `mcp:<tool_name>` (example: `mcp:web_scrape`, `mcp:create_repo_report`)
- Custom user tools are policy-addressable as `user_tool:<uuid>` (and you can match all with `user_tool:*`)

Constraints (optional, stored in `constraints` JSON):
- `allowed_domains`: list of allowed hostnames/suffixes for `url`/`repo_url` args
- `deny_private_networks`: boolean; blocks `localhost`, private IPs, `.local`, `.internal`
- `max_cost_tier`: `"low"|"medium"|"high"`

Endpoints:
- `GET /api/v1/tools/registry` (includes built-in + your custom tools; shows `allowed` / `require_approval`)
- `POST /api/v1/tools/evaluate` (debug: evaluate a tool call against current policies)
- `GET/POST/DELETE /api/v1/tools/policies` (your per-user policy rules)
- `GET/POST/DELETE /api/v1/admin/tool-policies` (admin policy rules)
- `GET /api/v1/audit/tools` and `POST /api/v1/audit/tools/{audit_id}/approve|reject|run` (owner or admin)

Approvals:
- Tools marked `require_approval` use a dual-approval model by default: resource owner + admin must approve before `run`.

Bootstrap (optional):
- Seed a recommended baseline (approval gates on network/write tools):
  - `python3 scripts/bootstrap_tool_policies.py --dry-run`
  - `python3 scripts/bootstrap_tool_policies.py`
- Optionally restrict network tools to specific domains:
  - `python3 scripts/bootstrap_tool_policies.py --allowed-domains wiki.company.com,github.com,gitlab.company.com`

### Data Source Configuration

#### GitLab Integration
1. Create a Personal Access Token in GitLab
2. Add to environment variables
3. Configure repositories in the admin panel

#### GitHub Integration
- Create a Personal Access Token in GitHub with repo read permissions.
- In Admin → Data Sources, create a source with `source_type: "github"` and config like:
  {
    "token": "ghp_...",
    "repos": ["owner/repo1", {"owner": "org", "repo": "repo2"}],
    "include_files": true,
    "include_issues": true,
    "include_pull_requests": false,
    "include_wiki": false,
    "file_extensions": [".md", ".txt", ".py"]
  }
- Start sync from the Admin panel to index content.

- Optional keys:
  - `ignore_globs`: glob patterns to exclude paths (e.g., ["**/node_modules/**", "**/dist/**"]).
  - `incremental_files` (default true): only fetch files changed since last sync.
  - `max_pages` (default 10): pagination cap for issues/commits.
  - `use_gitignore` (default false): auto-merge root .gitignore patterns into filters.

### Source Scheduling
- In each source's `config`, you can set:
  - `auto_sync`: boolean — enable automatic syncs.
  - `sync_interval_minutes`: number — run at this interval (e.g., 60 for hourly).
  - `cron`: string — optional cron expression (e.g., `0 2 * * *` for 2 AM daily). If present, it supersedes interval. Validated via croniter.
  - `sync_only_changed`: boolean — if false, scheduled runs force a full sync (heavy).

Admin UI exposes toggles for Auto Sync and Sync Only Changed, an Interval field, and shows ETAs during runs. You can also run a Dry Run preview and Cancel ongoing syncs.

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

#### Agents
- `POST /api/v1/agent/chat` - Agentic chat (tool calling + routing)
- `GET /api/v1/agent/tools` - List available agent tools
- `GET /api/v1/agent/capabilities` - List routing capabilities
- `GET /api/v1/agent/agents?search=...` - List agent definitions (admin UI)
- `POST /api/v1/agent/agents` - Create agent definition (admin only)
- `PUT /api/v1/agent/agents/{id}` - Update agent definition (admin only)
- `DELETE /api/v1/agent/agents/{id}` - Delete agent definition (admin only)
- `POST /api/v1/agent/agents/{id}/duplicate` - Duplicate agent definition (admin only)

#### Autonomous RnD
- `POST /api/v1/agent-jobs/quick-start/domain-research` - Launch a bounded domain-research job
- `GET /api/v1/agent-jobs/checkpoint-queue` - Review pending approvals and follow-up checkpoints
- `GET /api/v1/domain-research-profiles` - List recurring domain monitors
- `POST /api/v1/domain-research-profiles` - Create a recurring domain monitor
- `GET /api/v1/research-portfolios` - List research fleets
- `POST /api/v1/research-portfolios` - Create a research fleet
- `GET /api/v1/scientific-sandbox-profiles` - List scientific validation environments
- `POST /api/v1/experiments/plans/generate` - Create experiment plans from research findings

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
npm install --legacy-peer-deps
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

# Backend coverage gate (CI-style)
pytest --cov=app --cov-report=term-missing --cov-report=html --cov-report=xml --cov-fail-under=70

# Frontend tests
cd frontend
npm run test:ci

# Frontend typecheck
./node_modules/.bin/tsc --noEmit -p tsconfig.json

# Integration tests
docker compose -f docker-compose.test.yml up --abort-on-container-exit
```

## 📦 Production Deployment

### Docker Production Setup
```bash
# Build production images
docker compose -f docker-compose.prod.yml build

# Deploy with environment-specific config
docker compose -f docker-compose.prod.yml up -d
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
- Backend `/health` - Application health status
- Backend `/api/v1/health` - Detailed service health
- Nginx `/health` (http://localhost:3000/health) - Frontend reverse proxy health
- MinIO live check (http://localhost:9000/minio/health/live)

### Logs
```bash
# View application logs
docker compose logs -f backend

# View specific service logs
docker compose logs -f ollama
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
- Check vector store is running (Qdrant container) or Chroma directory permissions (if using Chroma)
- Verify embedding model is downloaded
- Restart backend/celery and re-ingest documents if needed

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
