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
   mkdir -p data/documents data/embeddings data/chroma_db data/logs data/postgres-init
   ```

4. **Start all services**:
   ```bash
   docker compose up -d
   ```

5. **Wait for services to be ready** (especially Ollama):
   ```bash
   # Check service status
   docker compose ps
   
   # View logs
   docker compose logs -f
   ```

6. **Pull Ollama model** (in a new terminal):
   ```bash
   # Connect to Ollama container
   docker exec -it knowledge_db_ollama ollama pull llama3.2:1b
   
   # Or use local Ollama if installed
   ollama pull llama3.2:1b
   ```

7. **Run database migrations**:
   ```bash
   make db-migrate
   # equivalently: docker exec -it knowledge_db_backend alembic upgrade head
   ```
   The container entrypoint already runs migrations on start; this is for
   applying new ones without a restart.

8. **Access the application**:
   - Frontend: http://localhost:23000
   - Backend API: http://localhost:28000
   - API Documentation: http://localhost:28000/docs
   - Ollama: http://localhost:11434

### LDAP / Active Directory configuration (Optional)

LDAP supports directory password sign-in, account provisioning, group-to-role
mapping, and reconciliation of existing directory-managed accounts. Configure
it in `backend/.env`; the Admin page reports settings but does not edit them.
Keep a working local administrator account for setup and directory outages.

#### 1. Choose a directory configuration

Replace the example host, DNs, and password below. Use a service account that
can read eligible users, their mapped attributes, and groups. Directory writes
are not required. The backend and Celery worker must be able to resolve and
reach the directory host.

**Active Directory** (short account name or user principal name sign-in):

```dotenv
LDAP_ENABLED=true
LDAP_URI=ldaps://dc01.corp.example.com:636
LDAP_START_TLS=false
LDAP_ALLOW_PLAINTEXT=false
LDAP_INSECURE_SKIP_TLS_VERIFY=false
LDAP_CA_CERT_FILE=/app/certs/corporate-ca.pem
LDAP_BIND_DN=CN=KnowledgeDB Reader,OU=Service Accounts,DC=corp,DC=example,DC=com
LDAP_BIND_PASSWORD=replace-with-service-account-password
LDAP_BASE_DN=DC=corp,DC=example,DC=com
LDAP_USER_DN_TEMPLATE=
LDAP_USER_SEARCH_FILTER=(&(objectClass=user)(!(objectClass=computer))(!(userAccountControl:1.2.840.113556.1.4.803:=2))(|(sAMAccountName={username})(userPrincipalName={username})))
LDAP_IMPORT_FILTER=(&(objectClass=user)(!(objectClass=computer))(!(userAccountControl:1.2.840.113556.1.4.803:=2)))
LDAP_USERNAME_ATTRIBUTE=sAMAccountName
LDAP_EMAIL_ATTRIBUTE=mail
LDAP_FULL_NAME_ATTRIBUTE=displayName
LDAP_GROUPS_ATTRIBUTE=memberOf
LDAP_USER_ATTRIBUTES=sAMAccountName,userPrincipalName,mail,displayName,cn,memberOf
LDAP_UNIQUE_ID_ATTRIBUTE=objectGUID
LDAP_CREATE_USER_ON_LOGIN=true
LDAP_SYNC_ON_LOGIN=true
```

The filters exclude computer and disabled accounts. Narrow the base DN or
filters further to the people who should have application access.
`LDAP_IMPORT_FILTER` also determines who retains access during reconciliation.
`objectGUID` preserves the local account identity when an AD entry moves or
is renamed. The application requests this attribute automatically.

**OpenLDAP** (search-based sign-in with `uid`):

```dotenv
LDAP_ENABLED=true
LDAP_URI=ldaps://ldap.example.com:636
LDAP_START_TLS=false
LDAP_ALLOW_PLAINTEXT=false
LDAP_INSECURE_SKIP_TLS_VERIFY=false
LDAP_CA_CERT_FILE=/app/certs/corporate-ca.pem
LDAP_BIND_DN=cn=reader,dc=example,dc=com
LDAP_BIND_PASSWORD=replace-with-service-account-password
LDAP_BASE_DN=ou=People,dc=example,dc=com
LDAP_USER_DN_TEMPLATE=
LDAP_USER_SEARCH_FILTER=(&(objectClass=inetOrgPerson)(uid={username}))
LDAP_IMPORT_FILTER=(objectClass=inetOrgPerson)
LDAP_USERNAME_ATTRIBUTE=uid
LDAP_EMAIL_ATTRIBUTE=mail
LDAP_FULL_NAME_ATTRIBUTE=displayName
LDAP_GROUPS_ATTRIBUTE=memberOf
LDAP_USER_ATTRIBUTES=uid,mail,displayName,cn,memberOf
LDAP_UNIQUE_ID_ATTRIBUTE=entryUUID
LDAP_CREATE_USER_ON_LOGIN=true
LDAP_SYNC_ON_LOGIN=true
```

Adapt the OpenLDAP filters to your schema and access policy; the example
includes every `inetOrgPerson` under the base DN. If entries lack `mail`, set
`LDAP_DEFAULT_EMAIL_DOMAIN=example.com` to generate `username@example.com`.
New accounts require an email address.

For direct-DN login instead of service-account lookup, set, for example,
`LDAP_USER_DN_TEMPLATE=uid={username},ou=People,dc=example,dc=com`. A nonempty
template takes precedence over search-based login. Import, reconciliation, and
the password-free user diagnostic still require a service account. Clear the
template copied from `backend/env.example` when using either search example.

#### 2. Configure TLS and apply settings

For StartTLS, replace the URI with `ldap://directory.example.com:389` and set
`LDAP_START_TLS=true`. Certificates are verified in both modes. Use a hostname
covered by the server certificate and keep
`LDAP_INSECURE_SKIP_TLS_VERIFY=false` and `LDAP_ALLOW_PLAINTEXT=false`.

`LDAP_CA_CERT_FILE` is a path **inside the process/container**, containing the
trusted CA certificates in PEM format. Leave it empty to use system trust.
With the development Compose mounts, placing the CA bundle at
`backend/certs/corporate-ca.pem` makes it available as
`/app/certs/corporate-ca.pem` in the backend and Celery containers. Other
deployments must mount the bundle into every process that connects to LDAP.
Do not commit service-account credentials or private keys.

For the development Compose stack, which mounts `backend/.env`, apply edits
with:

```bash
docker compose restart backend celery celery_beat
```

If your deployment injects settings through container environment variables,
update that environment and recreate the affected containers instead; a
restart does not replace injected environment variables. Environment variables
take precedence over `.env` settings. In manual development, restart the API,
Celery worker, and Celery beat processes after editing their configuration.

#### 3. Map groups to application roles

```dotenv
LDAP_ADMIN_GROUP_DNS=CN=KnowledgeDB Admins,OU=Groups,DC=corp,DC=example,DC=com
LDAP_VIEWER_GROUP_DNS=CN=KnowledgeDB Viewers,OU=Groups,DC=corp,DC=example,DC=com
```

Separate multiple group DNs with **semicolons**, not commas. Admin membership
wins over viewer membership. With a mapping configured, unmatched users get
`user`; with both settings empty, existing roles are preserved and new login
accounts default to `user`. Viewer accounts have read-only restrictions.

Groups are read from `LDAP_GROUPS_ATTRIBUTE` (default `memberOf`). If OpenLDAP
has no `memberOf` overlay, configure a membership search:

```dotenv
LDAP_GROUP_SEARCH_BASE=ou=Groups,dc=example,dc=com
LDAP_GROUP_SEARCH_FILTER=(|(member={user_dn})(uniqueMember={user_dn})(memberUid={username}))
```

For AD nested membership, use a group search with AD's recursive member filter:

```dotenv
LDAP_GROUP_SEARCH_BASE=OU=Groups,DC=corp,DC=example,DC=com
LDAP_GROUP_SEARCH_FILTER=(member:1.2.840.113556.1.4.1941:={user_dn})
```

#### 4. Verify sign-in and provision accounts

1. Sign in as a local administrator and open **Admin → Directory (LDAP)**.
2. Check the configuration problems and transport/certificate status.
3. Run the connection test, optionally supplying a directory login name. It
   checks settings, transport, service-account access, user lookup, and role
   mapping without needing that user's password. It does not validate their
   password; test an actual directory sign-in separately.
4. Use the import preview to inspect users and roles, then apply it. Import
   defaults to 200 entries (API maximum 5,000 per request); adjust its limit
   and filter for the intended population. Passwords are never imported.

Import is optional when `LDAP_CREATE_USER_ON_LOGIN=true`: eligible users get
an account on their first successful directory login. Set it to `false` to
require provisioning by import before sign-in. `LDAP_SYNC_ON_LOGIN` controls
profile updates (email/name); role mapping is still applied.

Local accounts can still sign in with their local password. When a local
password does not match, LDAP can be tried if configured. Login and import
use the same linking safeguards: an existing local account can be linked
when its email matches, or when `LDAP_LINK_LOCAL_USERS_BY_USERNAME=true`.
Accounts created through self-registration are never linked, even with that
override. Inspect refused imports instead of assuming a shared username is
the same person.

`AUTH_SELF_REGISTRATION` unset allows self-registration only when LDAP is off.
Set it explicitly to `true` or `false` to override that behavior; the sign-in
page follows this setting. A directory-managed account cannot set a local
password. A directory password rejection is final; the optional
`LDAP_LOCAL_FALLBACK_WHEN_UNAVAILABLE=true` only permits an existing stored
password during a directory outage, and defaults to `false`.

#### 5. Reconcile accounts between logins

On the same Admin page, preview **Sync**, review role changes and missing
users, then apply it. Sync updates existing LDAP-managed accounts; use import
or first login to create new accounts. Local accounts are untouched.

To enable periodic reconciliation, add:

```dotenv
LDAP_SYNC_INTERVAL_MINUTES=15
LDAP_SYNC_MAX_DEACTIVATE_FRACTION=0.2
LDAP_SYNC_MAX_ENTRIES=50000
LDAP_SEARCH_PAGE_SIZE=200
```

The default interval is `0` (off). Scheduled sync requires Redis, a Celery
worker, and Celery beat with the same directory settings and CA access. Beat
checks every five minutes, so the configured interval is not an exact timer.
Restart the affected processes after changing it.

Sync deactivates accounts absent from `LDAP_IMPORT_FILTER` and applies group
role changes. It reactivates accounts only if a previous sync deactivated
them; an administrator's deactivation is preserved. An unreachable directory
changes no accounts. Empty results, or missing accounts exceeding the default
20% of all managed accounts, block deactivation and explain why in the report.
Check the base DN/filter before raising that threshold for an intentional
large removal. Set `LDAP_SYNC_MAX_ENTRIES` high enough to read the entire
eligible population; a truncated search must not be used for reconciliation.
The latest sync report is stored in Redis and shown on the Admin page.

#### Admin API and troubleshooting

All routes require an administrator bearer token:

| Method and path | Purpose / request body |
| --- | --- |
| `GET /api/v1/admin/ldap/status` | Configuration problems, TLS status, and last sync report |
| `POST /api/v1/admin/ldap/test` | Diagnostic; `{}` or `{"username":"alice"}` |
| `POST /api/v1/admin/ldap/import` | Preview: `{"dry_run":true,"limit":200}`; apply with `dry_run:false` |
| `POST /api/v1/admin/ldap/sync` | Preview: `{"dry_run":true}`; apply with `dry_run:false` |

| Symptom | Check |
| --- | --- |
| TLS or hostname verification fails | Correct directory hostname, certificate validity, and CA bundle path/trust inside both backend and worker containers |
| Service-account bind fails | Bind DN/password and the service account's directory status |
| User is not found or lookup is ambiguous | Base DN, search filter, username attribute, and uniqueness; clear the DN template for search mode |
| Import cannot create an account | Required username/email, fallback email domain, unique email conflicts, and refused local-account linking |
| Expected role is missing | Full group DNs separated by `;`, readable `memberOf`, and membership search base/filter |
| Sync never runs | Nonzero interval, Redis connectivity, Celery worker and beat, and settings loaded by the worker |
| Deactivation is held back | Read the report; verify the filter, base DN, search cap, and removal fraction |
| Directory sign-in returns 503 | Directory connectivity, TLS, timeouts, and service-account access; outages are distinct from incorrect credentials |

Connection and receive timeouts default to 8 and 10 seconds
(`LDAP_CONNECT_TIMEOUT_SECONDS`, `LDAP_RECEIVE_TIMEOUT_SECONDS`).

The optional `./scripts/ldap_ad_bootstrap.sh` helper writes basic AD settings,
restarts the backend, runs diagnostics, and previews an import by default.
It does not configure `objectGUID`, scheduled sync, or exclusion of disabled
accounts by default; complete those settings above before using it for a
reconciled AD deployment.

LDAP is covered by in-memory integration tests and an opt-in real OpenLDAP
suite in `backend/tests/test_ldap_real_server.py`. Real Active Directory
validation remains outstanding.

### Install and inspect research programs

Research programs are trusted Python packages installed into the backend image
at build time. They contribute domain tools and sandbox image references through
the `kdbc.research_programs` Python entry point. Installing a program does not
build its sandbox images; follow that program's instructions to make those
images available to the sandbox daemon.

#### Install from a local checkout (development Compose)

Set these values in the **repository-root `.env`**, creating it from
`.env.example` if it does not exist:

```dotenv
RESEARCH_PROGRAM_PATH=../kdbc-uarch-lab
RESEARCH_PROGRAMS=/programs
```

The path identifies the program checkout on the host; `/programs` is its named
build context inside Docker. Use an absolute host path if the checkout is
elsewhere. Packages are installed with `--no-deps` against core's dependencies.
The source mount is temporarily writable so package backends can generate
metadata and build files; these writes are discarded after the build step and
do not change the host checkout.

Build the shared image, then recreate the API, general worker, agent worker,
and scheduler from it:

```bash
docker compose build backend
docker compose up -d --no-build backend celery celery_agents celery_beat
```

A restart alone cannot install or remove packages. These services share
`knowledge_db_backend:latest` in the development stack. Rebuild any derived
images separately if they also need the updated package set. In another
deployment, build and distribute the resulting image to every API and agent
worker instance.

For a release, `RESEARCH_PROGRAMS` can instead contain space-separated pip
targets, such as a Git URL pinned to a commit or tag. Leave
`RESEARCH_PROGRAM_PATH` empty when no local checkout is needed. Leave
`RESEARCH_PROGRAMS` empty and rebuild to create a core-only image.
Private Git installs can use the Dockerfile's optional BuildKit secret
`github_token`; pass it through your build configuration rather than placing a
token in a package URL or build argument. The default Compose file does not
wire that secret automatically.

#### Disable without uninstalling

Set the program's name in **`backend/.env`**, for example:

```dotenv
RESEARCH_PROGRAMS_DISABLED=uarch_lab
```

Use commas for multiple names. An empty value loads every installed program.
Restart the API and workers after changing this runtime setting. Disabling a
program removes its contributed tools and allowed sandbox images from that
process; it does not uninstall the package or delete independently configured
image allowlist entries.

#### Public compiler APIs for program authors

Use `app.services.agent_compiler_sandbox.clean_flags(flags)` to validate and
strip compiler flags (`None` means unsupported characters; an empty string is
valid), and `mca_summary(payload)` to read the first region's `SummaryView` from
an llvm-mca JSON report (an empty dictionary means no valid summary).

Execute sandbox commands through
`app.services.agent_sandbox_runtime.run_in_sandbox(script, workdir, image=..., timeout_seconds=...)`.
It returns `(returncode, stdout, stderr)` and manages timeout/cancellation
cleanup. Callers must check execution enablement and image allowlisting before
invoking this low-level runner. The compiler module retains `_clean_flags`,
`_mca_summary`, and `_run` for compatibility with older installed programs.
Upgrade core before installing a program version that uses the public helpers.

#### Verify the API and workers

Open **Admin → Research Programs** (or `/admin?tab=research-programs`) to see
loaded programs, their versions, tools, and sandbox images. Failed programs
include the reason they could not load; disabled programs are listed separately.
Invalid catalog declarations and duplicate tool names refuse the whole program,
so its tools cannot displace built-in tools or another successfully loaded program.
Optional evidence-tool lists, image origins, methodology references, and
alternative-parameter rules are also checked for valid shapes before loading.
The page reads `GET /api/v1/admin/research-programs` and requires an admin account.

Status describes the API process. Install the same program packages in worker
images so agents can execute their tools. `RESEARCH_PROGRAMS_DISABLED` is a
comma-separated list of programs to leave out. Programs load once per process:
restart the affected services after installation or enablement changes.
**Refresh** retrieves the current status without reloading programs.

Check the agent worker's own loaded programs after deployment as well:

```bash
docker compose exec -T celery_agents python -c 'import json; from app.agent_core.research_programs import status; print(json.dumps(status(), indent=2))'
```

Investigate any failed programs before starting research jobs that depend on
their tools. Missing imports, malformed declarations, and name collisions are
reported as failures. If a program is absent from all lists, verify that its
package and entry point were installed in the image actually running. A failed
package installation fails the image build; import-time failures are reported
by the running API or worker.

Provider factories run separately for each executor. If initialization fails,
or providers do not handle exactly the program's declared tools, that program's
tools return `research_program_unavailable` for the affected run. Core tools and
other programs remain usable; partial provider registration is discarded.
Check worker logs for the program name and reason. A later executor retries
initialization; these run-specific failures do not change the API's load report.

Methodology modules are loaded only when a plan uses their tools. If a program's
module is missing, raises an exception, or returns malformed notes, the run's
methodology notes and worker logs explicitly report that the methodology is
incomplete. Valid notes from other modules remain available. These runtime
warnings do not change the API's load report; inspect the run and worker logs.

#### Run the Docker smoke test

With a local backend image already built, run from `backend/` using your test
Python environment:

```bash
TEST_RESEARCH_PROGRAM_DOCKER_IMAGE=knowledge_db_backend:latest python -m pytest tests/test_research_program_docker.py -q --no-cov
```

This opt-in test builds a disposable image using the real program-install
instruction and current core source, installs a small test program, and checks
discovery, tool execution, evidence/image registration, and disablement. It also
checks that installation failure fails the build and that build-time metadata
does not alter the source checkout. It requires Docker/BuildKit access and may
download a Python build dependency. The test removes its temporary image and
does not start or restart the application stack. It skips when the image
environment variable is unset; this smoke test does not rebuild core's full
dependency image or run domain simulators.

### Docker Commands

```bash
# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f [service_name]

# Restart a service
docker compose restart [service_name]

# Rebuild after code changes
docker compose build [service_name]
docker compose up -d [service_name]

# Stop and remove volumes (clean slate)
docker compose down -v
```

### Production Docker Setup

For production deployment:

```bash
# Use production compose file
docker compose -f docker-compose.prod.yml up -d

# Or build first
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
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
ollama pull llama3.2:1b
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
DATABASE_URL=postgresql://user:password@localhost:25432/knowledge_db
REDIS_URL=redis://localhost:26379/0
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

# Build the schema. Alembic is the only supported way to do this: creating
# tables from model metadata skips migrations and leaves the database
# unstamped, which is how the schema silently drifted in the past.
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
echo "REACT_APP_API_URL=http://localhost:28000" > .env
echo "REACT_APP_WS_URL=ws://localhost:28000" >> .env

# Start development server
npm start
```

The frontend will open at http://localhost:23000

---

## Verification

### Check Backend Health

```bash
# Using curl
curl http://localhost:28000/health

# Or visit in browser
open http://localhost:28000/docs
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

1. Open http://localhost:23000
2. Click "Register" or use the API:
   ```bash
   curl -X POST http://localhost:28000/api/v1/auth/register \
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
docker compose logs [service_name]

# Restart specific service
docker compose restart [service_name]

# Rebuild containers
docker compose build --no-cache
docker compose up -d
```

#### 6. Vector Store Issues
```bash
# Recommended: reset via the admin API (works for Qdrant or Chroma)
# POST /api/v1/admin/vector-store/reset  (admin auth required)

# If you're using Chroma (VECTOR_STORE_PROVIDER=chroma), data lives on disk:
ls -la data/chroma_db

# If you're using Qdrant (VECTOR_STORE_PROVIDER=qdrant), data lives in the qdrant volume:
docker volume ls | rg qdrant
```

#### 7. Module Import Errors
```bash
# Reinstall dependencies
cd backend
source venv/bin/activate
pip install -r requirements.txt --force-reinstall

# Or for Docker
docker compose build --no-cache backend
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
   docker compose -f docker-compose.prod.yml build
   docker compose -f docker-compose.prod.yml up -d
   ```

3. **Set up reverse proxy** (nginx example):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://localhost:23000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /api {
           proxy_pass http://localhost:28000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }

       location /ws {
           proxy_pass http://localhost:28000;
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
- Frontend: http://localhost:23000
- Backend API: http://localhost:28000
- API Docs: http://localhost:28000/docs
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
make doctor         # Validate env + health
make test           # Run all tests
make fmt            # Format backend code
make lint           # Lint backend code
make pull-model     # Pull Ollama model
```

#### Using Docker Compose
```bash
docker compose up -d              # Start all services
docker compose logs -f           # View logs
docker compose restart backend    # Restart backend
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
npm run test:ci                   # Run tests (non-interactive)
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
