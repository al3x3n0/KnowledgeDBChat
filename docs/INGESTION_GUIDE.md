# Data Ingestion Pipeline Guide

This guide explains how to set up and use the automated data ingestion pipelines for your Knowledge Database.

## Overview

The Knowledge Database supports automated ingestion from multiple data sources:

- **GitLab**: Repositories, wikis, issues, and merge requests
- **Confluence**: Pages, attachments, and comments
- **Web Scraping**: Internal websites and documentation
- **File Upload**: Direct file uploads via API

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │   Connectors    │    │  Background     │
│                 │    │                 │    │     Tasks       │
│ • GitLab        │───►│ • GitLabConn    │───►│ • Celery        │
│ • Confluence    │    │ • ConfluenceConn│    │ • Redis Queue   │
│ • Web Sites     │    │ • WebConnector  │    │ • Scheduling    │
│ • Files         │    │ • FileProcessor │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │  Text Processing│    │  Vector Store   │
│                 │◄───│                 │───►│                 │
│ • Documents     │    │ • Extraction    │    │ • ChromaDB      │
│ • Metadata      │    │ • Chunking      │    │ • Embeddings    │
│ • History       │    │ • Cleaning      │    │ • Search Index  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Setting Up Data Sources

### Method 1: Using the Setup Script

1. **Create Configuration File**
   ```bash
   cd backend
   python scripts/setup_sources.py example
   ```

2. **Edit the Configuration**
   ```json
   {
     "gitlab_sources": [
       {
         "name": "Company GitLab",
         "gitlab_url": "https://gitlab.company.com",
         "token": "glpat-xxxxxxxxxxxxxxxxxxxx",
         "projects": [
           {
             "id": "group/project-name",
             "include_files": true,
             "include_wikis": true,
             "include_issues": true
           }
         ],
         "include_wikis": true,
         "include_issues": true
       }
     ],
     "confluence_sources": [
       {
         "name": "Company Confluence",
         "confluence_url": "https://company.atlassian.net",
         "username": "your-email@company.com",
         "api_token": "ATATT3xFfGF0...",
         "spaces": [
           {"key": "DOCS", "name": "Documentation"},
           {"key": "KB", "name": "Knowledge Base"}
         ],
         "include_attachments": true
       }
     ],
     "web_sources": [
       {
         "name": "Internal Documentation",
         "base_urls": [
           "https://docs.company.com",
           "https://wiki.company.com"
         ],
         "allowed_domains": ["docs.company.com", "wiki.company.com"],
         "max_depth": 3,
         "max_pages": 200
       }
     ]
   }
   ```

3. **Load the Configuration**
   ```bash
   python scripts/setup_sources.py sources_config.json
   ```

### Method 2: Using the API

1. **Create GitLab Source**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/documents/sources/" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Company GitLab",
       "source_type": "gitlab",
       "config": {
         "gitlab_url": "https://gitlab.company.com",
         "token": "glpat-xxxxxxxxxxxxxxxxxxxx",
         "projects": [
           {"id": "group/project-name", "include_files": true}
         ]
       }
     }'
   ```

2. **Create Confluence Source**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/documents/sources/" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Company Confluence",
       "source_type": "confluence",
       "config": {
         "confluence_url": "https://company.atlassian.net",
         "username": "your-email@company.com",
         "api_token": "ATATT3xFfGF0...",
         "spaces": [{"key": "DOCS"}]
       }
     }'
   ```

3. **Create Web Source**
   ```bash
   curl -X POST "http://localhost:8000/api/v1/documents/sources/" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "name": "Internal Documentation",
       "source_type": "web",
       "config": {
         "base_urls": ["https://docs.company.com"],
         "max_depth": 3,
         "max_pages": 100
       }
     }'
   ```

## Configuration Options

### GitLab Connector

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `gitlab_url` | string | required | GitLab instance URL |
| `token` | string | required | Personal access token |
| `projects` | array | [] | List of projects to sync |
| `include_wikis` | boolean | true | Include wiki pages |
| `include_issues` | boolean | true | Include issues |
| `include_merge_requests` | boolean | false | Include merge requests |
| `file_extensions` | array | ['.md', '.txt', ...] | File types to include |

**Project Configuration:**
```json
{
  "id": "group/project-name",  // Project ID or path
  "include_files": true,       // Include repository files
  "include_wikis": true,       // Include project wiki
  "include_issues": true,      // Include issues
  "include_merge_requests": false
}
```

### Confluence Connector

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `confluence_url` | string | required | Confluence instance URL |
| `username` | string | required | Username or email |
| `api_token` | string | required | API token |
| `spaces` | array | [] | List of spaces to sync |
| `include_attachments` | boolean | true | Include file attachments |
| `include_comments` | boolean | false | Include page comments |
| `page_limit` | number | 1000 | Maximum pages per space |

**Space Configuration:**
```json
{
  "key": "DOCS",              // Space key
  "name": "Documentation"     // Space name (optional)
}
```

### Web Connector

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `base_urls` | array | required | Starting URLs for crawling |
| `allowed_domains` | array | [] | Domains to restrict crawling |
| `max_depth` | number | 3 | Maximum crawling depth |
| `max_pages` | number | 100 | Maximum pages to crawl |
| `excluded_patterns` | array | [] | URL patterns to exclude |
| `included_patterns` | array | [] | URL patterns to include |
| `respect_robots` | boolean | true | Respect robots.txt |
| `crawl_delay` | number | 1.0 | Delay between requests |

## Automated Scheduling

The system includes automated synchronization schedules:

| Source Type | Schedule | Description |
|-------------|----------|-------------|
| GitLab | Every hour | Sync all active GitLab sources |
| Confluence | Every 2 hours | Sync all active Confluence sources |
| Web | Daily at 2 AM | Sync all active web sources |
| Cleanup | Weekly | Clean up old logs and data |
| Health Check | Every 15 minutes | System health monitoring |

### Custom Scheduling

To modify schedules, edit `backend/app/core/celery.py`:

```python
celery_app.conf.beat_schedule = {
    "sync-gitlab-sources": {
        "task": "app.tasks.sync_tasks.sync_all_gitlab_sources",
        "schedule": crontab(minute=0, hour="*/2"),  # Every 2 hours
    },
}
```

## Manual Synchronization

### Sync All Sources
```bash
curl -X POST "http://localhost:8000/api/v1/admin/sync/all" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Sync Specific Source
```bash
curl -X POST "http://localhost:8000/api/v1/admin/sync/source/SOURCE_ID" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Using the CLI
```bash
cd backend
python -c "
from app.tasks.sync_tasks import sync_all_sources
task = sync_all_sources.delay()
print(f'Task ID: {task.id}')
"
```

## Monitoring and Troubleshooting

### Check System Health
```bash
curl "http://localhost:8000/api/v1/admin/health" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### View System Statistics
```bash
curl "http://localhost:8000/api/v1/admin/stats" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Check Task Status
```bash
curl "http://localhost:8000/api/v1/admin/tasks/status" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### View Logs
```bash
curl "http://localhost:8000/api/v1/admin/logs?lines=100" \
  -H "Authorization: Bearer YOUR_ADMIN_TOKEN"
```

### Common Issues

**1. GitLab Connection Failed**
- Check GitLab URL and token
- Verify token has necessary permissions
- Ensure GitLab is accessible from your network

**2. Confluence Authentication Error**
- Verify username and API token
- Check if API token has required permissions
- Ensure two-factor authentication is properly configured

**3. Web Scraping Blocked**
- Check robots.txt restrictions
- Verify allowed domains configuration
- Reduce crawl rate if being rate-limited

**4. Vector Store Issues**
- Check ChromaDB permissions and disk space
- Verify embedding model is downloaded
- Restart vector store service if needed

### Performance Tuning

**Celery Workers**
```bash
# Increase worker concurrency
celery -A app.core.celery worker --concurrency=4

# Use different queues for different task types
celery -A app.core.celery worker -Q ingestion,processing
```

**Database Connection Pool**
```python
# In app/core/database.py
engine = create_async_engine(
    async_database_url,
    pool_size=20,           # Increase pool size
    max_overflow=30,        # Allow overflow connections
    pool_pre_ping=True,
    pool_recycle=3600,      # Recycle connections hourly
)
```

**Vector Store Batch Size**
```python
# In app/services/vector_store.py
# Process documents in batches
batch_size = 10
for i in range(0, len(documents), batch_size):
    batch = documents[i:i + batch_size]
    await self.add_document_chunks(batch)
```

## Security Considerations

### Token Security
- Store API tokens in environment variables
- Use minimal required permissions
- Rotate tokens regularly
- Monitor token usage

### Network Security
- Restrict access to internal networks only
- Use VPN for external access
- Implement IP whitelisting
- Enable HTTPS for all communications

### Data Privacy
- Review data before ingestion
- Implement content filtering
- Respect privacy policies
- Audit access logs regularly

## Best Practices

1. **Start Small**: Begin with a limited set of documents and gradually expand
2. **Monitor Resources**: Keep an eye on disk space, memory, and CPU usage
3. **Regular Backups**: Backup both PostgreSQL and ChromaDB data
4. **Test Configurations**: Validate source configurations before production
5. **Update Dependencies**: Keep connectors and dependencies updated
6. **Document Changes**: Track configuration changes and their impacts

## API Reference

### List Sources
```
GET /api/v1/documents/sources/
```

### Create Source
```
POST /api/v1/documents/sources/
{
  "name": "Source Name",
  "source_type": "gitlab|confluence|web",
  "config": { ... }
}
```

### Update Source
```
PUT /api/v1/documents/sources/{source_id}
```

### Delete Source
```
DELETE /api/v1/documents/sources/{source_id}
```

### Trigger Sync
```
POST /api/v1/documents/sources/{source_id}/sync
```

For complete API documentation, visit: http://localhost:8000/docs






