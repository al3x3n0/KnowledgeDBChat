#!/usr/bin/env python3
"""
Script to set up initial data sources for the Knowledge Database.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import AsyncSessionLocal
from app.models.document import DocumentSource
from app.services.document_service import DocumentService


async def create_gitlab_source(
    name: str,
    gitlab_url: str,
    token: str,
    projects: list = None,
    include_wikis: bool = True,
    include_issues: bool = True
) -> DocumentSource:
    """Create a GitLab data source."""
    config = {
        "gitlab_url": gitlab_url,
        "token": token,
        "projects": projects or [],
        "include_wikis": include_wikis,
        "include_issues": include_issues,
        "include_merge_requests": False,
        "file_extensions": [".md", ".txt", ".rst", ".py", ".js", ".ts", ".java", ".cpp", ".h"]
    }
    
    async with AsyncSessionLocal() as db:
        document_service = DocumentService()
        source = await document_service.create_document_source(
            name=name,
            source_type="gitlab",
            config=config,
            db=db
        )
        return source


async def create_confluence_source(
    name: str,
    confluence_url: str,
    username: str,
    api_token: str,
    spaces: list = None,
    include_attachments: bool = True
) -> DocumentSource:
    """Create a Confluence data source."""
    config = {
        "confluence_url": confluence_url,
        "username": username,
        "api_token": api_token,
        "spaces": spaces or [],
        "include_attachments": include_attachments,
        "include_comments": False,
        "page_limit": 1000
    }
    
    async with AsyncSessionLocal() as db:
        document_service = DocumentService()
        source = await document_service.create_document_source(
            name=name,
            source_type="confluence",
            config=config,
            db=db
        )
        return source


async def create_web_source(
    name: str,
    base_urls: list,
    allowed_domains: list = None,
    max_depth: int = 3,
    max_pages: int = 100
) -> DocumentSource:
    """Create a web scraping data source."""
    config = {
        "base_urls": base_urls,
        "allowed_domains": allowed_domains or [],
        "max_depth": max_depth,
        "max_pages": max_pages,
        "excluded_patterns": ["/api/", "/admin/", "/login/", "/.git/"],
        "included_patterns": [],
        "respect_robots": True,
        "crawl_delay": 1.0,
        "headers": {
            "User-Agent": "Knowledge-DB-Bot/1.0"
        }
    }
    
    async with AsyncSessionLocal() as db:
        document_service = DocumentService()
        source = await document_service.create_document_source(
            name=name,
            source_type="web",
            config=config,
            db=db
        )
        return source


async def load_sources_from_config(config_file: str):
    """Load and create data sources from a configuration file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        sources_created = []
        
        # Create GitLab sources
        for gitlab_config in config.get("gitlab_sources", []):
            print(f"Creating GitLab source: {gitlab_config['name']}")
            source = await create_gitlab_source(**gitlab_config)
            sources_created.append(source)
            print(f"✅ Created GitLab source: {source.name} (ID: {source.id})")
        
        # Create Confluence sources
        for confluence_config in config.get("confluence_sources", []):
            print(f"Creating Confluence source: {confluence_config['name']}")
            source = await create_confluence_source(**confluence_config)
            sources_created.append(source)
            print(f"✅ Created Confluence source: {source.name} (ID: {source.id})")
        
        # Create Web sources
        for web_config in config.get("web_sources", []):
            print(f"Creating Web source: {web_config['name']}")
            source = await create_web_source(**web_config)
            sources_created.append(source)
            print(f"✅ Created Web source: {source.name} (ID: {source.id})")
        
        print(f"\n🎉 Successfully created {len(sources_created)} data sources!")
        return sources_created
        
    except Exception as e:
        print(f"❌ Error loading sources from config: {e}")
        return []


async def create_example_config():
    """Create an example configuration file."""
    example_config = {
        "gitlab_sources": [
            {
                "name": "Company GitLab",
                "gitlab_url": "https://gitlab.company.com",
                "token": "your-gitlab-token-here",
                "projects": [
                    {"id": "project-id-or-name", "include_files": True, "include_wikis": True, "include_issues": True}
                ],
                "include_wikis": True,
                "include_issues": True
            }
        ],
        "confluence_sources": [
            {
                "name": "Company Confluence",
                "confluence_url": "https://company.atlassian.net",
                "username": "your-username",
                "api_token": "your-api-token",
                "spaces": [
                    {"key": "DOCS", "name": "Documentation"},
                    {"key": "KB", "name": "Knowledge Base"}
                ],
                "include_attachments": True
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
    
    config_file = "sources_config.json"
    with open(config_file, 'w') as f:
        json.dump(example_config, f, indent=2)
    
    print(f"📄 Created example configuration file: {config_file}")
    print("Please edit this file with your actual configuration and run:")
    print(f"python setup_sources.py {config_file}")


async def list_existing_sources():
    """List all existing data sources."""
    async with AsyncSessionLocal() as db:
        document_service = DocumentService()
        sources = await document_service.get_document_sources(db)
        
        if not sources:
            print("No data sources found.")
            return
        
        print("Existing data sources:")
        print("-" * 60)
        for source in sources:
            status = "✅ Active" if source.is_active else "❌ Inactive"
            last_sync = source.last_sync.strftime("%Y-%m-%d %H:%M") if source.last_sync else "Never"
            print(f"Name: {source.name}")
            print(f"Type: {source.source_type}")
            print(f"Status: {status}")
            print(f"Last Sync: {last_sync}")
            print(f"ID: {source.id}")
            print("-" * 60)


async def main():
    """Main function."""
    if len(sys.argv) == 1:
        print("Knowledge Database - Data Source Setup")
        print("=" * 40)
        print()
        print("Usage:")
        print("  python setup_sources.py list                    - List existing sources")
        print("  python setup_sources.py example                 - Create example config file")
        print("  python setup_sources.py <config_file.json>      - Load sources from config file")
        print()
        return
    
    command = sys.argv[1]
    
    if command == "list":
        await list_existing_sources()
    elif command == "example":
        await create_example_config()
    elif command.endswith(".json"):
        await load_sources_from_config(command)
    else:
        print(f"Unknown command: {command}")
        print("Use 'list', 'example', or provide a JSON configuration file.")


if __name__ == "__main__":
    asyncio.run(main())








