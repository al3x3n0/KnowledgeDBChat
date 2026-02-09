#!/usr/bin/env python3
"""
Team AI Cloud CLI - Command-line interface for the Knowledge Platform API.

Usage:
    taic configure             # Set up API connection
    taic search "query"        # Search documents
    taic chat "question"       # Ask a question using RAG
    taic documents list        # List documents
    taic documents upload file # Upload a document
    taic sources list          # List data sources
    taic agent chat            # Interactive agent chat
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

try:
    import requests
    from requests.exceptions import ConnectionError, Timeout
except ImportError:
    print("Error: 'requests' package is required. Install with: pip install requests")
    sys.exit(1)

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.markdown import Markdown
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Prompt, Confirm
    from rich.syntax import Syntax
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better output formatting: pip install rich")

# Configuration
CONFIG_DIR = Path.home() / ".taic"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Console for rich output
console = Console() if RICH_AVAILABLE else None


def print_output(text: str, style: str = None):
    """Print with optional rich formatting."""
    if console and style:
        console.print(text, style=style)
    else:
        print(text)


def print_error(text: str):
    """Print error message."""
    if console:
        console.print(f"[red]Error:[/red] {text}")
    else:
        print(f"Error: {text}", file=sys.stderr)


def print_success(text: str):
    """Print success message."""
    if console:
        console.print(f"[green]✓[/green] {text}")
    else:
        print(f"✓ {text}")


def print_json(data: Any):
    """Print JSON data nicely formatted."""
    if console:
        console.print_json(json.dumps(data, indent=2, default=str))
    else:
        print(json.dumps(data, indent=2, default=str))


class TAICClient:
    """Team AI Cloud API Client."""

    def __init__(self, base_url: str = None, api_key: str = None):
        self.config = self._load_config()
        self.base_url = (base_url or self.config.get("base_url", "")).rstrip("/")
        self.api_key = api_key or self.config.get("api_key")
        self.timeout = 30

    def _load_config(self) -> dict:
        """Load configuration from file."""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_config(self, config: dict):
        """Save configuration to file."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        os.chmod(CONFIG_FILE, 0o600)  # Secure the config file

    def configure(self, base_url: str, api_key: str):
        """Configure the client."""
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._save_config({
            "base_url": self.base_url,
            "api_key": self.api_key,
        })
        print_success(f"Configuration saved to {CONFIG_FILE}")

    def _headers(self) -> dict:
        """Get request headers."""
        return {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        """Make an API request."""
        if not self.base_url:
            print_error("API URL not configured. Run: taic configure")
            sys.exit(1)
        if not self.api_key:
            print_error("API key not configured. Run: taic configure")
            sys.exit(1)

        url = f"{self.base_url}/api/v1{endpoint}"
        headers = self._headers()

        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))

        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                timeout=self.timeout,
                **kwargs
            )
            response.raise_for_status()
            return response.json() if response.content else {}
        except ConnectionError:
            print_error(f"Cannot connect to {self.base_url}")
            sys.exit(1)
        except Timeout:
            print_error("Request timed out")
            sys.exit(1)
        except requests.HTTPError as e:
            try:
                error_detail = e.response.json().get("detail", str(e))
            except Exception:
                error_detail = str(e)
            print_error(f"API error: {error_detail}")
            sys.exit(1)

    def get(self, endpoint: str, **kwargs) -> dict:
        return self._request("GET", endpoint, **kwargs)

    def post(self, endpoint: str, **kwargs) -> dict:
        return self._request("POST", endpoint, **kwargs)

    def delete(self, endpoint: str, **kwargs) -> dict:
        return self._request("DELETE", endpoint, **kwargs)

    # ========== API Methods ==========

    def search(self, query: str, limit: int = 10, mode: str = "smart") -> dict:
        """Search documents."""
        return self.get("/search/", params={
            "q": query,
            "page_size": limit,
            "mode": mode,
        })

    def chat(self, message: str, session_id: str = None) -> dict:
        """Send a chat message."""
        data = {"message": message}
        if session_id:
            data["session_id"] = session_id
        return self.post("/chat/message", json=data)

    def list_documents(self, limit: int = 20, source: str = None) -> dict:
        """List documents."""
        params = {"limit": limit}
        if source:
            params["source"] = source
        return self.get("/documents/", params=params)

    def get_document(self, doc_id: str) -> dict:
        """Get document details."""
        return self.get(f"/documents/{doc_id}")

    def upload_document(self, file_path: str, title: str = None, tags: List[str] = None) -> dict:
        """Upload a document."""
        path = Path(file_path)
        if not path.exists():
            print_error(f"File not found: {file_path}")
            sys.exit(1)

        with open(path, "rb") as f:
            files = {"file": (path.name, f)}
            data = {}
            if title:
                data["title"] = title
            if tags:
                data["tags"] = ",".join(tags)

            headers = {"X-API-Key": self.api_key}
            url = f"{self.base_url}/api/v1/documents/upload"

            response = requests.post(url, files=files, data=data, headers=headers, timeout=60)
            response.raise_for_status()
            return response.json()

    def list_sources(self) -> dict:
        """List data sources."""
        return self.get("/admin/sources")

    def get_stats(self) -> dict:
        """Get system statistics."""
        return self.get("/admin/stats")

    def agent_chat(self, message: str, conversation_id: str = None) -> dict:
        """Chat with the AI agent."""
        data = {"message": message}
        if conversation_id:
            return self.post(f"/agent/conversations/{conversation_id}/message", json=data)
        else:
            # Create new conversation and send message
            conv = self.post("/agent/conversations", json={"title": f"CLI Chat {datetime.now().strftime('%H:%M')}"})
            conv_id = conv["id"]
            return self.post(f"/agent/conversations/{conv_id}/message", json=data)

    def list_workflows(self) -> dict:
        """List workflows."""
        return self.get("/workflows/")

    def run_workflow(self, workflow_id: str, inputs: dict = None) -> dict:
        """Run a workflow."""
        return self.post(f"/workflows/{workflow_id}/execute", json={"inputs": inputs or {}})


# ========== CLI Commands ==========

def cmd_configure(args):
    """Configure API connection."""
    client = TAICClient()

    if RICH_AVAILABLE:
        console.print(Panel.fit(
            "[bold]Team AI Cloud CLI Configuration[/bold]\n\n"
            "You'll need:\n"
            "1. Your API base URL (e.g., http://localhost:8000)\n"
            "2. An API key (create one at /api-keys in the web UI)",
            title="Setup"
        ))
        base_url = Prompt.ask("API Base URL", default=client.base_url or "http://localhost:8000")
        api_key = Prompt.ask("API Key", password=True)
    else:
        print("Team AI Cloud CLI Configuration")
        print("-" * 40)
        base_url = input(f"API Base URL [{client.base_url or 'http://localhost:8000'}]: ").strip()
        base_url = base_url or client.base_url or "http://localhost:8000"
        api_key = input("API Key: ").strip()

    if not api_key:
        print_error("API key is required")
        return

    client.configure(base_url, api_key)

    # Test the connection
    print("\nTesting connection...")
    try:
        client.get("/system/health")
        print_success("Connection successful!")
    except Exception as e:
        print_error(f"Connection test failed: {e}")


def cmd_search(args):
    """Search documents."""
    client = TAICClient()
    result = client.search(args.query, limit=args.limit, mode=args.mode)

    if console:
        table = Table(title=f"Search Results for '{args.query}'")
        table.add_column("Title", style="cyan", max_width=50)
        table.add_column("Source", style="green")
        table.add_column("Score", justify="right")
        table.add_column("Snippet", max_width=60)

        for doc in result.get("results", []):
            table.add_row(
                doc.get("title", "Untitled")[:50],
                doc.get("source", "Unknown"),
                f"{doc.get('relevance_score', 0):.2f}",
                (doc.get("snippet", "")[:60] + "...") if doc.get("snippet") else ""
            )

        console.print(table)
        console.print(f"\n[dim]Found {result.get('total', 0)} results in {result.get('took_ms', 0)}ms[/dim]")
    else:
        print(f"\nSearch Results for '{args.query}'")
        print("-" * 60)
        for doc in result.get("results", []):
            print(f"\n• {doc.get('title', 'Untitled')}")
            print(f"  Source: {doc.get('source', 'Unknown')} | Score: {doc.get('relevance_score', 0):.2f}")
            if doc.get("snippet"):
                print(f"  {doc['snippet'][:100]}...")
        print(f"\nFound {result.get('total', 0)} results")


def cmd_chat(args):
    """Ask a question using RAG."""
    client = TAICClient()

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Thinking...", total=None)
            result = client.chat(args.question)
    else:
        print("Thinking...")
        result = client.chat(args.question)

    response = result.get("response", result.get("content", "No response"))

    if console:
        console.print(Panel(Markdown(response), title="Answer", border_style="green"))
        if result.get("sources"):
            console.print("\n[dim]Sources:[/dim]")
            for src in result["sources"][:3]:
                console.print(f"  • {src.get('title', 'Unknown')}")
    else:
        print("\n" + "=" * 60)
        print("Answer:")
        print("=" * 60)
        print(response)
        if result.get("sources"):
            print("\nSources:")
            for src in result["sources"][:3]:
                print(f"  • {src.get('title', 'Unknown')}")


def cmd_documents_list(args):
    """List documents."""
    client = TAICClient()
    result = client.list_documents(limit=args.limit, source=args.source)

    documents = result if isinstance(result, list) else result.get("documents", [])

    if console:
        table = Table(title="Documents")
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Title", style="cyan", max_width=40)
        table.add_column("Source", style="green")
        table.add_column("Type")
        table.add_column("Created")

        for doc in documents:
            created = doc.get("created_at", "")[:10] if doc.get("created_at") else ""
            table.add_row(
                doc.get("id", "")[:8],
                doc.get("title", "Untitled")[:40],
                doc.get("source", "Unknown"),
                doc.get("file_type", ""),
                created
            )

        console.print(table)
    else:
        print("Documents:")
        print("-" * 60)
        for doc in documents:
            print(f"\n{doc.get('id', '')[:8]}  {doc.get('title', 'Untitled')}")
            print(f"   Source: {doc.get('source', 'Unknown')} | Type: {doc.get('file_type', 'N/A')}")


def cmd_documents_upload(args):
    """Upload a document."""
    client = TAICClient()

    if console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task("Uploading...", total=None)
            result = client.upload_document(args.file, title=args.title, tags=args.tags)
    else:
        print(f"Uploading {args.file}...")
        result = client.upload_document(args.file, title=args.title, tags=args.tags)

    print_success(f"Document uploaded! ID: {result.get('document_id', 'Unknown')}")


def cmd_documents_get(args):
    """Get document details."""
    client = TAICClient()
    doc = client.get_document(args.id)

    if console:
        console.print(Panel.fit(
            f"[bold]{doc.get('title', 'Untitled')}[/bold]\n\n"
            f"ID: {doc.get('id')}\n"
            f"Source: {doc.get('source', 'Unknown')}\n"
            f"Type: {doc.get('file_type', 'N/A')}\n"
            f"Author: {doc.get('author', 'Unknown')}\n"
            f"Created: {doc.get('created_at', '')[:19]}\n"
            f"Tags: {', '.join(doc.get('tags', [])) or 'None'}",
            title="Document Details"
        ))
        if doc.get("summary"):
            console.print("\n[bold]Summary:[/bold]")
            console.print(Markdown(doc["summary"]))
    else:
        print(f"\nDocument: {doc.get('title', 'Untitled')}")
        print("-" * 40)
        print(f"ID: {doc.get('id')}")
        print(f"Source: {doc.get('source', 'Unknown')}")
        print(f"Type: {doc.get('file_type', 'N/A')}")
        if doc.get("summary"):
            print(f"\nSummary:\n{doc['summary']}")


def cmd_sources_list(args):
    """List data sources."""
    client = TAICClient()
    result = client.list_sources()

    sources = result if isinstance(result, list) else result.get("sources", [])

    if console:
        table = Table(title="Data Sources")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Status")
        table.add_column("Documents", justify="right")

        for src in sources:
            status = "[green]Active[/green]" if src.get("is_active") else "[red]Inactive[/red]"
            table.add_row(
                src.get("name", "Unknown"),
                src.get("source_type", "Unknown"),
                status,
                str(src.get("document_count", 0))
            )

        console.print(table)
    else:
        print("Data Sources:")
        print("-" * 40)
        for src in sources:
            status = "Active" if src.get("is_active") else "Inactive"
            print(f"\n• {src.get('name', 'Unknown')} ({src.get('source_type', 'Unknown')})")
            print(f"  Status: {status} | Documents: {src.get('document_count', 0)}")


def cmd_stats(args):
    """Show system statistics."""
    client = TAICClient()
    stats = client.get_stats()

    if console:
        console.print(Panel.fit(
            f"[bold]Documents:[/bold] {stats.get('documents', {}).get('total', 0)}\n"
            f"[bold]Sources:[/bold] {stats.get('sources', {}).get('active', 0)} active\n"
            f"[bold]Users:[/bold] {stats.get('users', {}).get('total', 0)}\n"
            f"[bold]Chat Sessions:[/bold] {stats.get('chat', {}).get('total_sessions', 0)}",
            title="System Statistics"
        ))
    else:
        print("\nSystem Statistics")
        print("-" * 40)
        print(f"Documents: {stats.get('documents', {}).get('total', 0)}")
        print(f"Sources: {stats.get('sources', {}).get('active', 0)} active")
        print(f"Users: {stats.get('users', {}).get('total', 0)}")


def cmd_agent(args):
    """Interactive agent chat."""
    client = TAICClient()
    conversation_id = None

    if console:
        console.print(Panel.fit(
            "[bold]Team AI Cloud Agent[/bold]\n\n"
            "Chat with the AI agent. It can search documents, run workflows,\n"
            "generate content, and more.\n\n"
            "Type 'exit' or 'quit' to end the session.",
            title="Agent Chat"
        ))
    else:
        print("\n" + "=" * 50)
        print("Team AI Cloud Agent")
        print("Type 'exit' or 'quit' to end the session.")
        print("=" * 50)

    while True:
        try:
            if console:
                user_input = Prompt.ask("\n[bold cyan]You[/bold cyan]")
            else:
                user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print_output("\nGoodbye!", "green")
                break

            # Send message to agent
            if console:
                with Progress(
                    SpinnerColumn(),
                    TextColumn("[progress.description]{task.description}"),
                    transient=True,
                ) as progress:
                    progress.add_task("Agent is thinking...", total=None)
                    result = client.agent_chat(user_input, conversation_id)
            else:
                print("Agent is thinking...")
                result = client.agent_chat(user_input, conversation_id)

            # Update conversation ID
            if not conversation_id:
                conversation_id = result.get("conversation_id")

            # Display response
            response = result.get("response", result.get("content", "No response"))

            if console:
                console.print(f"\n[bold green]Agent[/bold green]")
                console.print(Markdown(response))

                # Show tool calls if any
                if result.get("tool_calls"):
                    console.print("\n[dim]Tools used:[/dim]")
                    for tool in result["tool_calls"]:
                        console.print(f"  • {tool.get('tool_name', 'Unknown')}")
            else:
                print(f"\nAgent: {response}")

        except KeyboardInterrupt:
            print_output("\n\nGoodbye!", "green")
            break
        except Exception as e:
            print_error(f"Error: {e}")


def cmd_workflows_list(args):
    """List workflows."""
    client = TAICClient()
    result = client.list_workflows()

    workflows = result if isinstance(result, list) else result.get("workflows", [])

    if console:
        table = Table(title="Workflows")
        table.add_column("ID", style="dim", max_width=8)
        table.add_column("Name", style="cyan")
        table.add_column("Status")
        table.add_column("Trigger")

        for wf in workflows:
            status = "[green]Active[/green]" if wf.get("is_active") else "[red]Inactive[/red]"
            trigger = wf.get("trigger_config", {}).get("type", "manual")
            table.add_row(
                wf.get("id", "")[:8],
                wf.get("name", "Untitled"),
                status,
                trigger
            )

        console.print(table)
    else:
        print("Workflows:")
        print("-" * 40)
        for wf in workflows:
            status = "Active" if wf.get("is_active") else "Inactive"
            print(f"\n{wf.get('id', '')[:8]}  {wf.get('name', 'Untitled')} [{status}]")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Team AI Cloud CLI - Command-line interface for the Knowledge Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  taic configure                    # Set up API connection
  taic search "compiler optimization"    # Search documents
  taic chat "What is loop unrolling?"    # Ask a question
  taic documents list               # List documents
  taic agent                        # Start interactive agent chat

For more information, visit the documentation or use --help on subcommands.
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Configure
    subparsers.add_parser("configure", help="Configure API connection")

    # Search
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-l", "--limit", type=int, default=10, help="Max results (default: 10)")
    search_parser.add_argument("-m", "--mode", default="smart", choices=["smart", "keyword", "exact"],
                               help="Search mode (default: smart)")

    # Chat
    chat_parser = subparsers.add_parser("chat", help="Ask a question using RAG")
    chat_parser.add_argument("question", help="Question to ask")

    # Documents
    docs_parser = subparsers.add_parser("documents", help="Document management")
    docs_subparsers = docs_parser.add_subparsers(dest="docs_command")

    docs_list = docs_subparsers.add_parser("list", help="List documents")
    docs_list.add_argument("-l", "--limit", type=int, default=20, help="Max documents")
    docs_list.add_argument("-s", "--source", help="Filter by source")

    docs_upload = docs_subparsers.add_parser("upload", help="Upload a document")
    docs_upload.add_argument("file", help="File to upload")
    docs_upload.add_argument("-t", "--title", help="Document title")
    docs_upload.add_argument("--tags", nargs="+", help="Tags for the document")

    docs_get = docs_subparsers.add_parser("get", help="Get document details")
    docs_get.add_argument("id", help="Document ID")

    # Sources
    sources_parser = subparsers.add_parser("sources", help="Data source management")
    sources_subparsers = sources_parser.add_subparsers(dest="sources_command")
    sources_subparsers.add_parser("list", help="List data sources")

    # Stats
    subparsers.add_parser("stats", help="Show system statistics")

    # Agent
    subparsers.add_parser("agent", help="Interactive agent chat")

    # Workflows
    wf_parser = subparsers.add_parser("workflows", help="Workflow management")
    wf_subparsers = wf_parser.add_subparsers(dest="wf_command")
    wf_subparsers.add_parser("list", help="List workflows")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route commands
    if args.command == "configure":
        cmd_configure(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "chat":
        cmd_chat(args)
    elif args.command == "documents":
        if args.docs_command == "list":
            cmd_documents_list(args)
        elif args.docs_command == "upload":
            cmd_documents_upload(args)
        elif args.docs_command == "get":
            cmd_documents_get(args)
        else:
            docs_parser.print_help()
    elif args.command == "sources":
        if args.sources_command == "list":
            cmd_sources_list(args)
        else:
            sources_parser.print_help()
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "agent":
        cmd_agent(args)
    elif args.command == "workflows":
        if args.wf_command == "list":
            cmd_workflows_list(args)
        else:
            wf_parser.print_help()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
