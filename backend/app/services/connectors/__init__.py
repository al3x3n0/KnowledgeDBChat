"""
Data source connectors for various platforms.
"""

from .arxiv_connector import ArxivConnector
from .base_connector import BaseConnector
from .confluence_connector import ConfluenceConnector
from .github_connector import GitHubConnector
from .gitlab_connector import GitLabConnector
from .web_connector import WebConnector

__all__ = [
    "BaseConnector",
    "GitLabConnector",
    "GitHubConnector",
    "ConfluenceConnector",
    "WebConnector",
    "ArxivConnector",
]
