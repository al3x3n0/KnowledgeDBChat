"""
Web connector for scraping content from internal websites and documentation.
"""

import httpx
import hashlib
from typing import List, Dict, Any, Optional, Set
from datetime import datetime
from urllib.parse import urljoin, urlparse, urlunparse
from bs4 import BeautifulSoup
from loguru import logger

from .base_connector import BaseConnector, DocumentInfo


class WebConnector(BaseConnector):
    """Connector for web scraping internal sites and documentation."""
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.base_urls = []
        self.max_depth = 3
        self.max_pages = 100
        self.allowed_domains = []
        self.excluded_patterns = []
        self.included_patterns = []
        self.respect_robots = True
        self.crawl_delay = 1.0
        self.headers = {}
        self.discovered_urls = set()
        self.scraped_urls = set()
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize web connector with configuration."""
        try:
            self.config = config
            self.base_urls = config.get("base_urls", [])
            self.max_depth = config.get("max_depth", 3)
            self.max_pages = config.get("max_pages", 100)
            self.allowed_domains = config.get("allowed_domains", [])
            self.excluded_patterns = config.get("excluded_patterns", [])
            self.included_patterns = config.get("included_patterns", [])
            self.respect_robots = config.get("respect_robots", True)
            self.crawl_delay = config.get("crawl_delay", 1.0)
            
            # Custom headers
            default_headers = {
                "User-Agent": "Knowledge-DB-Bot/1.0 (+https://your-domain.com/bot)"
            }
            self.headers = {**default_headers, **config.get("headers", {})}
            
            if not self.base_urls:
                raise ValueError("At least one base URL is required")
            
            # Extract allowed domains from base URLs if not specified
            if not self.allowed_domains:
                for url in self.base_urls:
                    parsed = urlparse(url)
                    if parsed.netloc:
                        self.allowed_domains.append(parsed.netloc)
            
            # Initialize HTTP client
            self.client = httpx.AsyncClient(
                headers=self.headers,
                timeout=30.0,
                follow_redirects=True
            )
            
            # Test connection
            if await self.test_connection():
                self.is_initialized = True
                logger.info(f"Web connector initialized for {len(self.base_urls)} URLs")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize web connector: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test connection to the first base URL."""
        try:
            if self.base_urls:
                response = await self.client.get(self.base_urls[0])
                return response.status_code == 200
            return False
        except Exception as e:
            logger.error(f"Web connection test failed: {e}")
            return False
    
    async def list_documents(self) -> List[Dict[str, Any]]:
        """Discover and list all documents from web crawling."""
        self._ensure_initialized()
        
        documents = []
        self.discovered_urls.clear()
        self.scraped_urls.clear()
        
        # Start crawling from base URLs
        for base_url in self.base_urls:
            await self._crawl_recursive(base_url, 0)
        
        # Convert discovered URLs to document list
        for url in list(self.discovered_urls)[:self.max_pages]:
            try:
                doc_info = await self._create_document_info(url)
                if doc_info:
                    documents.append(doc_info)
            except Exception as e:
                logger.error(f"Error creating document info for {url}: {e}")
                continue
        
        logger.info(f"Found {len(documents)} web documents")
        return documents
    
    async def get_document_content(self, identifier: str) -> str:
        """Get content of a specific web page."""
        self._ensure_initialized()
        
        try:
            # Identifier is the URL
            url = identifier
            
            response = await self.client.get(url)
            if response.status_code != 200:
                return ""
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content
            content = self._extract_text_content(soup)
            
            return content
            
        except Exception as e:
            logger.error(f"Error getting web document content {identifier}: {e}")
            return ""
    
    async def get_document_metadata(self, identifier: str) -> Dict[str, Any]:
        """Get metadata for a specific web page."""
        self._ensure_initialized()
        
        try:
            url = identifier
            
            response = await self.client.get(url)
            if response.status_code != 200:
                return {}
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            metadata = {
                "url": url,
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "content_length": len(response.content),
                "type": "web_page"
            }
            
            # Extract meta tags
            meta_tags = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                if name and content:
                    meta_tags[name] = content
            
            if meta_tags:
                metadata["meta_tags"] = meta_tags
            
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata["html_title"] = title_tag.get_text().strip()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error getting web document metadata {identifier}: {e}")
            return {}
    
    async def _crawl_recursive(self, url: str, depth: int):
        """Recursively crawl web pages."""
        if depth > self.max_depth:
            return
        
        if url in self.scraped_urls:
            return
        
        if len(self.discovered_urls) >= self.max_pages:
            return
        
        try:
            # Check if URL should be crawled
            if not self._should_crawl_url(url):
                return
            
            self.scraped_urls.add(url)
            
            # Add crawl delay
            if self.crawl_delay > 0 and len(self.scraped_urls) > 1:
                import asyncio
                await asyncio.sleep(self.crawl_delay)
            
            # Fetch the page
            response = await self.client.get(url)
            
            if response.status_code != 200:
                return
            
            # Add to discovered URLs
            self.discovered_urls.add(url)
            
            # Parse HTML to find links
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract links for further crawling
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link['href']
                absolute_url = urljoin(url, href)
                
                # Clean URL (remove fragments, query params if needed)
                parsed = urlparse(absolute_url)
                clean_url = urlunparse((
                    parsed.scheme,
                    parsed.netloc,
                    parsed.path,
                    parsed.params,
                    parsed.query,
                    ""  # Remove fragment
                ))
                
                # Recursively crawl if within bounds
                if clean_url not in self.scraped_urls and len(self.discovered_urls) < self.max_pages:
                    await self._crawl_recursive(clean_url, depth + 1)
            
        except Exception as e:
            logger.error(f"Error crawling URL {url}: {e}")
    
    def _should_crawl_url(self, url: str) -> bool:
        """Check if URL should be crawled."""
        parsed = urlparse(url)
        
        # Check allowed domains
        if self.allowed_domains and parsed.netloc not in self.allowed_domains:
            return False
        
        # Check excluded patterns
        for pattern in self.excluded_patterns:
            if pattern in url:
                return False
        
        # Check included patterns (if specified, URL must match at least one)
        if self.included_patterns:
            for pattern in self.included_patterns:
                if pattern in url:
                    break
            else:
                return False
        
        # Skip common non-content URLs
        skip_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', 
                          '.zip', '.tar', '.gz', '.jpg', '.jpeg', '.png', '.gif', 
                          '.mp3', '.mp4', '.avi', '.exe', '.dmg']
        
        for ext in skip_extensions:
            if url.lower().endswith(ext):
                return False
        
        # Skip common non-content paths
        skip_paths = ['/api/', '/ajax/', '/json/', '/xml/', '/rss/', '/feed/',
                     '/admin/', '/login/', '/logout/', '/register/', '/search/']
        
        for path in skip_paths:
            if path in url.lower():
                return False
        
        return True
    
    async def _create_document_info(self, url: str) -> Optional[Dict[str, Any]]:
        """Create document info for a URL."""
        try:
            response = await self.client.get(url)
            if response.status_code != 200:
                return None
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = "Untitled"
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Try to get last modified from headers
            last_modified = None
            last_modified_header = response.headers.get('last-modified')
            if last_modified_header:
                try:
                    from email.utils import parsedate_to_datetime
                    last_modified = parsedate_to_datetime(last_modified_header)
                except:
                    pass
            
            # Generate URL hash as identifier
            url_hash = hashlib.md5(url.encode()).hexdigest()
            
            return {
                "identifier": url,
                "title": title,
                "url": url,
                "last_modified": last_modified,
                "file_type": "html",
                "metadata": {
                    "url_hash": url_hash,
                    "domain": urlparse(url).netloc,
                    "path": urlparse(url).path,
                    "content_type": response.headers.get("content-type", ""),
                    "content_length": len(response.content),
                    "status_code": response.status_code,
                    "type": "web_page"
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating document info for {url}: {e}")
            return None
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract meaningful text content from HTML."""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        # Try to find main content areas
        main_content = None
        
        # Look for common content containers
        content_selectors = [
            'main',
            '[role="main"]',
            '.main-content',
            '.content',
            '.post-content',
            '.entry-content',
            '.article-content',
            '#content',
            '#main-content',
            '.documentation',
            '.wiki-content'
        ]
        
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        # If no main content found, use body
        if not main_content:
            main_content = soup.find('body') or soup
        
        # Extract text
        text = main_content.get_text(separator=' ', strip=True)
        
        # Clean up whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Add title if available
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text().strip()
            if title and title not in text[:200]:  # Don't duplicate if title is in content
                text = f"Title: {title}\n\n{text}"
        
        return text
    
    async def cleanup(self):
        """Clean up the HTTP client."""
        if self.client:
            await self.client.aclose()







