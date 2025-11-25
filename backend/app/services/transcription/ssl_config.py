#!/usr/bin/env python3
"""
SSL configuration utilities for handling self-signed certificates
"""

import os
import ssl
import urllib.request
import warnings
import logging

logger = logging.getLogger(__name__)


def patch_requests_for_ssl():
    """Patch requests library to skip SSL verification"""
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        import urllib3
        
        # Disable SSL warnings
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # Create a custom session that doesn't verify SSL
        original_request = requests.Session.request
        
        def patched_request(self, method, url, **kwargs):
            kwargs['verify'] = False
            return original_request(self, method, url, **kwargs)
        
        # Monkey patch the session
        requests.Session.request = patched_request
        
        # Also patch the module-level functions
        original_get = requests.get
        original_post = requests.post
        
        def patched_get(url, **kwargs):
            kwargs['verify'] = False
            return original_get(url, **kwargs)
            
        def patched_post(url, **kwargs):
            kwargs['verify'] = False
            return original_post(url, **kwargs)
            
        requests.get = patched_get
        requests.post = patched_post
        
        logger.info("Patched requests library to skip SSL verification")
        return True
        
    except ImportError:
        logger.warning("requests library not available for patching")
        return False


def patch_huggingface_hub():
    """Patch HuggingFace Hub to skip SSL verification"""
    try:
        # Set HuggingFace specific environment variables
        os.environ['HF_HUB_DISABLE_SSL_VERIFY'] = '1'
        os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.expanduser('~/.cache/huggingface')
        
        # Try to patch huggingface_hub if available
        try:
            import huggingface_hub
            from huggingface_hub import file_download
            
            # Patch the download function
            if hasattr(file_download, '_download_file_from_hub'):
                original_download = file_download._download_file_from_hub
                
                def patched_download(*args, **kwargs):
                    # Force verify=False in any requests calls
                    import requests
                    original_request = requests.request
                    
                    def no_verify_request(method, url, **req_kwargs):
                        req_kwargs['verify'] = False
                        return original_request(method, url, **req_kwargs)
                    
                    requests.request = no_verify_request
                    try:
                        return original_download(*args, **kwargs)
                    finally:
                        requests.request = original_request
                
                file_download._download_file_from_hub = patched_download
                
        except ImportError:
            pass
            
        logger.info("Configured HuggingFace Hub for SSL bypass")
        return True
        
    except Exception as e:
        logger.warning(f"Could not patch HuggingFace Hub: {e}")
        return False


def configure_ssl_for_self_signed():
    """Configure SSL to accept self-signed certificates"""
    
    # Check if SSL verification should be skipped
    skip_ssl = os.environ.get('SKIP_SSL_VERIFY', '').lower() in ['1', 'true', 'yes']
    
    if skip_ssl:
        logger.warning("⚠️  SSL verification disabled - accepting self-signed certificates")
        
        # Create unverified SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Monkey patch ssl default context
        ssl._create_default_https_context = lambda: ssl_context
        
        # Set environment variables for various libraries
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_VERIFY'] = 'false'
        os.environ['PYTHONHTTPSVERIFY'] = '0'
        os.environ['PYTHONWARNINGS'] = 'ignore:Unverified HTTPS request'
        
        # Disable SSL warnings
        warnings.filterwarnings('ignore', message='Unverified HTTPS request')
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        
        # Patch requests library
        patch_requests_for_ssl()
        
        # Patch HuggingFace Hub
        patch_huggingface_hub()
        
        # Try to disable urllib3 warnings if installed
        try:
            import urllib3
            urllib3.disable_warnings()
        except ImportError:
            pass
            
        return True
    
    return False


def get_ssl_instructions():
    """Get instructions for handling SSL certificate issues"""
    return """
If you're experiencing SSL certificate errors, you have several options:

1. **Temporary workaround (not recommended for production):**
   Set environment variable before running:
   ```bash
   export SKIP_SSL_VERIFY=1
   python download_models.py
   ```

2. **Command line flag:**
   ```bash
   python download_models.py --skip-ssl-verify
   ```

3. **Permanent environment variable:**
   Add to your shell profile (~/.bashrc or ~/.zshrc):
   ```bash
   export SKIP_SSL_VERIFY=1
   ```

4. **For specific proxy/firewall issues:**
   ```bash
   export HTTP_PROXY=http://your-proxy:port
   export HTTPS_PROXY=http://your-proxy:port
   ```

⚠️  WARNING: Disabling SSL verification can expose you to security risks.
Only use in trusted environments or when downloading from known sources.
"""