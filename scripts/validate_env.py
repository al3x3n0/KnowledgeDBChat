#!/usr/bin/env python3
"""
Environment validation script for Knowledge Database Chat.
Validates that all required environment variables are set correctly.
"""

import os
import sys
from urllib.parse import urlparse

OPTIONAL_VARS = [
    "GITLAB_URL",
    "GITLAB_TOKEN",
    "CONFLUENCE_URL",
    "CONFLUENCE_USER",
    "CONFLUENCE_API_TOKEN"
]


def validate_url(url_string: str) -> bool:
    """Validate URL format."""
    try:
        result = urlparse(url_string)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def check_database_connection(database_url: str) -> bool:
    """Check if database is accessible."""
    try:
        # Try to parse the URL
        parsed = urlparse(database_url.replace("postgresql+asyncpg://", "postgresql://"))
        # For now, just validate the format
        return parsed.scheme == "postgresql" and parsed.hostname
    except Exception:
        return False


def check_redis_connection(redis_url: str) -> bool:
    """Check if Redis is accessible."""
    try:
        parsed = urlparse(redis_url)
        return parsed.scheme == "redis"
    except Exception:
        return False


def check_ollama_connection(ollama_url: str) -> bool:
    """Check if Ollama is accessible."""
    try:
        import requests  # type: ignore
        response = requests.get(f"{ollama_url}/api/tags", timeout=5)
        return response.status_code == 200
    except ImportError:
        return False
    except Exception:
        return False


def required_vars_for_provider(llm_provider: str) -> dict:
    base_required = {
        "DATABASE_URL": {
            "required": True,
            "validate": lambda v: v.startswith(("postgresql://", "postgresql+asyncpg://")),
            "error": "DATABASE_URL must start with postgresql:// or postgresql+asyncpg://",
        },
        "REDIS_URL": {
            "required": True,
            "validate": lambda v: v.startswith("redis://"),
            "error": "REDIS_URL must start with redis://",
        },
        "SECRET_KEY": {
            "required": True,
            "validate": lambda v: len(v) >= 32 and v != "your-secret-key-here",
            "error": "SECRET_KEY must be at least 32 characters and not the default value",
        },
        "LLM_PROVIDER": {
            "required": True,
            "validate": lambda v: v in {"ollama", "deepseek"},
            "error": "LLM_PROVIDER must be 'ollama' or 'deepseek'",
        },
    }

    if llm_provider == "deepseek":
        base_required.update(
            {
                "DEEPSEEK_API_KEY": {
                    "required": True,
                    "validate": lambda v: len(v.strip()) > 0,
                    "error": "DEEPSEEK_API_KEY must be set when LLM_PROVIDER=deepseek",
                },
                "DEEPSEEK_API_BASE": {
                    "required": False,
                    "validate": lambda v: v.startswith(("http://", "https://")),
                    "error": "DEEPSEEK_API_BASE must be a valid HTTP/HTTPS URL",
                },
            }
        )
    else:
        base_required.update(
            {
                "OLLAMA_BASE_URL": {
                    "required": True,
                    "validate": lambda v: v.startswith(("http://", "https://")),
                    "error": "OLLAMA_BASE_URL must be a valid HTTP/HTTPS URL",
                }
            }
        )

    return base_required


def main():
    """Main validation function."""
    print("🔍 Validating Environment Configuration")
    print("=" * 50)
    print()
    
    # Load environment from .env file if it exists
    env_file = os.path.join("backend", ".env")
    if os.path.exists(env_file):
        print(f"📄 Loading environment from {env_file}")
        try:
            from dotenv import load_dotenv  # type: ignore
        except ImportError:
            print("⚠️  python-dotenv is not installed; skipping .env loading (using system environment).")
        else:
            load_dotenv(env_file)
    else:
        print(f"⚠️  {env_file} not found. Using system environment variables.")
    print()
    
    errors = []
    warnings = []

    llm_provider = os.getenv("LLM_PROVIDER", "ollama")
    required_vars = required_vars_for_provider(llm_provider)
    
    # Check required variables
    print("Checking required environment variables...")
    for var_name, config in required_vars.items():
        value = os.getenv(var_name)
        
        if not value:
            if config["required"]:
                errors.append(f"❌ {var_name}: Not set (required)")
            continue
        
        # Validate format
        if "validate" in config:
            if not config["validate"](value):
                errors.append(f"❌ {var_name}: {config.get('error', 'Invalid format')}")
            else:
                print(f"✅ {var_name}: Set and valid")
        else:
            print(f"✅ {var_name}: Set")
    
    print()
    
    # Check optional variables
    print("Checking optional environment variables...")
    for var_name in OPTIONAL_VARS:
        value = os.getenv(var_name)
        if value:
            print(f"✅ {var_name}: Set")
        else:
            print(f"⚪ {var_name}: Not set (optional)")
    
    print()
    
    # Test connections
    print("Testing service connections...")
    
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        if check_database_connection(database_url):
            print("✅ Database URL format is valid")
        else:
            warnings.append("⚠️  Database URL format may be invalid")
    
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        if check_redis_connection(redis_url):
            print("✅ Redis URL format is valid")
        else:
            warnings.append("⚠️  Redis URL format may be invalid")
    
    if llm_provider == "ollama":
        ollama_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_url:
            if check_ollama_connection(ollama_url):
                print("✅ Ollama is accessible")
            else:
                warnings.append("⚠️  Ollama is not accessible (may not be running, or 'requests' not installed)")
    
    print()
    
    # Summary
    print("=" * 50)
    if errors:
        print("❌ Validation failed with the following errors:")
        for error in errors:
            print(f"  {error}")
        print()
        return 1
    
    if warnings:
        print("⚠️  Validation passed with warnings:")
        for warning in warnings:
            print(f"  {warning}")
        print()
    
    print("✅ Environment validation passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
