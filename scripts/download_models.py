#!/usr/bin/env python3
"""
Download all necessary models for Knowledge Database Chat application.

This script downloads:
1. Ollama LLM models (llama2, etc.)
2. Sentence Transformer embedding models
3. Cross-encoder reranking models
"""

import os
import sys
import subprocess
import requests
from pathlib import Path
from typing import List, Tuple

# Models to download
# For Mac users: Use smaller models like llama3.2:3b or llama3.2:1b
# For more powerful systems: Use llama2, mistral:7b, or llama3.2
OLLAMA_MODELS = [
    "llama3.2:3b",  # Default LLM model (Mac-friendly, ~2GB)
    # Alternatives:
    # "llama3.2:1b",  # Smallest option (~1GB, faster but less capable)
    # "llama2",  # Original default (requires more memory, ~4GB+)
    # "phi3:mini",  # Microsoft's small model (~2GB)
    # "gemma:2b",  # Google's small model (~1.5GB)
]

EMBEDDING_MODELS = [
    "all-MiniLM-L6-v2",  # Default embedding model (small, fast)
    "all-mpnet-base-v2",  # Better quality (optional)
    "paraphrase-multilingual-mpnet-base-v2",  # Multilingual support (optional)
]

RERANKING_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",  # Default reranking model
]

# Colors for output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def print_success(text: str):
    """Print success message."""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_error(text: str):
    """Print error message."""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_info(text: str):
    """Print info message."""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.END}")


def check_ollama_installed() -> bool:
    """Check if Ollama is installed and running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_ollama_accessible(url: str = "http://localhost:11434") -> bool:
    """Check if Ollama API is accessible."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def download_ollama_model(model_name: str, ollama_url: str = "http://localhost:11434") -> bool:
    """Download an Ollama model."""
    print_info(f"Downloading Ollama model: {model_name}")
    
    # Check if model already exists
    try:
        response = requests.get(f"{ollama_url}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if any(m.get("name", "").startswith(model_name) for m in models):
                print_success(f"Model {model_name} already exists")
                return True
    except Exception as e:
        print_warning(f"Could not check existing models: {e}")
    
    # Download using ollama CLI
    try:
        print(f"  Pulling {model_name} (this may take several minutes)...")
        result = subprocess.run(
            ["ollama", "pull", model_name],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print_success(f"Successfully downloaded {model_name}")
            return True
        else:
            print_error(f"Failed to download {model_name}: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print_error(f"Timeout while downloading {model_name}")
        return False
    except FileNotFoundError:
        print_error("Ollama CLI not found. Please install Ollama first.")
        print_info("Install from: https://ollama.ai")
        return False
    except Exception as e:
        print_error(f"Error downloading {model_name}: {e}")
        return False


def download_sentence_transformer_model(model_name: str) -> bool:
    """Download a Sentence Transformer model."""
    print_info(f"Downloading Sentence Transformer model: {model_name}")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Check if model cache exists
        cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
        model_path = cache_dir / model_name
        
        if model_path.exists():
            print_success(f"Model {model_name} already cached")
            return True
        
        print(f"  Loading {model_name} (will be cached automatically)...")
        # Loading the model will automatically download it
        model = SentenceTransformer(model_name)
        
        # Verify it loaded correctly
        if model is not None:
            print_success(f"Successfully downloaded and cached {model_name}")
            # Test encoding
            test_embedding = model.encode("test")
            if test_embedding is not None and len(test_embedding) > 0:
                print_success(f"Model {model_name} is working correctly")
            return True
        else:
            print_error(f"Failed to load {model_name}")
            return False
            
    except ImportError:
        print_error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print_error(f"Error downloading {model_name}: {e}")
        return False


def download_cross_encoder_model(model_name: str) -> bool:
    """Download a Cross-encoder model."""
    print_info(f"Downloading Cross-encoder model: {model_name}")
    
    try:
        from sentence_transformers import CrossEncoder
        
        # Check if model cache exists
        cache_dir = Path.home() / ".cache" / "torch" / "sentence_transformers"
        model_path = cache_dir / model_name.replace("/", "_")
        
        if model_path.exists():
            print_success(f"Model {model_name} already cached")
            return True
        
        print(f"  Loading {model_name} (will be cached automatically)...")
        # Loading the model will automatically download it
        model = CrossEncoder(model_name)
        
        # Verify it loaded correctly
        if model is not None:
            print_success(f"Successfully downloaded and cached {model_name}")
            # Test prediction
            test_scores = model.predict([["query", "document"]])
            if test_scores is not None:
                print_success(f"Model {model_name} is working correctly")
            return True
        else:
            print_error(f"Failed to load {model_name}")
            return False
            
    except ImportError:
        print_error("sentence-transformers not installed. Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print_error(f"Error downloading {model_name}: {e}")
        return False


def main():
    """Main function to download all models."""
    print_header("Knowledge Database Chat - Model Download Script")
    
    print("This script will download all necessary models for the application.")
    print("This may take a while and require significant disk space.")
    print()
    
    # Check dependencies
    print("Checking dependencies...")
    
    # Check Ollama
    ollama_installed = check_ollama_installed()
    ollama_accessible = check_ollama_accessible()
    
    if ollama_installed or ollama_accessible:
        print_success("Ollama is available")
    else:
        print_warning("Ollama is not installed or not running")
        print_info("Ollama models will be skipped")
        print_info("Install Ollama from: https://ollama.ai")
        print_info("Or start Ollama service if using Docker")
    
    # Check sentence-transformers
    try:
        import sentence_transformers
        print_success("sentence-transformers is installed")
    except ImportError:
        print_error("sentence-transformers is not installed")
        print_info("Install with: pip install sentence-transformers")
        sys.exit(1)
    
    print()
    
    # Download models
    results = {
        "ollama": [],
        "embedding": [],
        "reranking": []
    }
    
    # Download Ollama models
    if ollama_installed or ollama_accessible:
        print_header("Downloading Ollama LLM Models")
        for model in OLLAMA_MODELS:
            success = download_ollama_model(model)
            results["ollama"].append((model, success))
    else:
        print_warning("Skipping Ollama models (Ollama not available)")
        for model in OLLAMA_MODELS:
            results["ollama"].append((model, False))
    
    # Download embedding models
    print_header("Downloading Sentence Transformer Embedding Models")
    for model in EMBEDDING_MODELS:
        success = download_sentence_transformer_model(model)
        results["embedding"].append((model, success))
    
    # Download reranking models
    print_header("Downloading Cross-encoder Reranking Models")
    for model in RERANKING_MODELS:
        success = download_cross_encoder_model(model)
        results["reranking"].append((model, success))
    
    # Summary
    print_header("Download Summary")
    
    total = 0
    successful = 0
    
    if results["ollama"]:
        print("\nOllama Models:")
        for model, success in results["ollama"]:
            total += 1
            if success:
                successful += 1
                print_success(f"  {model}")
            else:
                print_error(f"  {model}")
    
    print("\nEmbedding Models:")
    for model, success in results["embedding"]:
        total += 1
        if success:
            successful += 1
            print_success(f"  {model}")
        else:
            print_error(f"  {model}")
    
    print("\nReranking Models:")
    for model, success in results["reranking"]:
        total += 1
        if success:
            successful += 1
            print_success(f"  {model}")
        else:
            print_error(f"  {model}")
    
    print()
    print(f"Downloaded {successful}/{total} models successfully")
    
    if successful == total:
        print_success("All models downloaded successfully!")
        return 0
    elif successful > 0:
        print_warning("Some models failed to download. Check errors above.")
        return 1
    else:
        print_error("No models were downloaded. Please check your setup.")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nDownload interrupted by user.")
        sys.exit(1)

