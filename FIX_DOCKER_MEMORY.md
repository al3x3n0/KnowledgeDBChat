# Fix Docker Desktop Memory Issue

## Problem
Docker Desktop only has **7.65GB** total memory allocated, but we need more for Ollama + other containers.

## Solution: Increase Docker Desktop Memory

### Step 1: Open Docker Desktop Settings
1. Click the **Docker icon** in your Mac menu bar
2. Select **Settings** (or **Preferences**)

### Step 2: Increase Memory Allocation
1. Go to **Resources** → **Advanced**
2. Find the **Memory** slider
3. **Increase to at least 12GB** (16GB if you have it available)
4. Click **Apply & Restart**

### Step 3: Verify
```bash
# Check new memory allocation
docker system info | grep -i "total memory"
```

You should see at least 12GiB or more.

### Step 4: Restart Services
```bash
# Restart Ollama with new memory limits
docker-compose restart ollama

# Check memory usage
docker stats knowledge_db_ollama --no-stream
```

## Alternative: Use Smaller Quantization

If you can't increase Docker Desktop memory, try a smaller model quantization:

```bash
# Try Q4 quantization (smaller than Q8)
docker exec knowledge_db_ollama ollama pull llama3.2:1b-q4_0

# Update config to use it
# In backend/app/core/config.py:
# DEFAULT_MODEL: str = "llama3.2:1b-q4_0"
```

## Current Configuration

- **Docker Desktop Memory**: 7.65GB (TOO LOW - needs 12GB+)
- **Ollama Container Limit**: 4GB (reduced from 6GB)
- **Model**: llama3.2:1b (Q8_0 quantization, ~1.3GB)

## Recommended Settings

For a Mac with 16GB+ RAM:
- **Docker Desktop Memory**: 12GB
- **Ollama Container**: 6GB limit
- **Other containers**: 6GB total

For a Mac with 8GB RAM:
- **Docker Desktop Memory**: 6GB (maximum)
- **Ollama Container**: 3GB limit
- **Use Q4 quantization**: `llama3.2:1b-q4_0`

## Quick Check Commands

```bash
# Check Docker memory
docker system info | grep -i memory

# Check Ollama memory usage
docker stats knowledge_db_ollama

# Test model
docker exec knowledge_db_ollama ollama run llama3.2:1b "test"
```


