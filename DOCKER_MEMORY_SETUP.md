# Docker Memory Configuration for Mac

This guide helps you configure Docker Desktop memory settings for running Ollama models on Mac.

## Issue
Ollama models require significant memory. If you see errors like:
```
model requires more system memory than is currently available
```

You need to increase Docker Desktop's memory allocation.

## Steps to Fix

### 1. Increase Docker Desktop Memory

1. **Open Docker Desktop**
   - Click the Docker icon in your menu bar
   - Select "Settings" (or "Preferences")

2. **Go to Resources**
   - Click "Resources" in the left sidebar
   - Click "Advanced" tab

3. **Adjust Memory Settings**
   - **For 8GB Mac**: Set to **4-6GB** (leave 2-4GB for macOS)
   - **For 16GB Mac**: Set to **8-10GB** (leave 6GB for macOS)
   - **For 32GB+ Mac**: Set to **12-16GB**

4. **Apply & Restart**
   - Click "Apply & Restart"
   - Wait for Docker to restart

### 2. Update docker-compose.yml Memory Limits

After increasing Docker Desktop memory, you may need to adjust the memory limit in `docker-compose.yml`:

```yaml
deploy:
  resources:
    limits:
      # For 8GB Mac: use 4G
      # For 16GB Mac: use 6-8G
      memory: 6G
    reservations:
      memory: 2G
```

### 3. Restart Services

After making changes:

```bash
# Stop services
docker-compose down

# Start services with new memory limits
docker-compose up -d

# Check Ollama container memory usage
docker stats knowledge_db_ollama
```

### 4. Download a Smaller Model

If you still have memory issues, use the smallest model:

```bash
# Download the smallest model
docker exec -it knowledge_db_ollama ollama pull llama3.2:1b

# Update your .env file
# DEFAULT_MODEL=llama3.2:1b
```

## Model Size Reference

| Model | Size | RAM Required | Mac Compatibility |
|-------|------|--------------|------------------|
| llama3.2:1b | ~1GB | 2-3GB | ✅ Best for 8GB Mac |
| llama3.2:3b | ~2GB | 4-5GB | ✅ Good for 16GB Mac |
| phi3:mini | ~2GB | 4-5GB | ✅ Good for 16GB Mac |
| llama2 | ~4GB | 6-8GB | ⚠️ Requires 16GB+ Mac |
| mistral:7b | ~4GB | 6-8GB | ⚠️ Requires 16GB+ Mac |

## Verify Configuration

Check if Docker has enough memory:

```bash
# Check Docker Desktop memory allocation
docker system info | grep -i memory

# Check Ollama container memory
docker stats knowledge_db_ollama --no-stream

# Test model loading
docker exec -it knowledge_db_ollama ollama run llama3.2:3b "Hello"
```

## Troubleshooting

### Still getting memory errors?

1. **Reduce model size**: Use `llama3.2:1b` instead of `llama3.2:3b`
2. **Close other applications**: Free up macOS memory
3. **Reduce Docker memory limit**: Lower the `memory: 6G` to `4G` in docker-compose.yml
4. **Check available memory**: `docker stats` to see actual usage

### Docker Desktop won't allocate more memory?

- Make sure you have enough free RAM on your Mac
- Close other memory-intensive applications
- Restart Docker Desktop completely

## Current Configuration

The `docker-compose.yml` is configured with:
- **Memory limit**: 6GB (adjustable)
- **Shared memory**: 2GB (for model loading)
- **CPU-only mode**: Enabled (no GPU required)
- **Max loaded models**: 1 (prevents loading multiple models)

Adjust these values based on your Mac's available RAM.


