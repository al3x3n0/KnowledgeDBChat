# Docker Tuning Guide for Mac

## Current Configuration

The `docker-compose.yml` has been optimized for Mac with:

### Ollama Container Settings:
- **Memory Limit**: 6GB
- **Memory Reservation**: 2GB
- **CPU Limit**: 4 cores
- **Shared Memory**: 2GB
- **GPU**: Disabled (CPU-only mode)
- **Model**: llama3.2:1b (smallest, ~1GB)

### Environment Variables:
- `OLLAMA_NUM_GPU=0` - Force CPU-only
- `CUDA_VISIBLE_DEVICES=""` - Disable CUDA
- `OLLAMA_MAX_LOADED_MODELS=1` - Only one model at a time
- `OLLAMA_NUM_PARALLEL=1` - Single request at a time
- `OLLAMA_KEEP_ALIVE=5m` - Unload model after 5 minutes
- `OLLAMA_NUM_THREAD=4` - Limit CPU threads

## Docker Desktop Settings

### Required Settings:

1. **Open Docker Desktop**
   - Click Docker icon → Settings/Preferences

2. **Resources → Advanced**
   - **Memory**: Set to **8GB minimum** (12GB recommended)
   - **CPUs**: Use at least **4 cores**
   - **Swap**: Set to **2GB**
   - **Disk image size**: At least **64GB**

3. **Apply & Restart**

### Verify Settings:

```bash
# Check Docker memory allocation
docker system info | grep -i memory

# Check Ollama container resources
docker stats knowledge_db_ollama --no-stream

# Test model loading
docker exec knowledge_db_ollama ollama run llama3.2:1b "Hello"
```

## Troubleshooting

### Still Getting Memory Errors?

1. **Reduce Docker Memory Limit** (in docker-compose.yml):
   ```yaml
   limits:
     memory: 4G  # Instead of 6G
   ```

2. **Use Even Smaller Model**:
   ```bash
   # Already using llama3.2:1b (smallest)
   # No smaller option available
   ```

3. **Check Available Memory**:
   ```bash
   # Check Mac's available memory
   vm_stat | head -5
   
   # Check Docker memory usage
   docker stats --no-stream
   ```

4. **Close Other Applications**:
   - Free up macOS memory
   - Close memory-intensive apps

### GPU Errors?

The configuration should prevent GPU usage, but if you still see GPU errors:

1. **Verify Environment Variables**:
   ```bash
   docker exec knowledge_db_ollama env | grep OLLAMA
   ```

2. **Check Docker Desktop**:
   - Settings → Resources → Advanced
   - Ensure no GPU passthrough is enabled

3. **Restart Ollama Container**:
   ```bash
   docker-compose restart ollama
   ```

## Model Options

| Model | Size | RAM Needed | Speed | Quality |
|-------|------|------------|-------|---------|
| llama3.2:1b | ~1GB | 2-3GB | Fastest | Basic |
| llama3.2:3b | ~2GB | 4-5GB | Fast | Good |
| phi3:mini | ~2GB | 4-5GB | Fast | Good |
| llama2 | ~4GB | 6-8GB | Medium | Better |
| mistral:7b | ~4GB | 6-8GB | Medium | Best |

**Current**: `llama3.2:1b` (best for 8GB Mac)

## Performance Tips

1. **Keep Model Loaded**: The `OLLAMA_KEEP_ALIVE=5m` keeps the model in memory for 5 minutes, reducing reload time.

2. **Single Request**: `OLLAMA_NUM_PARALLEL=1` ensures only one request at a time, preventing memory spikes.

3. **Monitor Usage**: Use `docker stats` to monitor memory usage in real-time.

4. **Restart if Needed**: If memory issues persist, restart the Ollama container:
   ```bash
   docker-compose restart ollama
   ```

## Quick Commands

```bash
# Restart Ollama with new settings
docker-compose restart ollama

# Check Ollama status
docker logs knowledge_db_ollama --tail 50

# Test model
docker exec knowledge_db_ollama ollama run llama3.2:1b "test"

# Monitor memory
docker stats knowledge_db_ollama

# List available models
docker exec knowledge_db_ollama ollama list
```


