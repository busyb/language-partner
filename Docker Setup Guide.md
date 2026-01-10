# Docker Setup Guide

## Project Structure
```
project/
├── backend/
│   └── main.py           # Main FastAPI application
├── utility/
│   ├── prompt_manager.py
│   ├── session_manager.py
│   ├── tts_manager.py
│   └── ...
├── static/
│   ├── stream.html
│   └── pronunciation.html
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .dockerignore
```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the application
docker-compose up --build

# Run in detached mode (background)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Option 2: Using Docker Directly

```bash
# Build the image
docker build -t language-learning-app .

# Run the container
docker run -d \
  --name language-learning \
  -p 8000:8000 \
  -v $(pwd)/static:/app/static \
  language-learning-app

# View logs
docker logs -f language-learning

# Stop and remove
docker stop language-learning
docker rm language-learning
```

## Accessing the Application

Once running, access the application at:
- **Main UI**: http://localhost:8000/
- **Pronunciation Practice**: http://localhost:8000/pronunciation
- **API Docs**: http://localhost:8000/docs (FastAPI auto-generated)

## Environment Variables

You can customize the application by setting environment variables:

```yaml
# In docker-compose.yml
environment:
  - PYTHONUNBUFFERED=1
  - PYTHONPATH=/app
  - MODEL_CACHE_DIR=/root/.cache/huggingface
  - SESSION_TTL_HOURS=2
  - CLEANUP_INTERVAL_SECONDS=300
```

Or with docker run:
```bash
docker run -d \
  -e SESSION_TTL_HOURS=4 \
  -p 8000:8000 \
  language-learning-app
```

## Volumes

### Model Cache Volume
The Docker setup includes a persistent volume for HuggingFace models:
```yaml
volumes:
  - model-cache:/root/.cache/huggingface
```

This ensures models are downloaded once and reused across container restarts.

### Development Volumes
For development, the docker-compose.yml mounts your local code:
```yaml
volumes:
  - ./backend:/app/backend
  - ./static:/app/static
  - ./utility:/app/utility
```

**For production**, remove these mounts to use the code baked into the image.

## First Run

On first startup, the application will:
1. Download the Qwen2.5-3B-Instruct-GGUF model (~2GB)
2. Initialize Faster-Whisper models
3. Download TTS models

**This may take 5-10 minutes depending on your internet connection.**

Monitor progress with:
```bash
docker-compose logs -f
```

## Health Checks

The container includes a health check that runs every 30 seconds:
```bash
# Check container health
docker ps

# Inspect health status
docker inspect language-learning-app | grep -A 10 Health
```

## Troubleshooting

### Container Fails to Start

**Check logs:**
```bash
docker-compose logs app
```

**Common issues:**
- Port 8000 already in use: Change port mapping to `-p 8001:8000`
- Out of memory: Increase Docker memory limit (Preferences → Resources)
- Model download failed: Check internet connection and retry

### GPU Support (Optional)

To use GPU acceleration with CUDA:

1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

2. Update docker-compose.yml:
```yaml
services:
  app:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

3. Rebuild with GPU support:
```bash
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
```

### Rebuild After Code Changes

```bash
# Rebuild and restart
docker-compose up --build

# Or force rebuild
docker-compose build --no-cache
docker-compose up
```

### Clear Everything and Start Fresh

```bash
# Stop containers
docker-compose down

# Remove volumes (including model cache)
docker-compose down -v

# Remove images
docker rmi language-learning-app

# Rebuild from scratch
docker-compose up --build
```

## Production Deployment

### 1. Remove Development Volumes
Edit `docker-compose.yml`:
```yaml
services:
  app:
    volumes:
      # REMOVE these for production:
      # - ./backend:/app/backend
      # - ./static:/app/static
      # - ./utility:/app/utility

      # KEEP this:
      - model-cache:/root/.cache/huggingface
```

### 2. Use Environment File
Create `.env` file:
```env
SESSION_TTL_HOURS=2
CLEANUP_INTERVAL_SECONDS=300
PYTHONUNBUFFERED=1
```

Update docker-compose.yml:
```yaml
services:
  app:
    env_file:
      - .env
```

### 3. Add Nginx Reverse Proxy
```yaml
services:
  app:
    # ... existing config ...
    expose:
      - "8000"
    # Remove direct port mapping

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app
```

### 4. Add Redis for Distributed Sessions (Optional)
```yaml
services:
  app:
    depends_on:
      - redis
    environment:
      - REDIS_URL=redis://redis:6379

  redis:
    image: redis:alpine
    volumes:
      - redis-data:/data

volumes:
  redis-data:
```

## Resource Requirements

### Minimum
- **CPU**: 4 cores
- **RAM**: 8GB
- **Disk**: 10GB

### Recommended
- **CPU**: 8+ cores
- **RAM**: 16GB+
- **Disk**: 20GB+ (for models and audio cache)
- **GPU**: NVIDIA GPU with 4GB+ VRAM (optional, for faster inference)

## Monitoring

### View Resource Usage
```bash
docker stats language-learning-app
```

### Check Logs
```bash
# Last 100 lines
docker-compose logs --tail=100 app

# Follow logs
docker-compose logs -f app

# Logs since 1 hour ago
docker-compose logs --since 1h app
```

### Session Statistics
Add an endpoint to monitor active sessions:
```python
@app.get("/admin/stats")
async def get_stats():
    return {
        "active_sessions": app.state.session_manager.get_session_count(),
        "uptime": time.time() - app.state.start_time
    }
```

Access at: http://localhost:8000/admin/stats

## Backup and Restore

### Backup Model Cache
```bash
docker run --rm \
  -v language-learning_model-cache:/data \
  -v $(pwd):/backup \
  alpine tar czf /backup/model-cache-backup.tar.gz -C /data .
```

### Restore Model Cache
```bash
docker run --rm \
  -v language-learning_model-cache:/data \
  -v $(pwd):/backup \
  alpine tar xzf /backup/model-cache-backup.tar.gz -C /data
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build image
        run: docker build -t myregistry/language-learning:latest .

      - name: Push image
        run: |
          echo "${{ secrets.DOCKER_PASSWORD }}" | docker login -u "${{ secrets.DOCKER_USERNAME }}" --password-stdin
          docker push myregistry/language-learning:latest
```

## Security Best Practices

1. **Don't run as root** (update Dockerfile):
```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

2. **Use secrets for sensitive data**:
```yaml
services:
  app:
    secrets:
      - api_key

secrets:
  api_key:
    file: ./secrets/api_key.txt
```

3. **Scan for vulnerabilities**:
```bash
docker scan language-learning-app
```

4. **Keep base image updated**:
```bash
docker pull python:3.11-slim
docker-compose build --no-cache
```

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f`
2. Verify health: `docker ps`
3. Test connectivity: `curl http://localhost:8000/`
4. Review resource usage: `docker stats`

Happy coding! 🚀