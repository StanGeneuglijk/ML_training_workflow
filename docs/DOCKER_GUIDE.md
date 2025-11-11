# Docker Deployment Guide

**Step-by-step guide for containerized ML workflow**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Docker Setup](#docker-setup)
3. [Docker Compose Services](#docker-compose-services)
4. [Usage](#usage)
5. [Troubleshooting](#troubleshooting)

---

## Overview

The ML Workflow can be deployed using Docker containers for:
- ✅ **Consistent environment** - Same Python, dependencies everywhere
- ✅ **Easy deployment** - Build once, run anywhere
- ✅ **Service isolation** - MLflow, training, serving in separate containers
- ✅ **Scalability** - Easy to scale inference API

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Docker Compose Stack                                   │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   MLflow     │  │   Training   │  │   Serving    │  │
│  │   Server     │  │  Application │  │     API      │  │
│  │              │  │              │  │              │  │
│  │  Port: 5000  │  │  (one-shot)  │  │  Port: 8000  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  │
│         │                 │                 │          │
│         └─────────────────┴─────────────────┘          │
│                     Shared Volumes:                     │
│                     - ./mlruns                          │
│                     - ./data                            │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## Docker Setup

### Prerequisites

```bash
# Install Docker
# macOS: Docker Desktop
# Linux: Docker Engine + Docker Compose

# Verify installation
docker --version
docker-compose --version
```

### Build Image

```bash
# Build the ML workflow image
docker build -t ml-workflow:latest .
```

**What the build does:**
1. Uses Python 3.13-slim base image
2. Installs Poetry
3. Installs all dependencies from `pyproject.toml`
4. Copies application code
5. Creates required directories

**Build time:** ~3-5 minutes (first build)
**Image size:** ~1.5GB

---

## Docker Compose Services

### Service: mlflow

**Purpose:** MLflow tracking server

**Configuration:**
```yaml
mlflow:
  ports: ["5000:5000"]
  volumes:
    - ./mlruns:/app/mlruns  # Persist experiments
    - ./data:/app/data      # Access data
  command: mlflow ui --host 0.0.0.0 --port 5000
```

**Access:** http://localhost:5000

**What it provides:**
- Experiment tracking UI
- Model registry
- Artifact storage
- Run comparison

### Service: ml-training

**Purpose:** Run training application

**Configuration:**
```yaml
ml-training:
  environment:
    - MLFLOW_TRACKING_URI=http://mlflow:5000
  volumes:
    - ./data:/app/data
    - ./mlruns:/app/mlruns
  depends_on: mlflow (healthcheck)
  command: python application_training.py
```

**What it does:**
1. Waits for MLflow to be healthy
2. Runs `application_training.py`
3. Logs to MLflow server
4. Exits when training completes

### Service: ml-serving

**Purpose:** REST API for inference

**Configuration:**
```yaml
ml-serving:
  ports: ["8000:8000"]
  environment:
    - MLFLOW_TRACKING_URI=http://mlflow:5000
  command: >
    python application_inference_serving.py serve
    --model-name gradient_boosting_classifier_registered
    --model-stage Production
    --host 0.0.0.0
    --port 8000
```

**Access:** http://localhost:8000
**API Docs:** http://localhost:8000/docs

---

## Usage

### Quick Start

```bash
# 1. Build images
docker-compose build

# 2. Start MLflow server only
docker-compose up mlflow -d

# 3. Generate data (on host)
poetry run python data/generate_database.py

# 4. Run training
docker-compose up ml-training

# 5. Start inference API
docker-compose up ml-serving -d
```

### Start All Services

```bash
# Generate data first (required)
poetry run python data/generate_database.py

# Start all services
docker-compose up -d
```

**Services started:**
- MLflow UI: http://localhost:5000
- Inference API: http://localhost:8000

### Run Training in Container

```bash
# One-time training run
docker-compose run --rm ml-training
```

**Result:** Model trained and registered in MLflow

### Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

---

## Volume Mounts

### Shared Volumes

```yaml
volumes:
  - ./mlruns:/app/mlruns      # MLflow experiments and models
  - ./data:/app/data          # SQLite database
  - ./logs:/app/logs          # Application logs
```

**Why volumes:**
- ✅ Persist MLflow experiments across restarts
- ✅ Share data between containers
- ✅ Access logs from host
- ✅ Inspect database with external tools

### Data Persistence

**Persisted on host:**
- `mlruns/` - All MLflow experiments and registered models
- `data/database/sample_data.db` - Training data
- `logs/` - Application logs

**Generated in container:**
- `.venv/` - Python virtual environment (not mounted)
- `feature_repo/` - Feast repository (ephemeral)

---

## Environment Variables

### MLflow Configuration

```yaml
# In ml-training and ml-serving
environment:
  - MLFLOW_TRACKING_URI=http://mlflow:5000
```

**Why:** Points to MLflow container instead of local file storage

### Optional Variables

```yaml
environment:
  - LOG_LEVEL=INFO              # Logging verbosity
  - MLFLOW_EXPERIMENT=prod      # Default experiment
  - MODEL_STAGE=Production      # Model stage to load
```

---

## Advanced Usage

### Development with Hot Reload

```yaml
# Add to docker-compose.override.yml
services:
  ml-serving:
    volumes:
      - .:/app  # Mount code for hot reload
    command: >
      python application_inference_serving.py serve
      --reload  # Enable auto-reload on code changes
```

### Custom Training Command

```bash
# Run with custom parameters
docker-compose run --rm ml-training \
  python application_training.py --custom-args
```

### Access Container Shell

```bash
# Debug inside container
docker-compose run --rm ml-training bash

# Inside container:
$ python data/generate_database.py
$ python application_training.py
$ pytest
```

### Multi-Stage Build (Optimization)

```dockerfile
# Dockerfile.prod - optimized for production
FROM python:3.13-slim as builder
WORKDIR /app
RUN pip install poetry
COPY pyproject.toml poetry.lock ./
RUN poetry export -f requirements.txt -o requirements.txt --without-hashes

FROM python:3.13-slim
WORKDIR /app
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["mlflow", "ui", "--host", "0.0.0.0"]
```

---

## Deployment Patterns

### Pattern 1: Development

```bash
# Run MLflow and develop locally
docker-compose up mlflow -d

# Train on host (faster iteration)
poetry run python application_training.py

# View in MLflow container
# http://localhost:5000
```

### Pattern 2: Production

```bash
# Generate data
docker-compose run --rm ml-training python data/generate_database.py

# Train model
docker-compose run --rm ml-training

# Deploy API
docker-compose up ml-serving -d

# Scale API
docker-compose up --scale ml-serving=3 -d
```

### Pattern 3: CI/CD

```yaml
# In GitHub Actions
- name: Build and test
  run: |
    docker-compose build
    docker-compose run --rm ml-training python data/generate_database.py
    docker-compose run --rm ml-training pytest
```

---

## Troubleshooting

### Issue 1: Port Already in Use

```
Error: port 5000 is already allocated
```

**Solution:**
```bash
# Check what's using the port
lsof -i :5000

# Kill the process or change port in docker-compose.yml
ports: ["5001:5000"]
```

### Issue 2: Volume Permission Errors

```
Error: Permission denied: '/app/mlruns'
```

**Solution:**
```bash
# On Linux, fix permissions
sudo chown -R $USER:$USER mlruns/ data/ logs/

# Or run with --user flag
docker-compose run --rm --user $(id -u):$(id -g) ml-training
```

### Issue 3: Database Not Found

```
ERROR: Database not found at .../sample_data.db
```

**Solution:**
```bash
# Generate data on host (before starting containers)
poetry run python data/generate_database.py

# Or in container
docker-compose run --rm ml-training python data/generate_database.py
```

### Issue 4: Container Build Fails

```
ERROR: failed to solve: process "/bin/sh -c poetry install" did not complete
```

**Solution:**
```bash
# Clear Docker cache
docker-compose build --no-cache

# Or use docker buildkit
DOCKER_BUILDKIT=1 docker-compose build
```

### Issue 5: MLflow Not Accessible

```
Connection refused to http://mlflow:5000
```

**Solution:**
```bash
# Check MLflow health
docker-compose ps

# View MLflow logs
docker-compose logs mlflow

# Restart MLflow
docker-compose restart mlflow
```

---

## Best Practices

### 1. Generate Data Before Training

```bash
# Always generate data first
poetry run python data/generate_database.py

# Then start Docker services
docker-compose up
```

### 2. Use Volume Mounts for Development

```yaml
# Mount code for development
volumes:
  - .:/app
```

**Benefits:** Code changes reflected immediately

### 3. Health Checks

```yaml
# Ensure services are ready
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:5000"]
  interval: 30s
```

### 4. Resource Limits

```yaml
# Prevent resource exhaustion
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
```

### 5. Use .dockerignore

```
# .dockerignore
.venv/
__pycache__/
*.pyc
.git/
mlruns/
htmlcov/
.pytest_cache/
```

---

## Docker Commands Reference

### Build

```bash
# Build specific service
docker-compose build mlflow

# Build with no cache
docker-compose build --no-cache

# Build with progress
docker-compose build --progress=plain
```

### Run

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up mlflow -d

# Run one-off command
docker-compose run --rm ml-training pytest

# View logs
docker-compose logs -f mlflow
```

### Manage

```bash
# List running containers
docker-compose ps

# Stop services
docker-compose stop

# Remove containers
docker-compose down

# Remove containers and volumes
docker-compose down -v
```

### Debug

```bash
# Access container shell
docker-compose exec mlflow bash

# View container logs
docker-compose logs --tail=100 ml-training

# Inspect container
docker inspect ml-workflow-app
```

---

## Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml ml-workflow

# Scale services
docker service scale ml-workflow_ml-serving=3
```

### Using Kubernetes

```yaml
# k8s-deployment.yaml (example)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-serving
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: ml-serving
        image: ml-workflow:latest
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-service:5000"
```

---

## Summary

### Docker Benefits

✅ **Consistency** - Same environment everywhere
✅ **Isolation** - No dependency conflicts
✅ **Portability** - Run on any Docker host
✅ **Scalability** - Easy to scale services
✅ **CI/CD Ready** - Automated builds and tests

### Quick Commands

```bash
# Build
docker-compose build

# Train model
docker-compose run --rm ml-training

# Start MLflow
docker-compose up mlflow -d

# Start API
docker-compose up ml-serving -d

# Stop all
docker-compose down
```

### File Locations

- **Dockerfile** - Image definition
- **docker-compose.yml** - Service orchestration
- **.dockerignore** - Exclude files from build

**Ready for containerized deployment!** 🐳

---

## Next Steps

- **CI/CD Guide**: See `CICD_GUIDE.md` for automated testing
- **Module Guide**: See `MODULE_GUIDE.md` for code architecture
- **Applications Guide**: See `APPLICATIONS_GUIDE.md` for training/inference

