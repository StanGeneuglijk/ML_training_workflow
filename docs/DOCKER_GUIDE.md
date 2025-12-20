# Docker Deployment Guide

**Containerized ML workflow deployment**

---

## Overview

Docker containers provide consistent environments for MLflow and training.

### Architecture

```
┌──────────────┐  ┌──────────────┐
│   MLflow     │  │   Training   │
│   Server     │  │  Application │
│  Port: 5000  │  │  (one-shot)  │
└──────┬───────┘  └──────┬───────┘
       │                 │
       └─────────────────┘
    Shared Volumes:
    - ./mlruns
    - ./data
```

---

## Quick Start

### Build and Run

```bash
# Build images
docker-compose build

# Generate data (on host)
poetry run python data/generate_delta_lake.py

# Start MLflow server
docker-compose up mlflow -d

# Run training
docker-compose up ml-training
```

### Services

**mlflow:** MLflow tracking server on port 5000
- Volumes: `./mlruns`, `./data`
- Access: http://localhost:5000

**ml-training:** Training application
- Depends on: mlflow (healthcheck)
- Runs: `application_training.py`
- Exits when complete

---

## Common Commands

```bash
# Start all services
docker-compose up -d

# Run training in container
docker-compose run --rm ml-training

# Stop services
docker-compose down

# View logs
docker-compose logs -f mlflow
```

---

## Troubleshooting

**Port 5000 in use:**
```bash
lsof -i :5000
# Change port in docker-compose.yml
```

**Data not found:**
```bash
poetry run python data/generate_delta_lake.py
```

**MLflow not accessible:**
```bash
docker-compose restart mlflow
```

---

