# CI/CD Pipeline Guide

**Automated testing and deployment with GitHub Actions**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Workflows](#workflows)
4. [Setup](#setup)
5. [Advanced Configuration](#advanced-configuration)

---

## Overview

The ML Workflow includes a **GitHub Actions CI/CD pipeline** that automatically:

- ✅ **Tests code** - Runs **225 tests** (189 unit + 36 integration) on every push
- ✅ **Unit tests** - Fast, isolated component tests
- ✅ **Integration tests** - End-to-end workflow validation
- ✅ **Checks quality** - Linting, formatting, type checking
- ✅ **Tests training** - Verifies complete ML workflow
- ✅ **Builds Docker** - Validates containerization
- ✅ **Multi-version** - Tests Python 3.9-3.13

### Test Coverage
- **189 unit tests** ✅ - Component testing
- **36 integration tests** ✅ - Workflow testing
- **82% code coverage** ✅ - Comprehensive validation

### Pipeline Triggers

```yaml
on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]
```

**Runs on:**
- Every push to `main` or `develop`
- Every pull request to `main` or `develop`

---

## Pipeline Architecture

### Four-Job Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  CI/CD Pipeline (GitHub Actions)                        │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Job 1: Test                                            │
│  ├── Matrix: Python 3.9, 3.10, 3.11, 3.12, 3.13        │
│  ├── Install dependencies                               │
│  ├── Generate test data                                 │
│  ├── Run unit tests (220+ tests)                        │
│  └── Upload coverage to Codecov                         │
│                                                          │
│  Job 2: Lint                                            │
│  ├── Check black (code formatting)                      │
│  ├── Check isort (import sorting)                       │
│  └── Check mypy (type checking)                         │
│                                                          │
│  Job 3: Training                                        │
│  ├── Depends on: Test                                   │
│  ├── Generate database                                  │
│  ├── Run application_training.py                        │
│  ├── Verify MLflow artifacts                            │
│  └── Upload MLflow artifacts                            │
│                                                          │
│  Job 4: Docker                                          │
│  ├── Build Docker image                                 │
│  └── Validate docker-compose.yml                        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Total runtime:** ~10-15 minutes per push

---

## Workflows

### Workflow File

**Location:** `.github/workflows/ci.yml`

### Job 1: Test

**Purpose:** Run comprehensive test suite

```yaml
test:
  runs-on: ubuntu-latest
  strategy:
    matrix:
      python-version: ['3.9', '3.10', '3.11', '3.12', '3.13']
  
  steps:
    - Checkout code
    - Set up Python ${{ matrix.python-version }}
    - Install Poetry
    - Cache dependencies
    - Install dependencies
    - Generate test data
    - Run unit tests with coverage
    - Upload coverage to Codecov
```

**What it tests:**
- All unit tests (220+)
- Code coverage (69%)
- Python 3.9-3.13 compatibility

**Artifacts:**
- Coverage report uploaded to Codecov
- Test results in workflow logs

### Job 2: Lint

**Purpose:** Code quality checks

```yaml
lint:
  runs-on: ubuntu-latest
  steps:
    - Check code formatting (black)
    - Check import sorting (isort)
    - Type checking (mypy)
```

**Checks:**
```bash
poetry run black --check .
poetry run isort --check-only .
poetry run mypy module/ specs/ src/
```

**Note:** Continues on error (won't fail the build)

### Job 3: Training

**Purpose:** Test end-to-end ML workflow

```yaml
training:
  needs: test  # Only runs if tests pass
  steps:
    - Generate database
    - Run application_training.py
    - Verify MLflow artifacts created
    - Upload MLflow artifacts
```

**What it verifies:**
- ✅ Data generation works
- ✅ Training application runs successfully
- ✅ MLflow tracking works
- ✅ Model registered in registry

**Artifacts:**
- MLflow runs uploaded (7-day retention)

### Job 4: Docker

**Purpose:** Validate containerization

```yaml
docker:
  steps:
    - Build Docker image
    - Validate docker-compose.yml
```

**What it checks:**
- ✅ Dockerfile builds successfully
- ✅ docker-compose.yml is valid
- ✅ Image size reasonable

---

## Setup

### Step 1: Push to GitHub

```bash
# Initialize git (if not done)
git init
git add .
git commit -m "Initial commit"

# Add remote
git remote add origin https://github.com/username/ml-workflow.git

# Push
git push -u origin main
```

**Result:** CI/CD pipeline runs automatically

### Step 2: Enable GitHub Actions

1. Go to repository on GitHub
2. Click "Actions" tab
3. Workflows should appear automatically
4. View pipeline runs

### Step 3: Configure Codecov (Optional)

```bash
# 1. Go to https://codecov.io
# 2. Sign in with GitHub
# 3. Enable repository
# 4. Get upload token (automatic for public repos)
```

**No secrets needed** - Codecov action works automatically

### Step 4: View Results

```
GitHub Repository → Actions tab → Latest workflow run
```

**Each run shows:**
- Test results for all Python versions
- Code quality checks
- Training application output
- Docker build status

---

## Advanced Configuration

### Caching Dependencies

```yaml
- name: Cache dependencies
  uses: actions/cache@v3
  with:
    path: .venv
    key: venv-${{ runner.os }}-${{ matrix.python-version }}-${{ hashFiles('**/poetry.lock') }}
```

**Benefits:**
- ✅ Faster builds (~3 min → ~1 min)
- ✅ Reduced network usage
- ✅ More reliable (fewer download failures)

### Matrix Testing

```yaml
strategy:
  matrix:
    python-version: ['3.9', '3.10', '3.11', '3.12', '3.13']
    os: [ubuntu-latest, macos-latest, windows-latest]
```

**Tests:** 3 OS × 5 Python = 15 combinations

### Conditional Steps

```yaml
- name: Upload coverage
  if: matrix.python-version == '3.13'
  uses: codecov/codecov-action@v3
```

**Why:** Only upload coverage once (not 5 times)

### Secrets Management

```yaml
# For production deployments
environment:
  - MLFLOW_TRACKING_URI=${{ secrets.MLFLOW_SERVER }}
  - AWS_ACCESS_KEY_ID=${{ secrets.AWS_KEY }}
```

**Setup:**
1. Repository → Settings → Secrets
2. Add secrets
3. Reference in workflow

---

## Continuous Deployment

### Deploy to Production

```yaml
deploy:
  name: Deploy to Production
  runs-on: ubuntu-latest
  needs: [test, training]
  if: github.ref == 'refs/heads/main'
  
  steps:
    - name: Deploy to server
      run: |
        ssh user@production-server << 'EOF'
          cd /app/ml-workflow
          git pull
          docker-compose pull
          docker-compose up -d
        EOF
```

**Triggers:** Only on push to `main` after tests pass

### Model Deployment

```yaml
- name: Promote model to production
  run: |
    poetry run python -c "
    import mlflow
    client = mlflow.tracking.MlflowClient()
    # Get latest version
    versions = client.search_model_versions('name=\"gradient_boosting_classifier_registered\"')
    latest = sorted(versions, key=lambda x: int(x.version))[-1]
    # Promote to production
    client.transition_model_version_stage(
        name='gradient_boosting_classifier_registered',
        version=latest.version,
        stage='Production'
    )
    "
```

---

## Monitoring

### View Pipeline Status

```bash
# GitHub CLI
gh workflow list
gh run list
gh run view <run-id>

# Or GitHub web UI
https://github.com/username/repo/actions
```

### Pipeline Badges

Add to README.md:

```markdown
![Tests](https://github.com/username/ml-workflow/actions/workflows/ci.yml/badge.svg)
![Coverage](https://codecov.io/gh/username/ml-workflow/branch/main/graph/badge.svg)
```

### Email Notifications

**Automatic:**
- GitHub sends emails on workflow failures
- Configure in GitHub Settings → Notifications

---

## Best Practices

### 1. Keep Tests Fast

```yaml
# Run fast tests in CI
- name: Run unit tests
  run: poetry run pytest tests/unit_tests/ -v

# Run slow tests nightly
- name: Run integration tests
  if: github.event.schedule
  run: poetry run pytest tests/ -v
```

### 2. Fail Fast

```yaml
strategy:
  fail-fast: false  # Continue testing other versions
  matrix:
    python-version: ['3.9', '3.13']  # Test min and max
```

### 3. Cache Aggressively

```yaml
# Cache pip packages
- uses: actions/cache@v3
  with:
    path: ~/.cache/pip
    key: pip-${{ hashFiles('**/poetry.lock') }}

# Cache Poetry virtualenv
- uses: actions/cache@v3
  with:
    path: .venv
    key: venv-${{ hashFiles('**/poetry.lock') }}
```

### 4. Parallel Jobs

```yaml
# Jobs run in parallel by default
jobs:
  test:     # ─┐
  lint:     # ─┼─ Run in parallel
  docker:   # ─┘
  
  training:  # Runs after test completes
    needs: test
```

### 5. Security Scanning

```yaml
- name: Security audit
  run: |
    pip install safety
    poetry export -f requirements.txt | safety check --stdin
```

---

## Example Workflows

### Daily Model Training

```yaml
# .github/workflows/daily-training.yml
name: Daily Model Training

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Train model
        run: |
          poetry install
          python data/generate_database.py
          python application_training.py
      - name: Upload artifacts
        uses: actions/upload-artifact@v3
        with:
          name: daily-model
          path: mlruns/
```

### Release Workflow

```yaml
# .github/workflows/release.yml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build and push Docker
        run: |
          docker build -t ml-workflow:${{ github.ref_name }} .
          docker push ml-workflow:${{ github.ref_name }}
```

---

## Summary

### CI/CD Capabilities

✅ **Automated Testing** - 220+ tests on every push
✅ **Multi-version** - Python 3.9-3.13 tested
✅ **Code Quality** - Linting and type checking
✅ **End-to-End** - Full training workflow tested
✅ **Docker Build** - Container validation
✅ **Coverage Tracking** - Codecov integration

### Pipeline Files

```
.github/workflows/
└── ci.yml              # Main CI/CD pipeline

docker-compose.yml      # Service orchestration
Dockerfile              # Container definition
```

### Workflow Results

**On every push:**
- ✅ All tests run (220+)
- ✅ Code quality checked
- ✅ Training tested end-to-end
- ✅ Docker image built
- ✅ Coverage reported

**Time:** ~10-15 minutes per run

**Ready for production CI/CD!** 🚀

---

## Next Steps

- **Docker Guide**: See `DOCKER_GUIDE.md` for container deployment
- **Module Guide**: See `MODULE_GUIDE.md` for code architecture
- **Testing**: See `tests/README.md` for test details

