# CI/CD Pipeline Guide

**Automated testing and deployment**

---

## Overview

GitHub Actions CI/CD pipeline runs automatically on every push to `main` or `develop`.

### Pipeline Jobs

1. **Test** - Unit and integration tests with coverage
2. **Lint** - Code quality (black, mypy)
3. **Training** - End-to-end ML workflow validation
4. **Docker** - Docker image build and validation

---

## Workflow File

**Location:** `.github/workflows/ci-cd.yml`

### Job: Test

```yaml
test:
  - Install dependencies
  - Generate test data
  - Run unit tests with coverage
  - Run integration tests
  - Upload coverage to Codecov
```

### Job: Lint

```yaml
lint:
  - Check code formatting (black)
  - Type checking (mypy)
```

### Job: Training

```yaml
training:
  needs: test
  - Generate test data
  - Run application_training.py
  - Verify MLflow artifacts
```

### Job: Docker

```yaml
docker:
  - Build Docker image
  - Validate docker-compose.yml
```

---

## Setup

1. Push code to GitHub
2. Workflow runs automatically
3. View results in GitHub Actions tab

**Runtime:** ~5-10 minutes per push

---

## Capabilities

✅ Automated testing on every push
✅ Code quality checks
✅ End-to-end workflow validation
✅ Docker build validation
✅ Coverage tracking

---

**Ready for production CI/CD!** 🚀
