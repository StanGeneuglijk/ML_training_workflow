# ML Workflow Training

**Production-ready ML engineering framework with spec-driven design**

> **Note:** This project focuses on **model training**. For model inference and serving, see the companion project **[ML_WORKFLOW_INFERENCE_SERVING](../ML_WORKFLOW_INFERENCE_SERVING/)**.

---

## Quick Start

### Prerequisites

- Python 3.9-3.13 (tested on 3.13)
- Poetry (dependency management)

### Installation

```bash
curl -sSL https://install.python-poetry.org | python3 -
cd ML_workflow_training
poetry install
```

### Train Your First Model

```bash
# 1. Generate synthetic data
poetry run python data/generate_delta_lake.py

# 2. Train model with MLflow tracking
poetry run python application_training.py

# 3. View results in MLflow UI
poetry run mlflow ui --port 5000
# Open http://127.0.0.1:5000
```

---

## Project Structure

```
ML_workflow_training/
├── data/                    # Data generation and loading
├── specs/                   # Pydantic specifications (config layer)
├── module/                  # ML modules (implementation layer)
├── src_training/            # Workflow orchestration
├── feature_store/           # Feast integration (optional)
├── tests/                   # Comprehensive test suite
├── docs/                    # Technical documentation
├── application_training.py  # Training application entry point
└── pyproject.toml          # Dependencies (Poetry)
```

---

## Core Components

### Specifications (Config Layer)

Type-safe configuration with Pydantic validation:

```python
from specs_training import FeatureSpecBuilder, ModelSpecBuilder

feature_specs = FeatureSpecBuilder()\
    .add_numeric_group(["age", "income"], scaler_type="standard")\
    .build()

model_spec = ModelSpecBuilder()\
    .add_classifier("gb_model", hyperparameters={"n_estimators": 100})\
    .build()[0]
```

### Modules (Implementation Layer)

Core ML engineering implementations:

```python
from module.classifier import GradientBoostingClassifierImpl
from module.pre_processing import FeatureSpecPipeline
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', FeatureSpecPipeline(feature_specs)),
    ('classifier', GradientBoostingClassifierImpl(model_spec))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

### Orchestrator (Workflow Layer)

Coordinate complete ML workflow:

```python
from src_training.orchestrator import run_ml_workflow

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y,
    validation_strategy="cross_validation",
    tuning_spec=tuning_spec,
    calibration_spec=calib_spec,
    mlflow_spec=mlflow_spec
)
```

---

## Workflow Example

```python
from specs_training import (
    FeatureSpecBuilder,
    ModelSpecBuilder,
    GridSearchSpecBuilder,
    MLflowSpecBuilder
)
from src_training.orchestrator import run_ml_workflow
from data import load_data

# Load data
X, y = load_data("classification_data")

# Configure
feature_specs = FeatureSpecBuilder()\
    .add_numeric_group([f"feature_{i}" for i in range(20)])\
    .build()

model_spec = ModelSpecBuilder().add_classifier("gb_model").build()[0]

tuning_spec = GridSearchSpecBuilder()\
    .set_param_grid({"classifier__n_estimators": [50, 100, 200]})\
    .set_cv_folds(5)\
    .build()

mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("production_models")\
    .enable_model_registry(stage="Staging")\
    .build()

# Run workflow
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y,
    validation_strategy="cross_validation",
    tuning_spec=tuning_spec,
    mlflow_spec=mlflow_spec
)

print(f"CV Accuracy: {results['cv_score']:.3f}")
```

---

## Testing

### Run Tests

```bash
# All tests
poetry run pytest

# With coverage
poetry run pytest --cov

# Specific module
poetry run pytest tests/unit_tests/test_classifier.py -v
```

### Test Organization

```
tests/
├── unit_tests/              # Fast, isolated tests
└── integration_tests/       # End-to-end tests
```

---

## Dependencies

### Core ML

- scikit-learn >= 1.3.0
- numpy >= 1.24.0
- pandas >= 2.0.0

### Configuration

- pydantic >= 2.0.0

### Experiment Tracking

- mlflow >= 2.0.0

### Feature Store (Optional)

- feast >= 0.38.0

---

## Documentation

### Technical Guides

- **[Module Guide](docs/MODULE_GUIDE.md)** - ML modules
- **[Specs Guide](docs/SPECS_GUIDE.md)** - Configuration architecture
- **[Orchestrator Guide](docs/ORCHESTRATOR_GUIDE.md)** - Workflow coordination
- **[Data Guide](docs/DATA_GUIDE.md)** - Data management
- **[Feature Store Guide](docs/FEATURE_STORE.md)** - Feast integration

### Deployment Guides

- **[Docker Guide](docs/DOCKER_GUIDE.md)** - Container deployment
- **[CI/CD Guide](docs/CICD_GUIDE.md)** - Automated testing

---

## Architecture

### Three-Layer Design

```
Specs (WHAT) → Modules (HOW) → Orchestrator (WHEN)
```

- **Specs** - Type-safe configuration
- **Modules** - ML implementations
- **Orchestrator** - Workflow coordination

### Key Patterns

1. **Spec-Driven** - Configuration as validated objects
2. **Modular** - Composable components via Pipeline
3. **Sklearn-Compatible** - Standard fit/predict API
4. **Production-Ready** - Error handling, logging, testing

---

## Summary

✅ **Spec-driven design** - Type-safe configuration
✅ **MLflow integration** - Experiment tracking
✅ **Modular architecture** - Composable components
✅ **Comprehensive testing** - Unit and integration tests
✅ **Production-ready** - Error handling, logging, validation

**Perfect for:** Learning, prototyping, and building production ML training systems! 🚀

---

## Quick Links

- **Technical Guides:** [`docs/`](docs/)
- **Tests:** [`tests/`](tests/)
- **Training App:** [`application_training.py`](application_training.py)

**Get started in 3 commands:**
```bash
poetry install                          # 1. Setup
poetry run python data/generate_delta_lake.py  # 2. Data
poetry run python application_training.py      # 3. Train
```
