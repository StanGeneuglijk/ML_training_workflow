# ML Workflow Training

**Production-ready ML engineering framework with spec-driven design and comprehensive experiment tracking**

> **Note:** This project focuses on **model training**. For model inference and serving, see the companion project **[ML_WORKFLOW_INFERENCE_SERVING](../ML_WORKFLOW_INFERENCE_SERVING/)**.

---

## 🎯 Overview

A modern machine learning workflow framework focused on **ML engineering best practices**:

- ✅ **Spec-driven architecture** - Configuration via Pydantic validation
- ✅ **MLflow integration** - Experiment tracking and model registry
- ✅ **Modular design** - Composable, testable components
- ✅ **Type safety** - Full type hints and validation
- ✅ **Production patterns** - Error handling, logging, testing

**Philosophy:** Demonstrate production-ready ML engineering without infrastructure complexity

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9-3.13 (tested on 3.13)
- Poetry (dependency management)

### Installation

```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
cd ML_workflow_training
poetry install
```

### Train Your First Model

```bash
# 1. Generate synthetic data
poetry run python data/generate_database.py

# 2. Train model with MLflow tracking
poetry run python application_training.py

# 3. View results in MLflow UI
poetry run mlflow ui --port 5000
# Open http://127.0.0.1:5000
```

**That's it!** You've trained a model with:
- Automated preprocessing
- Hyperparameter tuning
- Model calibration
- Cross-validation
- MLflow tracking
- Model registry

---

## 📊 Project Structure

```
ML_workflow_training/
├── data/                    # Data generation and loading
│   ├── generate_database.py    # Create synthetic data
│   └── __init__.py              # SQLite data loader
│
├── specs/                   # Pydantic specifications (config layer)
│   ├── feature_spec.py          # Feature preprocessing config
│   ├── model_spec.py            # Model configuration
│   ├── params_tuning_spec.py    # Hyperparameter tuning config
│   ├── calibration_spec.py      # Calibration config
│   ├── mlflow_spec.py           # MLflow tracking config
│   └── feature_store_spec.py    # Feature store config
│
├── module/                  # ML modules (implementation layer)
│   ├── classifier.py            # Model implementations
│   ├── pre_processing.py        # Feature transformation
│   ├── calibration.py           # Probability calibration
│   ├── params_tuning.py         # Hyperparameter optimization
│   └── mlflow_tracker.py        # Experiment tracking
│
├── src/                     # Workflow orchestration
│   └── orchestrator.py          # ML workflow coordinator
│
├── feature_store/           # Feast integration (optional)
│   ├── feast_manager.py         # Feature store manager
│   ├── feature_definitions.py   # Feature views & entities
│   └── data_sources.py          # Data source config
│
├── tests/                   # Comprehensive test suite
│   ├── unit_tests/              # 207 passing tests
│   └── integration_tests/       # End-to-end tests
│
├── docs/                    # Technical documentation
│   ├── MODULE_GUIDE.md          # ML modules deep dive
│   ├── SPECS_GUIDE.md           # Configuration specs
│   ├── ORCHESTRATOR_GUIDE.md    # Workflow orchestration
│   ├── DATA_GUIDE.md            # Data management
│   └── FEATURE_STORE.md         # Feature store integration
│
├── application_training.py      # Training application entry point
├── utils.py                     # Shared utilities
└── pyproject.toml              # Dependencies (Poetry)
```

---

## 🎓 Technical Documentation

### Core Guides

1. **[Module Guide](docs/MODULE_GUIDE.md)** - Deep dive into ML modules
   - Classifier implementation
   - Preprocessing pipeline
   - Calibration & tuning
   - MLflow tracking

2. **[Specs Guide](docs/SPECS_GUIDE.md)** - Configuration architecture
   - Pydantic specifications
   - Builder patterns
   - Validation rules
   - Integration patterns

3. **[Orchestrator Guide](docs/ORCHESTRATOR_GUIDE.md)** - Workflow coordination
   - Complete execution flow
   - Integration points
   - Error handling
   - Advanced usage

4. **[Data Guide](docs/DATA_GUIDE.md)** - Data management
   - Database generation
   - Data loading
   - Schema details
   - Migration patterns

5. **[Feature Store Guide](docs/FEATURE_STORE.md)** - Feast integration
   - Feature store architecture
   - Feature definitions
   - Point-in-time correctness
   - Integration patterns

---

## 🔧 Core Components

### Specifications (Config Layer)

**Purpose:** Type-safe configuration with Pydantic validation

```python
from specs import FeatureSpecBuilder, ModelSpecBuilder, MLflowSpecBuilder

# Configure features
feature_specs = FeatureSpecBuilder()\
    .add_numeric_group(["age", "income"], scaler_type="standard")\
    .build()

# Configure model
model_spec = ModelSpecBuilder()\
    .add_classifier("gb_model", hyperparameters={"n_estimators": 100})\
    .build()[0]

# Configure MLflow
mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("my_experiment")\
    .enable_model_registry(stage="Staging")\
    .build()
```

**See:** `docs/SPECS_GUIDE.md` for complete reference

### Modules (Implementation Layer)

**Purpose:** Core ML engineering implementations

```python
from module.classifier import GradientBoostingClassifierImpl
from module.pre_processing import FeatureSpecPipeline
from sklearn.pipeline import Pipeline

# Build pipeline
pipeline = Pipeline([
    ('preprocessor', FeatureSpecPipeline(feature_specs)),
    ('classifier', GradientBoostingClassifierImpl(model_spec))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
```

**See:** `docs/MODULE_GUIDE.md` for implementation details

### Orchestrator (Workflow Layer)

**Purpose:** Coordinate complete ML workflow

```python
from src.orchestrator import run_ml_workflow

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y,
    validation_strategy="cross_validation",
    tuning_spec=tuning_spec,
    calibration_spec=calib_spec,
    mlflow_spec=mlflow_spec
)

# Returns: pipeline, metrics, MLflow run ID
```

**See:** `docs/ORCHESTRATOR_GUIDE.md` for workflow details

---

## 🧪 Testing

### Test Suite

- **207 unit tests** - 100% passing
- **68% code coverage**
- **Comprehensive validation** - All specs, modules, integrations tested

### Run Tests

```bash
# All tests
poetry run pytest

# With coverage
poetry run pytest --cov

# Specific module
poetry run pytest tests/unit_tests/test_classifier.py -v

# Fast tests only (skip slow integration tests)
poetry run pytest -m "not slow"
```

### Test Organization

```
tests/
├── unit_tests/              # Fast, isolated tests
│   ├── test_specs/          # Pydantic validation tests
│   ├── test_modules/        # ML module tests
│   └── test_orchestrator/   # Workflow tests
└── conftest.py             # Shared fixtures
```

**Coverage by module:**
- Classifier: 97%
- Pre-processing: 88%
- Calibration: 72%
- MLflow Tracker: 68%
- Specs: 80%+

---

## 📦 Dependencies

### Core ML

```toml
python = ">=3.9,<3.14"
scikit-learn = "^1.3.0"
numpy = "^1.24.0"
pandas = "^2.0.0"
```

### Configuration & Validation

```toml
pydantic = "^2.0.0"
```

### Experiment Tracking

```toml
mlflow = "^2.0.0"
joblib = "^1.0.0"
```

### Feature Store (Optional)

```toml
feast = "^0.38.0"
pyarrow = ">=14.0.0,<22.0.0"
```


### Development

```toml
pytest = "^7.0.0"
pytest-cov = "^4.0.0"
black = "^23.0.0"
mypy = "^1.0.0"
```

---

## 🎨 Development

### Code Quality

```bash
# Format code
poetry run black .

# Sort imports
poetry run isort .

# Type checking
poetry run mypy specs/ module/ src/

# Linting
poetry run flake8 specs/ module/ src/
```

### Project Standards

- **Google-style docstrings** - Clear documentation
- **Type hints everywhere** - Better IDE support
- **Pydantic validation** - Runtime type checking
- **100% test coverage goal** - Comprehensive testing
- **Logging** - Structured logging throughout

---

## 🏗️ Architecture Highlights

### 1. Spec-Driven Design

**Configuration as validated objects:**
- All configs are Pydantic models
- Validation happens at construction time
- Type-safe throughout workflow
- Serializable for versioning

### 2. Modular Composition

**Mix and match components:**
- Preprocessor + Classifier = Pipeline
- Pipeline + Tuning = Optimized Pipeline  
- Pipeline + Calibration = Calibrated Pipeline
- All tracked by MLflow

### 3. Clean Abstractions

**Three-layer architecture:**
```
Specs (WHAT) → Modules (HOW) → Orchestrator (WHEN)
```

Each layer independent and testable

### 4. Production Patterns

- **Error handling** - Graceful degradation
- **Logging** - Comprehensive event tracking
- **Testing** - 207 unit tests
- **Documentation** - Step-by-step guides
- **Versioning** - Model registry integration

---

## 📈 Workflow Example

**Complete training workflow:**

```python
from specs import (
    FeatureSpecBuilder,
    ModelSpecBuilder,
    GridSearchSpecBuilder,
    CalibrationSpecBuilder,
    MLflowSpecBuilder
)
from src.orchestrator import run_ml_workflow
from data import load_data

# 1. Load data
X, y = load_data("classification_data")

# 2. Configure features
feature_specs = FeatureSpecBuilder()\
    .add_numeric_group(
        [f"feature_{i}" for i in range(20)],
        imputer_strategy="mean",
        scaler_type="standard"
    ).build()

# 3. Configure model
model_spec = ModelSpecBuilder()\
    .add_classifier("gradient_boost_v1")\
    .build()[0]

# 4. Configure hyperparameter tuning
tuning_spec = GridSearchSpecBuilder()\
    .set_param_grid({
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1]
    })\
    .set_cv_folds(5)\
    .build()

# 5. Configure calibration
calib_spec = CalibrationSpecBuilder()\
    .set_method("sigmoid")\
    .set_cv_strategy(5)\
    .build()

# 6. Configure MLflow
mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("production_models")\
    .enable_model_registry(stage="Staging")\
    .add_tags({"version": "1.0", "team": "ml-eng"})\
    .build()

# 7. Run complete workflow
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y,
    validation_strategy="cross_validation",
    tuning_spec=tuning_spec,
    calibration_spec=calib_spec,
    mlflow_spec=mlflow_spec
)

# 8. Results
print(f"CV Accuracy: {results['cv_score']:.3f} ± {results['cv_std']:.3f}")
print(f"Best params: {results['tuning_summary']['best_params']}")
print(f"MLflow Run: {results['mlflow_run_id']}")
```

**Everything automated, tracked, and validated!** ✨

---

## 🔬 Advanced Features

### Parameter Tuning

```python
# Grid search over hyperparameter space
from module.params_tuning import GridSearch

tuner = GridSearch(tuning_spec)
tuner.fit(pipeline, X, y)

best_model = tuner.get_best_estimator()
best_params = tuner.get_best_params()
best_score = tuner.get_best_score()
```

### Probability Calibration

```python
# Calibrate for reliable probabilities
from module.calibration import ClassifierCalibration

calibrator = ClassifierCalibration(calib_spec)
calibrator.fit(pipeline, X_val, y_val)

calibrated_proba = calibrator.predict_proba(X_test)
# Now probabilities reflect true confidence!
```


### Feature Store Integration

```python
# Load features from Feast (optional)
from specs import FeatureStoreSpecBuilder

fs_spec = FeatureStoreSpecBuilder()\
    .enable()\
    .set_repo_path("feature_repo")\
    .set_n_features(20)\
    .build()

# Orchestrator handles feature loading
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=None,  # Loaded from feature store
    y=None,
    feature_store_spec=fs_spec
)
```

---

## 📚 Documentation

### Technical Guides

| Guide | Description | Size |
|-------|-------------|------|
| [MODULE_GUIDE.md](docs/MODULE_GUIDE.md) | ML modules deep dive | 27KB |
| [SPECS_GUIDE.md](docs/SPECS_GUIDE.md) | Configuration architecture | 22KB |
| [ORCHESTRATOR_GUIDE.md](docs/ORCHESTRATOR_GUIDE.md) | Workflow coordination | 23KB |
| [DATA_GUIDE.md](docs/DATA_GUIDE.md) | Data management | 17KB |
| [FEATURE_STORE.md](docs/FEATURE_STORE.md) | Feast integration | 17KB |

### Deployment Guides

| Guide | Description | Size |
|-------|-------------|------|
| [DOCKER_GUIDE.md](docs/DOCKER_GUIDE.md) | Container deployment | 12KB |
| [CICD_GUIDE.md](docs/CICD_GUIDE.md) | Automated testing | 11KB |

---

## 🧬 Architecture

### Three-Layer Design

```
┌─────────────────────────────────────┐
│  CONFIGURATION LAYER (specs/)       │
│  - Pydantic validation              │
│  - Type safety                      │
│  - Builder patterns                 │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  IMPLEMENTATION LAYER (module/)     │
│  - ML algorithms                    │
│  - Feature processing               │
│  - Experiment tracking              │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│  ORCHESTRATION LAYER (src/)         │
│  - Workflow coordination            │
│  - Component integration            │
│  - Error handling                   │
└─────────────────────────────────────┘
```

### Key Patterns

**1. Specification Pattern**
- Configuration separated from implementation
- Validation at build time, not runtime
- Easy to serialize/version

**2. Builder Pattern**
- Fluent API for configuration
- Sensible defaults
- Progressive disclosure of complexity

**3. Dependency Injection**
- Components receive dependencies via parameters
- Easy to test with mocks
- Clear dependency graph

**4. Pipeline Composition**
- Preprocessing + Model = Pipeline
- Composable, reusable components
- Sklearn-compatible

---

## 📊 Metrics & Results

### Current Performance

**Synthetic Dataset:**
- 1000 samples
- 20 features
- Binary classification

**Baseline Results:**
- **CV Accuracy:** 92.1% ± 2.2%
- **Training time:** ~2 seconds
- **Tuned accuracy:** ~94%+

### MLflow Tracking

**Automatically logs:**
- ✅ All hyperparameters
- ✅ CV metrics (mean, std)
- ✅ Test set metrics
- ✅ Model artifacts
- ✅ Feature importance
- ✅ Calibration plots
- ✅ Model signature
- ✅ Training duration

---

## 🧪 Testing

### Test Coverage

```bash
# Run all tests
poetry run pytest

# With coverage report
poetry run pytest --cov --cov-report=html
# Open htmlcov/index.html
```

### Test Summary

- **Unit Tests:** 207 passing (100%)
- **Coverage:** 68% overall
  - Classifier: 97%
  - Pre-processing: 88%
  - Specs: 80%+
  - Feature Store: 100%

### Test Categories

```bash
# Fast unit tests only
poetry run pytest tests/unit_tests/

# Specific module
poetry run pytest tests/unit_tests/test_classifier.py -v

# With markers
poetry run pytest -m "not slow"
```

---

## 🚀 Production Deployment

### Model Training

```bash
# Production training with full pipeline
poetry run python application_training.py
```

**What happens:**
1. Loads data from SQLite
2. Builds preprocessing pipeline
3. Runs hyperparameter tuning
4. Trains best model
5. Calibrates probabilities
6. Validates with 5-fold CV
7. Registers in MLflow
8. Saves to model registry

**Output:** Model v1 registered in "Staging" stage

### Model Serving

For inference and serving capabilities, see the separate **ML_WORKFLOW_INFERENCE_SERVING** project:
```bash
cd ../ML_WORKFLOW_INFERENCE_SERVING
# See that project's README for inference and serving options
```

---

## 💡 Design Decisions

### Why Pydantic Specs?

**Benefits:**
- ✅ Type safety - Catch errors early
- ✅ Documentation - Self-documenting code
- ✅ Validation - Runtime checks
- ✅ Serialization - Easy config versioning

**Alternative:** Dict-based configs (error-prone, no validation)

### Why SQLite?

**For demo/development:**
- ✅ No setup required
- ✅ Single file database
- ✅ Fast for <100K samples
- ✅ Easy to inspect

**For production:** Swap for PostgreSQL/Snowflake without changing ML code

### Why Sklearn Compatibility?

**Benefits:**
- ✅ Standard interface (fit/predict)
- ✅ Works with sklearn utilities
- ✅ Easy to understand
- ✅ Production-proven

**All custom components follow sklearn API**

### Why MLflow?

**Industry standard for:**
- Experiment tracking
- Model versioning
- Model registry
- Deployment tracking
- Collaboration

**Built-in throughout workflow**

---

## 🎯 Use Cases

### 1. ML Engineering Demo

Perfect for showcasing:
- Production ML patterns
- Configuration management
- Experiment tracking
- Model versioning

### 2. Rapid Prototyping

Quick iteration on:
- Feature engineering
- Model selection
- Hyperparameter tuning
- Validation strategies

### 3. Education & Learning

Learn production ML:
- Spec-driven design
- Pipeline composition
- MLflow integration
- Testing patterns

### 4. Production Foundation

Start small, scale up:
- SQLite → PostgreSQL
- Local MLflow → Remote server
- Synthetic data → Real data
- Single model → Multi-model

---

## 🔄 Workflow Variations

### Quick Baseline

```python
# Minimal workflow for quick results
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y
)
```

### With Tuning

```python
# Add hyperparameter optimization
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y,
    tuning_spec=tuning_spec  # ← Grid/Random search
)
```

### Production Pipeline

```python
# Full production workflow
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X_train, y=y_train,
    X_test=X_test, y_test=y_test,
    tuning_spec=tuning_spec,
    calibration_spec=calib_spec,
    mlflow_spec=mlflow_spec
)
```

---

## 📝 Code Statistics

| Component | Lines | Purpose |
|-----------|-------|---------|
| **Specs** | ~1,600 | Configuration layer |
| **Modules** | ~1,800 | ML implementations |
| **Orchestrator** | ~340 | Workflow coordination |
| **Data** | ~350 | Data generation/loading |
| **Feature Store** | ~250 | Feast integration |
| **Tests** | ~3,000+ | Comprehensive testing |
| **Total** | ~7,300+ | Production-ready ML training framework |

---

## 🎓 Learning Path

**Recommended order:**

1. **Start:** Run quick start to see it working
2. **Data:** Read `DATA_GUIDE.md` - understand data flow
3. **Specs:** Read `SPECS_GUIDE.md` - learn configuration
4. **Modules:** Read `MODULE_GUIDE.md` - understand implementations
5. **Orchestrator:** Read `ORCHESTRATOR_GUIDE.md` - see how it all connects
6. **Advanced:** Explore feature store, serving, tuning

---

## 🤝 Contributing

### Adding New Features

1. **Add spec** in `specs/` for configuration
2. **Add module** in `module/` for implementation
3. **Integrate** in `src/orchestrator.py`
4. **Test** in `tests/unit_tests/`
5. **Document** in `docs/`

### Code Style

- Follow Google docstring convention
- Use type hints everywhere
- Pydantic for configuration
- Comprehensive logging
- Unit tests required

---

## 📜 License

MIT License - See LICENSE file

---

## 🎉 Summary

**ML Workflow Training** is a production-ready ML engineering framework that demonstrates:

✅ **Best practices** - Spec-driven design, modular architecture
✅ **Type safety** - Pydantic validation throughout
✅ **Experiment tracking** - MLflow integration
✅ **Testability** - 207 passing tests, 68% coverage
✅ **Documentation** - Comprehensive technical guides
✅ **Simplicity** - SQLite-based, no infrastructure needed
✅ **Production-ready** - Error handling, logging, versioning

**Companion Project:** For inference and serving, see **[ML_WORKFLOW_INFERENCE_SERVING](../ML_WORKFLOW_INFERENCE_SERVING/)**

**Perfect for:** Learning, prototyping, and building production ML training systems! 🚀

---

## 📞 Quick Links

- **Technical Guides:** [`docs/`](docs/)
- **Tests:** [`tests/`](tests/)
- **Specifications:** [`specs/`](specs/)
- **ML Modules:** [`module/`](module/)
- **Orchestrator:** [`src/orchestrator.py`](src/orchestrator.py)
- **Training App:** [`application_training.py`](application_training.py)

**Get started in 3 commands:**
```bash
poetry install                          # 1. Setup
python data/generate_database.py       # 2. Data
python application_training.py          # 3. Train
```

Happy ML Engineering! 🎯
