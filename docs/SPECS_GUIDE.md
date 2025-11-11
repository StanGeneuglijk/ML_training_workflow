# Specs Folder Technical Guide

**Technical step-by-step breakdown of Pydantic specifications**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Specification Details](#specification-details)
4. [Integration Patterns](#integration-patterns)
5. [Best Practices](#best-practices)

---

## Overview

The `specs/` folder contains Pydantic-based specifications that define configuration for all ML workflow components. These specs provide:

- **Type Safety** - Runtime validation of all configurations
- **Documentation** - Self-documenting configuration schema
- **Immutability** - Validated configuration that can't be corrupted
- **Builder Pattern** - Fluent API for easy configuration
- **Serialization** - Easy JSON/YAML export for versioning

### Specs Structure

```
specs/
├── __init__.py                # Spec exports
├── feature_spec.py            # Feature processing config (304 lines)
├── model_spec.py              # Model configuration (193 lines)
├── params_tuning_spec.py      # Tuning configuration (243 lines)
├── calibration_spec.py        # Calibration config (158 lines)
├── mlflow_spec.py             # MLflow tracking config (431 lines)
└── feature_store_spec.py      # Feature store config (262 lines)
```

**Total: ~1,600 lines of configuration specifications**

---

## Architecture

### Why Pydantic Specs?

**Traditional approach (error-prone):**
```python
def train_model(
    model_name: str,
    algorithm: str,
    hyperparameters: dict,  # ❌ No validation!
    random_state: int = 42
):
    # What if hyperparameters = "not_a_dict"?
    # What if algorithm = "invalid_algo"?
    # Runtime errors only!
    ...
```

**Spec-driven approach (validated):**
```python
class ModelSpec(BaseModel):
    model_name: str = Field(..., min_length=1)
    algorithm: Literal["gradient_boosting", ...]  # ✅ Validated!
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    random_state: Optional[int] = Field(None, ge=0)

def train_model(model_spec: ModelSpec):
    # Guaranteed valid at this point!
    ...
```

### Design Pattern: Builder + Spec

```python
# Builder Pattern (Fluent API)
spec = ModelSpecBuilder()\
    .add_classifier("my_model")\
    .set_hyperparameters({"n_estimators": 100})\
    .set_random_state(42)\
    .build()  # Returns validated ModelSpec

# Spec (Immutable Configuration)
# Once built, cannot be corrupted
spec.model_name  # "my_model" ✅
spec.algorithm   # "gradient_boosting" ✅
```

---

## Specification Details

### Feature Spec

**File:** `specs/feature_spec.py` (304 lines)

**Purpose:** Configure feature preprocessing (imputation, scaling, encoding)

#### Class Hierarchy

```python
FeatureSpec (Abstract Base)
    ├── NumericFeatureSpec
    │   ├── imputer_strategy: "mean" | "median" | "constant"
    │   ├── scaler_type: "standard" | "minmax" | "robust" | "none"
    │   └── enabled: bool
    └── CategoricalFeatureSpec
        ├── imputer_strategy: "most_frequent" | "constant"
        ├── encoder_type: "onehot" | "ordinal" | "target" | "none"
        └── enabled: bool

FeatureSelectionSpec
    ├── selection_mode: "all" | "indices" | "names" | "range"
    ├── feature_indices: Optional[List[int]]
    ├── feature_names: Optional[List[str]]
    └── max_features: Optional[int]

FeatureSpecBuilder
    └── Fluent API for building feature specs
```

#### Step-by-Step: Feature Configuration

**Step 1: Define Numeric Features**
```python
from specs import FeatureSpecBuilder

builder = FeatureSpecBuilder()

# Add group of numeric features with same preprocessing
builder.add_numeric_group(
    feature_names=["age", "income", "credit_score"],
    imputer_strategy="mean",       # Missing values → mean
    scaler_type="standard"          # Z-score normalization
)
```

**What this creates:**
```python
[
    NumericFeatureSpec(
        feature_name="age",
        imputer_strategy="mean",
        scaler_type="standard",
        enabled=True
    ),
    NumericFeatureSpec(feature_name="income", ...),
    NumericFeatureSpec(feature_name="credit_score", ...)
]
```

**Step 2: Add Categorical Features**
```python
builder.add_categorical_group(
    feature_names=["gender", "city", "occupation"],
    imputer_strategy="most_frequent",
    encoder_type="onehot"
)
```

**Step 3: Build**
```python
feature_specs = builder.build()
# Returns: List[FeatureSpec] - validated and ready
```

#### Feature Selection Spec

**Purpose:** Select subset of features for training

```python
from specs.feature_spec import FeatureSelectionSpec

# Select specific features by name
selection = FeatureSelectionSpec(
    selection_mode="names",
    feature_names=["feature_0", "feature_5", "feature_10"]
)

# Or by index
selection = FeatureSelectionSpec(
    selection_mode="indices",
    feature_indices=[0, 5, 10, 15]
)

# Or by range
selection = FeatureSelectionSpec(
    selection_mode="range",
    feature_range=(0, 10),  # Features 0-10
    exclude_features=["feature_5"]  # Except this one
)

# Apply selection
all_features = [f"feature_{i}" for i in range(20)]
selected = selection.get_selected_features(all_features)
# Returns: ["feature_0", "feature_1", ..., "feature_10"]  (excluding feature_5)
```

#### Validation Rules

**Numeric Features:**
- `imputer_strategy="constant"` requires `imputer_fill_value` to be set
- `scaler_type` must be one of the allowed values
- Feature name cannot be empty

**Categorical Features:**
- `imputer_strategy="constant"` requires `imputer_fill_value` to be set
- `encoder_type` must be valid encoder
- Feature name cannot be empty

**Pydantic ensures these at construction time!**

---

### Model Spec

**File:** `specs/model_spec.py` (193 lines)

**Purpose:** Configure ML model selection and hyperparameters

#### Class Hierarchy

```python
ModelSpec (Abstract Base)
    └── ClassifierModelSpec
        ├── model_name: str
        ├── algorithm: Literal["gradient_boosting"]
        ├── hyperparameters: Dict[str, Any]
        ├── random_state: Optional[int]
        └── evaluation_metrics: List[str]

ModelSpecBuilder
    └── Fluent API for building model specs
```

#### Step-by-Step: Model Configuration

**Step 1: Create Builder**
```python
from specs import ModelSpecBuilder

builder = ModelSpecBuilder()
```

**Step 2: Add Classifier**
```python
builder.add_classifier(
    model_name="gradient_boost_v1",
    hyperparameters={
        "n_estimators": 200,
        "learning_rate": 0.05,
        "max_depth": 5,
        "min_samples_split": 10
    },
    random_state=42
)
```

**What this creates:**
```python
ClassifierModelSpec(
    model_name="gradient_boost_v1",
    algorithm="gradient_boosting",  # Inferred
    hyperparameters={"n_estimators": 200, ...},
    random_state=42,
    evaluation_metrics=["accuracy", "roc_auc", "f1_score"]  # Defaults
)
```

**Step 3: Build**
```python
model_specs = builder.build()
# Returns: List[ModelSpec]

# Usually just one model:
model_spec = model_specs[0]
```

#### Validation Rules

- `model_name` must be non-empty string
- `algorithm` must be supported ("gradient_boosting")
- `hyperparameters` must be dict
- `random_state` must be >= 0 if provided
- `evaluation_metrics` must be list of strings

#### Usage with Classifier

```python
from module.classifier import GradientBoostingClassifierImpl

# Spec → Classifier
classifier = GradientBoostingClassifierImpl(model_spec)

# Spec values used to configure sklearn model
classifier.fit(X, y)
# Internally creates: GradientBoostingClassifier(**model_spec.hyperparameters)
```

---

### Parameter Tuning Spec

**File:** `specs/params_tuning_spec.py` (243 lines)

**Purpose:** Configure hyperparameter optimization

#### Class Hierarchy

```python
ParamTuningSpec (Abstract Base)
    ├── tuning_name: str
    ├── scoring: str | Dict[str, str]
    ├── cv_folds: int
    └── n_jobs: int

GridSearchSpec(ParamTuningSpec)
    └── param_grid: Dict[str, List[Any]]

RandomizedSearchSpec(ParamTuningSpec)
    ├── param_distributions: Dict[str, Any]
    └── n_iter: int

Builders:
    ├── GridSearchSpecBuilder
    └── RandomizedSearchSpecBuilder
```

#### Step-by-Step: Grid Search Configuration

**Step 1: Define Parameter Grid**
```python
from specs import GridSearchSpecBuilder

spec = GridSearchSpecBuilder()\
    .set_name("gb_tuning")\
    .set_param_grid({
        "classifier__n_estimators": [50, 100, 200],
        "classifier__learning_rate": [0.01, 0.1],
        "classifier__max_depth": [3, 5, 7]
    })
```

**Step 2: Configure Search Strategy**
```python
spec.set_cv_folds(5)\           # 5-fold cross-validation
    .set_scoring("accuracy")\    # Optimization metric
    .set_refit_score("accuracy")\  # Refit best model
    .set_n_jobs(-1)\             # Use all CPUs
    .set_verbose(1)              # Progress output
```

**Step 3: Build**
```python
tuning_spec = spec.build()
```

**What you get:**
```python
GridSearchSpec(
    tuning_name="gb_tuning",
    param_grid={"classifier__n_estimators": [50, 100, 200], ...},
    scoring="accuracy",
    refit_score="accuracy",
    cv_folds=5,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)
```

#### Multi-Metric Optimization

```python
spec = GridSearchSpecBuilder()\
    .set_scoring({
        'accuracy': 'accuracy',
        'f1': 'f1_weighted',
        'roc_auc': 'roc_auc'
    })\
    .set_refit_score('f1')  # Optimize for F1, track all 3\
    .build()
```

**Result:** All metrics logged, model optimized for F1

#### Validation Rules

- `param_grid` must be non-empty dict
- All parameter names must be strings
- `cv_folds` must be >= 2
- `n_jobs` must be >= -1
- `scoring` must be valid sklearn scorer

---

### Calibration Spec

**File:** `specs/calibration_spec.py` (158 lines)

**Purpose:** Configure probability calibration

#### Class Structure

```python
CalibrationSpec (Abstract Base)
    └── ClassifierCalibrationSpec
        ├── calibration_name: str
        ├── method: Literal["sigmoid", "isotonic"]
        ├── cv_strategy: int | Literal["prefit"]
        └── ensemble: bool

CalibrationSpecBuilder
    └── Fluent API
```

#### Step-by-Step: Calibration Configuration

**Step 1: Basic Calibration**
```python
from specs import CalibrationSpecBuilder

calib_spec = CalibrationSpecBuilder()\
    .set_name("probability_calibration")\
    .set_method("sigmoid")\    # Platt scaling
    .set_cv_strategy(5)\        # 5-fold CV
    .build()
```

**Step 2: Advanced Configuration**
```python
calib_spec = CalibrationSpecBuilder()\
    .set_method("isotonic")\    # More flexible, needs more data
    .set_cv_strategy(10)\        # 10-fold CV for stability
    .enable_ensemble()\          # Average all folds
    .build()
```

#### Calibration Methods

**Sigmoid (Platt Scaling):**
- Fits logistic regression: `P(y=1) = 1 / (1 + exp(A*f + B))`
- Good for: Small datasets, near-linear probability relationship
- Fast to compute

**Isotonic:**
- Non-parametric: fits stepwise-constant function
- Good for: Larger datasets, any probability relationship
- More flexible but needs 100+ samples

#### CV Strategies

**Integer (e.g., 5):**
- Creates N calibrated models via cross-validation
- Each trained on different fold
- If `ensemble=True`: averages predictions from all N
- If `ensemble=False`: uses best fold

**"prefit" (deprecated):**
- Assumes model already fitted
- Single calibrator on validation set
- Faster but less robust

---

### MLflow Spec

**File:** `specs/mlflow_spec.py` (431 lines)

**Purpose:** Configure experiment tracking and model registry

#### Class Structure

```python
MLflowSpec
    ├── enabled: bool
    ├── experiment_name: str
    ├── run_name: Optional[str]
    ├── tracking_uri: Optional[str]
    ├── artifact_location: Optional[str]
    ├── register_model: bool
    ├── model_stage: Literal["None", "Staging", "Production", "Archived"]
    ├── tags: Dict[str, str]
    ├── description: Optional[str]
    ├── log_model_signature: bool
    ├── log_input_example: bool
    └── log_artifacts: bool

MLflowSpecBuilder
    └── Fluent API with 30+ configuration methods
```

#### Step-by-Step: MLflow Configuration

**Step 1: Basic Experiment Tracking**
```python
from specs import MLflowSpecBuilder

mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("classification_experiments")\
    .set_run_name("gb_baseline_v1")\
    .build()
```

**Step 2: Add Model Registry**
```python
mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("classification_experiments")\
    .enable_model_registry(stage="Staging")\  # Auto-register
    .set_registered_model_name("gb_classifier")\
    .build()
```

**Step 3: Add Metadata**
```python
mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("classification_experiments")\
    .add_tags({
        "team": "ml-engineering",
        "model_version": "2.0",
        "dataset": "production_data",
        "environment": "staging"
    })\
    .set_description("Gradient boosting with isotonic calibration")\
    .build()
```

**Step 4: Configure Logging**
```python
mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .enable_model_signature()\      # Log input/output schema
    .enable_input_example()\         # Log sample inputs
    .enable_artifact_logging()\      # Log plots, data
    .build()
```

#### Helper Methods

```python
# Check if should register
if mlflow_spec.should_register_model():
    register_to_registry()

# Check if should log artifacts
if mlflow_spec.should_log_artifacts():
    log_plots_and_data()

# Get full tracking URI
uri = mlflow_spec.get_tracking_uri()  # With defaults
```

#### Validation Rules

- `experiment_name` cannot be empty if enabled
- `model_stage` must be valid stage name
- `tags` must be dict with string values
- `tracking_uri` validated as proper URI if provided
- `registered_model_name` auto-generated if not provided

---

### Feature Store Spec

**File:** `specs/feature_store_spec.py` (262 lines)

**Purpose:** Configure Feast feature store integration

#### Class Structure

```python
FeatureStoreSpec
    ├── enabled: bool
    ├── repo_path: str
    ├── n_features: int
    ├── initialize_on_start: bool
    ├── force_recreate: bool
    ├── sample_indices: Optional[List[int]]
    └── timestamp: Optional[datetime]

FeatureStoreSpecBuilder
    └── Fluent API for feature store config
```

#### Step-by-Step: Feature Store Configuration

**Step 1: Basic Setup**
```python
from specs import FeatureStoreSpecBuilder

fs_spec = FeatureStoreSpecBuilder()\
    .enable()\
    .set_repo_path("feature_repo")\
    .set_n_features(20)\
    .build()
```

**Step 2: Advanced Configuration**
```python
fs_spec = FeatureStoreSpecBuilder()\
    .enable()\
    .set_repo_path("feature_repo")\
    .set_n_features(20)\
    .set_initialize_on_start(True)\   # Auto-initialize
    .set_force_recreate(False)\        # Reuse if exists
    .build()
```

**Step 3: Sample Filtering**
```python
from datetime import datetime

fs_spec = FeatureStoreSpecBuilder()\
    .enable()\
    .set_sample_indices([0, 10, 20, 30])\  # Only these samples
    .set_timestamp(datetime(2025, 1, 1))\   # Point-in-time
    .build()
```

#### Helper Methods

```python
# Check if should initialize
if fs_spec.should_initialize():
    initialize_feature_store()

# Get feature references for Feast
feature_refs = fs_spec.get_feature_references()
# Returns: ["classification_features:feature_0", ...]
```

---

## Integration Patterns

### Pattern 1: End-to-End Workflow

```python
from specs import (
    FeatureSpecBuilder,
    ModelSpecBuilder,
    GridSearchSpecBuilder,
    CalibrationSpecBuilder,
    MLflowSpecBuilder
)
from src.orchestrator import run_ml_workflow

# 1. Configure features
feature_specs = FeatureSpecBuilder()\
    .add_numeric_group([f"feature_{i}" for i in range(20)])\
    .build()

# 2. Configure model
model_spec = ModelSpecBuilder()\
    .add_classifier("gb_model")\
    .build()[0]

# 3. Configure tuning
tuning_spec = GridSearchSpecBuilder()\
    .set_param_grid({"classifier__n_estimators": [50, 100, 200]})\
    .build()

# 4. Configure calibration
calib_spec = CalibrationSpecBuilder()\
    .set_method("sigmoid")\
    .set_cv_strategy(5)\
    .build()

# 5. Configure MLflow
mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("my_experiment")\
    .enable_model_registry()\
    .build()

# 6. Run workflow
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X_train,
    y=y_train,
    tuning_spec=tuning_spec,
    calibration_spec=calib_spec,
    mlflow_spec=mlflow_spec
)
```

### Pattern 2: Spec Serialization

**Save configuration:**
```python
# Specs are Pydantic models - easy to serialize
import json

config = {
    "model": model_spec.model_dump(),
    "features": [fs.model_dump() for fs in feature_specs],
    "mlflow": mlflow_spec.model_dump()
}

with open("config.json", "w") as f:
    json.dump(config, f, indent=2)
```

**Load configuration:**
```python
with open("config.json") as f:
    config = json.load(f)

model_spec = ClassifierModelSpec(**config["model"])
feature_specs = [NumericFeatureSpec(**fs) for fs in config["features"]]
mlflow_spec = MLflowSpec(**config["mlflow"])
```

### Pattern 3: Configuration Inheritance

```python
# Base configuration
base_mlflow = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("classification")\
    .add_tags({"team": "ml-eng"})

# Experiment variations
experiment_1 = base_mlflow\
    .set_run_name("experiment_1")\
    .add_tags({"variant": "baseline"})\
    .build()

experiment_2 = base_mlflow\
    .set_run_name("experiment_2")\
    .add_tags({"variant": "tuned"})\
    .build()
```

---

## Best Practices

### 1. Always Validate Early

**✅ Do:**
```python
try:
    spec = ModelSpecBuilder().add_classifier("").build()
except ValidationError as e:
    print(f"Invalid configuration: {e}")
    # Fix config before running workflow
```

**❌ Don't:**
```python
# Wait for runtime error during training
config = {"model_name": ""}  # Invalid!
# ... hours of training later ...
# Error: model_name cannot be empty
```

### 2. Use Builders, Not Direct Construction

**✅ Do:**
```python
spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("my_exp")\
    .build()
```

**❌ Don't:**
```python
spec = MLflowSpec(
    enabled=True,
    experiment_name="my_exp",
    run_name=None,
    tracking_uri=None,
    # ... 20 more fields
)
```

**Why:** Builders provide defaults and validation

### 3. Version Your Configurations

```python
# Save specs with model for reproducibility
config = {
    "version": "1.0",
    "date": "2025-11-08",
    "model_spec": model_spec.model_dump(),
    "feature_specs": [fs.model_dump() for fs in feature_specs],
    "tuning_spec": tuning_spec.model_dump()
}

# Can recreate exact training run later
```

### 4. Use Type Hints

**✅ Do:**
```python
def train_model(
    model_spec: ClassifierModelSpec,
    feature_specs: List[FeatureSpec],
    mlflow_spec: MLflowSpec
) -> Pipeline:
    # Type checker ensures correct specs
    ...
```

**Benefits:**
- IDE autocomplete
- Static type checking
- Self-documenting code

### 5. Document Configuration Choices

```python
# Add metadata explaining why
model_spec = ModelSpecBuilder()\
    .add_classifier(
        "gb_production",
        hyperparameters={
            "n_estimators": 200,  # Balanced accuracy vs speed
            "learning_rate": 0.05,  # Slower learning = better generalization
            "max_depth": 5  # Prevent overfitting on this dataset
        }
    )\
    .build()[0]

model_spec.description = "Tuned for production: accuracy=0.92, latency<100ms"
```

---

## Spec Comparison

| Spec | Purpose | Key Fields | Builder Methods |
|------|---------|------------|-----------------|
| **FeatureSpec** | Feature preprocessing | imputer, scaler, encoder | add_numeric_group, add_categorical_group |
| **ModelSpec** | Model configuration | algorithm, hyperparameters | add_classifier |
| **ParamTuningSpec** | Hyperparameter search | param_grid, cv_folds | set_param_grid, set_cv_folds |
| **CalibrationSpec** | Probability calibration | method, cv_strategy | set_method, set_cv_strategy |
| **MLflowSpec** | Experiment tracking | experiment, tags, registry | set_experiment, add_tags, enable_model_registry |
| **FeatureStoreSpec** | Feature store config | repo_path, n_features | set_repo_path, set_n_features |

---

## Summary

### Why Specs?

**Benefits:**
1. ✅ **Type safety** - Catch errors before training
2. ✅ **Documentation** - Self-documenting configuration
3. ✅ **Validation** - Pydantic validates all values
4. ✅ **Serialization** - Easy to save/load/version
5. ✅ **Separation** - Config separate from implementation
6. ✅ **Testability** - Easy to test with different configs

### Spec Workflow

```
1. Create Builder → 2. Configure → 3. Validate → 4. Build → 5. Use
                         ↓              ↓           ↓          ↓
                    Fluent API    Pydantic    Immutable   ML Module
```

### Code Organization

```
specs/          ← Configuration layer (WHAT to do)
    ↓
module/         ← Implementation layer (HOW to do it)
    ↓
src/orchestrator ← Workflow layer (WHEN to do it)
```

**This separation enables:**
- Change configuration without touching code
- Reuse modules with different configs
- Test modules with mock specs
- Version control configuration separately

---

## Next Steps

- **Module Guide**: See `MODULE_GUIDE.md` for module implementations
- **Orchestrator Guide**: See `ORCHESTRATOR_GUIDE.md` for workflow orchestration
- **Feature Store**: See `FEATURE_STORE.md` for feature management
- **Data Loading**: See `DATA_GUIDE.md` for data ingestion

All specs validated with **207 passing tests** and production-ready! 🚀

