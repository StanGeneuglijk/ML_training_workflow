# Specs Guide

**Pydantic specifications for ML workflow configuration**

---

## Overview

The `specs/` folder contains Pydantic-based specifications providing type-safe, validated configuration for all ML components.

### Key Features

- **Type Safety** - Runtime validation
- **Builder Pattern** - Fluent API for configuration
- **Immutability** - Validated configs can't be corrupted

### Specs Structure

```
specs/
├── feature_spec.py            # Feature preprocessing config
├── model_spec.py              # Model configuration
├── params_tuning_spec.py      # Tuning configuration
├── calibration_spec.py        # Calibration config
├── mlflow_spec.py             # MLflow tracking config
└── feature_store_spec.py      # Feature store config
```

---

## Architecture

### Why Pydantic Specs?

**Traditional (error-prone):**
```python
def train_model(hyperparameters: dict):  
    ...
```

**Spec-driven (validated):**
```python
class ModelSpec(BaseModel):
    algorithm: Literal["gradient_boosting"] 
    hyperparameters: Dict[str, Any]

def train_model(model_spec: ModelSpec):
    # Guaranteed valid!
    ...
```

### Builder Pattern

```python
spec = ModelSpecBuilder()\
    .add_classifier("my_model")\
    .set_hyperparameters({"n_estimators": 100})\
    .build()  
```

---

## Core Specs

### Feature Spec

**Purpose:** Configure feature preprocessing

```python
from specs_training import FeatureSpecBuilder

feature_specs = (
    FeatureSpecBuilder()
    .add_numeric_group(
        feature_names=["age", "income"],
        imputer_strategy="mean",
        scaler_type="standard"
    )
    .add_categorical_group(
        feature_names=["category"],
        encoder_type="one_hot"
    )
    .build()
)
```

### Model Spec

**Purpose:** Configure ML model

```python
from specs_training import ModelSpecBuilder

model_spec = (
    ModelSpecBuilder()
    .add_classifier(
        model_name="gb_model",
        hyperparameters={"n_estimators": 100}
    )
    .build()[0]
)
```

### Tuning Spec

**Purpose:** Configure hyperparameter tuning

```python
from specs_training import GridSearchSpecBuilder

tuning_spec = (
    GridSearchSpecBuilder()
    .set_param_grid({
        "classifier__n_estimators": [50, 100, 200]
    })
    .set_cv_folds(5)
    .build()
)
```

### Calibration Spec

**Purpose:** Configure probability calibration

```python
from specs_training import CalibrationSpecBuilder

calib_spec = (
    CalibrationSpecBuilder()
    .set_method("sigmoid")
    .set_cv_strategy(5)
    .build()
)
```

### MLflow Spec

**Purpose:** Configure experiment tracking

```python
from specs_training import MLflowSpecBuilder

mlflow_spec = (
    MLflowSpecBuilder()
    .enable()
    .set_experiment("my_experiment")
    .enable_model_registry(stage="Staging")
    .build()
)
```

### Feature Store Spec

**Purpose:** Configure feature store

```python
from specs_training import FeatureStoreSpecBuilder

feature_store_spec = (
    FeatureStoreSpecBuilder()
    .enable()
    .set_dataset_name("classification_data")
    .set_n_features(20)
    .build()
)
```

---

## Usage Pattern

```python
# 1. Create specs
feature_specs = FeatureSpecBuilder().add_numeric_group(...).build()
model_spec = ModelSpecBuilder().add_classifier(...).build()[0]

# 2. Use in workflow
from src_training.orchestrator import run_ml_workflow

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y
)
```

---

## Validation

All specs validate at construction time:
- Required fields must be provided
- Field types must match
- Enum values must be valid
- Constraints must be satisfied

**Benefits:**
- Catch errors early
- Clear error messages
- No runtime surprises

---

**Ready for type-safe ML configuration!** 🎯
