# Orchestrator Guide

**ML workflow coordination**

---

## Overview

The `src/orchestrator.py` is the central coordinator that orchestrates preprocessing, training, tuning, calibration, validation, and MLflow tracking.

### Design

- **Single Responsibility** - Coordinate workflow steps
- **Dependency Injection** - Receives components via specs
- **Error Handling** - Graceful degradation for optional steps

---

## Core Functions

### build_ml_pipeline()

**Purpose:** Construct sklearn pipeline from specs

```python
from src.orchestrator import build_ml_pipeline

pipeline = build_ml_pipeline(
    feature_specs=feature_specs,
    model_spec=model_spec
)
# Returns: sklearn Pipeline with preprocessing + model
```

### run_ml_workflow()

**Purpose:** Execute complete ML workflow

```python
from src.orchestrator import run_ml_workflow

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y,
    validation_strategy="cross_validation",
    validation_params={"cv_folds": 5},
    tuning_spec=tuning_spec,        
    calibration_spec=calib_spec,    
    mlflow_spec=mlflow_spec,        
    feature_store_spec=fs_spec     
)
```

**Returns:**
- `pipeline` - Trained pipeline
- `cv_score` - Cross-validation score
- `tuning_summary` - Best params/scores (if tuning enabled)
- `calibration_summary` - Calibration info (if enabled)
- `mlflow_run_id` - Run ID (if MLflow enabled)

### get_workflow_summary()

**Purpose:** Format results for display

```python
from src.orchestrator import get_workflow_summary

summary = get_workflow_summary(results)
print(summary)
```

---

## Workflow Execution

### Step Order

1. **Data Loading** - From Delta Lake or feature store
2. **Pipeline Construction** - Build preprocessing + model pipeline
3. **Parameter Tuning** (optional) - Optimize hyperparameters
4. **Training** - Fit pipeline on data
5. **Calibration** (optional) - Calibrate probabilities
6. **Validation** - Cross-validation or train-test split
7. **MLflow Logging** (optional) - Track experiment
8. **Return Results** - Pipeline and metrics

### Validation Strategies

**Cross-Validation:**
```python
validation_strategy="cross_validation"
validation_params={"cv_folds": 5}
```

**Train-Test Split:**
```python
validation_strategy="train_test"
validation_params={"test_size": 0.2, "random_state": 42}
```

**No Validation:**
```python
validation_strategy="none"
```

---

## Integration Points

### With Feature Store

```python
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=None,  
    y=None,
    feature_store_spec=feature_store_spec
)
```

### With MLflow

```python
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y,
    mlflow_spec=mlflow_spec  
)
```

---

## Error Handling

Optional steps fail gracefully:
- Tuning fails → Continues with default hyperparameters
- Calibration fails → Uses uncalibrated model
- MLflow fails → Training continues without tracking

All errors are logged with warnings.

---

**Ready for complete ML workflow orchestration!** 🎯
