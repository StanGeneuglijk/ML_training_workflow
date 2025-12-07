# Module Guide

**Core ML engineering components**

---

## Overview

The `module/` folder contains ML components with spec-driven configuration and sklearn-compatible interfaces.

### Design Principles

- **Spec-Driven** - Configuration via Pydantic specs
- **Sklearn-Compatible** - Standard fit/predict API
- **Composable** - Work together via Pipeline
- **MLflow-Integrated** - Automatic experiment tracking

### Module Structure

```
module/
├── classifier.py            # Model implementations
├── pre_processing.py        # Feature transformation
├── calibration.py           # Probability calibration
├── params_tuning.py         # Hyperparameter optimization
└── mlflow_tracker.py        # Experiment tracking
```

---

## Core Modules

### Classifier

**Purpose:** ML model implementations

```python
from module.classifier import GradientBoostingClassifierImpl
from specs import ModelSpecBuilder

model_spec = ModelSpecBuilder().add_classifier("gb_model").build()[0]
classifier = GradientBoostingClassifierImpl(model_spec)

classifier.fit(X, y)
predictions = classifier.predict(X_new)
probabilities = classifier.predict_proba(X_new)
```

### Pre-processing

**Purpose:** Feature transformation pipeline

```python
from module.pre_processing import FeatureSpecPipeline

pipeline = FeatureSpecPipeline(feature_specs)
pipeline.fit(X_train, y)
X_transformed = pipeline.transform(X_test)
```

**Creates transformers for:**
- Numeric: Imputation + Scaling
- Categorical: Imputation + Encoding

### Calibration

**Purpose:** Calibrate probabilities for better uncertainty estimates

```python
from module.calibration import ClassifierCalibration

calibrator = ClassifierCalibration(calib_spec)
calibrator.fit(pipeline, X_val, y_val)
calibrated_proba = calibrator.predict_proba(X_test)
```

**Methods:** sigmoid (Platt scaling) or isotonic

### Parameter Tuning

**Purpose:** Hyperparameter optimization

```python
from module.params_tuning import GridSearch

tuner = GridSearch(tuning_spec)
tuner.fit(pipeline, X, y)
best_pipeline = tuner.get_best_estimator()
best_params = tuner.get_best_params()
```

**Strategies:**
- GridSearch - Exhaustive search
- RandomizedSearch - Random sampling

### MLflow Tracker

**Purpose:** Experiment tracking and model registry

```python
from module.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(mlflow_spec)
tracker.start_run()
tracker.log_params({"n_estimators": 100})
tracker.log_metrics({"accuracy": 0.92})
tracker.log_model(pipeline, "model", X_sample)
tracker.end_run()
```

---

## Integration

### Complete Pipeline

```python
from sklearn.pipeline import Pipeline

# Build full pipeline
ml_pipeline = Pipeline([
    ('preprocessor', FeatureSpecPipeline(feature_specs)),
    ('classifier', GradientBoostingClassifierImpl(model_spec))
])

# Train
ml_pipeline.fit(X_train, y_train)

# Predict (preprocessing automatic)
predictions = ml_pipeline.predict(X_test)
```

### With Orchestrator

```python
from src.orchestrator import run_ml_workflow

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y,
    tuning_spec=tuning_spec,
    calibration_spec=calib_spec,
    mlflow_spec=mlflow_spec
)
```

---

## Key Features

✅ **Type-safe configuration** via Pydantic specs
✅ **Standard sklearn API** - fit, predict, transform
✅ **Pipeline composition** - Preprocessing + Model
✅ **Automatic tracking** - MLflow integration throughout
✅ **Production-ready** - Error handling, logging, validation

---

**Ready for production ML engineering!** 🎯
