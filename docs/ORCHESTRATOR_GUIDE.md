# Orchestrator Guide

**Technical step-by-step breakdown of the ML workflow orchestrator**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Functions](#core-functions)
4. [Workflow Execution](#workflow-execution)
5. [Integration Points](#integration-points)
6. [Advanced Usage](#advanced-usage)

---

## Overview

The `src/orchestrator.py` (336 lines) is the **central coordinator** of the ML workflow. It orchestrates the execution of:

- Feature preprocessing
- Model training  
- Parameter tuning
- Calibration
- Validation
- MLflow tracking
- Feature store integration

### Design Philosophy

**Single Responsibility:** Coordinate workflow steps in correct order

**Dependency Injection:** Receives all components via specs (testable!)

**Error Handling:** Graceful degradation if optional steps fail

**Logging:** Comprehensive logging of each step

---

## Architecture

### Workflow Layers

```
Application Layer (application_training.py)
    ↓ calls
Orchestrator Layer (src/orchestrator.py)
    ↓ coordinates
Module Layer (module/*)
    ├── FeatureSpecPipeline (preprocessing)
    ├── GradientBoostingClassifierImpl (model)
    ├── GridSearch (tuning)
    ├── ClassifierCalibration (calibration)
    └── MLflowTracker (tracking)
```

### Key Functions

```python
src/orchestrator.py
├── build_ml_pipeline()        # Construct sklearn pipeline
├── run_ml_workflow()          # Execute complete workflow
└── get_workflow_summary()     # Format results for display
```

---

## Core Functions

### build_ml_pipeline()

**Purpose:** Construct sklearn-compatible pipeline from specs

**Signature:**
```python
def build_ml_pipeline(
    feature_specs: List[FeatureSpec],
    model_spec: ClassifierModelSpec
) -> Pipeline:
    """
    Build sklearn pipeline from specifications.
    
    Args:
        feature_specs: Feature processing configurations
        model_spec: Model configuration
        
    Returns:
        Sklearn Pipeline with preprocessing + model
    """
```

**Step-by-Step Execution:**

1. **Create Preprocessor**
   ```python
   preprocessor = FeatureSpecPipeline(feature_specs)
   ```
   - Validates feature_specs is list
   - Creates transformer pipeline
   - Ready to fit on data

2. **Create Classifier**
   ```python
   classifier = GradientBoostingClassifierImpl(model_spec)
   ```
   - Validates model_spec
   - Creates classifier wrapper
   - Ready to fit on transformed features

3. **Combine into Pipeline**
   ```python
   pipeline = Pipeline([
       ('preprocessor', preprocessor),
       ('classifier', classifier)
   ])
   ```
   - Named steps for parameter tuning
   - Sequential execution: preprocess → classify
   - Single fit/predict interface

4. **Return Pipeline**
   - Unfitted pipeline
   - Ready for training or tuning

**Example:**
```python
pipeline = build_ml_pipeline(feature_specs, model_spec)
pipeline.fit(X_train, y_train)  # Fits both steps
predictions = pipeline.predict(X_test)  # Transforms then predicts
```

---

### run_ml_workflow()

**Purpose:** Execute complete ML workflow with all optional components

**Signature:**
```python
def run_ml_workflow(
    feature_specs: List[FeatureSpec],
    model_spec: ClassifierModelSpec,
    X: Any,  # Feature matrix or None (if using feature store)
    y: Any,  # Target vector or None (if using feature store)
    validation_strategy: str = "cross_validation",
    validation_params: Optional[Dict] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    X_test: Optional[Any] = None,
    y_test: Optional[Any] = None,
    tuning_spec: Optional[ParamTuningSpec] = None,
    calibration_spec: Optional[CalibrationSpec] = None,
    mlflow_spec: Optional[MLflowSpec] = None,
    feature_store_spec: Optional[FeatureStoreSpec] = None,
) -> Dict[str, Any]:
    """
    Run complete ML workflow.
    
    Returns:
        Dictionary with pipeline, metrics, and tracking info
    """
```

**Step-by-Step Execution:**

#### Phase 1: Initialization

**1.1 Start MLflow Tracking (if enabled)**
```python
if mlflow_spec and mlflow_spec.enabled:
    mlflow_tracker = create_mlflow_tracker(mlflow_spec, model_spec)
    mlflow_tracker.start_run()
    logger.info("Started MLflow run: %s", mlflow_tracker.run_id)
```

**1.2 Load Features from Feature Store (if enabled)**
```python
if feature_store_spec and feature_store_spec.enabled:
    feast_manager = create_feast_manager(
        repo_path=feature_store_spec.repo_path,
        n_features=feature_store_spec.n_features,
        initialize=feature_store_spec.should_initialize(),
        force_recreate=feature_store_spec.force_recreate
    )
    
    X, y = feast_manager.get_training_data(
        sample_indices=feature_store_spec.sample_indices,
        timestamp=feature_store_spec.timestamp
    )
    logger.info("Retrieved features from feature store")
```

**1.3 Validate Data**
```python
if X is None or y is None:
    raise ValueError("No data provided and feature store not configured")

X_array = np.asarray(X)
y_array = np.asarray(y)
logger.info("Data shape: X=%s, y=%s", X_array.shape, y_array.shape)
```

#### Phase 2: Pipeline Construction

**2.1 Build Pipeline**
```python
pipeline = build_ml_pipeline(feature_specs, model_spec)
logger.info("Built ML pipeline: %s", model_spec.model_name)
```

**2.2 Log Initial Parameters to MLflow**
```python
if mlflow_tracker:
    params = {
        "model_name": model_spec.model_name,
        "algorithm": model_spec.algorithm,
        **model_spec.hyperparameters,
        "n_features": len(feature_specs),
        "n_samples": len(X_array)
    }
    mlflow_tracker.log_params(params)
```

#### Phase 3: Parameter Tuning (Optional)

**3.1 Run Tuning**
```python
if tuning_spec and tuning_spec.enabled:
    try:
        logger.info("Running parameter tuning: %s", tuning_spec.tuning_name)
        
        tuning = create_tuning(tuning_spec)
        tuning.fit(pipeline, X_array, y_array)
        
        results['tuning_summary'] = tuning.get_results_summary()
        pipeline = tuning.get_best_estimator()
        results['pipeline'] = pipeline
        
        logger.info("Tuning completed: best score=%.4f", 
                   results['tuning_summary']['best_score'])
    except Exception as e:
        logger.warning("Tuning failed (%s). Proceeding without tuning.", e)
```

**What happens:**
- Runs GridSearch or RandomizedSearch
- Tests multiple hyperparameter combinations
- Selects best based on CV score
- Replaces pipeline with best estimator
- Gracefully continues if tuning fails

**3.2 Log Tuning Results**
```python
if mlflow_tracker and 'tuning_summary' in results:
    mlflow_tracker.log_params(results['tuning_summary']['best_params'])
    mlflow_tracker.log_metrics({
        "tuning_best_score": results['tuning_summary']['best_score']
    })
```

#### Phase 4: Model Training

**4.1 Fit Pipeline**
```python
logger.info("Fitting pipeline on training data")
pipeline.fit(X_array, y_array)
```

**What happens:**
- Fits preprocessor on X_array
- Fits classifier on transformed features
- Both steps executed in sequence
- Pipeline now ready for predictions

**Note:** If tuning was run, best estimator is already fitted!

#### Phase 5: Calibration (Optional)

**5.1 Run Calibration**
```python
if calibration_spec and calibration_spec.enabled:
    try:
        logger.info("Running calibration: %s", calibration_spec.calibration_name)
        
        calibration = create_calibration(calibration_spec)
        calibration.fit(pipeline, X_array, y_array)
        
        results['calibration_summary'] = calibration.get_results_summary()
        pipeline = calibration.calibrated_
        results['pipeline'] = pipeline
        
        logger.info("Calibration completed")
    except Exception as e:
        logger.warning("Calibration failed (%s). Proceeding without.", e)
```

**What happens:**
- Wraps pipeline in CalibratedClassifierCV
- Fits calibrators via cross-validation
- Replaces pipeline with calibrated version
- Gracefully continues if calibration fails

#### Phase 6: Validation

**6.1 Cross-Validation**
```python
if validation_strategy == "cross_validation":
    cv_folds = validation_params.get('cv_folds', 5)
    
    scores = cross_val_score(
        pipeline,
        X_array,
        y_array,
        cv=cv_folds,
        scoring='accuracy'
    )
    
    results['cv_score'] = float(scores.mean())
    results['cv_std'] = float(scores.std())
    results['validation_strategy'] = 'cross_validation'
    results['cv_folds'] = cv_folds
    
    logger.info("CV Accuracy: %.4f ± %.4f", results['cv_score'], results['cv_std'])
```

**6.2 Train-Test Split**
```python
elif validation_strategy == "train_test":
    X_train, X_val, y_train, y_val = train_test_split(
        X_array, y_array,
        test_size=test_size,
        random_state=random_state
    )
    
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_val, y_val)
    
    results['val_score'] = float(score)
    results['validation_strategy'] = 'train_test'
```

**6.3 No Validation**
```python
elif validation_strategy == "none":
    results['validation_strategy'] = 'none'
    logger.info("Skipping validation")
```

#### Phase 7: Test Set Evaluation (Optional)

**7.1 If Test Set Provided**
```python
if X_test is not None and y_test is not None:
    X_test_array = np.asarray(X_test)
    y_test_array = np.asarray(y_test)
    
    test_predictions = pipeline.predict(X_test_array)
    test_accuracy = (test_predictions == y_test_array).mean()
    
    results['test_predictions'] = test_predictions
    results['test_accuracy'] = float(test_accuracy)
    
    logger.info("Test Accuracy: %.4f", test_accuracy)
```

#### Phase 8: MLflow Logging

**8.1 Log All Results**
```python
if mlflow_tracker and mlflow_spec:
    # Prepare sample for model signature
    X_sample = X_array[:min(100, len(X_array))] if mlflow_spec.log_model_signature else None
    
    # Log everything
    mlflow_tracker.log_workflow_results(
        results=results,
        X_sample=X_sample,
        register_model=mlflow_spec.should_register_model()
    )
    
    # Store MLflow info in results
    results['mlflow_run_id'] = mlflow_tracker.run_id
    results['mlflow_experiment_id'] = mlflow_tracker.experiment_id
    
    # End run
    mlflow_tracker.end_run(status="FINISHED")
    logger.info("MLflow tracking completed successfully")
```

**What gets logged:**
- All metrics (cv_score, test_accuracy, etc.)
- All hyperparameters
- Model architecture
- Model artifacts
- Feature importance plots
- Model signature

#### Phase 9: Return Results

**9.1 Package Results**
```python
logger.info("ML workflow completed successfully")
return results
```

**Results dictionary contains:**
```python
{
    'pipeline': trained_pipeline,          # Fitted pipeline
    'cv_score': 0.92,                     # Validation score
    'cv_std': 0.02,                       # Score std dev
    'validation_strategy': 'cross_validation',
    'tuning_summary': {...},              # If tuning enabled
    'calibration_summary': {...},         # If calibration enabled
    'mlflow_run_id': 'abc123...',         # If MLflow enabled
    'mlflow_experiment_id': '12345',
    'feature_store_enabled': True/False
}
```

---

## Workflow Execution Flow

### Complete Execution Sequence

```
1. START WORKFLOW
    ↓
2. INITIALIZE MLFLOW (if enabled)
    ↓
3. LOAD DATA
    ├── From feature store (if enabled)
    └── From provided X, y
    ↓
4. BUILD PIPELINE
    ├── Create preprocessor from feature_specs
    ├── Create classifier from model_spec
    └── Combine into sklearn Pipeline
    ↓
5. LOG INITIAL PARAMS (if MLflow enabled)
    ↓
6. PARAMETER TUNING (if enabled)
    ├── Run GridSearch/RandomizedSearch
    ├── Get best estimator
    ├── Replace pipeline with best
    └── Log tuning results
    ↓
7. TRAIN PIPELINE
    ├── Fit preprocessor
    ├── Fit classifier
    └── Pipeline ready for predictions
    ↓
8. CALIBRATION (if enabled)
    ├── Wrap in CalibratedClassifierCV
    ├── Fit calibrators
    ├── Replace pipeline with calibrated
    └── Log calibration results
    ↓
9. VALIDATION
    ├── Cross-validation OR
    ├── Train-test split OR
    ├── No validation
    └── Log scores
    ↓
10. TEST SET EVALUATION (if provided)
    ├── Predict on test set
    ├── Calculate metrics
    └── Log test scores
    ↓
11. MLFLOW LOGGING (if enabled)
    ├── Log all metrics
    ├── Log model artifacts
    ├── Register in model registry
    └── End run
    ↓
12. RETURN RESULTS
    └── Dictionary with pipeline, metrics, tracking info
```

---

## Integration Points

### With Feature Store

```python
# Orchestrator handles feature loading
if feature_store_spec and feature_store_spec.enabled:
    feast_manager = create_feast_manager(...)
    X, y = feast_manager.get_training_data(...)
else:
    # Use provided X, y
    X_array = np.asarray(X)
    y_array = np.asarray(y)
```

**Benefits:**
- Single code path for both scenarios
- Feature store transparent to downstream modules
- Can switch data sources without changing workflow

### With Parameter Tuning

```python
# Before tuning
pipeline = build_ml_pipeline(feature_specs, model_spec)
# Pipeline unfitted, with default hyperparameters

# After tuning
if tuning_spec and tuning_spec.enabled:
    tuning = create_tuning(tuning_spec)
    tuning.fit(pipeline, X, y)
    pipeline = tuning.get_best_estimator()
# Pipeline now has best hyperparameters AND is fitted!

# Don't fit again - already fitted during grid search
# Just validate
scores = cross_val_score(pipeline, X, y, cv=5)
```

**Key insight:** Tuning returns fitted estimator, skip redundant fitting

### With Calibration

```python
# After training
pipeline.fit(X_train, y_train)  # Fitted pipeline

# Calibrate
if calibration_spec and calibration_spec.enabled:
    calibration = create_calibration(calibration_spec)
    calibration.fit(pipeline, X_array, y_array)  # Uses CV to calibrate
    pipeline = calibration.calibrated_  # Wrapped in calibrator

# Calibrated pipeline ready
proba = pipeline.predict_proba(X_test)  # Calibrated probabilities!
```

### With MLflow

```python
# MLflow tracks entire workflow
mlflow_tracker = create_mlflow_tracker(mlflow_spec, model_spec)

# Start
mlflow_tracker.start_run()

# Log throughout workflow
mlflow_tracker.log_params({...})
mlflow_tracker.log_metrics({...})

# End
mlflow_tracker.log_workflow_results(results, X_sample)
mlflow_tracker.end_run(status="FINISHED")
```

---

## Advanced Usage

### Error Handling Strategy

**Graceful Degradation:**
```python
# Tuning failure doesn't stop workflow
if tuning_spec and tuning_spec.enabled:
    try:
        tuning.fit(pipeline, X, y)
        pipeline = tuning.get_best_estimator()
    except Exception as e:
        logger.warning("Tuning failed (%s). Proceeding without tuning.", e)
        # Continue with original pipeline
```

**Why:**
- Tuning might fail (timeout, insufficient data, etc.)
- Calibration might fail (too few samples)
- Workflow continues with baseline pipeline
- Logged as warning, not error

### Validation Strategies

**Strategy 1: Cross-Validation (Recommended)**
```python
results = run_ml_workflow(
    ...,
    validation_strategy="cross_validation",
    validation_params={'cv_folds': 5}
)
```

**Use when:**
- Limited data
- Need robust performance estimate
- Have time for multiple training runs

**Returns:** `cv_score`, `cv_std`

**Strategy 2: Train-Test Split**
```python
results = run_ml_workflow(
    ...,
    validation_strategy="train_test",
    test_size=0.2
)
```

**Use when:**
- Large dataset
- Faster validation needed
- Single holdout is sufficient

**Returns:** `val_score`

**Strategy 3: None (Skip Validation)**
```python
results = run_ml_workflow(
    ...,
    validation_strategy="none"
)
```

**Use when:**
- Validation done externally
- Just want to train and log
- Have separate test set

### Custom Validation Metrics

**Orchestrator computes:**
```python
# Standard metrics
'cv_score': mean accuracy across folds
'cv_std': standard deviation
'test_accuracy': accuracy on test set

# Can extend in modules:
from sklearn.metrics import precision_score, recall_score, f1_score

metrics = {
    'precision': precision_score(y_true, y_pred),
    'recall': recall_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred)
}
mlflow_tracker.log_metrics(metrics)
```

### Workflow Customization

**Extend orchestrator for custom workflows:**
```python
# Custom workflow with feature selection
def run_custom_workflow(feature_specs, model_spec, X, y):
    # 1. Build initial pipeline
    pipeline = build_ml_pipeline(feature_specs, model_spec)
    
    # 2. Feature selection
    from sklearn.feature_selection import SelectKBest, f_classif
    selector = SelectKBest(f_classif, k=10)
    X_selected = selector.fit_transform(X, y)
    
    # 3. Train on selected features
    pipeline.fit(X_selected, y)
    
    # 4. Use standard validation
    scores = cross_val_score(pipeline, X_selected, y, cv=5)
    
    return {'pipeline': pipeline, 'cv_score': scores.mean()}
```

---

## get_workflow_summary()

**Purpose:** Format results dictionary for display

**Signature:**
```python
def get_workflow_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract human-readable summary from workflow results.
    
    Args:
        results: Results dict from run_ml_workflow()
        
    Returns:
        Formatted summary dict
    """
```

**What it extracts:**
```python
{
    'model_name': model_spec.model_name,
    'algorithm': model_spec.algorithm,
    'validation_strategy': 'cross_validation',
    'cv_score': 0.92,
    'cv_std': 0.02,
    'n_features': 20,
    'tuning_enabled': True,
    'calibration_enabled': True,
    'mlflow_enabled': True
}
```

**Usage:**
```python
results = run_ml_workflow(...)
summary = get_workflow_summary(results)

print(f"Model: {summary['model_name']}")
print(f"CV Accuracy: {summary['cv_score']:.3f} ± {summary['cv_std']:.3f}")
print(f"Tuning: {'Yes' if summary.get('tuning_enabled') else 'No'}")
```

---

## Common Workflows

### Workflow 1: Quick Baseline

```python
# Minimal configuration for quick baseline
from src.orchestrator import run_ml_workflow

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X_train,
    y=y_train,
    validation_strategy="cross_validation"
)

print(f"Baseline CV: {results['cv_score']:.3f}")
```

### Workflow 2: Full Production Pipeline

```python
# Complete workflow with all bells and whistles
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X_train,
    y=y_train,
    X_test=X_test,
    y_test=y_test,
    validation_strategy="cross_validation",
    validation_params={'cv_folds': 10},
    random_state=42,
    tuning_spec=tuning_spec,           # Hyperparameter optimization
    calibration_spec=calib_spec,        # Probability calibration
    mlflow_spec=mlflow_spec,            # Experiment tracking
    feature_store_spec=None             # Direct data loading
)

# Everything tracked, tuned, calibrated!
```

### Workflow 3: With Feature Store

```python
# Data loaded from Feast feature store
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=None,  # Loaded from feature store
    y=None,  # Loaded from feature store
    validation_strategy="cross_validation",
    mlflow_spec=mlflow_spec,
    feature_store_spec=feature_store_spec  # ← Data source
)
```

### Workflow 4: Hyperparameter Search

```python
# Focus on finding best hyperparameters
tuning_spec = GridSearchSpecBuilder()\
    .set_param_grid({
        "classifier__n_estimators": [50, 100, 150, 200],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__max_depth": [3, 5, 7, 9]
    })\
    .set_cv_folds(5)\
    .set_n_jobs(-1)\  # Parallel search
    .build()

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X_train,
    y=y_train,
    tuning_spec=tuning_spec,
    validation_strategy="none"  # Skip CV, tuning does it
)

best_params = results['tuning_summary']['best_params']
best_score = results['tuning_summary']['best_score']
```

---

## Orchestrator Design Decisions

### Why Single Function Instead of Class?

**Current: Functional approach**
```python
results = run_ml_workflow(
    feature_specs=...,
    model_spec=...,
    X=X, y=y,
    tuning_spec=...,
    mlflow_spec=...
)
```

**Pros:**
- ✅ Stateless - no hidden state
- ✅ Easy to test - pure function
- ✅ Clear inputs/outputs
- ✅ Functional programming style

**Alternative: Class-based**
```python
orchestrator = MLOrchestrator(feature_specs, model_spec)
orchestrator.set_data(X, y)
orchestrator.enable_tuning(tuning_spec)
orchestrator.enable_mlflow(mlflow_spec)
results = orchestrator.run()
```

**Pros:**
- Step-by-step configuration
- Reusable across runs

**Decision:** Functional approach chosen for simplicity and testability

### Why Dependency Injection?

**All dependencies injected via parameters:**
```python
def run_ml_workflow(
    feature_specs: List[FeatureSpec],  # ← Injected
    model_spec: ClassifierModelSpec,   # ← Injected
    tuning_spec: Optional[...] = None, # ← Injected
    ...
)
```

**Benefits:**
- ✅ Easy to test with mock specs
- ✅ No hidden dependencies
- ✅ Explicit configuration
- ✅ Can swap implementations

### Why Optional Components?

**Tuning, calibration, MLflow all optional:**
```python
tuning_spec: Optional[ParamTuningSpec] = None
calibration_spec: Optional[CalibrationSpec] = None
mlflow_spec: Optional[MLflowSpec] = None
```

**Rationale:**
- Quick prototyping without setup
- Production adds tracking incrementally
- Graceful degradation if components fail
- Flexible for different use cases

---

## Summary

### Orchestrator Responsibilities

✅ **Coordinate** - Execute steps in correct order
✅ **Validate** - Check data and configurations
✅ **Handle errors** - Graceful degradation
✅ **Log** - Comprehensive logging of all steps
✅ **Track** - Integrate with MLflow
✅ **Return** - Package results for consumption

### Workflow Pattern

```
Specs → Orchestrator → Modules → Results → MLflow
  ↓          ↓            ↓         ↓         ↓
Config   Coordinate   Execute   Package   Track
```

### Why This Design?

1. **Separation of Concerns**
   - Specs: WHAT to do
   - Modules: HOW to do it
   - Orchestrator: WHEN to do it

2. **Testability**
   - Easy to test with mock specs
   - Each module tested independently
   - Integration tested via orchestrator

3. **Flexibility**
   - Enable/disable features via specs
   - Swap implementations without changing workflow
   - Add new components without refactoring

4. **Production-Ready**
   - Error handling
   - Logging
   - Experiment tracking
   - Model versioning

---

## Next Steps

- **Module Details**: See `MODULE_GUIDE.md` for module implementations
- **Spec Configuration**: See `SPECS_GUIDE.md` for all specification options
- **Feature Store**: See `FEATURE_STORE.md` for feature management
- **Application Guide**: See `APPLICATIONS_GUIDE.md` for end-to-end examples

The orchestrator is the **heart of the ML workflow** - 336 lines coordinating everything! 🎯

