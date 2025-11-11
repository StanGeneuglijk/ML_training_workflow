# Module Folder Technical Guide

**Technical step-by-step breakdown of the ML modules**

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Module Details](#module-details)
   - [Classifier](#classifier-module)
   - [Pre-processing](#pre-processing-module)
   - [Calibration](#calibration-module)
   - [Parameter Tuning](#parameter-tuning-module)
   - [MLflow Tracker](#mlflow-tracker-module)
   - [Inference](#inference-module)
4. [Integration Patterns](#integration-patterns)
5. [Best Practices](#best-practices)

---

## Overview

The `module/` folder contains the core ML engineering components that implement the actual machine learning functionality. Each module is designed with:

- **Separation of concerns** - Each module has a single, well-defined responsibility
- **Spec-driven design** - Configuration via Pydantic specifications
- **Sklearn compatibility** - Follows sklearn's fit/predict API
- **MLflow integration** - Automatic experiment tracking
- **Production-ready patterns** - Error handling, logging, validation

### Module Structure

```
module/
├── __init__.py              # Module exports
├── classifier.py            # Model implementations (154 lines)
├── pre_processing.py        # Feature transformation (201 lines)
├── calibration.py           # Probability calibration (87 lines)
├── params_tuning.py         # Hyperparameter optimization (211 lines)
├── mlflow_tracker.py        # Experiment tracking (605 lines)
└── inference.py             # Model serving and prediction (534 lines)
```

**Total: ~1,800 lines of ML engineering code**

---

## Architecture

### Design Principles

1. **Spec-Driven Configuration**
   ```python
   # Configuration via specs (validation + documentation)
   model_spec = ModelSpecBuilder().add_classifier("gb_model").build()[0]
   classifier = GradientBoostingClassifierImpl(model_spec)
   
   # NOT: Hard-coded parameters
   # classifier = GradientBoostingClassifier(n_estimators=100, ...)
   ```

2. **Sklearn-Compatible Interfaces**
   ```python
   # All modules follow sklearn's API
   classifier.fit(X, y)
   predictions = classifier.predict(X_new)
   probabilities = classifier.predict_proba(X_new)
   ```

3. **Composable Components**
   ```python
   # Modules work together via sklearn Pipeline
   pipeline = Pipeline([
       ('preprocessor', FeatureSpecPipeline(feature_specs)),
       ('classifier', GradientBoostingClassifierImpl(model_spec))
   ])
   pipeline.fit(X, y)
   ```

4. **Automatic Logging**
   ```python
   # MLflow tracking integrated throughout
   tracker = MLflowTracker(mlflow_spec)
   tracker.start_run()
   tracker.log_workflow_results(results)  # Logs metrics, params, models
   ```

---

## Module Details

### Classifier Module

**File:** `module/classifier.py` (154 lines)

**Purpose:** Implements machine learning classifiers with spec-driven configuration

#### Class Hierarchy

```python
BaseClassifier (BaseEstimator, ClassifierMixin)
    ├── Abstract base with common sklearn interface
    └── GradientBoostingClassifierImpl
        └── Concrete gradient boosting implementation
```

#### Step-by-Step: How It Works

**Step 1: Initialization**
```python
from module.classifier import GradientBoostingClassifierImpl
from specs import ModelSpecBuilder

# Create spec
model_spec = ModelSpecBuilder()\
    .add_classifier("my_model", hyperparameters={"n_estimators": 100})\
    .build()[0]

# Initialize classifier
classifier = GradientBoostingClassifierImpl(model_spec)
```

**What happens internally:**
- Validates model_spec is a ClassifierModelSpec
- Stores spec for later reference
- Initializes `self.model_` to None (lazy initialization)
- Sets `self.is_fitted_` flag to False
- Logs creation event

**Step 2: Training (fit)**
```python
classifier.fit(X_train, y_train)
```

**What happens:**
1. **Validation**: Check X and y are valid numpy arrays/DataFrames
2. **Conversion**: Convert to numpy arrays if needed
3. **Model Creation**: Create sklearn `GradientBoostingClassifier` with:
   - Base parameters (n_estimators=100, learning_rate=0.1, etc.)
   - Merge with spec.hyperparameters
   - Apply random_state if provided
4. **Training**: Call `self.model_.fit(X, y)`
5. **State Update**: Set `self.is_fitted_ = True`
6. **Logging**: Log training completion

**Step 3: Prediction (predict)**
```python
predictions = classifier.predict(X_new)
```

**What happens:**
1. Check `is_fitted_` flag
2. Convert input to numpy if needed
3. Call `self.model_.predict(X_new)`
4. Return predictions

**Step 4: Probability Prediction (predict_proba)**
```python
probabilities = classifier.predict_proba(X_new)
```

**What happens:**
1. Check `is_fitted_` flag
2. Convert input to numpy
3. Call `self.model_.predict_proba(X_new)`
4. Return probability matrix [n_samples, n_classes]

#### Key Methods

| Method | Purpose | Returns |
|--------|---------|---------|
| `__init__(model_spec)` | Initialize with configuration | None |
| `fit(X, y)` | Train the model | self |
| `predict(X)` | Make predictions | np.ndarray |
| `predict_proba(X)` | Get class probabilities | np.ndarray |
| `get_params(deep)` | Get sklearn parameters | dict |
| `set_params(**params)` | Set sklearn parameters | self |

#### Integration with Specs

```python
# model_spec contains:
model_spec.model_name          # "my_classifier"
model_spec.algorithm           # "gradient_boosting"
model_spec.hyperparameters     # {"n_estimators": 100, ...}
model_spec.random_state        # 42
model_spec.evaluation_metrics  # ["accuracy", "roc_auc"]

# Classifier uses these to configure sklearn model
```

---

### Pre-processing Module

**File:** `module/pre_processing.py` (201 lines)

**Purpose:** Feature transformation pipeline with spec-driven configuration

#### Core Classes

```python
FeatureSpecTransformerFactory
    └── Creates sklearn transformers from feature specs

FeatureSpecPipeline (BaseEstimator, TransformerMixin)
    └── Orchestrates feature transformations
```

#### Step-by-Step: How It Works

**Step 1: Create Feature Specifications**
```python
from specs import FeatureSpecBuilder

builder = FeatureSpecBuilder()
feature_specs = builder.add_numeric_group(
    feature_names=["age", "income", "score"],
    imputer_strategy="mean",
    scaler_type="standard"
).build()
```

**Step 2: Initialize Pipeline**
```python
from module.pre_processing import FeatureSpecPipeline

pipeline = FeatureSpecPipeline(feature_specs)
```

**What happens:**
- Stores feature_specs
- Initializes `transformer_` to None
- Extracts feature names from specs
- Sets up logging

**Step 3: Fit Pipeline**
```python
pipeline.fit(X_train, y)
```

**What happens internally:**

1. **Convert to DataFrame**
   ```python
   if isinstance(X, pd.DataFrame):
       X_df = X
   else:
       X_df = pd.DataFrame(X, columns=self.feature_names_)
   ```

2. **Create Transformers for Each Feature**
   ```python
   for spec in self.feature_specs:
       if spec.feature_name in X_df.columns and spec.enabled:
           transformer = FeatureSpecTransformerFactory.create_transformer(spec)
           transformers.append((spec.feature_name, transformer, [spec.feature_name]))
   ```

3. **Build ColumnTransformer**
   ```python
   self.transformer_ = ColumnTransformer(
       transformers=transformers,
       remainder='passthrough',
       verbose_feature_names_out=False
   )
   ```

4. **Fit Transformers**
   ```python
   self.transformer_.fit(X_df)
   ```

**Step 4: Transform Data**
```python
X_transformed = pipeline.transform(X_test)
```

**What happens:**
1. Convert input to DataFrame
2. Apply fitted transformers
3. Return numpy array

#### Transformer Factory

**For Numeric Features:**
```python
# Creates Pipeline with:
1. SimpleImputer (mean/median/constant)
2. StandardScaler/MinMaxScaler/RobustScaler (if enabled)

Example:
Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
```

**For Categorical Features:**
```python
# Creates Pipeline with:
1. SimpleImputer (most_frequent/constant)
2. OneHotEncoder/OrdinalEncoder (if enabled)

Example:
Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])
```

#### Integration Example

```python
# In orchestrator or training script:
from sklearn.pipeline import Pipeline

# Create full ML pipeline
ml_pipeline = Pipeline([
    ('preprocessor', FeatureSpecPipeline(feature_specs)),
    ('classifier', GradientBoostingClassifierImpl(model_spec))
])

# Train end-to-end
ml_pipeline.fit(X_train, y_train)

# Predict (preprocessing happens automatically)
predictions = ml_pipeline.predict(X_test)
```

---

### Calibration Module

**File:** `module/calibration.py` (87 lines)

**Purpose:** Calibrate predicted probabilities for better uncertainty estimates

#### Core Class

```python
ClassifierCalibration
    └── Wraps sklearn's CalibratedClassifierCV
```

#### Step-by-Step: How Calibration Works

**Step 1: Create Calibration Spec**
```python
from specs import CalibrationSpecBuilder

calib_spec = CalibrationSpecBuilder()\
    .set_name("my_calibration")\
    .set_method("sigmoid")  # or "isotonic"\
    .set_cv_strategy(5)  # 5-fold CV\
    .enable_ensemble()\
    .build()
```

**Step 2: Initialize Calibrator**
```python
from module.calibration import ClassifierCalibration

calibrator = ClassifierCalibration(calib_spec)
```

**Step 3: Calibrate Pre-trained Model**
```python
# Requires a fitted classifier or pipeline
fitted_pipeline.fit(X_train, y_train)

# Calibrate probabilities
calibrator.fit(fitted_pipeline, X_val, y_val)
```

**What happens:**
1. **Creates CalibratedClassifierCV**
   ```python
   self.calibrated_ = CalibratedClassifierCV(
       base_estimator=pipeline,
       method=spec.method,  # 'sigmoid' or 'isotonic'
       cv=spec.cv_strategy,  # int for n-folds
       ensemble=spec.ensemble  # Average calibrators?
   )
   ```

2. **Fits Calibration**
   - If cv_strategy is int (e.g., 5): Creates 5 calibrated models via CV
   - If ensemble=True: Predictions averaged from all 5
   - If ensemble=False: Uses best fold

3. **Stores State**
   - `self.calibrated_` contains calibrated model
   - `self.is_fitted_` = True

**Step 4: Get Calibrated Probabilities**
```python
calibrated_proba = calibrator.predict_proba(X_test)
```

**Result:** Better calibrated probabilities that reflect true confidence

#### Calibration Methods

**Sigmoid (Platt Scaling):**
- Fits logistic regression on classifier outputs
- Good for small datasets
- Assumes monotonic relationship

**Isotonic:**
- Non-parametric, piecewise-constant calibration
- More flexible than sigmoid
- Needs more data (100+ samples per class)

---

### Parameter Tuning Module

**File:** `module/params_tuning.py` (211 lines)

**Purpose:** Hyperparameter optimization via GridSearch or RandomizedSearch

#### Core Classes

```python
BaseTuning
    ├── Abstract base for tuning strategies
    ├── GridSearch
    │   └── Exhaustive grid search
    └── RandomizedSearch
        └── Random sampling of parameter space
```

#### Step-by-Step: Grid Search

**Step 1: Define Parameter Grid**
```python
from specs import GridSearchSpecBuilder

tuning_spec = GridSearchSpecBuilder()\
    .set_name("gb_tuning")\
    .set_param_grid({
        "classifier__n_estimators": [50, 100, 200],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.1, 0.2]
    })\
    .set_scoring("accuracy")\
    .set_cv_folds(5)\
    .set_n_jobs(-1)\
    .build()
```

**Step 2: Initialize Tuner**
```python
from module.params_tuning import GridSearch

tuner = GridSearch(tuning_spec)
```

**Step 3: Run Tuning**
```python
# Provide untrained pipeline
pipeline = build_ml_pipeline(feature_specs, model_spec)

# Tune hyperparameters
tuner.fit(pipeline, X_train, y_train)
```

**What happens internally:**

1. **Validate Input**
   - Check param_grid is non-empty dict
   - Convert X, y to numpy arrays

2. **Create GridSearchCV**
   ```python
   self.search_cv = GridSearchCV(
       estimator=pipeline,
       param_grid=spec.param_grid,
       scoring=spec.scoring,
       cv=spec.cv_folds,
       n_jobs=spec.n_jobs,
       return_train_score=True,
       verbose=spec.verbose
   )
   ```

3. **Run Search**
   - Tests all parameter combinations
   - For example: 3 × 3 × 3 = 27 models × 5 folds = 135 fits
   - Uses cross-validation for each combination
   - Tracks best parameters and score

4. **Store Results**
   - `self.search_cv.best_params_`
   - `self.search_cv.best_score_`
   - `self.search_cv.best_estimator_`
   - `self.search_cv.cv_results_`

**Step 4: Get Best Model**
```python
best_pipeline = tuner.get_best_estimator()
best_params = tuner.get_best_params()
best_score = tuner.get_best_score()

print(f"Best accuracy: {best_score:.3f}")
print(f"Best params: {best_params}")
```

**Step 5: Use in Production**
```python
# In orchestrator:
if tuning_spec and tuning_spec.enabled:
    tuning = create_tuning(tuning_spec)
    tuning.fit(pipeline, X, y)
    pipeline = tuning.get_best_estimator()  # Replace with best
    
# Continue with best pipeline
pipeline.fit(X_train, y_train)  # Already fitted during search
```

#### Randomized Search

**When to use:**
- Parameter space too large for exhaustive search
- Want faster results
- Continuous parameters

**Example:**
```python
from module.params_tuning import RandomizedSearch
from specs import RandomizedSearchSpecBuilder

spec = RandomizedSearchSpecBuilder()\
    .set_param_distributions({
        "classifier__learning_rate": uniform(0.01, 0.3),
        "classifier__n_estimators": randint(50, 200)
    })\
    .set_n_iter(50)  # Try 50 random combinations\
    .build()

tuner = RandomizedSearch(spec)
tuner.fit(pipeline, X, y)
```

---

### MLflow Tracker Module

**File:** `module/mlflow_tracker.py` (605 lines)

**Purpose:** Experiment tracking, model logging, and registry management

#### Core Class

```python
MLflowTracker
    ├── Experiment management
    ├── Run tracking
    ├── Metrics and parameters logging
    ├── Model artifacts logging
    └── Model registry operations
```

#### Step-by-Step: Experiment Tracking

**Step 1: Initialize Tracker**
```python
from module.mlflow_tracker import MLflowTracker
from specs import MLflowSpecBuilder

mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("my_experiment")\
    .set_run_name("gradient_boost_v1")\
    .enable_model_registry(stage="Staging")\
    .add_tags({"team": "ml-eng", "version": "1.0"})\
    .build()

tracker = MLflowTracker(mlflow_spec)
```

**What happens:**
1. Sets tracking URI (defaults to `./mlruns`)
2. Creates/gets experiment by name
3. Prepares tags, description
4. Validates model stage if registry enabled

**Step 2: Start Run**
```python
tracker.start_run()
```

**What happens:**
1. Creates new MLflow run with experiment_id
2. Applies run_name and tags
3. Stores run_id and experiment_id
4. Logs start event

**Step 3: Log Parameters**
```python
tracker.log_params({
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5
})
```

**What happens:**
- Logs each parameter to MLflow
- Parameters are immutable (can't change after logging)
- Visible in MLflow UI

**Step 4: Log Metrics**
```python
tracker.log_metrics({
    "accuracy": 0.92,
    "f1_score": 0.91,
    "roc_auc": 0.95
})
```

**What happens:**
- Logs each metric with optional step number
- Metrics can be logged multiple times (e.g., per epoch)
- Creates time-series plots in UI

**Step 5: Log Model**
```python
tracker.log_model(
    model=trained_pipeline,
    artifact_path="model",
    X_sample=X_train[:100]  # For signature inference
)
```

**What happens:**
1. **Infer Model Signature**
   ```python
   signature = infer_signature(X_sample, model.predict(X_sample))
   # Stores input/output schema
   ```

2. **Log Model Artifact**
   - Serializes model using joblib/pickle
   - Stores in MLflow artifact store
   - Includes conda environment spec
   - Adds model signature

3. **Optional: Register Model**
   ```python
   if mlflow_spec.register_model:
       mlflow.register_model(
           model_uri=f"runs:/{run_id}/model",
           name=model_spec.model_name
       )
   ```

**Step 6: Log Workflow Results**
```python
tracker.log_workflow_results(
    results={
        'pipeline': pipeline,
        'cv_score': 0.92,
        'cv_std': 0.02,
        'test_accuracy': 0.91
    },
    X_sample=X_train[:100],
    register_model=True
)
```

**What happens:**
1. Logs all metrics from results dict
2. Logs model with signature
3. Registers in model registry if requested
4. Transitions to specified stage (e.g., "Staging")
5. Logs model architecture summary

**Step 7: End Run**
```python
tracker.end_run(status="FINISHED")
```

**What happens:**
- Marks run as complete
- Closes MLflow run context
- Final state persisted to storage

#### MLflow Tracker Flow

```
Initialize → Start Run → Log Params → Log Metrics → Log Model → Register → End Run
                            ↓              ↓            ↓            ↓
                        Immutable     Time-series   Artifacts    Registry
```

---

### Inference Module

**File:** `module/inference.py` (534 lines)

**Purpose:** Model loading and prediction serving

#### Core Classes

```python
ModelLoader
    ├── load_from_registry() - Load from model registry
    ├── load_from_run() - Load from specific run
    └── load_from_path() - Load from local path

InferenceEngine
    ├── predict() - Make predictions
    ├── predict_proba() - Get probabilities
    ├── predict_batch() - Batch predictions
    └── get_feature_importance() - Feature importance
```

#### Step-by-Step: Model Loading

**Option 1: From Model Registry**
```python
from module.inference import ModelLoader

loader = ModelLoader()
model = loader.load_from_registry(
    model_name="gradient_boosting_classifier_registered",
    stage="Production"  # or "Staging" or version="1"
)
```

**What happens:**
1. Connects to MLflow registry
2. Queries for model by name and stage/version
3. Gets model URI: `models:/model_name/Production`
4. Downloads model artifacts
5. Deserializes sklearn model
6. Returns loaded model

**Option 2: From Run ID**
```python
model = loader.load_from_run(run_id="abc123...")
```

**What happens:**
1. Constructs URI: `runs:/abc123.../model`
2. Downloads from run's artifacts
3. Loads model

**Option 3: From Local Path**
```python
model = loader.load_from_path("/path/to/model")
```

**What happens:**
1. Loads model from local filesystem
2. No MLflow server needed

#### Step-by-Step: Inference

**Step 1: Create Inference Engine**
```python
from module.inference import InferenceEngine

engine = InferenceEngine(model=loaded_model)
```

**What happens:**
- Validates model has `predict` method
- Checks for `predict_proba` support
- Sets `supports_proba` flag
- Initializes logging

**Step 2: Single Prediction**
```python
# Dict input
sample = {"feature_0": 1.5, "feature_1": -0.3, ...}
prediction = engine.predict_single(sample)
# Returns: 1 (class label)

# With probability
result = engine.predict_single(sample, return_proba=True)
# Returns: {"prediction": 1, "probability": 0.87}
```

**Step 3: Batch Prediction**
```python
# NumPy array
X_batch = np.random.randn(100, 20)
predictions = engine.predict(X_batch)
# Returns: array([1, 0, 1, ...])

# DataFrame
df = pd.DataFrame(X_batch, columns=[f'feature_{i}' for i in range(20)])
predictions = engine.predict(df)
```

**Step 4: Probability Predictions**
```python
probabilities = engine.predict_proba(X_batch)
# Returns: array([[0.2, 0.8], [0.7, 0.3], ...])
#          Shape: (n_samples, n_classes)
```

**Step 5: Get Feature Importance**
```python
importance = engine.get_feature_importance()
# Returns: array([0.15, 0.08, 0.23, ...])
```

**What it checks:**
1. `model.feature_importances_` (direct model)
2. `model.named_steps['classifier'].feature_importances_` (pipeline)
3. `model.named_steps['classifier'].model_.feature_importances_` (wrapped)
4. Returns None if not available

---

## Integration Patterns

### Pattern 1: Full ML Pipeline

```python
# 1. Create specifications
feature_specs = FeatureSpecBuilder().add_numeric_group([...]).build()
model_spec = ModelSpecBuilder().add_classifier("gb").build()[0]
mlflow_spec = MLflowSpecBuilder().enable().set_experiment("exp").build()

# 2. Build pipeline
from module.pre_processing import FeatureSpecPipeline
from module.classifier import GradientBoostingClassifierImpl
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('preprocessor', FeatureSpecPipeline(feature_specs)),
    ('classifier', GradientBoostingClassifierImpl(model_spec))
])

# 3. Train with MLflow
from module.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(mlflow_spec)
tracker.start_run()

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
accuracy = (predictions == y_test).mean()

tracker.log_metrics({"accuracy": accuracy})
tracker.log_model(pipeline, "model", X_train[:10])
tracker.end_run(status="FINISHED")
```

### Pattern 2: With Parameter Tuning

```python
# 1. Create tuning spec
from specs import GridSearchSpecBuilder

tuning_spec = GridSearchSpecBuilder()\
    .set_param_grid({
        "classifier__n_estimators": [50, 100, 200]
    })\
    .set_cv_folds(5)\
    .build()

# 2. Tune
from module.params_tuning import GridSearch

tuner = GridSearch(tuning_spec)
tuner.fit(pipeline, X, y)

# 3. Get best model
best_pipeline = tuner.get_best_estimator()
best_params = tuner.get_best_params()

# 4. Log to MLflow
tracker.log_params(best_params)
tracker.log_metrics({"best_cv_score": tuner.get_best_score()})
```

### Pattern 3: With Calibration

```python
# 1. Train model
pipeline.fit(X_train, y_train)

# 2. Calibrate on validation set
from module.calibration import ClassifierCalibration
from specs import CalibrationSpecBuilder

calib_spec = CalibrationSpecBuilder()\
    .set_method("isotonic")\
    .set_cv_strategy(5)\
    .build()

calibrator = ClassifierCalibration(calib_spec)
calibrator.fit(pipeline, X_val, y_val)

# 3. Get calibrated predictions
calibrated_proba = calibrator.predict_proba(X_test)
```

### Pattern 4: Complete Orchestrated Workflow

```python
from src.orchestrator import run_ml_workflow

results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X_train,
    y=y_train,
    validation_strategy='cross_validation',
    validation_params={'cv_folds': 5},
    tuning_spec=tuning_spec,
    calibration_spec=calib_spec,
    mlflow_spec=mlflow_spec
)

# Returns dict with:
# - pipeline: Trained (tuned, calibrated) pipeline
# - cv_score: Cross-validation accuracy
# - tuning_summary: Best params and scores
# - calibration_summary: Calibration info
# - mlflow_run_id: Run ID for retrieval
```

---

## Best Practices

### 1. Always Use Specs

**❌ Don't:**
```python
classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1)
```

**✅ Do:**
```python
model_spec = ModelSpecBuilder().add_classifier(
    "gb",
    hyperparameters={"n_estimators": 100, "learning_rate": 0.1}
).build()[0]
classifier = GradientBoostingClassifierImpl(model_spec)
```

**Why:** Specs provide validation, documentation, and MLflow tracking

### 2. Use Pipeline for Composition

**✅ Always combine preprocessing + model:**
```python
pipeline = Pipeline([
    ('preprocessor', FeatureSpecPipeline(feature_specs)),
    ('classifier', GradientBoostingClassifierImpl(model_spec))
])
```

**Benefits:**
- Single fit/predict call
- Prevents data leakage
- Preprocessing always applied consistently
- Can tune preprocessing + model together

### 3. Enable MLflow Tracking

**✅ Track all experiments:**
```python
mlflow_spec = MLflowSpecBuilder()\
    .enable()\
    .set_experiment("classification")\
    .enable_model_registry()\
    .build()
```

**Benefits:**
- Reproducibility
- Experiment comparison
- Model versioning
- Audit trail

### 4. Tune Before Final Training

**✅ Tuning workflow:**
```python
# 1. Tune on subset/CV
tuner.fit(pipeline, X_train, y_train)
best_pipeline = tuner.get_best_estimator()

# 2. Already fitted during tuning!
predictions = best_pipeline.predict(X_test)
```

**Note:** GridSearchCV with refit=True fits the best model on full data

### 5. Calibrate for Production

**✅ For probabilistic predictions:**
```python
# Train
pipeline.fit(X_train, y_train)

# Calibrate on held-out set
calibrator.fit(pipeline, X_val, y_val)

# Use calibrated for production
calibrated_model = calibrator.calibrated_
```

**When to calibrate:**
- Need reliable probability estimates
- Making decisions based on confidence
- Comparing predictions from different models

### 6. Use Orchestrator for Consistency

**✅ Recommended:**
```python
results = run_ml_workflow(
    feature_specs=feature_specs,
    model_spec=model_spec,
    X=X, y=y,
    tuning_spec=tuning_spec,
    calibration_spec=calib_spec,
    mlflow_spec=mlflow_spec
)
```

**Benefits:**
- Consistent order of operations
- Proper error handling
- Automatic MLflow logging
- Reusable across projects

---

## Advanced Topics

### Custom Classifiers

To add new classifier algorithms:

```python
# 1. Create in classifier.py
class RandomForestClassifierImpl(BaseClassifier):
    def __init__(self, model_spec: ClassifierModelSpec):
        super().__init__(model_spec)
    
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        
        self.model_ = RandomForestClassifier(
            n_estimators=100,
            random_state=self.model_spec.random_state,
            **self.model_spec.hyperparameters
        )
        self.model_.fit(X, y)
        self.is_fitted_ = True
        return self

# 2. Update ModelSpecBuilder to support it
# 3. Add to factory pattern
```

### Feature Engineering in Pipeline

```python
# Can add feature engineering steps:
from sklearn.preprocessing import PolynomialFeatures

pipeline = Pipeline([
    ('preprocessor', FeatureSpecPipeline(feature_specs)),
    ('poly', PolynomialFeatures(degree=2)),
    ('classifier', GradientBoostingClassifierImpl(model_spec))
])
```

### Multi-Metric Optimization

```python
# Tune with multiple metrics:
tuning_spec = GridSearchSpecBuilder()\
    .set_scoring({
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    })\
    .set_refit_score('f1')  # Optimize for F1\
    .build()
```

---

## Summary

The `module/` folder implements production-ready ML engineering patterns:

✅ **Spec-driven design** - Configuration as code with validation
✅ **Sklearn compatibility** - Standard fit/predict API
✅ **Composable components** - Mix and match via Pipeline
✅ **MLflow integration** - Automatic experiment tracking
✅ **Production patterns** - Error handling, logging, validation

**Total Code:** ~1,800 lines of focused ML engineering

**Test Coverage:** 68% with 207 passing unit tests

**Ready for:** Training, tuning, calibration, tracking, serving

