# ML Workflow Tests

**Test suite for the ML training workflow**

---

## Test Structure

```
tests/
├── unit_tests/              
│   ├── conftest.py          # Shared fixtures (10 fixtures)
│   ├── test_calibration_spec.py
│   ├── test_calibration.py
│   ├── test_classifier.py
│   ├── test_data_sources.py
│   ├── test_feast_manager.py
│   ├── test_feature_definitions.py
│   ├── test_feature_spec.py
│   ├── test_feature_store_spec.py
│   ├── test_mlflow_spec.py
│   ├── test_model_spec.py
│   ├── test_orchestrator.py
│   ├── test_params_tuning_spec.py
│   ├── test_params_tuning.py
│   └── test_pre_processing.py
│
└── integration_tests/       
    └── test_training_workflow.py
```

---

## Running Tests

### Run All Tests

```bash
poetry run pytest tests/ -v
```

### Run Unit Tests Only

```bash
python3 -m pytest tests/unit_tests/ -v
```

### Run Integration Tests Only

```bash
python3 -m pytest tests/integration_tests/ -v
```

### Run Tests by Marker

```bash
# Run only integration tests
python3 -m pytest -m integration -v

# Run only unit tests
python3 -m pytest -m unit -v
```

---

## Test Markers

- **`@pytest.mark.unit`** - Fast unit tests (< 1 second each)
- **`@pytest.mark.integration`** - Integration tests (may take longer)

---

## Unit Tests

### Purpose

Test individual components in isolation:

- Feature specifications
- Model specifications
- Preprocessing pipelines
- Classifiers
- Calibration
- Parameter tuning
- Feature store components
- Orchestrator functions

### Shared Fixtures

All fixtures in `tests/unit_tests/conftest.py`:

- `sample_classification_data` - DataFrame and Series
- `sample_classification_array` - Numpy arrays
- `simple_feature_specs` - Feature specifications
- `simple_model_spec` - Model specification
- `numeric_spec_default` - Default numeric feature spec
- `numeric_spec_no_preprocessing` - Numeric spec without preprocessing
- `numeric_data` - Simple numeric test data
- `categorical_spec_default` - Default categorical feature spec
- `fitted_pipeline` - Fitted preprocessing pipeline
- `temp_repo_path` - Temporary repository path

---

## Integration Tests

### Purpose

Test complete workflows:

- End-to-end ML training workflow
- Data loading
- Feature preprocessing
- Model training
- Hyperparameter tuning
- Calibration
- Cross-validation

### Core Tests

- `test_basic_training_workflow` - Core end-to-end workflow
- `test_workflow_with_hyperparameter_tuning` - Tuning integration
- `test_workflow_with_calibration` - Calibration integration
- `test_full_workflow_with_tuning_and_calibration` - Complete workflow

---

## Coverage

### Generate Coverage Report

```bash
python3 -m pytest tests/ \
    --cov=module --cov=specs --cov=src_training --cov=data --cov=feature_store \
    --cov-report=html --cov-report=term
```

**Target:** > 70% coverage

---

## Best Practices

1. **Test Isolation** - Each test independent, use fixtures
2. **Naming** - `test_<what>_<condition>_<expected>`
3. **Assertions** - Descriptive messages, one concept per test
4. **Mocking** - Mock external dependencies in unit tests
5. **Markers** - Use `@pytest.mark.unit` or `@pytest.mark.integration`

---

## Troubleshooting

**Data not found:**
```bash
poetry run python data/generate_delta_lake.py
```

**Module import errors:**
```bash
poetry shell
poetry run pytest tests/
```

---

**Happy Testing! 🧪✅**
