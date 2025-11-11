# ML Workflow Tests

Comprehensive test suite for the ML training workflow with SQLite data management.

---

## Test Structure

```
tests/
├── unit_tests/              # Fast, isolated unit tests
│   ├── test_calibration.py
│   ├── test_classifier.py
│   ├── test_data_ingestion.py
│   ├── test_feature_spec.py
│   ├── test_mlflow_spec.py
│   ├── test_mlflow_tracker.py
│   ├── test_model_spec.py
│   ├── test_orchestrator.py
│   ├── test_params_tuning.py
│   ├── test_pre_processing.py
│   └── test_utils.py
│
└── integration_tests/       # End-to-end integration tests
    ├── test_ingestion_workflow.py
    └── test_ml_workflow_integration.py
```

---

## Running Tests

### Run All Tests
```bash
cd /path/to/ML_workflow_training
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

### Run Specific Test File
```bash
python3 -m pytest tests/unit_tests/test_data_ingestion.py -v
```

### Run Tests by Marker
```bash
# Run only integration tests
python3 -m pytest -m integration -v

# Run only unit tests
python3 -m pytest -m unit -v

# Skip slow tests
python3 -m pytest -m "not slow" -v
```

---

## Test Markers

Tests are marked with pytest markers for flexible execution:

- **`@pytest.mark.unit`** - Fast unit tests (< 1 second each)
- **`@pytest.mark.integration`** - Integration tests (may take longer)
- **`@pytest.mark.slow`** - Slow tests (> 10 seconds)

---

## Unit Tests

### Purpose
Test individual components in isolation:
- Data ingestion functions
- Feature specifications
- Model specifications
- Preprocessing pipelines
- Classifiers
- MLflow tracking
- Calibration
- Parameter tuning

### Characteristics
- ✅ Fast (< 1 second per test)
- ✅ No external dependencies (mocked)
- ✅ Isolated (no side effects)
- ✅ Deterministic (same result every time)

### Example
```python
def test_generate_data():
    """Test data generation returns correct shapes."""
    X, y = generate_classification_data(
        n_samples=100,
        n_features=10,
        random_state=42
    )
    assert X.shape == (100, 10)
    assert y.shape == (100,)
```

---

## Integration Tests

### Purpose
Test complete workflows and component interactions:
- End-to-end ML workflow
- MLflow integration
- Feature store integration
- Data loading and validation
- Model training and inference

### Characteristics
- ⏱️ Slower (may take several seconds)
- 🔗 Tests real interactions
- 📊 Uses actual data and models
- 🎯 Validates complete scenarios

### Example
```python
@pytest.mark.integration
def test_complete_workflow():
    """Test end-to-end: load data → train model → log to MLflow."""
    from data import load_data
    from src.orchestrator import run_ml_workflow
    
    # Load data
    X, y = load_data("classification_data")
    
    # Run workflow
    results = run_ml_workflow(
        feature_specs=feature_specs,
        model_spec=model_spec,
        X=X, y=y
    )
    
    # Verify results
    assert 'pipeline' in results
    assert results['cv_score'] > 0.8
```

---

## Coverage

### Generate Coverage Report
```bash
python3 -m pytest tests/ --cov=data --cov=specs --cov=module --cov=src --cov-report=html
```

View report:
```bash
open htmlcov/index.html
```

### Current Coverage Targets
- Unit tests: > 80% coverage
- Integration tests: Cover all main workflows

---

## Test Data

### Unit Tests
- Use small, synthetic datasets (100-500 samples)
- Use mocked Spark sessions where possible
- Use temporary directories for file operations

### Integration Tests
- Use realistic dataset sizes (500-10,000 samples)
- Use actual Spark sessions
- Clean up test data in teardown

---

## Writing New Tests

### Unit Test Template
```python
import pytest
from module import function_to_test

class TestFeatureName:
    """Tests for feature X."""
    
    def test_basic_functionality(self):
        """Test basic case."""
        result = function_to_test(input_data)
        assert result == expected_output
    
    def test_edge_case(self):
        """Test edge case."""
        with pytest.raises(ValueError):
            function_to_test(invalid_input)
```

### Integration Test Template
```python
import pytest
from data import load_data
from src.orchestrator import run_ml_workflow

@pytest.mark.integration
class TestWorkflow:
    """Test complete workflow."""
    
    def test_end_to_end(self, feature_specs, model_spec):
        """Test full ML workflow."""
        # Load data
        X, y = load_data("classification_data")
        
        # Execute workflow
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X, y=y
        )
        
        # Verify
        assert results['cv_score'] > 0.8
        assert 'pipeline' in results
```

---

## Best Practices

### 1. Test Isolation
- Each test should be independent
- Use fixtures for setup/teardown
- Clean up temporary files

### 2. Naming Convention
- `test_<what>_<condition>_<expected>`
- Example: `test_load_data_missing_file_raises_error`

### 3. Assertions
- Use descriptive assertion messages
- Test one concept per test
- Use appropriate assert helpers (`np.testing.assert_array_equal`, etc.)

### 4. Mocking
- Mock external dependencies in unit tests
- Use `monkeypatch` for path mocking
- Use `tmp_path` for temporary directories

### 5. Performance
- Keep unit tests fast (< 1 second)
- Mark slow tests with `@pytest.mark.slow`
- Use smaller datasets for tests

---

## Continuous Integration

### Pre-commit Checks
```bash
# Run unit tests (fast)
python3 -m pytest tests/unit_tests/ -v

# Run linting
flake8 tests/
black tests/ --check
```

### Full CI Pipeline
```bash
# Run all tests
python3 -m pytest tests/ -v

# Generate coverage
python3 -m pytest tests/ --cov=. --cov-report=xml

# Check coverage threshold
python3 -m pytest tests/ --cov=. --cov-fail-under=80
```

---

## Debugging Tests

### Run Single Test
```bash
python3 -m pytest tests/unit_tests/test_data_ingestion.py::TestGenerateClassificationData::test_generate_data -v
```

### Show Print Statements
```bash
python3 -m pytest tests/ -v -s
```

### Drop into Debugger on Failure
```bash
python3 -m pytest tests/ --pdb
```

### Verbose Output
```bash
python3 -m pytest tests/ -vv
```

---

## Test Requirements

### Required Dependencies
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0
- pydantic >= 2.0.0
- mlflow >= 2.0.0

All dependencies are defined in `pyproject.toml`.

---

## Troubleshooting

### Issue: Database Not Found
```bash
# Generate test database
poetry run python data/generate_database.py
```

### Issue: Module Import Errors
```bash
# Ensure virtual environment is activated
poetry shell

# Or use poetry run
poetry run pytest tests/
```

### Issue: Test Database Conflicts
```bash
# Clean up test artifacts
rm -f tests/*.db
rm -rf htmlcov/
rm -rf .pytest_cache/
```

### Issue: MLflow Conflicts
```bash
# Clean MLflow test runs
rm -rf mlruns_test/
```

---

## Performance Benchmarks

### Unit Tests
- **Target**: < 30 seconds total
- **Current**: ~25 seconds (all unit tests)

### Integration Tests
- **Target**: < 5 minutes total
- **Current**: ~3 minutes (all integration tests)

### Combined
- **Target**: < 6 minutes
- **Current**: ~3.5 minutes

---

## Examples

### Run Quick Smoke Test
```bash
# Run fast tests only
python3 -m pytest tests/unit_tests/ -m "not slow" -v
```

### Run Before Commit
```bash
# Quick validation
python3 -m pytest tests/unit_tests/ -v --maxfail=1
```

### Run Full Suite
```bash
# Complete test run with coverage
python3 -m pytest tests/ -v --cov=. --cov-report=term-missing
```

---

## Test Metrics

| Metric | Target | Current |
|--------|--------|---------|
| Unit Test Coverage | > 80% | ~85% |
| Integration Coverage | > 60% | ~70% |
| Test Execution Time | < 6 min | ~3.5 min |
| Flaky Tests | 0 | 0 |

---

## Contributing

When adding new features:

1. ✅ Write unit tests first (TDD)
2. ✅ Ensure unit tests pass
3. ✅ Write integration tests
4. ✅ Verify full test suite passes
5. ✅ Check coverage doesn't decrease

---

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [MLflow Testing](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.tracking.MlflowClient)
- [scikit-learn Testing](https://scikit-learn.org/stable/developers/develop.html#testing)

---

**Happy Testing! 🧪✅**
