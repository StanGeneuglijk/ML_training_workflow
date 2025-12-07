"""
Pytest configuration and shared fixtures.
"""
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def sample_classification_data():
    """Generate sample classification data for testing."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    y_series = pd.Series(y, name="target")
    return X_df, y_series


@pytest.fixture
def sample_classification_array():
    """Generate sample classification data as numpy arrays."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return X, y


@pytest.fixture
def simple_feature_specs():
    """Create simple feature specifications for testing."""
    from specs import FeatureSpecBuilder

    builder = FeatureSpecBuilder()
    specs = builder.add_numeric_group(
        ["feature_0", "feature_1", "feature_2"],
        imputer_strategy="mean",
        scaler_type="standard",
    ).build()
    return specs


@pytest.fixture
def simple_model_spec():
    """Create simple model specification for testing."""
    from specs import ModelSpecBuilder

    builder = ModelSpecBuilder()
    spec = builder.add_classifier(
        name="test_classifier",
        hyperparameters={"n_estimators": 10, "learning_rate": 0.1},
    ).build()[0]
    return spec


@pytest.fixture
def numeric_spec_default():
    """Create default numeric feature spec with imputer and scaler."""
    from specs import NumericFeatureSpec

    return NumericFeatureSpec(
        feature_name="num_feature",
        imputer_strategy="mean",
        scaler_type="standard",
    )


@pytest.fixture
def numeric_spec_no_preprocessing():
    """Create numeric feature spec with all preprocessing disabled."""
    from specs import NumericFeatureSpec

    return NumericFeatureSpec(
        feature_name="num_feature",
        imputer_enabled=False,
        scaler_enabled=False,
    )


@pytest.fixture
def numeric_data():
    """Create simple numeric test data."""
    np.random.seed(42)
    return np.random.randn(50, 3).astype(np.float32)


@pytest.fixture
def categorical_spec_default():
    """Create default categorical feature spec."""
    from specs import CategoricalFeatureSpec

    return CategoricalFeatureSpec(
        feature_name="cat_feature",
        encoder_type="onehot",
    )


@pytest.fixture
def fitted_pipeline(simple_feature_specs, sample_classification_array):
    """Create a fitted preprocessing pipeline."""
    from module.pre_processing import FeatureSpecPipeline

    X, _ = sample_classification_array
    pipeline = FeatureSpecPipeline(simple_feature_specs)
    pipeline.fit(X)
    return pipeline


@pytest.fixture
def temp_repo_path(tmp_path):
    """Create a temporary repository path."""
    repo_path = tmp_path / "test_feast_repo"
    yield str(repo_path)
    if repo_path.exists():
        shutil.rmtree(repo_path)

