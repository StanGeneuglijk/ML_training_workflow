"""
Pytest configuration and shared fixtures.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

import sys
import os

# Add parent directory to path
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
        random_state=42
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
        random_state=42
    )
    return X, y


@pytest.fixture
def simple_feature_specs():
    """Create simple feature specifications for testing."""
    from specs import FeatureSpecBuilder
    
    builder = FeatureSpecBuilder()
    specs = builder.add_numeric_group(
        ['feature_0', 'feature_1', 'feature_2'],
        imputer_strategy='mean',
        scaler_type='standard'
    ).build()
    return specs


@pytest.fixture
def simple_model_spec():
    """Create simple model specification for testing."""
    from specs import ModelSpecBuilder
    
    builder = ModelSpecBuilder()
    spec = builder.add_classifier(
        name='test_classifier',
        hyperparameters={'n_estimators': 10, 'learning_rate': 0.1}
    ).build()[0]
    return spec

