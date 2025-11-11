"""
Integration tests for data loading and preprocessing pipeline.

Tests the integration between:
- Data loading from SQLite
- Feature specification
- Preprocessing pipeline
- End-to-end data flow
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from data import load_data
from specs import FeatureSpecBuilder
from module.pre_processing import FeatureSpecPipeline


class TestDataLoadingIntegration:
    """Integration tests for data loading."""
    
    def test_load_classification_data(self):
        """Test loading classification data from SQLite."""
        X, y = load_data("classification_data")
        
        # Verify data shape and types
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 20  # 20 features
        assert len(np.unique(y)) == 2  # Binary classification
        
        # Verify data is valid
        assert not np.isnan(X).any()
        assert not np.isnan(y).any()
        assert all(label in [0, 1] for label in y)
    
    def test_data_consistency(self):
        """Test that loading data multiple times gives consistent results."""
        X1, y1 = load_data("classification_data")
        X2, y2 = load_data("classification_data")
        
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)
    
    def test_data_statistics(self):
        """Test that loaded data has expected statistical properties."""
        X, y = load_data("classification_data")
        
        # Check basic statistics
        assert X.shape[0] > 0
        assert X.shape[1] == 20
        
        # Check class balance is reasonable
        class_counts = np.bincount(y)
        class_ratio = class_counts.min() / class_counts.max()
        assert class_ratio > 0.3  # Not too imbalanced


class TestPreprocessingPipelineIntegration:
    """Integration tests for preprocessing pipeline."""
    
    def test_end_to_end_numeric_preprocessing(self):
        """Test complete numeric preprocessing pipeline."""
        # Load data
        X, y = load_data("classification_data")
        
        # Create feature specs
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        # Create and fit pipeline
        pipeline = FeatureSpecPipeline(feature_specs)
        X_transformed = pipeline.fit_transform(X)
        
        # Verify output
        assert X_transformed.shape == X.shape
        assert not np.isnan(X_transformed).any()
        
        # Check standardization (mean ~0, std ~1)
        assert np.abs(X_transformed.mean()) < 0.1
        assert np.abs(X_transformed.std() - 1.0) < 0.1
    
    def test_preprocessing_with_different_scalers(self):
        """Test preprocessing with different scaler types."""
        X, y = load_data("classification_data")
        X = X[:200]  # Use subset
        
        scaler_types = ["standard", "minmax", "robust"]
        
        for scaler_type in scaler_types:
            feature_specs = (
                FeatureSpecBuilder()
                .add_numeric_group(
                    feature_names=[f"feature_{i}" for i in range(20)],
                    imputer_strategy="mean",
                    scaler_type=scaler_type
                )
                .build()
            )
            
            pipeline = FeatureSpecPipeline(feature_specs)
            X_transformed = pipeline.fit_transform(X)
            
            assert X_transformed.shape == X.shape
            assert not np.isnan(X_transformed).any()
    
    def test_preprocessing_with_missing_values(self):
        """Test preprocessing handles missing values correctly."""
        X, y = load_data("classification_data")
        X = X[:100].copy()
        
        # Introduce missing values
        missing_indices = np.random.choice(X.size, size=50, replace=False)
        X.ravel()[missing_indices] = np.nan
        
        # Create feature specs with imputation
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        # Process
        pipeline = FeatureSpecPipeline(feature_specs)
        X_transformed = pipeline.fit_transform(X)
        
        # Verify no missing values remain
        assert not np.isnan(X_transformed).any()
        assert X_transformed.shape == X.shape
    
    def test_preprocessing_preserves_feature_count(self):
        """Test that preprocessing preserves the number of features."""
        X, y = load_data("classification_data")
        
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="median",
                scaler_type="minmax"
            )
            .build()
        )
        
        pipeline = FeatureSpecPipeline(feature_specs)
        X_transformed = pipeline.fit_transform(X)
        
        assert X_transformed.shape[1] == X.shape[1]
    
    def test_preprocessing_with_dataframe_input(self):
        """Test preprocessing works with DataFrame input."""
        X, y = load_data("classification_data")
        
        # Convert to DataFrame
        X_df = pd.DataFrame(
            X,
            columns=[f"feature_{i}" for i in range(20)]
        )
        
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        pipeline = FeatureSpecPipeline(feature_specs)
        X_transformed = pipeline.fit_transform(X_df)
        
        assert X_transformed.shape == X.shape
        assert not np.isnan(X_transformed).any()
    
    def test_preprocessing_transform_consistency(self):
        """Test that transform is consistent after fit."""
        X, y = load_data("classification_data")
        X_train = X[:800]
        X_test = X[800:]
        
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        pipeline = FeatureSpecPipeline(feature_specs)
        
        # Fit on training data
        pipeline.fit(X_train)
        
        # Transform test data multiple times
        X_test_transformed_1 = pipeline.transform(X_test)
        X_test_transformed_2 = pipeline.transform(X_test)
        
        # Should be identical
        np.testing.assert_array_equal(
            X_test_transformed_1,
            X_test_transformed_2
        )


class TestDataWorkflowIntegration:
    """Integration tests for complete data workflow."""
    
    def test_data_load_to_model_pipeline(self):
        """Test complete flow from data loading to model training."""
        from module.classifier import GradientBoostingClassifierImpl
        from specs import ModelSpecBuilder
        from sklearn.pipeline import Pipeline
        
        # Load data
        X, y = load_data("classification_data")
        X = X[:200]
        y = y[:200]
        
        # Create feature specs
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        # Create model spec
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        # Build pipeline
        preprocessor = FeatureSpecPipeline(feature_specs)
        classifier = GradientBoostingClassifierImpl(model_spec=model_spec)
        
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        
        # Train
        pipeline.fit(X, y)
        
        # Predict
        predictions = pipeline.predict(X)
        
        # Verify
        assert len(predictions) == len(y)
        assert all(pred in [0, 1] for pred in predictions)
        
        # Check accuracy is reasonable
        accuracy = (predictions == y).mean()
        assert accuracy > 0.6  # Should do better than random
    
    def test_train_test_split_workflow(self):
        """Test typical train-test split workflow."""
        from sklearn.model_selection import train_test_split
        from module.classifier import GradientBoostingClassifierImpl
        from specs import ModelSpecBuilder
        from sklearn.pipeline import Pipeline
        
        # Load and split data
        X, y = load_data("classification_data")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create specs
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier",
                hyperparameters={"n_estimators": 20}
            )
            .build()[0]
        )
        
        # Build and train pipeline
        preprocessor = FeatureSpecPipeline(feature_specs)
        classifier = GradientBoostingClassifierImpl(model_spec=model_spec)
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        train_score = pipeline.score(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        
        # Verify reasonable performance
        assert train_score > 0.7
        assert test_score > 0.6
        assert train_score >= test_score  # No severe underfitting

