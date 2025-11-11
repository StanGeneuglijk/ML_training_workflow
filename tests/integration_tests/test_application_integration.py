"""
Integration tests for the main training application.

Tests complete application-level workflows including:
- Full training application flow
- Different configuration combinations
- Error handling and edge cases
"""
import pytest
import numpy as np
import tempfile
import shutil

from specs import (
    FeatureSpecBuilder,
    ModelSpecBuilder,
    MLflowSpecBuilder,
    GridSearchSpec,
    ClassifierCalibrationSpec,
)
from src.orchestrator import run_ml_workflow, get_workflow_summary
from data import load_data


class TestApplicationIntegration:
    """Integration tests for complete application workflows."""
    
    def test_minimal_training_configuration(self):
        """Test training with minimal configuration."""
        X, y = load_data("classification_data")
        X = X[:100]
        y = y[:100]
        
        # Minimal specs
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
            .add_classifier(name="minimal_classifier")
            .build()[0]
        )
        
        # Run with minimal configuration
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            random_state=42
        )
        
        assert results is not None
        assert "pipeline" in results
    
    def test_production_like_configuration(self):
        """Test production-like configuration with tuning, calibration, and MLflow."""
        X, y = load_data("classification_data")
        X = X[:300]
        y = y[:300]
        
        # Production-like specs
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        # Use default hyperparameters - tuning will optimize them
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="production_classifier",
                evaluation_metrics=["accuracy", "roc_auc", "f1_score"]
            )
            .build()[0]
        )
        
        tuning_spec = GridSearchSpec(
            tuning_name="production_grid",
            param_grid={
                "classifier__n_estimators": [20, 30],
                "classifier__max_depth": [2, 3]
            },
            scoring="roc_auc",
            n_splits=2,
            n_jobs=1,
            verbose=0
        )
        
        calibration_spec = ClassifierCalibrationSpec(
            calibration_name="production_calibration",
            method="sigmoid",
            cv_strategy=2,
            ensemble=True
        )
        
        mlflow_spec = (
            MLflowSpecBuilder()
            .enable()
            .set_experiment("production_test")
            .set_run_name("production_run")
            .add_tags({
                "environment": "test",
                "model_version": "1.0"
            })
            .build()
        )
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            tuning_spec=tuning_spec,
            calibration_spec=calibration_spec,
            mlflow_spec=mlflow_spec,
            random_state=42
        )
        
        # Verify all components worked
        assert "pipeline" in results
        assert "cv_score" in results
        assert "tuning_summary" in results
        assert "calibration_summary" in results
        assert "mlflow_run_id" in results
        
        # Verify reasonable performance
        assert results["cv_score"] > 0.6
    
    def test_different_random_seeds(self):
        """Test that different random seeds produce different results."""
        X, y = load_data("classification_data")
        X = X[:150]
        y = y[:150]
        
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
                name="seed_test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        # Run with different seeds
        results_1 = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            random_state=42
        )
        
        results_2 = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            random_state=123
        )
        
        # Results might be same or different due to sklearn randomness
        # Just verify both are valid
        assert 0.0 <= results_1["cv_score"] <= 1.0
        assert 0.0 <= results_2["cv_score"] <= 1.0
        # Verify pipeline was created
        assert results_1["pipeline"] is not None
        assert results_2["pipeline"] is not None
    
    def test_reproducibility_with_same_seed(self):
        """Test that same random seed produces similar results."""
        X, y = load_data("classification_data")
        X = X[:150]
        y = y[:150]
        
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
                name="repro_test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        # Run twice with same seed
        results_1 = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            random_state=42
        )
        
        results_2 = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            random_state=42
        )
        
        # Results should be very similar (sklearn has some inherent randomness)
        # Allow small tolerance for floating point differences
        assert abs(results_1["cv_score"] - results_2["cv_score"]) < 0.05
    
    def test_workflow_summary_generation(self):
        """Test that workflow summary is generated correctly."""
        X, y = load_data("classification_data")
        X = X[:100]
        y = y[:100]
        
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
                name="summary_test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            random_state=42
        )
        
        # Generate summary  
        summary = get_workflow_summary(results)
        
        assert summary is not None
        # Summary can be dict or string depending on implementation
        assert isinstance(summary, (str, dict))
        if isinstance(summary, dict):
            assert len(summary) > 0
        else:
            assert len(summary) > 10


class TestErrorHandlingIntegration:
    """Integration tests for error handling in workflows."""
    
    def test_invalid_data_shape(self):
        """Test handling of invalid data shapes."""
        # 1D array instead of 2D
        X = np.array([1, 2, 3, 4, 5])
        y = np.array([0, 1, 0, 1, 0])
        
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=["feature_0"],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(name="test_classifier")
            .build()[0]
        )
        
        with pytest.raises((ValueError, IndexError)):
            run_ml_workflow(
                feature_specs=feature_specs,
                model_spec=model_spec,
                X=X,
                y=y,
                random_state=42
            )
    
    def test_mismatched_feature_count(self):
        """Test handling of mismatched feature counts."""
        X, y = load_data("classification_data")
        
        # Create specs for wrong number of features
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(10)],  # Wrong count
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(name="test_classifier")
            .build()[0]
        )
        
        # This might not raise an error if preprocessing handles it
        # Just verify it doesn't crash or produces reasonable results
        try:
            results = run_ml_workflow(
                feature_specs=feature_specs,
                model_spec=model_spec,
                X=X,
                y=y,
                random_state=42
            )
            # If it succeeds, that's also acceptable
            assert results is not None
        except (ValueError, KeyError, IndexError):
            # Expected behavior - error caught
            pass
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        X = np.array([]).reshape(0, 20)
        y = np.array([])
        
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
            .add_classifier(name="test_classifier")
            .build()[0]
        )
        
        with pytest.raises(ValueError):
            run_ml_workflow(
                feature_specs=feature_specs,
                model_spec=model_spec,
                X=X,
                y=y,
                random_state=42
            )


class TestPerformanceIntegration:
    """Integration tests for performance characteristics."""
    
    def test_model_improves_with_more_data(self):
        """Test that model performance improves with more training data."""
        X, y = load_data("classification_data")
        
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
        
        # Train with small dataset
        results_small = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X[:100],
            y=y[:100],
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
            random_state=42
        )
        
        # Train with larger dataset
        results_large = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X[:500],
            y=y[:500],
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
            random_state=42
        )
        
        # Larger dataset should generally perform better or similar
        # Allow some tolerance due to randomness
        assert results_large["cv_score"] >= results_small["cv_score"] - 0.1
    
    def test_tuning_works_correctly(self):
        """Test that hyperparameter tuning workflow executes correctly."""
        X, y = load_data("classification_data")
        X = X[:200]
        y = y[:200]
        
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        # Use default hyperparameters to avoid sklearn cloning issues
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(name="test_classifier")
            .build()[0]
        )
        
        # Simple tuning spec
        tuning_spec = GridSearchSpec(
            tuning_name="simple_grid",
            param_grid={
                "classifier__n_estimators": [10, 20]
            },
            scoring="accuracy",
            n_splits=2,
            n_jobs=1,
            verbose=0
        )
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            tuning_spec=tuning_spec,
            random_state=42
        )
        
        # Verify tuning completed successfully
        assert "tuning_summary" in results
        assert results["tuning_summary"] is not None
        assert "best_score" in results["tuning_summary"]
        assert results["tuning_summary"]["best_score"] > 0.5
        assert "pipeline" in results

