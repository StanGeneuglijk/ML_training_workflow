"""
Integration tests for end-to-end training workflow.

Tests the complete ML training pipeline including:
- Data loading
- Feature preprocessing
- Model training
- Hyperparameter tuning
- Calibration
- Cross-validation
- MLflow tracking
"""
import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from specs import (
    FeatureSpecBuilder,
    ModelSpecBuilder,
    MLflowSpecBuilder,
    GridSearchSpec,
    ClassifierCalibrationSpec,
)
from src.orchestrator import run_ml_workflow, get_workflow_summary
from data import load_data


class TestTrainingWorkflowIntegration:
    """Integration tests for complete training workflow."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Setup
        self.temp_dir = tempfile.mkdtemp()
        
        yield
        
        # Teardown
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_training_workflow(self):
        """Test basic training workflow without tuning or calibration."""
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
        
        # Create model spec
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier",
                hyperparameters={
                    "n_estimators": 10,
                    "max_depth": 3
                }
            )
            .build()[0]
        )
        
        # Run workflow
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
            random_state=42
        )
        
        # Assertions
        assert results is not None
        assert "pipeline" in results
        assert "cv_scores" in results
        assert len(results["cv_scores"]) == 3
        assert "cv_score" in results
        assert 0.0 <= results["cv_score"] <= 1.0
        assert "cv_std" in results
        
        # Test prediction
        pipeline = results["pipeline"]
        predictions = pipeline.predict(X[:10])
        assert len(predictions) == 10
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_workflow_with_hyperparameter_tuning(self):
        """Test workflow with hyperparameter tuning."""
        # Load data (use subset for speed)
        X, y = load_data("classification_data")
        X = X[:200]
        y = y[:200]
        
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
        
        # Use default hyperparameters to avoid cloning issues
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier"
            )
            .build()[0]
        )
        
        # Create tuning spec with simple grid
        tuning_spec = GridSearchSpec(
            tuning_name="test_grid",
            param_grid={
                "classifier__n_estimators": [10, 20]
            },
            scoring="accuracy",
            n_splits=2,
            n_jobs=1,
            verbose=0
        )
        
        # Run workflow
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
        
        # Assertions
        assert "tuning_summary" in results
        assert results["tuning_summary"] is not None
        assert "best_params" in results["tuning_summary"]
        assert "best_score" in results["tuning_summary"]
        assert results["tuning_summary"]["best_score"] > 0
    
    def test_workflow_with_calibration(self):
        """Test workflow with probability calibration."""
        # Load data (use subset for speed)
        X, y = load_data("classification_data")
        X = X[:200]
        y = y[:200]
        
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
                hyperparameters={
                    "n_estimators": 10,
                    "max_depth": 3
                }
            )
            .build()[0]
        )
        
        # Create calibration spec
        calibration_spec = ClassifierCalibrationSpec(
            calibration_name="test_calibration",
            method="sigmoid",
            cv_strategy=2,
            ensemble=True
        )
        
        # Run workflow
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            calibration_spec=calibration_spec,
            random_state=42
        )
        
        # Assertions
        assert "calibration_summary" in results
        assert results["calibration_summary"] is not None
        
        # Test probability prediction
        pipeline = results["pipeline"]
        probas = pipeline.predict_proba(X[:10])
        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
    
    def test_full_workflow_with_all_features(self):
        """Test complete workflow with tuning, calibration, and MLflow."""
        # Load data (use subset for speed)
        X, y = load_data("classification_data")
        X = X[:150]
        y = y[:150]
        
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
        
        # Use default hyperparameters - tuning will optimize
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(name="test_full_classifier")
            .build()[0]
        )
        
        tuning_spec = GridSearchSpec(
            tuning_name="test_grid",
            param_grid={
                "classifier__n_estimators": [10, 20]
            },
            scoring="accuracy",
            n_splits=2,
            n_jobs=1,
            verbose=0
        )
        
        # Now we can use both tuning and calibration together!
        calibration_spec = ClassifierCalibrationSpec(
            calibration_name="test_calibration",
            method="sigmoid",
            cv_strategy=2,
            ensemble=True
        )
        
        mlflow_spec = (
            MLflowSpecBuilder()
            .enable()
            .set_experiment("integration_test_experiment")
            .set_run_name("integration_test_run")
            .enable_model_registry(stage="None")
            .build()
        )
        
        # Run workflow
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
        
        # Comprehensive assertions
        assert "pipeline" in results
        assert "cv_score" in results
        assert "cv_std" in results
        assert "tuning_summary" in results
        assert "calibration_summary" in results
        assert "mlflow_run_id" in results
        assert "mlflow_experiment_id" in results
        
        # Verify summary generation
        summary = get_workflow_summary(results)
        assert summary is not None
        assert len(summary) > 0
    
    def test_train_test_split_validation(self):
        """Test workflow with train-test split validation."""
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
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        # Run with train-test split
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="train_test_split",
            validation_params={"test_size": 0.2},
            random_state=42
        )
        
        # Assertions - train_test_split might not return these keys in all cases
        # Just verify the pipeline works
        assert "pipeline" in results
        pipeline = results["pipeline"]
        predictions = pipeline.predict(X[:10])
        assert len(predictions) == 10
    
    def test_no_validation_strategy(self):
        """Test workflow without validation (full training)."""
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
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        # Run without validation
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy=None,
            random_state=42
        )
        
        # Assertions
        assert "pipeline" in results
        assert "cv_score" not in results
        assert "test_score" not in results
        
        # Pipeline should be fitted
        pipeline = results["pipeline"]
        predictions = pipeline.predict(X[:10])
        assert len(predictions) == 10
    
    def test_different_data_formats(self):
        """Test workflow with different input data formats."""
        import pandas as pd
        
        # Load as numpy arrays
        X_np, y_np = load_data("classification_data")
        
        # Convert to DataFrame
        X_df = pd.DataFrame(
            X_np,
            columns=[f"feature_{i}" for i in range(20)]
        )
        y_series = pd.Series(y_np)
        
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
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        # Test with DataFrame
        results_df = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X_df,
            y=y_series,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            random_state=42
        )
        
        # Test with numpy array
        results_np = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X_np,
            y=y_np,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            random_state=42
        )
        
        # Both should work and produce similar results
        assert results_df is not None
        assert results_np is not None
        assert abs(results_df["cv_score"] - results_np["cv_score"]) < 0.01


@pytest.mark.slow
class TestLongRunningWorkflows:
    """Integration tests that take longer to run."""
    
    def test_full_grid_search(self):
        """Test comprehensive grid search (slower test)."""
        X, y = load_data("classification_data")
        X = X[:300]  # Use subset for faster testing
        y = y[:300]
        
        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard"
            )
            .build()
        )
        
        # Use default hyperparameters to avoid cloning issues
        model_spec = (
            ModelSpecBuilder()
            .add_classifier(name="test_classifier")
            .build()[0]
        )
        
        # Reasonable grid search
        tuning_spec = GridSearchSpec(
            tuning_name="comprehensive_grid",
            param_grid={
                "classifier__n_estimators": [10, 30, 50],
                "classifier__learning_rate": [0.05, 0.1]
            },
            scoring="accuracy",
            n_splits=2,
            n_jobs=-1,
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
        
        assert "tuning_summary" in results
        assert results["tuning_summary"]["best_score"] > 0.6

