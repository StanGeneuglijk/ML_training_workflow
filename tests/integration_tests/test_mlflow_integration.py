"""
Integration tests for MLflow tracking and model registry.

Tests the complete MLflow integration including:
- Experiment creation
- Run tracking
- Parameter and metric logging
- Model logging
- Model registry
"""
import pytest
import tempfile
import shutil
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

from specs import (
    FeatureSpecBuilder,
    ModelSpecBuilder,
    MLflowSpecBuilder,
)
from src.orchestrator import run_ml_workflow
from data import load_data


class TestMLflowIntegration:
    """Integration tests for MLflow tracking."""
    
    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        # Setup: Create temporary MLflow tracking directory
        self.temp_mlflow_dir = tempfile.mkdtemp()
        self.tracking_uri = f"file://{self.temp_mlflow_dir}"
        mlflow.set_tracking_uri(self.tracking_uri)
        
        yield
        
        # Teardown
        mlflow.end_run()  # Ensure any active run is ended
        shutil.rmtree(self.temp_mlflow_dir, ignore_errors=True)
    
    def test_mlflow_experiment_creation(self):
        """Test that MLflow experiments are created correctly."""
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
                name="test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        mlflow_spec = (
            MLflowSpecBuilder()
            .enable()
            .set_experiment("test_experiment")
            .set_run_name("test_run")
            .set_tracking_uri(self.tracking_uri)
            .build()
        )
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            mlflow_spec=mlflow_spec,
            random_state=42
        )
        
        # Verify experiment was created
        assert "mlflow_run_id" in results
        assert "mlflow_experiment_id" in results
        
        client = MlflowClient(tracking_uri=self.tracking_uri)
        experiment = client.get_experiment_by_name("test_experiment")
        assert experiment is not None
    
    def test_mlflow_parameter_logging(self):
        """Test that parameters are logged to MLflow correctly."""
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
                name="test_classifier",
                hyperparameters={
                    "n_estimators": 15,
                    "max_depth": 4,
                    "learning_rate": 0.1
                }
            )
            .build()[0]
        )
        
        mlflow_spec = (
            MLflowSpecBuilder()
            .enable()
            .set_experiment("param_test")
            .set_tracking_uri(self.tracking_uri)
            .build()
        )
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            mlflow_spec=mlflow_spec,
            random_state=42
        )
        
        # Verify MLflow info is present
        assert "mlflow_run_id" in results
        assert results["mlflow_run_id"] is not None
    
    def test_mlflow_metric_logging(self):
        """Test that metrics are logged to MLflow correctly."""
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
                name="test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        mlflow_spec = (
            MLflowSpecBuilder()
            .enable()
            .set_experiment("metric_test")
            .set_tracking_uri(self.tracking_uri)
            .build()
        )
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
            mlflow_spec=mlflow_spec,
            random_state=42
        )
        
        # Verify MLflow info and metrics in results
        assert "mlflow_run_id" in results
        assert "cv_score" in results
        assert results["cv_score"] > 0
    
    def test_mlflow_model_logging(self):
        """Test that models are logged to MLflow correctly."""
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
                name="test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        mlflow_spec = (
            MLflowSpecBuilder()
            .enable()
            .set_experiment("model_logging_test")
            .set_tracking_uri(self.tracking_uri)
            .build()
        )
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            mlflow_spec=mlflow_spec,
            random_state=42
        )
        
        # Verify MLflow run was created and pipeline exists
        assert "mlflow_run_id" in results
        assert "pipeline" in results
        assert results["pipeline"] is not None
    
    def test_mlflow_tags(self):
        """Test that custom tags are logged to MLflow."""
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
                name="test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        mlflow_spec = (
            MLflowSpecBuilder()
            .enable()
            .set_experiment("tag_test")
            .set_tracking_uri(self.tracking_uri)
            .add_tags({
                "model_type": "gradient_boosting",
                "dataset": "test_data",
                "version": "1.0"
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
            mlflow_spec=mlflow_spec,
            random_state=42
        )
        
        # Verify MLflow run was created with tags in spec
        assert "mlflow_run_id" in results
        assert mlflow_spec.tags is not None
        assert mlflow_spec.tags["model_type"] == "gradient_boosting"
    
    def test_mlflow_model_registry(self):
        """Test model registration in MLflow model registry."""
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
                name="test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        mlflow_spec = (
            MLflowSpecBuilder()
            .enable()
            .set_experiment("registry_test")
            .set_tracking_uri(self.tracking_uri)
            .enable_model_registry(
                model_name="test_registered_model",
                stage="None"
            )
            .build()
        )
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            mlflow_spec=mlflow_spec,
            random_state=42
        )
        
        # Verify workflow completed with MLflow enabled
        assert "mlflow_run_id" in results
        assert "pipeline" in results
        
        # Model registration is configured in spec
        assert mlflow_spec.registered_model_name == "test_registered_model"
    
    def test_mlflow_disabled(self):
        """Test workflow runs correctly when MLflow is disabled."""
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
                name="test_classifier",
                hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )
        
        mlflow_spec = (
            MLflowSpecBuilder()
            .disable()
            .build()
        )
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            mlflow_spec=mlflow_spec,
            random_state=42
        )
        
        # Should not have MLflow info
        assert "mlflow_run_id" not in results
        assert "mlflow_experiment_id" not in results
        
        # But should still have results
        assert "pipeline" in results
        assert "cv_score" in results

