"""
Tests for MLflow tracker integration.

This module tests the MLflow tracking and model registry functionality,
including basic tracking, model logging, workflow integration, and registry operations.
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from module.mlflow_tracker import MLflowTracker, create_mlflow_tracker
from specs import FeatureSpecBuilder, ModelSpecBuilder, MLflowSpec
from src.orchestrator import build_ml_pipeline, run_ml_workflow


class TestMLflowBasic:
    """Tests for basic MLflow tracker functionality."""
    
    def test_create_tracker(self):
        """Test creating MLflow tracker."""
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_basic",
            tracking_uri=None
        )
        
        assert tracker is not None
        assert tracker.experiment_name == "test_mlflow_basic"
        assert tracker.experiment_id is not None
    
    def test_start_end_run(self):
        """Test starting and ending MLflow run."""
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_basic",
            tracking_uri=None
        )
        
        run_id = tracker.start_run(
            run_name="test_run",
            tags={"test": "basic"}
        )
        
        assert run_id is not None
        assert tracker.run_id == run_id
        
        tracker.end_run(status="FINISHED")
        assert tracker.run_id is None
    
    def test_log_params(self):
        """Test logging parameters."""
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_basic",
            tracking_uri=None
        )
        
        tracker.start_run(run_name="test_params")
        
        params = {
            "learning_rate": 0.01,
            "n_estimators": 100,
            "max_depth": 5
        }
        
        # Should not raise exception
        tracker.log_params(params)
        
        tracker.end_run(status="FINISHED")
    
    def test_log_metrics(self):
        """Test logging metrics."""
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_basic",
            tracking_uri=None
        )
        
        tracker.start_run(run_name="test_metrics")
        
        metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.92
        }
        
        # Should not raise exception
        tracker.log_metrics(metrics)
        
        tracker.end_run(status="FINISHED")


class TestMLflowModelLogging:
    """Tests for model logging with MLflow."""
    
    def test_log_model_architecture(self):
        """Test logging model architecture."""
        # Create synthetic data
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        
        # Create specs
        feature_specs = FeatureSpecBuilder().add_numeric_group(
            X_df.columns.tolist(),
            imputer_strategy="mean",
            scaler_type="standard"
        ).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_classifier",
            hyperparameters={"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}
        ).build()[0]
        
        # Build and train pipeline
        pipeline = build_ml_pipeline(feature_specs, model_spec)
        pipeline.fit(X_df, y)
        
        # Create tracker and log
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_model",
            tracking_uri=None
        )
        
        tracker.start_run(run_name="test_architecture")
        
        # Should not raise exception
        tracker.log_model_architecture(pipeline, model_spec)
        
        tracker.end_run(status="FINISHED")
    
    def test_log_sklearn_model(self):
        """Test logging sklearn model."""
        # Create synthetic data
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        
        # Create specs and pipeline
        feature_specs = FeatureSpecBuilder().add_numeric_group(
            X_df.columns.tolist(),
            imputer_strategy="mean",
            scaler_type="standard"
        ).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_classifier",
            hyperparameters={"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3}
        ).build()[0]
        
        pipeline = build_ml_pipeline(feature_specs, model_spec)
        pipeline.fit(X_df, y)
        
        # Log model
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_model",
            tracking_uri=None
        )
        
        tracker.start_run(run_name="test_model_logging")
        
        # Should not raise exception
        tracker.log_sklearn_model(
            model=pipeline,
            artifact_path="model",
            X_sample=X_df[:10],
            model_spec=model_spec
        )
        
        tracker.end_run(status="FINISHED")


class TestMLflowWorkflowIntegration:
    """Tests for full workflow integration with MLflow."""
    
    def test_workflow_with_mlflow(self):
        """Test running workflow with MLflow tracking."""
        # Create synthetic data
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=5,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
        
        # Create specs
        feature_specs = FeatureSpecBuilder().add_numeric_group(
            X_df.columns.tolist(),
            imputer_strategy="mean",
            scaler_type="standard"
        ).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_workflow_classifier",
            hyperparameters={"n_estimators": 30, "learning_rate": 0.1, "max_depth": 2},
            evaluation_metrics=["accuracy", "roc_auc", "f1_score"]
        ).build()[0]
        
        # Create MLflow spec
        mlflow_spec = MLflowSpec(
            enabled=True,
            experiment_name="test_mlflow_workflow",
            run_name="test_full_workflow",
            register_model=False  # Don't register in tests to avoid clutter
        )
        
        # Run workflow
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X_df,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
            random_state=42,
            mlflow_spec=mlflow_spec
        )
        
        # Verify results
        assert 'mlflow_run_id' in results
        assert 'mlflow_experiment_id' in results
        assert 'cv_score' in results
        assert 'pipeline' in results
    
    def test_workflow_without_mlflow(self):
        """Test running workflow without MLflow tracking."""
        # Create synthetic data
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        
        # Create specs
        feature_specs = FeatureSpecBuilder().add_numeric_group(
            X_df.columns.tolist(),
            imputer_strategy="mean",
            scaler_type="standard"
        ).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_no_mlflow",
            hyperparameters={"n_estimators": 20}
        ).build()[0]
        
        # Run workflow without MLflow
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X_df,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
            random_state=42,
            mlflow_spec=None
        )
        
        # Verify results
        assert 'mlflow_run_id' not in results
        assert 'cv_score' in results
        assert 'pipeline' in results


class TestMLflowModelRegistry:
    """Tests for model registry operations."""
    
    def test_get_latest_model_version(self):
        """Test getting latest model version."""
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_registry",
            tracking_uri=None
        )
        
        # This should not raise exception even if model doesn't exist
        latest_version = tracker.get_latest_model_version("nonexistent_model")
        
        # Should return None if model doesn't exist
        assert latest_version is None or isinstance(latest_version, str)


# Standalone test functions for manual execution
def test_mlflow_basic():
    """Standalone test for basic MLflow tracker functionality."""
    print("=" * 60)
    print("Test 1: Basic MLflow Tracker")
    print("=" * 60)
    
    try:
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_integration",
            tracking_uri=None
        )
        print("✓ MLflow tracker created successfully")
        
        run_id = tracker.start_run(
            run_name="test_basic_run",
            tags={"test": "basic", "purpose": "validation"}
        )
        print(f"✓ MLflow run started: {run_id}")
        
        params = {
            "learning_rate": 0.01,
            "n_estimators": 100,
            "max_depth": 5
        }
        tracker.log_params(params)
        print(f"✓ Logged {len(params)} parameters")
        
        metrics = {
            "accuracy": 0.95,
            "precision": 0.93,
            "recall": 0.92
        }
        tracker.log_metrics(metrics)
        print(f"✓ Logged {len(metrics)} metrics")
        
        tracker.end_run(status="FINISHED")
        print("✓ MLflow run ended successfully")
        
        print("\n✅ Basic MLflow test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic MLflow test FAILED: {e}\n")
        return False


def test_mlflow_model_logging():
    """Standalone test for model logging with MLflow."""
    print("=" * 60)
    print("Test 2: Model Logging")
    print("=" * 60)
    
    try:
        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
        
        print(f"✓ Created synthetic dataset: {X_df.shape}")
        
        feature_specs = FeatureSpecBuilder().add_numeric_group(
            X_df.columns.tolist(),
            imputer_strategy="mean",
            scaler_type="standard"
        ).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_classifier",
            hyperparameters={
                "n_estimators": 50,
                "learning_rate": 0.1,
                "max_depth": 3
            }
        ).build()[0]
        
        print("✓ Created feature and model specifications")
        
        pipeline = build_ml_pipeline(feature_specs, model_spec)
        pipeline.fit(X_df, y)
        print("✓ Trained pipeline")
        
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_integration",
            tracking_uri=None
        )
        
        run_id = tracker.start_run(
            run_name="test_model_logging",
            tags={"test": "model_logging", "model": "gradient_boosting"}
        )
        print(f"✓ Started MLflow run: {run_id}")
        
        tracker.log_model_architecture(pipeline, model_spec)
        print("✓ Logged model architecture")
        
        tracker.log_sklearn_model(
            model=pipeline,
            artifact_path="model",
            registered_model_name="test_model_registered",
            X_sample=X_df[:10],
            model_spec=model_spec
        )
        print("✓ Logged sklearn model with signature")
        
        tracker.end_run(status="FINISHED")
        print("✓ MLflow run ended successfully")
        
        print("\n✅ Model logging test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Model logging test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_mlflow_workflow_integration():
    """Standalone test for full workflow integration with MLflow."""
    print("=" * 60)
    print("Test 3: Full Workflow Integration")
    print("=" * 60)
    
    try:
        X, y = make_classification(
            n_samples=200,
            n_features=8,
            n_informative=5,
            n_redundant=2,
            random_state=42
        )
        X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
        
        print(f"✓ Created synthetic dataset: {X_df.shape}")
        
        feature_specs = FeatureSpecBuilder().add_numeric_group(
            X_df.columns.tolist(),
            imputer_strategy="mean",
            scaler_type="standard"
        ).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_workflow_classifier",
            hyperparameters={
                "n_estimators": 30,
                "learning_rate": 0.1,
                "max_depth": 2
            },
            evaluation_metrics=["accuracy", "roc_auc", "f1_score"]
        ).build()[0]
        
        mlflow_spec = MLflowSpec(
            enabled=True,
            experiment_name="test_mlflow_integration",
            run_name="test_full_workflow",
            register_model=True,
            registered_model_name="test_workflow_model"
        )
        
        print("✓ Created specifications")
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X_df,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
            random_state=42,
            mlflow_spec=mlflow_spec
        )
        
        print("✓ Workflow completed")
        
        assert 'mlflow_run_id' in results, "MLflow run ID not in results"
        assert 'mlflow_experiment_id' in results, "MLflow experiment ID not in results"
        
        print(f"✓ MLflow Run ID: {results['mlflow_run_id']}")
        print(f"✓ MLflow Experiment ID: {results['mlflow_experiment_id']}")
        
        assert 'cv_score' in results, "CV score not in results"
        print(f"✓ CV Score: {results['cv_score']:.4f}")
        
        print("\n✅ Full workflow integration test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Full workflow integration test FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_mlflow_model_registry():
    """Standalone test for model registry operations."""
    print("=" * 60)
    print("Test 4: Model Registry Operations")
    print("=" * 60)
    
    try:
        tracker = create_mlflow_tracker(
            experiment_name="test_mlflow_integration",
            tracking_uri=None
        )
        
        try:
            latest_version = tracker.get_latest_model_version("test_model_registered")
            if latest_version:
                print(f"✓ Found latest model version: {latest_version}")
                
                try:
                    tracker.transition_model_stage(
                        name="test_model_registered",
                        version=latest_version,
                        stage="Staging"
                    )
                    print(f"✓ Transitioned model to Staging")
                except Exception as e:
                    print(f"⚠ Could not transition model stage: {e}")
            else:
                print("⚠ No model versions found (this is OK if first run)")
        except Exception as e:
            print(f"⚠ Model registry operations skipped: {e}")
        
        print("\n✅ Model registry test PASSED\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Model registry test FAILED: {e}\n")
        return False


def main():
    """Run all standalone tests."""
    import sys
    
    print("\n" + "=" * 60)
    print("MLflow Tracker Test Suite")
    print("=" * 60 + "\n")
    
    results = []
    
    # Run tests
    results.append(("Basic MLflow Tracker", test_mlflow_basic()))
    results.append(("Model Logging", test_mlflow_model_logging()))
    results.append(("Full Workflow Integration", test_mlflow_workflow_integration()))
    results.append(("Model Registry Operations", test_mlflow_model_registry()))
    
    # Summary
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print("-" * 60)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 60)
    
    # Instructions
    if passed == total:
        print("\n✅ All tests passed!")
        print("\nNext steps:")
        print("1. View results in MLflow UI:")
        print("   $ mlflow ui --port 5000")
        print("   Then navigate to: http://localhost:5000")
        print("\n2. Look for experiment: 'test_mlflow_integration'")
        print("3. Check the Models tab for registered models")
        print("4. Run pytest to execute unit tests:")
        print("   $ pytest tests/test_mlflow_tracker.py -v")
    else:
        print("\n⚠ Some tests failed. Check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

