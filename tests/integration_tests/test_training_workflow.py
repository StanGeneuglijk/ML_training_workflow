"""
Integration tests for end-to-end training workflow.

Tests the core ML training pipeline capabilities:
- Data loading
- Feature preprocessing
- Model training
- Hyperparameter tuning
- Calibration
- Cross-validation
"""
from __future__ import annotations

import tempfile

import numpy as np
import pytest
import shutil

from data import load_data
from specs_training import (
    ClassifierCalibrationSpec,
    FeatureSpecBuilder,
    GridSearchSpec,
    ModelSpecBuilder,
)
from src_training.orchestrator import get_workflow_summary, run_ml_workflow


@pytest.mark.integration
class TestTrainingWorkflowIntegration:
    """Integration tests for complete training workflow."""

    @pytest.fixture(autouse=True)
    def setup_teardown(self):
        """Setup and teardown for each test."""
        self.temp_dir = tempfile.mkdtemp()
        yield
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_basic_training_workflow(self):
        """Test basic training workflow without tuning or calibration."""
        X, y = load_data("classification_data")

        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard",
            )
            .build()
        )

        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier",
                hyperparameters={"n_estimators": 10, "max_depth": 3},
            )
            .build()[0]
        )

        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
            random_state=42,
        )

        assert results is not None, "Results should not be None"
        assert "pipeline" in results, "Results should contain 'pipeline'"
        assert "cv_score" in results, "Results should contain 'cv_score'"
        assert 0.0 <= results["cv_score"] <= 1.0, (
            f"CV score should be between 0.0 and 1.0, got {results['cv_score']}"
        )
        assert "cv_std" in results, "Results should contain 'cv_std'"

        pipeline = results["pipeline"]
        predictions = pipeline.predict(X[:10])
        assert len(predictions) == 10, (
            f"Predictions should have length 10, got {len(predictions)}"
        )
        assert all(pred in [0, 1] for pred in predictions), (
            "All predictions should be binary (0 or 1)"
        )

    def test_workflow_with_hyperparameter_tuning(self):
        """Test workflow with hyperparameter tuning."""
        X, y = load_data("classification_data")
        X = X[:200]
        y = y[:200]

        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard",
            )
            .build()
        )

        model_spec = (
            ModelSpecBuilder().add_classifier(name="test_classifier").build()[0]
        )

        tuning_spec = GridSearchSpec(
            tuning_name="test_grid",
            param_grid={"classifier__n_estimators": [10, 20]},
            scoring="accuracy",
            n_splits=2,
            n_jobs=1,
            verbose=0,
        )

        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            tuning_spec=tuning_spec,
            random_state=42,
        )

        assert "tuning_summary" in results, (
            "Results should contain 'tuning_summary'"
        )
        assert results["tuning_summary"] is not None, (
            "Tuning summary should not be None"
        )
        assert "best_params" in results["tuning_summary"], (
            "Tuning summary should contain 'best_params'"
        )
        assert "best_score" in results["tuning_summary"], (
            "Tuning summary should contain 'best_score'"
        )
        assert results["tuning_summary"]["best_score"] > 0, (
            "Best score should be greater than 0"
        )

    def test_workflow_with_calibration(self):
        """Test workflow with probability calibration."""
        X, y = load_data("classification_data")
        X = X[:200]
        y = y[:200]

        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard",
            )
            .build()
        )

        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier",
                hyperparameters={"n_estimators": 10, "max_depth": 3},
            )
            .build()[0]
        )

        calibration_spec = ClassifierCalibrationSpec(
            calibration_name="test_calibration",
            method="sigmoid",
            cv_strategy=2,
            ensemble=True,
        )

        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 2},
            calibration_spec=calibration_spec,
            random_state=42,
        )

        assert "calibration_summary" in results, (
            "Results should contain 'calibration_summary'"
        )
        assert results["calibration_summary"] is not None, (
            "Calibration summary should not be None"
        )

        pipeline = results["pipeline"]
        probas = pipeline.predict_proba(X[:10])
        assert probas.shape == (10, 2), (
            f"Probability predictions should have shape (10, 2), got {probas.shape}"
        )
        assert np.allclose(probas.sum(axis=1), 1.0), (
            "Probability predictions should sum to 1.0"
        )

    def test_full_workflow_with_tuning_and_calibration(self):
        """Test complete workflow with tuning and calibration."""
        X, y = load_data("classification_data")
        X = X[:150]
        y = y[:150]

        feature_specs = (
            FeatureSpecBuilder()
            .add_numeric_group(
                feature_names=[f"feature_{i}" for i in range(20)],
                imputer_strategy="mean",
                scaler_type="standard",
            )
            .build()
        )

        model_spec = (
            ModelSpecBuilder().add_classifier(name="test_full_classifier").build()[0]
        )

        tuning_spec = GridSearchSpec(
            tuning_name="test_grid",
            param_grid={"classifier__n_estimators": [10, 20]},
            scoring="accuracy",
            n_splits=2,
            n_jobs=1,
            verbose=0,
        )

        calibration_spec = ClassifierCalibrationSpec(
            calibration_name="test_calibration",
            method="sigmoid",
            cv_strategy=2,
            ensemble=True,
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
            random_state=42,
        )

        assert "pipeline" in results, "Results should contain 'pipeline'"
        assert "cv_score" in results, "Results should contain 'cv_score'"
        assert "cv_std" in results, "Results should contain 'cv_std'"
        assert "tuning_summary" in results, (
            "Results should contain 'tuning_summary'"
        )
        assert "calibration_summary" in results, (
            "Results should contain 'calibration_summary'"
        )

        summary = get_workflow_summary(results)
        assert summary is not None, "Summary should not be None"
        assert len(summary) > 0, "Summary should not be empty"
