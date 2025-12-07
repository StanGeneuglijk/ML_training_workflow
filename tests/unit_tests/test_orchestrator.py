"""
Tests for orchestrator module.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.orchestrator import (
    build_ml_pipeline,
    get_workflow_summary,
    run_ml_workflow,
)
from specs import FeatureSpecBuilder, ModelSpecBuilder


@pytest.mark.unit
class TestBuildMLPipeline:
    """Tests for build_ml_pipeline function."""

    def test_build_pipeline(self, simple_feature_specs, simple_model_spec):
        """Test building ML pipeline."""
        pipeline = build_ml_pipeline(simple_feature_specs, simple_model_spec)

        assert pipeline is not None, (
            "Pipeline should not be None after building"
        )
        assert isinstance(pipeline, Pipeline), (
            f"Pipeline should be sklearn Pipeline instance, got {type(pipeline)}"
        )
        assert "preprocessor" in pipeline.named_steps, (
            "Pipeline should contain 'preprocessor' step"
        )
        assert "classifier" in pipeline.named_steps, (
            "Pipeline should contain 'classifier' step"
        )

    def test_pipeline_structure(self, simple_feature_specs, simple_model_spec):
        """Test pipeline structure."""
        pipeline = build_ml_pipeline(simple_feature_specs, simple_model_spec)

        steps = [name for name, _ in pipeline.steps]
        assert steps == ["preprocessor", "classifier"], (
            f"Pipeline steps should be ['preprocessor', 'classifier'], got {steps}"
        )

    def test_pipeline_has_correct_order(self, simple_feature_specs, simple_model_spec):
        """Test that pipeline steps are in the correct order."""
        pipeline = build_ml_pipeline(simple_feature_specs, simple_model_spec)

        step_names = [step[0] for step in pipeline.steps]
        preprocessor_idx = step_names.index("preprocessor")
        classifier_idx = step_names.index("classifier")

        assert preprocessor_idx < classifier_idx, (
            "Preprocessor step should come before classifier step"
        )


@pytest.mark.unit
class TestRunMLWorkflow:
    """Tests for run_ml_workflow function."""

    def test_cross_validation(self, sample_classification_array):
        """Test workflow with cross-validation."""
        X, y = sample_classification_array

        feature_builder = FeatureSpecBuilder()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        feature_specs = feature_builder.add_numeric_group(feature_names).build()

        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier", hyperparameters={"n_estimators": 10}
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

        assert "pipeline" in results, (
            "Results should contain 'pipeline' key"
        )
        assert "cv_score" in results, (
            "Results should contain 'cv_score' key for cross-validation"
        )
        assert "cv_std" in results, (
            "Results should contain 'cv_std' key for cross-validation"
        )
        assert 0.0 <= results["cv_score"] <= 1.0, (
            f"CV score should be between 0.0 and 1.0, got {results['cv_score']}"
        )
        assert isinstance(results["pipeline"], Pipeline), (
            f"Pipeline should be sklearn Pipeline instance, got {type(results['pipeline'])}"
        )

    def test_train_test_split(self, sample_classification_array):
        """Test workflow with train/test split."""
        X, y = sample_classification_array

        feature_builder = FeatureSpecBuilder()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        feature_specs = feature_builder.add_numeric_group(feature_names).build()

        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier", hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )

        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="train_test",
            test_size=0.3,
            random_state=42,
        )

        assert "pipeline" in results, (
            "Results should contain 'pipeline' key"
        )
        assert "test_score" in results, (
            "Results should contain 'test_score' key for train/test split"
        )
        assert "train_size" in results, (
            "Results should contain 'train_size' key"
        )
        assert "test_size" in results, (
            "Results should contain 'test_size' key"
        )
        assert 0.0 <= results["test_score"] <= 1.0, (
            f"Test score should be between 0.0 and 1.0, got {results['test_score']}"
        )
        assert results["train_size"] + results["test_size"] == len(X), (
            f"Train size + test size should equal total samples ({len(X)}), "
            f"got {results['train_size']} + {results['test_size']} = "
            f"{results['train_size'] + results['test_size']}"
        )

    def test_no_validation(self, sample_classification_array):
        """Test workflow without validation."""
        X, y = sample_classification_array

        feature_builder = FeatureSpecBuilder()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        feature_specs = feature_builder.add_numeric_group(feature_names).build()

        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier", hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )

        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="none",
            random_state=42,
        )

        assert "pipeline" in results, (
            "Results should contain 'pipeline' key"
        )
        assert "cv_score" not in results, (
            "Results should not contain 'cv_score' when validation_strategy is 'none'"
        )
        assert "test_score" not in results, (
            "Results should not contain 'test_score' when validation_strategy is 'none'"
        )
        assert results["validation_strategy"] == "none", (
            f"Validation strategy should be 'none', got '{results['validation_strategy']}'"
        )

    def test_dataframe_input(self, sample_classification_data):
        """Test workflow with DataFrame input."""
        X_df, y_series = sample_classification_data

        feature_builder = FeatureSpecBuilder()
        feature_specs = feature_builder.add_numeric_group(
            X_df.columns.tolist()
        ).build()

        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier", hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )

        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X_df,
            y=y_series,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
        )

        assert "cv_score" in results, (
            "Results should contain 'cv_score' key for cross-validation"
        )
        assert isinstance(results["pipeline"], Pipeline), (
            f"Pipeline should be sklearn Pipeline instance, got {type(results['pipeline'])}"
        )

    @pytest.mark.parametrize(
        "validation_strategy,expected_keys,unexpected_keys",
        [
            pytest.param(
                "cross_validation",
                ["cv_score", "cv_std"],
                ["test_score"],
                id="cross_validation_keys",
            ),
            pytest.param(
                "train_test",
                ["test_score", "train_size", "test_size"],
                ["cv_score", "cv_std"],
                id="train_test_keys",
            ),
            pytest.param(
                "none",
                [],
                ["cv_score", "test_score"],
                id="no_validation_keys",
            ),
        ],
    )
    def test_validation_strategy_keys(
        self,
        sample_classification_array,
        validation_strategy,
        expected_keys,
        unexpected_keys,
    ):
        """Test that workflow results contain correct keys for each validation strategy."""
        X, y = sample_classification_array

        feature_builder = FeatureSpecBuilder()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        feature_specs = feature_builder.add_numeric_group(feature_names).build()

        model_spec = (
            ModelSpecBuilder()
            .add_classifier(
                name="test_classifier", hyperparameters={"n_estimators": 10}
            )
            .build()[0]
        )

        validation_params = {"cv_folds": 3} if validation_strategy == "cross_validation" else None
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy=validation_strategy,
            validation_params=validation_params,
            random_state=42,
        )

        for key in expected_keys:
            assert key in results, (
                f"Results should contain '{key}' key for validation_strategy='{validation_strategy}'"
            )

        for key in unexpected_keys:
            assert key not in results, (
                f"Results should not contain '{key}' key for validation_strategy='{validation_strategy}'"
            )


@pytest.mark.unit
class TestGetWorkflowSummary:
    """Tests for get_workflow_summary function."""

    def test_summary_with_cv(self):
        """Test summary with cross-validation results."""
        results = {
            "model_spec": ModelSpecBuilder().add_classifier("test").build()[0],
            "cv_score": 0.85,
            "cv_std": 0.05,
            "train_metrics": {"accuracy": 0.87},
        }

        summary = get_workflow_summary(results)

        assert summary["model_name"] == "test", (
            f"Model name should be 'test', got '{summary['model_name']}'"
        )
        assert summary["validation_strategy"] == "cross_validation", (
            f"Validation strategy should be 'cross_validation', got '{summary['validation_strategy']}'"
        )
        assert summary["cv_score"] == 0.85, (
            f"CV score should be 0.85, got {summary['cv_score']}"
        )
        assert summary["cv_std"] == 0.05, (
            f"CV std should be 0.05, got {summary['cv_std']}"
        )
        assert summary["accuracy"] == 0.87, (
            f"Accuracy should be 0.87, got {summary.get('accuracy')}"
        )

    def test_summary_with_test(self):
        """Test summary with test split results."""
        results = {
            "model_spec": ModelSpecBuilder().add_classifier("test").build()[0],
            "test_score": 0.83,
            "train_metrics": {},
        }

        summary = get_workflow_summary(results)

        assert summary["validation_strategy"] == "train_test", (
            f"Validation strategy should be 'train_test', got '{summary['validation_strategy']}'"
        )
        assert summary["test_score"] == 0.83, (
            f"Test score should be 0.83, got {summary['test_score']}"
        )

    def test_summary_no_validation(self):
        """Test summary without validation."""
        results = {
            "model_spec": ModelSpecBuilder().add_classifier("test").build()[0],
            "train_metrics": {"accuracy": 0.90},
        }

        summary = get_workflow_summary(results)

        assert summary["validation_strategy"] == "none", (
            f"Validation strategy should be 'none', got '{summary['validation_strategy']}'"
        )
        assert "cv_score" not in summary, (
            "Summary should not contain 'cv_score' when no validation is performed"
        )
        assert "test_score" not in summary, (
            "Summary should not contain 'test_score' when no validation is performed"
        )
        assert summary["accuracy"] == 0.90, (
            f"Accuracy should be 0.90, got {summary.get('accuracy')}"
        )

    def test_summary_includes_algorithm(self):
        """Test that summary includes algorithm from model spec."""
        results = {
            "model_spec": ModelSpecBuilder().add_classifier("test").build()[0],
            "train_metrics": {},
        }

        summary = get_workflow_summary(results)

        assert "algorithm" in summary, (
            "Summary should contain 'algorithm' key from model spec"
        )
        assert summary["algorithm"] == "gradient_boosting", (
            f"Algorithm should be 'gradient_boosting', got '{summary['algorithm']}'"
        )

    def test_summary_includes_train_metrics(self):
        """Test that summary includes all train metrics."""
        results = {
            "model_spec": ModelSpecBuilder().add_classifier("test").build()[0],
            "train_metrics": {
                "accuracy": 0.92,
                "precision": 0.88,
                "recall": 0.91,
            },
        }

        summary = get_workflow_summary(results)

        assert "accuracy" in summary, (
            "Summary should include 'accuracy' from train_metrics"
        )
        assert "precision" in summary, (
            "Summary should include 'precision' from train_metrics"
        )
        assert "recall" in summary, (
            "Summary should include 'recall' from train_metrics"
        )
        assert summary["accuracy"] == 0.92, (
            f"Accuracy should be 0.92, got {summary['accuracy']}"
        )
        assert summary["precision"] == 0.88, (
            f"Precision should be 0.88, got {summary['precision']}"
        )
        assert summary["recall"] == 0.91, (
            f"Recall should be 0.91, got {summary['recall']}"
        )
