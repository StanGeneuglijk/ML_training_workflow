"""
Unit tests for the calibration module.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from module.calibration import ClassifierCalibration, create_calibration
from specs_training import CalibrationSpec, ClassifierCalibrationSpec


def _make_data(n_samples: int = 120):
    """Create test classification data."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=6,
        n_informative=4,
        n_redundant=1,
        random_state=42,
    )
    return X, y


def _make_pipeline():
    """Create a test pipeline with scaler and classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )


@pytest.mark.unit
class TestClassifierCalibration:
    """Tests for ClassifierCalibration."""

    def test_fit_predict_and_summary(self):
        """Test fit, predict, and summary functionality."""
        X, y = _make_data()
        pipeline = _make_pipeline().fit(X, y)
        spec = ClassifierCalibrationSpec(
            calibration_name="calib",
            method="sigmoid",
            cv_strategy="kfold",
            n_splits=3,
            ensemble=True,
        )

        calibration = ClassifierCalibration(spec).fit(pipeline, X, y)

        proba = calibration.predict_proba(X[:10])
        preds = calibration.predict(X[:10])

        assert proba.shape == (10, 2), (
            f"Probability predictions should have shape (10, 2), got {proba.shape}"
        )
        assert preds.shape == (10,), (
            f"Predictions should have shape (10,), got {preds.shape}"
        )
        
        summary = calibration.get_results_summary()
        
        assert summary["calibration_type"] == "classifier_calibration", (
            f"Summary should indicate 'classifier_calibration', "
            f"got '{summary.get('calibration_type')}'"
        )
        assert summary["method"] == "sigmoid", (
            f"Summary should indicate method='sigmoid', got '{summary.get('method')}'"
        )
        assert summary["n_calibrated_models"] >= 1, (
            f"Should have at least 1 calibrated model, got {summary.get('n_calibrated_models')}"
        )

    def test_predict_before_fit_raises(self):
        """Test predict before fit raises ValueError."""
        X, _ = _make_data()
        pipeline = _make_pipeline()
        spec = ClassifierCalibrationSpec(calibration_name="calib")
        calibration = ClassifierCalibration(spec)

        with pytest.raises(ValueError) as exc_info:
            calibration.predict(X)
        
        assert "fitted first" in str(exc_info.value).lower(), (
            f"Error message should mention 'fitted first', got: {exc_info.value}"
        )

        with pytest.raises(ValueError) as exc_info:
            calibration.predict_proba(X)
        
        assert "fitted first" in str(exc_info.value).lower(), (
            f"Error message for predict_proba should mention 'fitted first', "
            f"got: {exc_info.value}"
        )

    @pytest.mark.parametrize(
        "cv_strategy,expected_strategy",
        [
            pytest.param("stratified_kfold", "stratified_kfold", id="stratified_kfold"),
            pytest.param("kfold", "kfold", id="kfold"),
        ],
    )
    def test_summary_includes_cv_strategy(self, cv_strategy, expected_strategy):
        """Test summary includes correct CV strategy."""
        X, y = _make_data()
        pipeline = _make_pipeline().fit(X, y)
        spec = ClassifierCalibrationSpec(
            calibration_name="calib",
            cv_strategy=cv_strategy,
            n_splits=2,
        )

        calibration = ClassifierCalibration(spec).fit(pipeline, X, y)
        summary = calibration.get_results_summary()
        
        assert summary["cv_strategy"] == expected_strategy, (
            f"Summary should indicate cv_strategy='{expected_strategy}', "
            f"got '{summary.get('cv_strategy')}'"
        )

    def test_calibration_improves_probability_calibration(self):
        """Test that calibration improves probability calibration."""
        X, y = _make_data()
        pipeline = _make_pipeline().fit(X, y)
        spec = ClassifierCalibrationSpec(
            calibration_name="calib",
            method="isotonic",
            cv_strategy="kfold",
            n_splits=3,
        )

        calibration = ClassifierCalibration(spec).fit(pipeline, X, y)
        
        proba = calibration.predict_proba(X[:10])
        
        assert np.all(proba >= 0), (
            "Calibrated probabilities should be >= 0"
        )
        assert np.all(proba <= 1), (
            "Calibrated probabilities should be <= 1"
        )
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6), (
            "Calibrated probabilities should sum to 1.0 for each sample"
        )


@pytest.mark.unit
class TestCreateCalibrationFactory:
    """Tests for create_calibration factory function."""

    def test_factory_returns_classifier_calibration(self):
        """Test factory returns ClassifierCalibration instance."""
        spec = ClassifierCalibrationSpec(calibration_name="calib")
        calibration = create_calibration(spec)
        
        assert isinstance(calibration, ClassifierCalibration), (
            f"Factory should return ClassifierCalibration, got {type(calibration)}"
        )

    def test_factory_requires_spec_instance(self):
        """Test factory requires spec instance."""
        with pytest.raises(TypeError) as exc_info:
            create_calibration("not a spec")
        
        assert "TypeError" in type(exc_info.value).__name__ or "spec" in str(
            exc_info.value
        ).lower(), (
            f"Error should be TypeError or mention 'spec', got: {exc_info.value}"
        )

    def test_factory_rejects_unknown_spec_type(self):
        """Test factory rejects unsupported calibration spec type."""
        base_spec = CalibrationSpec(calibration_name="base")

        with pytest.raises(ValueError) as exc_info:
            create_calibration(base_spec)
        
        assert "unsupported calibration type" in str(exc_info.value).lower(), (
            f"Error message should mention 'unsupported calibration type', "
            f"got: {exc_info.value}"
        )
