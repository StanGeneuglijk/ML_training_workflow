"""
Unit tests for the classifier module.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from module.classifier import GradientBoostingClassifierImpl, create_classifier
from specs import ClassifierModelSpec


def _fit_classifier(spec: ClassifierModelSpec, sample):
    """Helper to fit classifier with sample data."""
    classifier = GradientBoostingClassifierImpl(spec)
    X, y = sample
    classifier.fit(X, y)
    return classifier, X, y


@pytest.mark.unit
class TestGradientBoostingClassifierImpl:
    """Tests for GradientBoostingClassifierImpl."""

    def test_initialization_requires_model_spec(self):
        """Test initialization requires ClassifierModelSpec."""
        with pytest.raises(TypeError) as exc_info:
            GradientBoostingClassifierImpl(model_spec=None)
        
        assert "ClassifierModelSpec" in str(exc_info.value), (
            f"Error should mention 'ClassifierModelSpec', got: {exc_info.value}"
        )

        spec = ClassifierModelSpec(model_name="gb", algorithm="gradient_boosting")
        classifier = GradientBoostingClassifierImpl(spec)
        
        assert classifier.model_spec == spec, (
            "Classifier should store the provided model spec"
        )

    def test_fit_sets_internal_state(self, simple_model_spec, sample_classification_array):
        """Test fit sets internal state correctly."""
        classifier, X, y = _fit_classifier(simple_model_spec, sample_classification_array)

        assert classifier.is_fitted_, (
            "Classifier should be marked as fitted after fit()"
        )
        assert classifier.n_features_in_ == X.shape[1], (
            f"Expected n_features_in_={X.shape[1]}, got {classifier.n_features_in_}"
        )
        assert len(classifier.classes_) == len(np.unique(y)), (
            f"Expected {len(np.unique(y))} classes, got {len(classifier.classes_)}"
        )

    def test_predict_before_fit_raises(self, simple_model_spec, sample_classification_array):
        """Test predict before fit raises ValueError."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        X, _ = sample_classification_array

        with pytest.raises(ValueError) as exc_info:
            classifier.predict(X)
        
        assert "fitted first" in str(exc_info.value).lower(), (
            f"Error message should mention 'fitted first', got: {exc_info.value}"
        )

    def test_predict_and_predict_proba_shapes(self, simple_model_spec, sample_classification_array):
        """Test predict and predict_proba return correct shapes."""
        classifier, X, _ = _fit_classifier(simple_model_spec, sample_classification_array)
        X_test = X[:5]

        preds = classifier.predict(X_test)
        probas = classifier.predict_proba(X_test)

        assert preds.shape == (5,), (
            f"Predictions should have shape (5,), got {preds.shape}"
        )
        assert probas.shape == (5, len(classifier.classes_)), (
            f"Probabilities should have shape (5, {len(classifier.classes_)}), "
            f"got {probas.shape}"
        )
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6), (
            "Probability predictions should sum to 1.0 for each sample"
        )

    @pytest.mark.parametrize(
        "wrong_features",
        [
            pytest.param(1, id="too_few_features"),
            pytest.param(20, id="too_many_features"),
        ],
    )
    def test_wrong_feature_count_raises(
        self, simple_model_spec, sample_classification_array, wrong_features
    ):
        """Test predict with wrong feature count raises ValueError."""
        classifier, X, _ = _fit_classifier(simple_model_spec, sample_classification_array)
        expected_features = X.shape[1]
        wrong_X = np.random.randn(1, wrong_features)

        with pytest.raises(ValueError) as exc_info:
            classifier.predict(wrong_X)
        
        assert "features" in str(exc_info.value).lower(), (
            f"Error message should mention 'features', got: {exc_info.value}"
        )

    def test_evaluate_returns_requested_metrics(
        self, simple_model_spec, sample_classification_array
    ):
        """Test evaluate returns requested metrics with valid values."""
        classifier, X, y = _fit_classifier(simple_model_spec, sample_classification_array)

        metrics = classifier.evaluate(X, y)

        assert "accuracy" in metrics, (
            "Metrics should include 'accuracy' key"
        )
        assert 0.0 <= metrics["accuracy"] <= 1.0, (
            f"Accuracy should be between 0 and 1, got {metrics['accuracy']}"
        )

    def test_get_feature_importance_requires_fit(
        self, simple_model_spec, sample_classification_array
    ):
        """Test get_feature_importance requires fitted classifier."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)

        with pytest.raises(ValueError) as exc_info:
            classifier.get_feature_importance()
        
        assert "fitted first" in str(exc_info.value).lower(), (
            f"Error message should mention 'fitted first', got: {exc_info.value}"
        )

        classifier.fit(*sample_classification_array)
        importance = classifier.get_feature_importance()
        X, _ = sample_classification_array
        
        assert importance.shape[0] == X.shape[1], (
            f"Feature importance should have shape ({X.shape[1]},), got {importance.shape}"
        )
        assert np.isclose(importance.sum(), 1.0, rtol=1e-2), (
            f"Feature importances should sum to approximately 1.0, got {importance.sum()}"
        )

    def test_get_and_set_params_round_trip(self, simple_model_spec):
        """Test get_params and set_params work correctly."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        params = classifier.get_params(deep=True)
        
        assert params["n_estimators"] == classifier.model_.n_estimators, (
            f"get_params should return n_estimators={classifier.model_.n_estimators}, "
            f"got {params.get('n_estimators')}"
        )
        assert any(key.startswith("model__") for key in params), (
            "get_params should include nested model parameters with 'model__' prefix"
        )

        classifier.set_params(n_estimators=200, max_depth=3)
        
        assert classifier.model_.n_estimators == 200, (
            f"set_params should update n_estimators to 200, got {classifier.model_.n_estimators}"
        )
        assert classifier.model_.max_depth == 3, (
            f"set_params should update max_depth to 3, got {classifier.model_.max_depth}"
        )

    def test_clone_resets_fitted_state(self, simple_model_spec, sample_classification_array):
        """Test clone resets fitted state."""
        classifier, X, _ = _fit_classifier(simple_model_spec, sample_classification_array)

        cloned = clone(classifier)

        assert not cloned.is_fitted_, (
            "Cloned classifier should not be fitted"
        )
        assert cloned.model_.n_estimators == classifier.model_.n_estimators, (
            f"Cloned classifier should have same n_estimators, "
            f"expected {classifier.model_.n_estimators}, got {cloned.model_.n_estimators}"
        )
        
        with pytest.raises(ValueError):
            cloned.predict(X)

    def test_sklearn_tags_and_estimator_type(self, simple_model_spec):
        """Test sklearn tags and estimator type are correctly set."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        tags = classifier.__sklearn_tags__()

        assert tags.estimator_type == "classifier", (
            f"Tags should indicate 'classifier', got '{tags.estimator_type}'"
        )
        assert classifier._estimator_type == "classifier", (
            f"_estimator_type should be 'classifier', got '{classifier._estimator_type}'"
        )
        assert isinstance(classifier, BaseEstimator), (
            "Classifier should be instance of BaseEstimator"
        )
        assert isinstance(classifier, ClassifierMixin), (
            "Classifier should be instance of ClassifierMixin"
        )

    def test_hyperparameters_applied_from_spec(self):
        """Test hyperparameters from spec are applied to model."""
        spec = ClassifierModelSpec(
            model_name="gb",
            hyperparameters={"n_estimators": 50, "learning_rate": 0.05, "max_depth": 2},
        )
        classifier = GradientBoostingClassifierImpl(spec)

        assert classifier.model_.n_estimators == 50, (
            f"n_estimators should be 50, got {classifier.model_.n_estimators}"
        )
        assert classifier.model_.learning_rate == 0.05, (
            f"learning_rate should be 0.05, got {classifier.model_.learning_rate}"
        )
        assert classifier.model_.max_depth == 2, (
            f"max_depth should be 2, got {classifier.model_.max_depth}"
        )


@pytest.mark.unit
class TestCreateClassifierFactory:
    """Tests for create_classifier factory function."""

    def test_supports_gradient_boosting(self, simple_model_spec):
        """Test factory creates GradientBoostingClassifierImpl."""
        classifier = create_classifier(simple_model_spec)
        
        assert isinstance(classifier, GradientBoostingClassifierImpl), (
            f"Factory should return GradientBoostingClassifierImpl, got {type(classifier)}"
        )

    def test_requires_classifier_spec(self):
        """Test factory requires ClassifierModelSpec."""
        with pytest.raises(TypeError) as exc_info:
            create_classifier("not a spec")
        
        assert "ClassifierModelSpec" in str(exc_info.value), (
            f"Error should mention 'ClassifierModelSpec', got: {exc_info.value}"
        )
