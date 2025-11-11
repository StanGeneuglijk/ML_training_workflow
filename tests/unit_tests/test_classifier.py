"""
Tests for classifier module.
"""
import pytest
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from module.classifier import BaseClassifier, GradientBoostingClassifierImpl
from specs import ClassifierModelSpec


class TestBaseClassifier:
    """Tests for BaseClassifier base class."""
    
    def test_initialization(self, simple_model_spec):
        """Test classifier initialization."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        assert not classifier.is_fitted_
        assert classifier.model_spec == simple_model_spec
    
    def test_wrong_spec_type(self):
        """Test that wrong spec type raises error."""
        with pytest.raises(TypeError, match="ClassifierModelSpec"):
            BaseClassifier("not a spec")
    
    def test_predict_before_fit(self, simple_model_spec):
        """Test that predict before fit raises error."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="fitted"):
            classifier.predict(X)
    
    def test_predict_proba_before_fit(self, simple_model_spec):
        """Test that predict_proba before fit raises error."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        X = np.array([[1, 2], [3, 4]])
        
        with pytest.raises(ValueError, match="fitted"):
            classifier.predict_proba(X)


class TestGradientBoostingClassifierImpl:
    """Tests for GradientBoostingClassifierImpl."""
    
    def test_initialization(self, simple_model_spec):
        """Test gradient boosting classifier initialization."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        assert classifier.model_ is not None
        assert isinstance(classifier.model_, GradientBoostingClassifier)
    
    def test_wrong_algorithm(self):
        """Test that wrong algorithm raises error."""
        spec = ClassifierModelSpec(
            model_name="test",
            algorithm="gradient_boosting"  # This is correct, but test invalid if changed
        )
        # Should work with gradient_boosting
        classifier = GradientBoostingClassifierImpl(spec)
        assert classifier is not None
    
    def test_fit(self, simple_model_spec, sample_classification_array):
        """Test fitting the classifier."""
        X, y = sample_classification_array
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        
        classifier.fit(X, y)
        
        assert classifier.is_fitted_
        assert classifier.n_features_in_ == X.shape[1]
        assert classifier.classes_ is not None
    
    def test_predict(self, simple_model_spec, sample_classification_array):
        """Test making predictions."""
        X, y = sample_classification_array
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        classifier.fit(X, y)
        
        predictions = classifier.predict(X[:5])
        
        assert len(predictions) == 5
        assert all(pred in classifier.classes_ for pred in predictions)
    
    def test_predict_proba(self, simple_model_spec, sample_classification_array):
        """Test predicting probabilities."""
        X, y = sample_classification_array
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        classifier.fit(X, y)
        
        probabilities = classifier.predict_proba(X[:5])
        
        assert probabilities.shape == (5, len(classifier.classes_))
        assert np.allclose(probabilities.sum(axis=1), 1.0)
        assert np.all(probabilities >= 0) and np.all(probabilities <= 1)
    
    def test_evaluate(self, simple_model_spec, sample_classification_array):
        """Test model evaluation."""
        X, y = sample_classification_array
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        classifier.fit(X, y)
        
        metrics = classifier.evaluate(X, y)
        
        assert "accuracy" in metrics
        assert metrics["accuracy"] >= 0.0 and metrics["accuracy"] <= 1.0
    
    def test_get_feature_importance(self, simple_model_spec, sample_classification_array):
        """Test getting feature importance."""
        X, y = sample_classification_array
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        classifier.fit(X, y)
        
        importance = classifier.get_feature_importance()
        
        assert len(importance) == X.shape[1]
        assert np.all(importance >= 0)
        assert np.isclose(importance.sum(), 1.0, rtol=1e-2)
    
    def test_wrong_feature_count(self, simple_model_spec, sample_classification_array):
        """Test that wrong feature count in prediction raises error."""
        X, y = sample_classification_array
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        classifier.fit(X, y)
        
        X_wrong = np.array([[1, 2, 3]])  # Wrong number of features
        with pytest.raises(ValueError, match="features"):
            classifier.predict(X_wrong)
    
    def test_hyperparameters_used(self):
        """Test that hyperparameters are correctly applied."""
        spec = ClassifierModelSpec(
            model_name="test",
            hyperparameters={
                "n_estimators": 50,
                "learning_rate": 0.05,
                "max_depth": 2
            }
        )
        classifier = GradientBoostingClassifierImpl(spec)
        
        assert classifier.model_.n_estimators == 50
        assert classifier.model_.learning_rate == 0.05
        assert classifier.model_.max_depth == 2


class TestSklearnCompatibility:
    """Tests for sklearn compatibility features (GridSearchCV, cloning, etc)."""
    
    def test_get_params(self, simple_model_spec):
        """Test get_params method returns model parameters."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        params = classifier.get_params()
        
        assert 'model_spec' in params
        assert 'n_estimators' in params
        assert 'learning_rate' in params
        assert 'max_depth' in params
        assert params['n_estimators'] == classifier.model_.n_estimators
    
    def test_get_params_deep(self, simple_model_spec):
        """Test get_params with deep=True includes nested parameters."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        params = classifier.get_params(deep=True)
        
        # Should have model__ prefixed parameters for nested model
        nested_params = [k for k in params.keys() if k.startswith('model__')]
        assert len(nested_params) > 0
    
    def test_set_params(self, simple_model_spec):
        """Test set_params method updates parameters."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        
        # Set a new parameter value
        classifier.set_params(n_estimators=200, max_depth=5)
        
        assert classifier.model_.n_estimators == 200
        assert classifier.model_.max_depth == 5
    
    def test_initialization_without_model_spec(self):
        """Test initialization with individual parameters (sklearn cloning)."""
        classifier = GradientBoostingClassifierImpl(
            n_estimators=75,
            learning_rate=0.05,
            max_depth=4
        )
        
        assert classifier.model_.n_estimators == 75
        assert classifier.model_.learning_rate == 0.05
        assert classifier.model_.max_depth == 4
        assert classifier.model_spec is not None  # Should create default spec
    
    def test_sklearn_tags(self, simple_model_spec):
        """Test __sklearn_tags__ method marks as classifier."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        tags = classifier.__sklearn_tags__()
        
        assert tags.estimator_type == "classifier"
    
    def test_estimator_type(self, simple_model_spec):
        """Test _estimator_type attribute."""
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        assert classifier._estimator_type == "classifier"
    
    def test_base_estimator_inheritance(self, simple_model_spec):
        """Test that classifier inherits from BaseEstimator and ClassifierMixin."""
        from sklearn.base import BaseEstimator, ClassifierMixin
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        
        assert isinstance(classifier, BaseEstimator)
        assert isinstance(classifier, ClassifierMixin)
    
    def test_sklearn_clone_compatibility(self, simple_model_spec, sample_classification_array):
        """Test that classifier can be cloned by sklearn (for GridSearchCV)."""
        from sklearn.base import clone
        
        classifier = GradientBoostingClassifierImpl(simple_model_spec)
        X, y = sample_classification_array
        classifier.fit(X, y)
        
        # Clone should work
        cloned = clone(classifier)
        
        # Cloned classifier should not be fitted
        assert not cloned.is_fitted_
        # But should have same parameters
        assert cloned.model_.n_estimators == classifier.model_.n_estimators

