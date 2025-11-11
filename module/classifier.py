"""
Classifier module for ML workflow version 1.

Core classifier implementation for gradient boosting models.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from specs.model_spec import ClassifierModelSpec
import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


class BaseClassifier(BaseEstimator, ClassifierMixin):
    """
    Base classifier with sklearn-compatible interface.
    
    Provides common functionality for all classifier implementations.
    """
    
    # Explicitly set estimator type for sklearn compatibility
    _estimator_type = "classifier"
    
    def __sklearn_tags__(self):
        """
        Return sklearn tags for modern sklearn versions (1.6+).
        
        This explicitly marks the estimator as a classifier.
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags
    
    def __init__(self, model_spec: ClassifierModelSpec):
        """
        Initialize classifier with model specification.
        
        Args:
            model_spec: Classifier model specification
        """
        if not isinstance(model_spec, ClassifierModelSpec):
            raise TypeError(f"Expected ClassifierModelSpec, got {type(model_spec).__name__}")
        
        self.model_spec = model_spec
        self.is_fitted_ = False
        self.n_features_in_ = None
        self.classes_ = None
        self.evaluation_metrics_ = {}
    
    def fit(self, X, y):
        """
        Fit the classifier.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            self
        """
        X, y = utils.validate_training_data(X, y)
        
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.is_fitted_ = True
        
        logger.info(f"Fitting {self.model_spec.model_name} classifier")
        return self
    
    def predict(self, X):
        """Make class predictions."""
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before making predictions")
        
        X = utils.validate_prediction_data(X, self.n_features_in_)
        logger.debug(f"Making predictions for {X.shape[0]} samples")
        return self._predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities."""
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before predict_proba")
        
        X = utils.validate_prediction_data(X, self.n_features_in_)
        return self._predict_proba(X)
    
    def _predict(self, X):
        """Internal prediction method (to be overridden)."""
        raise NotImplementedError
    
    def _predict_proba(self, X):
        """Internal probability prediction (to be overridden)."""
        raise NotImplementedError
    
    def evaluate(self, X, y) -> dict:
        """
        Evaluate classifier performance.
        
        Args:
            X: Test features
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        X, y = utils.validate_training_data(X, y)
        
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        metrics = {}
        
        if "accuracy" in self.model_spec.evaluation_metrics:
            metrics["accuracy"] = accuracy_score(y, predictions)
        
        if "roc_auc" in self.model_spec.evaluation_metrics and len(self.classes_) == 2:
            metrics["roc_auc"] = roc_auc_score(y, probabilities[:, 1])
        
        if "f1_score" in self.model_spec.evaluation_metrics:
            metrics["f1_score"] = f1_score(y, predictions, average='weighted')
        
        self.evaluation_metrics_ = metrics
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics


class GradientBoostingClassifierImpl(BaseClassifier):
    """
    Gradient boosting classifier implementation.
    
    Wraps sklearn's GradientBoostingClassifier with specification-driven configuration.
    """
    
    def __init__(self, model_spec: ClassifierModelSpec = None, 
                 n_estimators=100, learning_rate=0.1, max_depth=3,
                 min_samples_split=2, min_samples_leaf=1, subsample=1.0, **kwargs):
        """
        Initialize gradient boosting classifier.
        
        Can be initialized either with a model_spec (recommended) or with
        individual parameters (for sklearn cloning/GridSearchCV compatibility).
        
        Args:
            model_spec: Classifier model specification (if None, uses individual params)
            n_estimators: Number of boosting stages
            learning_rate: Learning rate shrinks the contribution of each tree
            max_depth: Maximum depth of individual trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            subsample: Fraction of samples for fitting individual trees
            **kwargs: Additional parameters (ignored, for sklearn compatibility)
        """
        # Handle sklearn cloning case (no model_spec provided)
        if model_spec is None:
            from specs.model_spec import ClassifierModelSpec
            model_spec = ClassifierModelSpec(
                model_name="gradient_boosting_classifier",
                algorithm="gradient_boosting",
                hyperparameters={
                    "n_estimators": n_estimators,
                    "learning_rate": learning_rate,
                    "max_depth": max_depth,
                    "min_samples_split": min_samples_split,
                    "min_samples_leaf": min_samples_leaf,
                    "subsample": subsample,
                }
            )
        
        super().__init__(model_spec)
        
        if model_spec.algorithm != "gradient_boosting":
            raise ValueError(f"Expected gradient_boosting algorithm, got {model_spec.algorithm}")
        
        # Extract hyperparameters (prefer model_spec, fallback to direct params)
        hp = model_spec.hyperparameters
        
        # Store individual parameters as instance attributes for sklearn cloning
        # This ensures get_params() returns values that match what was passed to __init__
        self.n_estimators = hp.get("n_estimators", n_estimators)
        self.learning_rate = hp.get("learning_rate", learning_rate)
        self.max_depth = hp.get("max_depth", max_depth)
        self.min_samples_split = hp.get("min_samples_split", min_samples_split)
        self.min_samples_leaf = hp.get("min_samples_leaf", min_samples_leaf)
        self.subsample = hp.get("subsample", subsample)
        
        self.model_ = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            subsample=self.subsample,
            random_state=model_spec.random_state,
            verbose=0
        )
        
        logger.info(f"Created gradient boosting classifier: {model_spec.model_name}")
    
    def fit(self, X, y):
        """Fit the gradient boosting classifier."""
        super().fit(X, y)
        
        X, y = utils.validate_training_data(X, y)
        
        logger.info(f"Training {self.model_spec.model_name} on {X.shape[0]} samples")
        self.model_.fit(X, y)
        logger.info(f"Training completed for {self.model_spec.model_name}")
        
        return self
    
    def _predict(self, X):
        """Make class predictions."""
        return self.model_.predict(X)
    
    def _predict_proba(self, X):
        """Predict class probabilities."""
        return self.model_.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Array of feature importance values
        """
        if not self.is_fitted_:
            raise ValueError("Classifier must be fitted before getting feature importance")
        return self.model_.feature_importances_
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Returns parameters from both the wrapper and the underlying sklearn model
        to enable GridSearchCV to tune hyperparameters.
        
        Args:
            deep: If True, return parameters for nested objects
            
        Returns:
            Dictionary of parameter names mapped to their values
        """
        params = {
            'model_spec': self.model_spec,
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'subsample': self.subsample,
        }
        
        if deep:
            # Include nested model parameters
            model_params = self.model_.get_params(deep=False)
            for key, value in model_params.items():
                params[f'model__{key}'] = value
        
        return params
    
    def set_params(self, **params):
        """
        Set parameters for this estimator.
        
        Allows GridSearchCV to set both wrapper and underlying model parameters.
        
        Args:
            **params: Estimator parameters
            
        Returns:
            self
        """
        # Separate model parameters from wrapper parameters
        model_params = {}
        wrapper_params = {}
        
        for key, value in params.items():
            if key.startswith('model__'):
                # Parameter for nested model
                model_params[key.replace('model__', '')] = value
            elif key in ['n_estimators', 'learning_rate', 'max_depth', 
                        'min_samples_split', 'min_samples_leaf', 'subsample']:
                # Direct model parameter - update both instance attribute and model
                setattr(self, key, value)
                model_params[key] = value
            else:
                # Wrapper parameter
                wrapper_params[key] = value
        
        # Set wrapper parameters
        for key, value in wrapper_params.items():
            setattr(self, key, value)
        
        # Set model parameters
        if model_params:
            self.model_.set_params(**model_params)
        
        return self

