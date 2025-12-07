"""
Classifier module./yes!
"""

from __future__ import annotations

import logging
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from specs.model_spec import ClassifierModelSpec
import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)

GRADIENT_BOOSTING_PARAMS = [
    "n_estimators",
    "learning_rate",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "subsample",
]

class BaseClassifier(BaseEstimator, ClassifierMixin):
    """
    Base classifier with sklearn-compatible interface.
    """
    
    _estimator_type = "classifier"
    
    def __sklearn_tags__(
        self
    ) -> dict[str, Any]: 
        """
        Return sklearn tags.

        Args:
            None

        Returns:
            Dictionary of sklearn tags
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"

        return tags
    
    def __init__(
        self, 
        model_spec: ClassifierModelSpec
    ) -> None:
        """
        Initialize classifier.
        
        Args:
            model_spec: Classifier model specification
        """
        if not isinstance(model_spec, ClassifierModelSpec):
            raise TypeError(f"Expected ClassifierModelSpec, got {type(model_spec).__name__}")
        self.model_spec = model_spec
        self.n_features_in_ = None
        self.classes_ = None
        self.is_fitted_ = False
        self.evaluation_metrics_ = {}
    
    def fit(
        self, 
        X: Any,
        y: Any
    ) -> self:
        """
        Fit the classifier.
        
        Args:
            X: Training features 
            y: Training labels 
            
        Returns:
            self
        """
        X_validated, y_validated = utils.validate_training_data(X, y)
        self.n_features_in_ = X_validated.shape[1]
        self.classes_ = np.unique(y_validated)
        
        logger.info(f"Fitting {self.model_spec.model_name} classifier")
        
        self._fit(X_validated, y_validated)
        
        self.is_fitted_ = True

        return self
    
    def _fit(
        self, 
        X: Any,
        y: Any
    ) -> None:
        """
        Internal fit method (to be overridden by subclasses).
        
        Args:
            X: Training features (already validated)
            y: Training labels (already validated)
            
        Returns:
            None
        """
        raise NotImplementedError
    
    def _ensure_fitted(
        self
    ) -> None:
        """
        Ensure the classifier is fitted.
  
        Args:
            None

        Returns:
            None
        """
        if not self.is_fitted_:
            raise ValueError("classifier must be fitted first")

    def predict(
        self, 
        X: Any
    ) -> np.ndarray:
        """
        Make class predictions.
        
        Args:
            X: Features to predict
            
        Returns:
            Predicted class labels 
        """
        self._ensure_fitted()
        
        X_validated = utils.validate_prediction_data(X, self.n_features_in_)
        predict = self._predict(X_validated)

        logger.info(f"Predicted {predict.shape[0]} samples")

        return predict
    
    def predict_proba(
        self, 
        X: Any
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict
            
        Returns:
            Class probabilities a
        """
        self._ensure_fitted()
        
        X_validated = utils.validate_prediction_data(X, self.n_features_in_)
        predict_proba = self._predict_proba(X_validated)

        logger.info(f"Predicted {predict_proba.shape[0]} samples")

        return predict_proba
    
    def _predict(
        self, 
        X: Any
    ) -> np.ndarray:
        """
        Internal prediction method (to be overridden by subclasses).
        
        Args:
            X: Features to predict (already validated)
            
        Returns:
            Predicted class labels
        """
        raise NotImplementedError
    
    def _predict_proba(
        self, 
        X: Any
    ) -> np.ndarray:
        """
        Internal probability prediction (to be overridden by subclasses).
        
        Args:
            X: Features to predict (already validated)
            
        Returns:
            Class probabilities
        """
        raise NotImplementedError
    
    def evaluate(
        self, 
        X: Any,
        y: Any
    ) -> dict[str, float]:
        """
        Evaluate classifier performance.
        
        Args:
            X: Test features 
            y: True labels 
            
        Returns:
            Dictionary of evaluation metrics
        """
        X_validated, y_validated = utils.validate_training_data(X, y)
        predictions = self.predict(X_validated)
        probabilities = self.predict_proba(X_validated)
        
        metrics = {}
        if "accuracy" in self.model_spec.evaluation_metrics:
            metrics["accuracy"] = accuracy_score(y_validated, predictions)
        if "roc_auc" in self.model_spec.evaluation_metrics and len(self.classes_) == 2:
            metrics["roc_auc"] = roc_auc_score(y_validated, probabilities[:, 1])
        self.evaluation_metrics_ = metrics

        logger.info(f"Evaluation metrics: {metrics}")

        return metrics


class GradientBoostingClassifierImpl(BaseClassifier):
    """
    Gradient boosting classifier implementation.
    """
    
    def __init__(
        self, 
        model_spec: ClassifierModelSpec = None,
        **kwargs):
        """
        Initialize gradient boosting classifier.
    
        Args:
            model_spec: Classifier model specification 
            **kwargs: Additional parameters
        """
        if not isinstance(model_spec, ClassifierModelSpec):
            raise TypeError(f"Expected ClassifierModelSpec, got {type(model_spec).__name__}")

        super().__init__(model_spec)
        
        hp = dict(model_spec.hyperparameters)

        for param in GRADIENT_BOOSTING_PARAMS:
            setattr(self, param, hp.get(param))

        gb_params = {**hp}
        gb_params.setdefault("random_state", model_spec.random_state)
        gb_params.setdefault("verbose", 0)
        
        self.model_ = GradientBoostingClassifier(**gb_params)
        
    
    def _fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> None:
        """
        Fit the gradient boosting model.
        
        Args:
            X: Training features (already validated)
            y: Training labels (already validated)
            
        Returns:
            None
        """
        self.model_.fit(X, y)
        logger.info(f"Training completed for {self.model_spec.model_name}")
    
    def _predict(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Make class predictions.
        
        Args:
            X: Features to predict (already validated)
            
        Returns:
            Predicted class labels
        """
        return self.model_.predict(X)
    
    def _predict_proba(
        self, 
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict (already validated)
            
        Returns:
            Class probabilities
        """
        return self.model_.predict_proba(X)
    
    def get_feature_importance(
        self
    ) -> np.ndarray:
        """
        Get feature importance scores.
        
        Returns:
            Array of feature importance values
        """
        self._ensure_fitted()

        importances = self.model_.feature_importances_

        return importances
    
    def get_params(
        self, 
        deep=True
    ) -> dict[str, Any]:
        """
        Get parameters for this estimator.
        
        Allows GridSearchCV to set both wrapper and underlying model parameters.
        
        Args:
            deep: If True, parameters for nested objects
            
        Returns:
            Dictionary of parameter names mapped to their values
        """
        params = {
            'model_spec': self.model_spec,
            **{param: getattr(self, param) for param in GRADIENT_BOOSTING_PARAMS},
        }
    
        if deep:
            model_params = self.model_.get_params(deep=False)
            for key, value in model_params.items():
                params[f'model__{key}'] = value
        

        return params
    
    def set_params(
        self, 
        **params
    ) -> self:
        """
        Set parameters for this estimator.
        
        Allows GridSearchCV to set both wrapper and underlying model parameters.
        
        Args:
            **params: Estimator parameters
            
        Returns:
            self
        """
        model_params = {}
        wrapper_params = {}
        
        for key, value in params.items():
            if key.startswith('model__'):
                model_params[key.replace('model__', '')] = value
            elif key in GRADIENT_BOOSTING_PARAMS:
                setattr(self, key, value)
                model_params[key] = value
            else:
                wrapper_params[key] = value
        
        for key, value in wrapper_params.items():
            setattr(self, key, value)
        
        if model_params:
            self.model_.set_params(**model_params)
        

        return self

def create_classifier(
    model_spec: ClassifierModelSpec
) -> BaseClassifier:
    """
    Create a classifier.

    Args:
        model_spec: The classifier model specification.

    Returns:
        The classifier.
    """
    if not isinstance(model_spec, ClassifierModelSpec):
        raise TypeError(f"Expected ClassifierModelSpec, got {type(model_spec).__name__}")
    if model_spec.algorithm == "gradient_boosting":
        return GradientBoostingClassifierImpl(model_spec)
    raise ValueError(f"Unsupported algorithm: {model_spec.algorithm}")


__all__ = [
    "BaseClassifier",
    "GradientBoostingClassifierImpl",
    "create_classifier",
]
 