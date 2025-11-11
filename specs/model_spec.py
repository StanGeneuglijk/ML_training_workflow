"""
Model specification module for ML workflow version 1.

Core model specifications for classifier models only.
"""
from __future__ import annotations

import abc
import logging
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

logger = logging.getLogger(__name__)


class ModelSpec(BaseModel, abc.ABC):
    """Abstract base class for model specifications with Pydantic validation."""
    
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    model_name: str = Field(..., min_length=1, description="Name of the model")
    enabled: bool = Field(default=True, description="Whether model is enabled")
    random_state: Optional[int] = Field(default=None, description="Random state for reproducibility")
    description: Optional[str] = Field(default=None, description="Description of the model")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model name is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Model name cannot be empty or whitespace")
        logger.debug(f"Validated model name: '{v.strip()}'")
        return v.strip()
    
    @abc.abstractmethod
    def get_model_type(self) -> str:
        """Return the model type identifier."""
        raise NotImplementedError


class ClassifierModelSpec(ModelSpec):
    """Configuration specification for classifier models with Pydantic validation."""
    
    algorithm: str = Field(
        default="gradient_boosting",
        description="Machine learning algorithm to use"
    )
    hyperparameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Hyperparameters for the model"
    )
    fit_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the fit method"
    )
    evaluation_metrics: List[str] = Field(
        default_factory=lambda: ["accuracy", "roc_auc", "f1_score"],
        description="Metrics to evaluate model performance"
    )
    
    @field_validator('algorithm')
    @classmethod
    def validate_algorithm(cls, v: str) -> str:
        """Validate that only gradient_boosting is supported."""
        if v != "gradient_boosting":
            raise ValueError(
                f"Only 'gradient_boosting' algorithm is supported, got '{v}'"
            )
        return v
    
    @field_validator('evaluation_metrics')
    @classmethod
    def validate_metrics(cls, v: List[str]) -> List[str]:
        """Validate evaluation metrics are valid."""
        valid_metrics = [
            "accuracy", "roc_auc", "f1_score", 
            "precision", "recall", "neg_log_loss"
        ]
        for metric in v:
            if metric not in valid_metrics:
                raise ValueError(
                    f"Invalid metric '{metric}'. Valid metrics: {valid_metrics}"
                )
        return v
    
    @field_validator('hyperparameters')
    @classmethod
    def validate_hyperparameters(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate hyperparameters for gradient boosting."""
        if not v:
            # Return default hyperparameters if none provided
            return {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "subsample": 1.0,
            }
        
        # Validate learning_rate
        if "learning_rate" in v:
            lr = v["learning_rate"]
            if not isinstance(lr, (int, float)):
                raise TypeError("learning_rate must be a number")
            if lr <= 0 or lr > 1:
                raise ValueError("learning_rate must be between 0 and 1 (exclusive of 0)")
        
        # Validate n_estimators
        if "n_estimators" in v:
            n_est = v["n_estimators"]
            if not isinstance(n_est, int):
                raise TypeError("n_estimators must be an integer")
            if n_est <= 0:
                raise ValueError("n_estimators must be a positive integer")
        
        # Validate max_depth
        if "max_depth" in v:
            md = v["max_depth"]
            if not isinstance(md, int):
                raise TypeError("max_depth must be an integer")
            if md <= 0:
                raise ValueError("max_depth must be a positive integer")
        
        # Validate min_samples_split
        if "min_samples_split" in v:
            mss = v["min_samples_split"]
            if not isinstance(mss, int):
                raise TypeError("min_samples_split must be an integer")
            if mss < 2:
                raise ValueError("min_samples_split must be at least 2")
        
        # Validate min_samples_leaf
        if "min_samples_leaf" in v:
            msl = v["min_samples_leaf"]
            if not isinstance(msl, int):
                raise TypeError("min_samples_leaf must be an integer")
            if msl < 1:
                raise ValueError("min_samples_leaf must be at least 1")
        
        # Validate subsample
        if "subsample" in v:
            ss = v["subsample"]
            if not isinstance(ss, (int, float)):
                raise TypeError("subsample must be a number")
            if ss <= 0 or ss > 1:
                raise ValueError("subsample must be between 0 and 1 (exclusive of 0)")
        
        logger.debug(f"Validated hyperparameters: {v}")
        return v
    
    def get_model_type(self) -> str:
        """Return 'classifier' as the model type."""
        return "classifier"


class ModelSpecBuilder:
    """Builder for creating model specifications."""
    
    def __init__(self) -> None:
        """Initialize builder."""
        self.specs: List[ModelSpec] = []
    
    def add_classifier(
        self, 
        name: str, 
        **kwargs: Any
    ) -> 'ModelSpecBuilder':
        """
        Add a classifier model specification.
        
        Args:
            name: Name of the classifier
            **kwargs: Additional parameters for ClassifierModelSpec
            
        Returns:
            Self for chaining
        """
        self.specs.append(ClassifierModelSpec(model_name=name, **kwargs))
        return self
    
    def build(self) -> List[ModelSpec]:
        """
        Build and return list of model specifications.
        
        Returns:
            List of model specs
        """
        return self.specs.copy()


__all__ = [
    "ModelSpec",
    "ClassifierModelSpec",
    "ModelSpecBuilder",
]
