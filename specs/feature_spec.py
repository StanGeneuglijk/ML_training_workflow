"""
Feature specification module for ML workflow version 1.

Core feature specifications for preprocessing.
"""
from __future__ import annotations

import abc
import logging
from typing import Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

logger = logging.getLogger(__name__)


class FeatureSpec(BaseModel, abc.ABC):
    """Abstract base class for feature specifications with Pydantic validation."""
    
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    feature_name: str = Field(..., min_length=1, description="Name of the feature")
    enabled: bool = Field(default=True, description="Whether feature is enabled")
    description: Optional[str] = Field(default=None, description="Description of the feature")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    @field_validator('feature_name')
    @classmethod
    def validate_feature_name(cls, v: str) -> str:
        """Validate feature name is not empty or whitespace."""
        if not v.strip():
            raise ValueError("Feature name cannot be empty or whitespace")
        logger.debug(f"Validated feature name: '{v.strip()}'")
        return v.strip()
    
    @abc.abstractmethod
    def get_feature_type(self) -> str:
        """Return the feature type identifier."""
        raise NotImplementedError


class NumericFeatureSpec(FeatureSpec):
    """Configuration for numeric feature processing with Pydantic validation."""
    
    imputer_strategy: Literal["mean", "median", "constant"] = Field(
        default="mean",
        description="Strategy for imputing missing values"
    )
    imputer_fill_value: Optional[float] = Field(
        default=None,
        description="Fill value for constant imputation strategy"
    )
    imputer_enabled: bool = Field(
        default=True,
        description="Whether imputation is enabled"
    )
    scaler_type: Literal["standard", "minmax", "robust", "none"] = Field(
        default="standard",
        description="Type of scaler to use for normalization"
    )
    scaler_enabled: bool = Field(
        default=True,
        description="Whether scaling is enabled"
    )
    
    @model_validator(mode='after')
    def validate_constant_strategy(self) -> 'NumericFeatureSpec':
        """Validate that fill_value is provided when strategy is 'constant'."""
        if self.imputer_strategy == "constant" and self.imputer_fill_value is None:
            raise ValueError(
                f"imputer_fill_value is required when imputer_strategy='constant' "
                f"for feature '{self.feature_name}'"
            )
        logger.debug(f"Validated numeric feature spec for '{self.feature_name}'")
        return self
    
    def get_feature_type(self) -> str:
        """Return 'numeric' as the feature type."""
        return "numeric"


class CategoricalFeatureSpec(FeatureSpec):
    """Configuration for categorical feature processing with Pydantic validation."""
    
    imputer_strategy: Literal["most_frequent", "constant"] = Field(
        default="most_frequent",
        description="Strategy for imputing missing values"
    )
    imputer_fill_value: Optional[str] = Field(
        default=None,
        description="Fill value for constant imputation strategy"
    )
    imputer_enabled: bool = Field(
        default=True,
        description="Whether imputation is enabled"
    )
    encoder_type: Literal["onehot", "ordinal", "target", "none"] = Field(
        default="onehot",
        description="Type of encoder to use"
    )
    encoder_enabled: bool = Field(
        default=True,
        description="Whether encoding is enabled"
    )
    
    @model_validator(mode='after')
    def validate_constant_strategy(self) -> 'CategoricalFeatureSpec':
        """Validate that fill_value is provided when strategy is 'constant'."""
        if self.imputer_strategy == "constant" and self.imputer_fill_value is None:
            raise ValueError(
                f"imputer_fill_value is required when imputer_strategy='constant' "
                f"for feature '{self.feature_name}'"
            )
        logger.debug(f"Validated categorical feature spec for '{self.feature_name}'")
        return self
    
    def get_feature_type(self) -> str:
        """Return 'categorical' as the feature type."""
        return "categorical"


class FeatureSelectionSpec(BaseModel):
    """
    Configuration for feature selection in ML training.
    
    Controls which features to use from the feature store for model training.
    Separate from feature store spec which controls data retrieval.
    
    Attributes:
        selection_mode: How to select features ('all', 'indices', 'names', 'range')
        feature_indices: Specific feature indices to use (0-based)
        feature_names: Specific feature names to use
        feature_range: Range of features to use (start, end)
        exclude_features: Features to exclude from selection
        max_features: Maximum number of features to use
    """
    
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    selection_mode: Literal["all", "indices", "names", "range"] = Field(
        default="all",
        description="Feature selection mode"
    )
    feature_indices: Optional[List[int]] = Field(
        default=None,
        description="Specific feature indices to include (0-based)"
    )
    feature_names: Optional[List[str]] = Field(
        default=None,
        description="Specific feature names to include"
    )
    feature_range: Optional[tuple[int, int]] = Field(
        default=None,
        description="Range of features (start, end) inclusive"
    )
    exclude_features: List[str] = Field(
        default_factory=list,
        description="Features to exclude from selection"
    )
    max_features: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum number of features to use"
    )
    
    @model_validator(mode='after')
    def validate_selection_mode(self) -> 'FeatureSelectionSpec':
        """Validate selection mode parameters."""
        if self.selection_mode == "indices" and not self.feature_indices:
            raise ValueError("feature_indices required when selection_mode='indices'")
        if self.selection_mode == "names" and not self.feature_names:
            raise ValueError("feature_names required when selection_mode='names'")
        if self.selection_mode == "range" and not self.feature_range:
            raise ValueError("feature_range required when selection_mode='range'")
        
        if self.feature_range:
            start, end = self.feature_range
            if start < 0 or end < start:
                raise ValueError(f"Invalid feature_range: ({start}, {end})")
        
        return self
    
    def get_selected_features(self, all_features: List[str]) -> List[str]:
        """
        Get list of selected feature names based on configuration.
        
        Args:
            all_features: List of all available feature names
            
        Returns:
            List of selected feature names
        """
        if self.selection_mode == "all":
            selected = all_features.copy()
        
        elif self.selection_mode == "indices":
            selected = [all_features[i] for i in self.feature_indices if i < len(all_features)]
        
        elif self.selection_mode == "names":
            selected = [f for f in self.feature_names if f in all_features]
        
        elif self.selection_mode == "range":
            start, end = self.feature_range
            selected = all_features[start:end+1]
        
        else:
            selected = all_features.copy()
        
        # Apply exclusions
        selected = [f for f in selected if f not in self.exclude_features]
        
        # Apply max_features limit
        if self.max_features and len(selected) > self.max_features:
            selected = selected[:self.max_features]
        
        return selected


class FeatureSpecBuilder:
    """Builder for creating feature specifications."""
    
    def __init__(self, selection_spec: Optional[FeatureSelectionSpec] = None) -> None:
        """
        Initialize builder.
        
        Args:
            selection_spec: Optional feature selection configuration
        """
        self.specs: List[FeatureSpec] = []
        self.selection_spec = selection_spec
    
    def add_numeric_group(
        self,
        feature_names: List[str],
        imputer_strategy: Literal["mean", "median", "constant"] = "mean",
        scaler_type: Literal["standard", "minmax", "robust", "none"] = "standard",
        **kwargs: Any
    ) -> 'FeatureSpecBuilder':
        """
        Add numeric feature specifications for a group of features.
        
        Args:
            feature_names: List of feature names
            imputer_strategy: Imputation strategy
            scaler_type: Scaling type
            **kwargs: Additional parameters
            
        Returns:
            Self for chaining
        """
        for name in feature_names:
            self.specs.append(NumericFeatureSpec(
                feature_name=name,
                imputer_strategy=imputer_strategy,
                scaler_type=scaler_type,
                **kwargs
            ))
        return self
    
    def add_categorical_group(
        self,
        feature_names: List[str],
        imputer_strategy: Literal["most_frequent", "constant"] = "most_frequent",
        encoder_type: Literal["onehot", "ordinal", "target", "none"] = "onehot",
        **kwargs: Any
    ) -> 'FeatureSpecBuilder':
        """
        Add categorical feature specifications for a group of features.
        
        Args:
            feature_names: List of feature names
            imputer_strategy: Imputation strategy
            encoder_type: Encoding type
            **kwargs: Additional parameters
            
        Returns:
            Self for chaining
        """
        for name in feature_names:
            self.specs.append(CategoricalFeatureSpec(
                feature_name=name,
                imputer_strategy=imputer_strategy,
                encoder_type=encoder_type,
                **kwargs
            ))
        return self
    
    def build(self) -> List[FeatureSpec]:
        """
        Build and return list of feature specifications.
        
        Returns:
            List of feature specs
        """
        return self.specs.copy()


__all__ = [
    "FeatureSpec",
    "NumericFeatureSpec",
    "CategoricalFeatureSpec",
    "FeatureSelectionSpec",
    "FeatureSpecBuilder",
]
