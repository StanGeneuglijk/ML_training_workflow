"""
Preprocessing module./yes!
"""

from __future__ import annotations
 
import logging
from typing import Any
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from specs_training.feature_spec import FeatureSpec, NumericFeatureSpec, CategoricalFeatureSpec

import utils
logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


class FeatureSpecTransformerFactory:
    """Factory for creating sklearn transformers."""
    
    @staticmethod
    def create_transformer(
        spec: FeatureSpec
    ) -> Pipeline:
        """
        Create sklearn transformer pipeline.
        
        Args:
            spec: Feature specification
            
        Returns:
            Sklearn Pipeline with imputation and scaling
        """
        steps = []

        if isinstance(spec, NumericFeatureSpec):
            if spec.imputer_enabled:
                steps.append((
                    'imputer',
                    SimpleImputer(
                        strategy=spec.imputer_strategy,
                        fill_value=spec.imputer_fill_value
                    )
                ))
            
            if spec.scaler_enabled and spec.scaler_type != 'none':
                if spec.scaler_type == 'standard':
                    scaler = StandardScaler()
                    steps.append(('scaler', scaler))
                else:
                    raise ValueError(f"Unknown scaler type: {spec.scaler_type}")
        
        elif isinstance(spec, CategoricalFeatureSpec):
            if spec.imputer_enabled:
                steps.append((
                    'imputer',
                    SimpleImputer(
                        strategy=spec.imputer_strategy,
                        fill_value=spec.imputer_fill_value
                    )
                ))
            
            if spec.encoder_enabled and spec.encoder_type != 'none':
                if spec.encoder_type == 'onehot':
                    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                    steps.append(('encoder', encoder))
                else:
                    raise ValueError(f"Unknown encoder type: {spec.encoder_type}")
                
        if not steps:
            return Pipeline(
                [('identity', FunctionTransformer())]
                )
        
        return Pipeline(steps)


class FeatureSpecPipeline(BaseEstimator, TransformerMixin):
    """
    Preprocessing pipeline.
    """
 
    _estimator_type = "transformer"
    
    def __sklearn_tags__(
        self
    ) -> Any: 
        """
        Return sklearn tags.
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"

        return tags
    
    def __init__(
        self, 
        feature_specs: list[FeatureSpec]
    ) -> None:
        """
        Initialize preprocessing pipeline.
        
        Args:
            feature_specs: List of feature specifications
        """
        if not feature_specs:
            raise ValueError("At least one feature specification is required")
        self.feature_specs = feature_specs

        self.feature_name_to_index = {}
        for idx, spec in enumerate(feature_specs):
            if 'column_index' in spec.metadata:
                col_idx = spec.metadata['column_index']
                if not isinstance(col_idx, int):
                    raise ValueError(
                        f"Feature '{spec.feature_name}': metadata['column_index'] must be an integer, "
                        f"got {type(col_idx).__name__}"
                    )
            else:
                col_idx = idx
                logger.info(
                    f"Feature '{spec.feature_name}' using position {idx} as column index. "
                )

            self.feature_name_to_index[spec.feature_name] = col_idx

        self.transformer_ = None
    
    
    def fit(
        self, 
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> self:
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: Training features 
            y: Target 
            
        Returns:
            self
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(X).__name__}")
        
        transformers = []
        
        for spec in self.feature_specs:
            col_idx = self.feature_name_to_index.get(spec.feature_name)
            if col_idx is None or col_idx >= X.shape[1]:
                logger.warning(
                    f"Feature '{spec.feature_name}' mapped to invalid index {col_idx}, skipping"
                )
                continue
            
            transformer = FeatureSpecTransformerFactory.create_transformer(spec)
            transformers.append(
                (spec.feature_name, transformer, [col_idx])
            )
        
        if not transformers:
            raise ValueError("No valid feature specifications found in data")
        
        self.transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        self.transformer_.fit(X)
        logger.info("Preprocessing pipeline fitted successfully")
        
        return self
    

    def transform(
        self, X: np.ndarray
    ) -> np.ndarray:
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Features to transform 
            
        Returns:
            Transformed features
        """
        if self.transformer_ is None:
            raise ValueError("Pipeline must be fitted before transform")
        
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected numpy array, got {type(X).__name__}")
        
        transformed = self.transformer_.transform(X)

        logger.info(f"Transformed features: {transformed.shape}")
        
        return transformed
    

    def fit_transform(
        self, 
        X: np.ndarray, 
        y: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Fit and transform features.
        
        Args:
            X: Training features 
            y: Target 
            
        Returns:
            Transformed features
        """
        transformed = self.fit(X, y).transform(X)
        return transformed
    
    def get_params(
        self, 
        deep: bool = True
    ) -> dict[str, Any]:
        """
        Get parameters for this estimator.
                
        Args:
            deep: If True, return parameters for nested objects
            
        Returns:
            Dictionary of parameter names mapped to their values
        """
        params = {"feature_specs": self.feature_specs}
        return params


    def set_params(
        self, 
        **params: Any
    ) -> self:
        """
        Set parameters for this estimator.
                
        Args:
            **params: Estimator parameters
            
        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


def create_preprocessing_pipeline(
    feature_specs: list[FeatureSpec]
) -> FeatureSpecPipeline:
    """
    Create preprocessing pipeline.
    
    Args:
        feature_specs: List of feature specifications
        
    Returns:
        Configured preprocessing pipeline
    """
    return FeatureSpecPipeline(feature_specs)

