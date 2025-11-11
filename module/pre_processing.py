"""
Preprocessing module for ML workflow version 1.

Core preprocessing pipeline for feature transformation.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

from specs.feature_spec import FeatureSpec, NumericFeatureSpec, CategoricalFeatureSpec
import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


class FeatureSpecTransformerFactory:
    """Factory for creating sklearn transformers from feature specifications."""
    
    @staticmethod
    def create_transformer(spec: FeatureSpec) -> Pipeline:
        """
        Create sklearn transformer pipeline from feature specification.
        
        Args:
            spec: Feature specification
            
        Returns:
            Sklearn Pipeline with imputation and scaling/encoding
        """
        steps = []
        
        if isinstance(spec, NumericFeatureSpec):
            # Imputation
            if spec.imputer_enabled:
                steps.append((
                    'imputer',
                    SimpleImputer(
                        strategy=spec.imputer_strategy,
                        fill_value=spec.imputer_fill_value
                    )
                ))
            
            # Scaling
            if spec.scaler_enabled and spec.scaler_type != 'none':
                if spec.scaler_type == 'standard':
                    scaler = StandardScaler()
                elif spec.scaler_type == 'minmax':
                    scaler = MinMaxScaler()
                elif spec.scaler_type == 'robust':
                    scaler = RobustScaler()
                else:
                    raise ValueError(f"Unknown scaler type: {spec.scaler_type}")
                
                steps.append(('scaler', scaler))
        
        elif isinstance(spec, CategoricalFeatureSpec):
            # Imputation
            if spec.imputer_enabled:
                steps.append((
                    'imputer',
                    SimpleImputer(
                        strategy=spec.imputer_strategy,
                        fill_value=spec.imputer_fill_value
                    )
                ))
            
            # Encoding
            if spec.encoder_enabled and spec.encoder_type != 'none':
                if spec.encoder_type == 'onehot':
                    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
                elif spec.encoder_type == 'ordinal':
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                else:
                    raise ValueError(f"Unknown encoder type: {spec.encoder_type}")
                
                steps.append(('encoder', encoder))
        
        if not steps:
            # Return identity transformer if no steps
            from sklearn.preprocessing import FunctionTransformer
            return Pipeline([('identity', FunctionTransformer())])
        
        return Pipeline(steps)


class FeatureSpecPipeline(BaseEstimator, TransformerMixin):
    """
    Preprocessing pipeline built from feature specifications.
    
    Creates sklearn ColumnTransformer from list of feature specifications.
    Inherits from BaseEstimator and TransformerMixin for sklearn compatibility.
    """
    
    # Explicitly set as transformer (not classifier/regressor)
    _estimator_type = "transformer"
    
    def __sklearn_tags__(self):
        """
        Return sklearn tags for modern sklearn versions (1.6+).
        
        This explicitly marks the estimator as a transformer.
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "transformer"
        return tags
    
    def __init__(self, feature_specs: list[FeatureSpec]):
        """
        Initialize preprocessing pipeline.
        
        Args:
            feature_specs: List of feature specifications
        """
        if not feature_specs:
            raise ValueError("At least one feature specification is required")
        
        self.feature_specs = feature_specs
        self.transformer_ = None
        self.feature_names_ = None
        
        logger.info(f"Creating preprocessing pipeline for {len(feature_specs)} features")
    
    def fit(self, X, y=None):
        """
        Fit the preprocessing pipeline.
        
        Args:
            X: Training features (DataFrame or array)
            y: Ignored (for sklearn compatibility)
            
        Returns:
            self
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X_df = X
        else:
            # Convert to DataFrame for processing
            X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
            self.feature_names_ = X_df.columns.tolist()
        
        # Create transformers
        transformers = []
        
        for spec in self.feature_specs:
            if spec.feature_name not in X_df.columns:
                logger.warning(f"Feature '{spec.feature_name}' not found in data, skipping")
                continue
            
            transformer = FeatureSpecTransformerFactory.create_transformer(spec)
            transformers.append((spec.feature_name, transformer, [spec.feature_name]))
        
        if not transformers:
            raise ValueError("No valid feature specifications found in data")
        
        # Create column transformer
        self.transformer_ = ColumnTransformer(
            transformers=transformers,
            remainder='passthrough',
            verbose_feature_names_out=False
        )
        
        logger.info(f"Fitting preprocessing pipeline on {X_df.shape[0]} samples")
        self.transformer_.fit(X_df)
        logger.info("Preprocessing pipeline fitted successfully")
        
        return self
    
    def transform(self, X):
        """
        Transform features using fitted pipeline.
        
        Args:
            X: Features to transform
            
        Returns:
            Transformed features as numpy array
        """
        if self.transformer_ is None:
            raise ValueError("Pipeline must be fitted before transform")
        
        if isinstance(X, pd.DataFrame):
            X_df = X
        else:
            X_df = pd.DataFrame(X, columns=self.feature_names_)
        
        transformed = self.transformer_.transform(X_df)
        logger.debug(f"Transformed {X_df.shape[0]} samples to {transformed.shape}")
        
        return transformed
    
    def fit_transform(self, X, y=None):
        """Fit and transform features."""
        return self.fit(X, y).transform(X)
    
    def get_params(self, deep=True):
        """
        Get parameters for this estimator.
        
        Required for sklearn compatibility and GridSearchCV.
        
        Args:
            deep: If True, return parameters for nested objects
            
        Returns:
            Dictionary of parameter names mapped to their values
        """
        return {"feature_specs": self.feature_specs}
    
    def set_params(self, **params):
        """
        Set parameters for this estimator.
        
        Required for sklearn compatibility and GridSearchCV.
        
        Args:
            **params: Estimator parameters
            
        Returns:
            self
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


def create_preprocessing_pipeline(feature_specs: list[FeatureSpec]) -> FeatureSpecPipeline:
    """
    Create preprocessing pipeline from feature specifications.
    
    Args:
        feature_specs: List of feature specifications
        
    Returns:
        Configured preprocessing pipeline
    """
    return FeatureSpecPipeline(feature_specs)

