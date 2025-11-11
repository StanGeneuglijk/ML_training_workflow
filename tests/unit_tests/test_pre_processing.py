"""
Tests for preprocessing module.
"""
import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from module.pre_processing import (
    FeatureSpecTransformerFactory,
    FeatureSpecPipeline,
    create_preprocessing_pipeline
)
from specs import NumericFeatureSpec, CategoricalFeatureSpec


class TestFeatureSpecTransformerFactory:
    """Tests for FeatureSpecTransformerFactory."""
    
    def test_create_numeric_transformer(self):
        """Test creating transformer for numeric feature."""
        spec = NumericFeatureSpec(
            feature_name="test",
            imputer_strategy="mean",
            scaler_type="standard"
        )
        transformer = FeatureSpecTransformerFactory.create_transformer(spec)
        
        assert transformer is not None
        assert len(transformer.steps) == 2  # imputer + scaler
    
    def test_create_numeric_transformer_no_scaling(self):
        """Test numeric transformer without scaling."""
        spec = NumericFeatureSpec(
            feature_name="test",
            scaler_type="none"
        )
        transformer = FeatureSpecTransformerFactory.create_transformer(spec)
        
        # Should only have imputer
        assert len(transformer.steps) == 1
    
    def test_create_categorical_transformer(self):
        """Test creating transformer for categorical feature."""
        spec = CategoricalFeatureSpec(
            feature_name="test",
            encoder_type="onehot"
        )
        transformer = FeatureSpecTransformerFactory.create_transformer(spec)
        
        assert transformer is not None
        assert len(transformer.steps) >= 1  # At least encoder
    
    def test_unknown_scaler_type(self):
        """Test that unknown scaler type raises error."""
        spec = NumericFeatureSpec(
            feature_name="test",
            scaler_type="standard"
        )
        # Manually set to invalid type for testing - now raises ValidationError due to Pydantic
        with pytest.raises(ValidationError):
            spec.scaler_type = "unknown"


class TestFeatureSpecPipeline:
    """Tests for FeatureSpecPipeline."""
    
    def test_initialization(self, simple_feature_specs):
        """Test pipeline initialization."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        assert pipeline.feature_specs == simple_feature_specs
    
    def test_empty_specs(self):
        """Test that empty specs raise error."""
        with pytest.raises(ValueError, match="At least one"):
            FeatureSpecPipeline([])
    
    def test_fit(self, simple_feature_specs, sample_classification_data):
        """Test fitting the pipeline."""
        X_df, y_series = sample_classification_data
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        
        pipeline.fit(X_df)
        
        assert pipeline.transformer_ is not None
        assert pipeline.feature_names_ is not None
    
    def test_fit_missing_feature(self, sample_classification_data):
        """Test fitting with missing feature (should skip with warning)."""
        X_df, _ = sample_classification_data
        from specs import NumericFeatureSpec
        
        # Create spec for feature that doesn't exist
        spec = NumericFeatureSpec(feature_name="nonexistent_feature")
        pipeline = FeatureSpecPipeline([spec])
        
        # Should raise error if no valid features
        with pytest.raises(ValueError, match="No valid feature"):
            pipeline.fit(X_df)
    
    def test_transform(self, simple_feature_specs, sample_classification_data):
        """Test transforming data."""
        X_df, _ = sample_classification_data
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        pipeline.fit(X_df)
        
        transformed = pipeline.transform(X_df)
        
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == X_df.shape[0]
    
    def test_transform_before_fit(self, simple_feature_specs, sample_classification_data):
        """Test that transform before fit raises error."""
        X_df, _ = sample_classification_data
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        
        with pytest.raises(ValueError, match="fitted"):
            pipeline.transform(X_df)
    
    def test_fit_transform(self, simple_feature_specs, sample_classification_data):
        """Test fit_transform method."""
        X_df, _ = sample_classification_data
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        
        transformed = pipeline.fit_transform(X_df)
        
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == X_df.shape[0]
    
    def test_array_input(self, simple_feature_specs, sample_classification_array):
        """Test pipeline with numpy array input."""
        X, _ = sample_classification_array
        # Need to rename specs to match array columns
        from specs import NumericFeatureSpec
        specs = [
            NumericFeatureSpec(feature_name=f"feature_{i}")
            for i in range(X.shape[1])
        ]
        
        pipeline = FeatureSpecPipeline(specs)
        transformed = pipeline.fit_transform(X)
        
        assert isinstance(transformed, np.ndarray)
        assert transformed.shape[0] == X.shape[0]


class TestSklearnCompatibility:
    """Tests for sklearn compatibility features."""
    
    def test_get_params(self, simple_feature_specs):
        """Test get_params method for GridSearchCV compatibility."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        params = pipeline.get_params()
        
        assert 'feature_specs' in params
        assert params['feature_specs'] == simple_feature_specs
    
    def test_set_params(self, simple_feature_specs):
        """Test set_params method for GridSearchCV compatibility."""
        from specs import NumericFeatureSpec
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        
        new_specs = [NumericFeatureSpec(feature_name="new_feature")]
        pipeline.set_params(feature_specs=new_specs)
        
        assert pipeline.feature_specs == new_specs
    
    def test_sklearn_tags(self, simple_feature_specs):
        """Test __sklearn_tags__ method."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        tags = pipeline.__sklearn_tags__()
        
        assert tags.estimator_type == "transformer"
    
    def test_estimator_type(self, simple_feature_specs):
        """Test _estimator_type attribute."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        assert pipeline._estimator_type == "transformer"
    
    def test_base_estimator_inheritance(self, simple_feature_specs):
        """Test that pipeline inherits from BaseEstimator."""
        from sklearn.base import BaseEstimator, TransformerMixin
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        
        assert isinstance(pipeline, BaseEstimator)
        assert isinstance(pipeline, TransformerMixin)


class TestCreatePreprocessingPipeline:
    """Tests for create_preprocessing_pipeline function."""
    
    def test_create_pipeline(self, simple_feature_specs):
        """Test creating preprocessing pipeline."""
        pipeline = create_preprocessing_pipeline(simple_feature_specs)
        
        assert isinstance(pipeline, FeatureSpecPipeline)
        assert pipeline.feature_specs == simple_feature_specs

