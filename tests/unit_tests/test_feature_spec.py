"""
Tests for feature specifications.
"""
import pytest
from pydantic import ValidationError

from specs.feature_spec import (
    FeatureSpec,
    NumericFeatureSpec,
    CategoricalFeatureSpec,
    FeatureSpecBuilder,
    FeatureSelectionSpec
)


class TestFeatureSpec:
    """Tests for FeatureSpec base class."""
    
    def test_abstract_class(self):
        """Test that FeatureSpec is abstract."""
        with pytest.raises(TypeError):
            FeatureSpec(feature_name="test")
    
    def test_feature_name_validation(self):
        """Test feature name validation."""
        # Valid name
        spec = NumericFeatureSpec(feature_name="valid_feature")
        assert spec.feature_name == "valid_feature"
        
        # Invalid: empty string
        with pytest.raises(ValidationError):
            NumericFeatureSpec(feature_name="")


class TestNumericFeatureSpec:
    """Tests for NumericFeatureSpec."""
    
    def test_default_values(self):
        """Test default values."""
        spec = NumericFeatureSpec(feature_name="test")
        assert spec.imputer_strategy == "mean"
        assert spec.scaler_type == "standard"
        assert spec.imputer_enabled is True
        assert spec.scaler_enabled is True
    
    def test_custom_values(self):
        """Test custom configuration."""
        spec = NumericFeatureSpec(
            feature_name="test",
            imputer_strategy="median",
            scaler_type="minmax",
            imputer_enabled=False
        )
        assert spec.imputer_strategy == "median"
        assert spec.scaler_type == "minmax"
        assert spec.imputer_enabled is False
    
    def test_invalid_imputer_strategy(self):
        """Test invalid imputer strategy."""
        with pytest.raises(ValueError):
            NumericFeatureSpec(
                feature_name="test",
                imputer_strategy="invalid"
            )
    
    def test_invalid_scaler_type(self):
        """Test invalid scaler type."""
        with pytest.raises(ValueError):
            NumericFeatureSpec(
                feature_name="test",
                scaler_type="invalid"
            )
    
    def test_constant_imputer_without_fill_value(self):
        """Test constant imputer requires fill_value."""
        with pytest.raises(ValidationError):
            NumericFeatureSpec(
                feature_name="test",
                imputer_strategy="constant"
            )
    
    def test_constant_imputer_with_fill_value(self):
        """Test constant imputer with fill_value."""
        spec = NumericFeatureSpec(
            feature_name="test",
            imputer_strategy="constant",
            imputer_fill_value=0.0
        )
        assert spec.imputer_fill_value == 0.0
    
    def test_get_feature_type(self):
        """Test get_feature_type method."""
        spec = NumericFeatureSpec(feature_name="test")
        assert spec.get_feature_type() == "numeric"


class TestCategoricalFeatureSpec:
    """Tests for CategoricalFeatureSpec."""
    
    def test_default_values(self):
        """Test default values."""
        spec = CategoricalFeatureSpec(feature_name="test")
        assert spec.imputer_strategy == "most_frequent"
        assert spec.encoder_type == "onehot"
        assert spec.imputer_enabled is True
        assert spec.encoder_enabled is True
    
    def test_custom_values(self):
        """Test custom configuration."""
        spec = CategoricalFeatureSpec(
            feature_name="test",
            imputer_strategy="constant",
            imputer_fill_value="missing",
            encoder_type="ordinal"
        )
        assert spec.imputer_strategy == "constant"
        assert spec.imputer_fill_value == "missing"
        assert spec.encoder_type == "ordinal"
    
    def test_invalid_imputer_strategy(self):
        """Test invalid imputer strategy for categorical."""
        with pytest.raises(ValueError):
            CategoricalFeatureSpec(
                feature_name="test",
                imputer_strategy="mean"  # Invalid for categorical
            )
    
    def test_invalid_encoder_type(self):
        """Test invalid encoder type."""
        with pytest.raises(ValueError):
            CategoricalFeatureSpec(
                feature_name="test",
                encoder_type="invalid"
            )
    
    def test_get_feature_type(self):
        """Test get_feature_type method."""
        spec = CategoricalFeatureSpec(feature_name="test")
        assert spec.get_feature_type() == "categorical"


class TestFeatureSpecBuilder:
    """Tests for FeatureSpecBuilder."""
    
    def test_build_empty(self):
        """Test building empty spec list."""
        builder = FeatureSpecBuilder()
        specs = builder.build()
        assert len(specs) == 0
        assert isinstance(specs, list)
    
    def test_add_numeric_group(self):
        """Test adding numeric feature specifications."""
        builder = FeatureSpecBuilder()
        builder.add_numeric_group(
            ['feature_1', 'feature_2'],
            imputer_strategy='mean',
            scaler_type='standard'
        )
        specs = builder.build()
        
        assert len(specs) == 2
        assert all(isinstance(spec, NumericFeatureSpec) for spec in specs)
        assert specs[0].feature_name == 'feature_1'
        assert specs[1].feature_name == 'feature_2'
    
    def test_add_categorical_group(self):
        """Test adding categorical feature specifications."""
        builder = FeatureSpecBuilder()
        builder.add_categorical_group(
            ['category_1', 'category_2'],
            encoder_type='onehot'
        )
        specs = builder.build()
        
        assert len(specs) == 2
        assert all(isinstance(spec, CategoricalFeatureSpec) for spec in specs)
    
    def test_fluent_interface(self):
        """Test builder fluent interface."""
        specs = (FeatureSpecBuilder()
                .add_numeric_group(['num1', 'num2'])
                .add_categorical_group(['cat1'])
                .build())
        
        assert len(specs) == 3
        assert isinstance(specs[0], NumericFeatureSpec)
        assert isinstance(specs[2], CategoricalFeatureSpec)

