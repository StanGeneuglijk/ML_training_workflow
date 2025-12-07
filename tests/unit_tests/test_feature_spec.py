"""
Unit tests for feature specifications.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from specs.feature_spec import (
    FeatureSpec,
    NumericFeatureSpec,
    CategoricalFeatureSpec,
    FeatureSpecBuilder,
    FeatureSelectionSpec,
)


@pytest.mark.unit
class TestFeatureSpec:
    """Tests for FeatureSpec base class."""

    def test_abstract_class(self):
        """Test that FeatureSpec is abstract and cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            FeatureSpec(feature_name="test")

        assert "abstract" in str(exc_info.value).lower() or "instantiate" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention abstract class or instantiation, got: {exc_info.value}"
        )

    def test_feature_name_validation(self):
        """Test feature name validation."""
        spec = NumericFeatureSpec(feature_name="valid_feature")
        assert spec.feature_name == "valid_feature", (
            f"Feature name should be 'valid_feature', got '{spec.feature_name}'"
        )

        with pytest.raises(ValidationError) as exc_info:
            NumericFeatureSpec(feature_name="")

        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention empty or whitespace, got: {exc_info.value}"
        )


@pytest.mark.unit
class TestNumericFeatureSpec:
    """Tests for NumericFeatureSpec."""

    def test_default_values(self):
        """Test default values for numeric feature spec."""
        spec = NumericFeatureSpec(feature_name="test")

        assert spec.imputer_strategy == "mean", (
            f"Default imputer_strategy should be 'mean', got '{spec.imputer_strategy}'"
        )
        assert spec.scaler_type == "standard", (
            f"Default scaler_type should be 'standard', got '{spec.scaler_type}'"
        )
        assert spec.imputer_enabled is True, (
            "Default imputer_enabled should be True"
        )
        assert spec.scaler_enabled is True, (
            "Default scaler_enabled should be True"
        )

    def test_custom_values(self):
        """Test custom configuration for numeric feature spec."""
        spec = NumericFeatureSpec(
            feature_name="test",
            imputer_strategy="median",
            scaler_type="none",
            imputer_enabled=False,
        )

        assert spec.imputer_strategy == "median", (
            f"imputer_strategy should be 'median', got '{spec.imputer_strategy}'"
        )
        assert spec.scaler_type == "none", (
            f"scaler_type should be 'none', got '{spec.scaler_type}'"
        )
        assert spec.imputer_enabled is False, (
            "imputer_enabled should be False"
        )

    @pytest.mark.parametrize(
        "invalid_strategy",
        [
            pytest.param("invalid", id="invalid_strategy"),
            pytest.param("unknown", id="unknown_strategy"),
        ],
    )
    def test_invalid_imputer_strategy(self, invalid_strategy):
        """Test invalid imputer strategy raises ValidationError."""
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            NumericFeatureSpec(
                feature_name="test",
                imputer_strategy=invalid_strategy,   
            )

        assert "strategy" in str(exc_info.value).lower() or "invalid" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention strategy or invalid, got: {exc_info.value}"
        )

    def test_invalid_scaler_type(self):
        """Test invalid scaler type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            NumericFeatureSpec(
                feature_name="test",
                scaler_type="invalid",  
            )

        assert "scaler" in str(exc_info.value).lower() or "invalid" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention scaler or invalid, got: {exc_info.value}"
        )

    def test_constant_imputer_without_fill_value(self):
        """Test constant imputer requires fill_value."""
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            NumericFeatureSpec(
                feature_name="test",
                imputer_strategy="constant",
            )

        assert "fill_value" in str(exc_info.value).lower(), (
            f"Error should mention 'fill_value', got: {exc_info.value}"
        )

    def test_constant_imputer_with_fill_value(self):
        """Test constant imputer with fill_value works correctly."""
        spec = NumericFeatureSpec(
            feature_name="test",
            imputer_strategy="constant",
            imputer_fill_value=0.0,
        )

        assert spec.imputer_fill_value == 0.0, (
            f"imputer_fill_value should be 0.0, got {spec.imputer_fill_value}"
        )

    def test_get_feature_type(self):
        """Test get_feature_type method returns correct type."""
        spec = NumericFeatureSpec(feature_name="test")

        assert spec.get_feature_type() == "numeric", (
            f"get_feature_type() should return 'numeric', got '{spec.get_feature_type()}'"
        )


@pytest.mark.unit
class TestCategoricalFeatureSpec:
    """Tests for CategoricalFeatureSpec."""

    def test_default_values(self):
        """Test default values for categorical feature spec."""
        spec = CategoricalFeatureSpec(feature_name="test")

        assert spec.imputer_strategy == "most_frequent", (
            f"Default imputer_strategy should be 'most_frequent', "
            f"got '{spec.imputer_strategy}'"
        )
        assert spec.encoder_type == "onehot", (
            f"Default encoder_type should be 'onehot', got '{spec.encoder_type}'"
        )
        assert spec.imputer_enabled is True, (
            "Default imputer_enabled should be True"
        )
        assert spec.encoder_enabled is True, (
            "Default encoder_enabled should be True"
        )

    def test_custom_values(self):
        """Test custom configuration for categorical feature spec."""
        spec = CategoricalFeatureSpec(
            feature_name="test",
            imputer_strategy="constant",
            imputer_fill_value="missing",
            encoder_type="none",
        )

        assert spec.imputer_strategy == "constant", (
            f"imputer_strategy should be 'constant', got '{spec.imputer_strategy}'"
        )
        assert spec.imputer_fill_value == "missing", (
            f"imputer_fill_value should be 'missing', got '{spec.imputer_fill_value}'"
        )
        assert spec.encoder_type == "none", (
            f"encoder_type should be 'none', got '{spec.encoder_type}'"
        )

    def test_invalid_imputer_strategy(self):
        """Test invalid imputer strategy for categorical raises error."""
        with pytest.raises((ValidationError, ValueError)) as exc_info:
            CategoricalFeatureSpec(
                feature_name="test",
                imputer_strategy="mean",  
            )

        assert "strategy" in str(exc_info.value).lower() or "invalid" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention strategy or invalid, got: {exc_info.value}"
        )

    def test_invalid_encoder_type(self):
        """Test invalid encoder type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            CategoricalFeatureSpec(
                feature_name="test",
                encoder_type="invalid", 
            )

        assert "encoder" in str(exc_info.value).lower() or "invalid" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention encoder or invalid, got: {exc_info.value}"
        )

    def test_get_feature_type(self):
        """Test get_feature_type method returns correct type."""
        spec = CategoricalFeatureSpec(feature_name="test")

        assert spec.get_feature_type() == "categorical", (
            f"get_feature_type() should return 'categorical', "
            f"got '{spec.get_feature_type()}'"
        )


@pytest.mark.unit
class TestFeatureSpecBuilder:
    """Tests for FeatureSpecBuilder."""

    def test_build_empty(self):
        """Test building empty spec list."""
        builder = FeatureSpecBuilder()
        specs = builder.build()

        assert len(specs) == 0, (
            f"Empty builder should return empty list, got {len(specs)} specs"
        )
        assert isinstance(specs, list), (
            f"build() should return a list, got {type(specs)}"
        )

    def test_add_numeric_group(self):
        """Test adding numeric feature specifications."""
        builder = FeatureSpecBuilder()
        builder.add_numeric_group(
            ["feature_1", "feature_2"],
            imputer_strategy="mean",
            scaler_type="standard",
        )
        specs = builder.build()

        assert len(specs) == 2, (
            f"Should have 2 specs, got {len(specs)}"
        )
        assert all(
            isinstance(spec, NumericFeatureSpec) for spec in specs
        ), (
            "All specs should be NumericFeatureSpec instances"
        )
        assert specs[0].feature_name == "feature_1", (
            f"First spec should be 'feature_1', got '{specs[0].feature_name}'"
        )
        assert specs[1].feature_name == "feature_2", (
            f"Second spec should be 'feature_2', got '{specs[1].feature_name}'"
        )

    def test_add_categorical_group(self):
        """Test adding categorical feature specifications."""
        builder = FeatureSpecBuilder()
        builder.add_categorical_group(
            ["category_1", "category_2"],
            encoder_type="onehot",
        )
        specs = builder.build()

        assert len(specs) == 2, (
            f"Should have 2 specs, got {len(specs)}"
        )
        assert all(
            isinstance(spec, CategoricalFeatureSpec) for spec in specs
        ), (
            "All specs should be CategoricalFeatureSpec instances"
        )

    def test_fluent_interface(self):
        """Test builder fluent interface with chaining."""
        specs = (
            FeatureSpecBuilder()
            .add_numeric_group(["num1", "num2"])
            .add_categorical_group(["cat1"])
            .build()
        )

        assert len(specs) == 3, (
            f"Should have 3 specs, got {len(specs)}"
        )
        assert isinstance(specs[0], NumericFeatureSpec), (
            "First spec should be NumericFeatureSpec"
        )
        assert isinstance(specs[2], CategoricalFeatureSpec), (
            "Third spec should be CategoricalFeatureSpec"
        )


@pytest.mark.unit
class TestFeatureSelectionSpec:
    """Tests for FeatureSelectionSpec."""

    def test_default_selection_mode(self):
        """Test default selection mode is 'all'."""
        spec = FeatureSelectionSpec()

        assert spec.selection_mode == "all", (
            f"Default selection_mode should be 'all', got '{spec.selection_mode}'"
        )

    def test_selection_by_indices(self):
        """Test feature selection by indices."""
        spec = FeatureSelectionSpec(
            selection_mode="indices",
            feature_indices=[0, 2, 4],
        )

        all_features = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
        selected = spec.get_selected_features(all_features)

        assert len(selected) == 3, (
            f"Should select 3 features, got {len(selected)}"
        )
        assert "feature_0" in selected, (
            "Should include feature_0"
        )
        assert "feature_2" in selected, (
            "Should include feature_2"
        )
        assert "feature_4" in selected, (
            "Should include feature_4"
        )

    def test_selection_by_names(self):
        """Test feature selection by names."""
        spec = FeatureSelectionSpec(
            selection_mode="names",
            feature_names=["feature_1", "feature_3"],
        )

        all_features = ["feature_0", "feature_1", "feature_2", "feature_3"]
        selected = spec.get_selected_features(all_features)

        assert len(selected) == 2, (
            f"Should select 2 features, got {len(selected)}"
        )
        assert "feature_1" in selected, (
            "Should include feature_1"
        )
        assert "feature_3" in selected, (
            "Should include feature_3"
        )

    def test_selection_with_exclude(self):
        """Test feature selection with exclusions."""
        spec = FeatureSelectionSpec(
            selection_mode="all",
            exclude_features=["feature_1", "feature_3"],
        )

        all_features = ["feature_0", "feature_1", "feature_2", "feature_3"]
        selected = spec.get_selected_features(all_features)

        assert "feature_1" not in selected, (
            "Should exclude feature_1"
        )
        assert "feature_3" not in selected, (
            "Should exclude feature_3"
        )
        assert "feature_0" in selected, (
            "Should include feature_0"
        )
        assert "feature_2" in selected, (
            "Should include feature_2"
        )
