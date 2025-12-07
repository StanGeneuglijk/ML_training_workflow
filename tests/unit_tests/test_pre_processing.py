"""
Unit tests for the preprocessing module.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from pydantic import ValidationError
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from module.pre_processing import (
    FeatureSpecTransformerFactory,
    FeatureSpecPipeline,
    create_preprocessing_pipeline,
)
from specs import NumericFeatureSpec, CategoricalFeatureSpec


@pytest.mark.unit
class TestFeatureSpecTransformerFactory:
    """Tests for FeatureSpecTransformerFactory."""

    def test_numeric_transformer_includes_imputer_and_scaler(self, numeric_spec_default):
        """Test numeric transformer includes both imputer and scaler."""
        transformer = FeatureSpecTransformerFactory.create_transformer(numeric_spec_default)

        steps = [name for name, _ in transformer.steps]
        assert steps == ["imputer", "scaler"], (
            f"Expected transformer to have ['imputer', 'scaler'] steps, got {steps}"
        )
        assert isinstance(transformer.steps[0][1], SimpleImputer), (
            "First step should be SimpleImputer instance"
        )
        assert isinstance(transformer.steps[1][1], StandardScaler), (
            "Second step should be StandardScaler instance"
        )

    def test_numeric_transformer_without_enabled_steps_returns_identity(
        self, numeric_spec_no_preprocessing
    ):
        """Test numeric transformer without any enabled steps returns identity."""
        transformer = FeatureSpecTransformerFactory.create_transformer(
            numeric_spec_no_preprocessing
        )

        steps = [name for name, _ in transformer.steps]
        assert steps == ["identity"], (
            f"Expected identity transformer, got steps: {steps}"
        )

    @pytest.mark.parametrize(
        "imputer_enabled,scaler_enabled,expected_steps",
        [
            pytest.param(True, False, ["imputer"], id="imputer_only"),
            pytest.param(False, True, ["scaler"], id="scaler_only"),
            pytest.param(True, True, ["imputer", "scaler"], id="both_enabled"),
            pytest.param(False, False, ["identity"], id="both_disabled"),
        ],
    )

    def test_numeric_transformer_step_combinations(
        self, imputer_enabled, scaler_enabled, expected_steps
    ):
        """Test numeric transformer with different step combinations."""
        spec = NumericFeatureSpec(
            feature_name="num_feature",
            imputer_enabled=imputer_enabled,
            scaler_enabled=scaler_enabled,
        )

        transformer = FeatureSpecTransformerFactory.create_transformer(spec)
        actual_steps = [name for name, _ in transformer.steps]

        assert actual_steps == expected_steps, (
            f"For imputer_enabled={imputer_enabled}, scaler_enabled={scaler_enabled}, "
            f"expected steps {expected_steps}, got {actual_steps}"
        )

    def test_categorical_transformer_includes_imputer_and_encoder(
        self, categorical_spec_default
    ):
        """Test categorical transformer includes imputer and encoder."""
        transformer = FeatureSpecTransformerFactory.create_transformer(categorical_spec_default)
        steps_dict = dict(transformer.steps)

        assert "imputer" in steps_dict, "Categorical transformer should include imputer step"
        assert "encoder" in steps_dict, "Categorical transformer should include encoder step"
        assert isinstance(steps_dict["encoder"], OneHotEncoder), (
            "Encoder should be OneHotEncoder instance"
        )

    @pytest.mark.parametrize(
        "scaler_type,should_raise",
        [
            pytest.param("standard", False, id="valid_standard"),
            pytest.param("none", False, id="valid_none"),
            pytest.param("unknown", True, id="invalid_type"),
        ],
    )
    def test_invalid_scaler_type_validation(self, scaler_type, should_raise):
        """Test scaler type validation."""
        if should_raise:
            with pytest.raises(ValidationError) as exc_info:
                NumericFeatureSpec(
                    feature_name="num_feature", scaler_type=scaler_type  
                )
            assert "scaler_type" in str(exc_info.value).lower() or "validation" in str(
                exc_info.value
            ).lower(), (
                f"ValidationError should mention scaler_type or validation, "
                f"got: {exc_info.value}"
            )
        else:
            spec = NumericFeatureSpec(feature_name="num_feature", scaler_type=scaler_type)
            assert spec.scaler_type == scaler_type, (
                f"Spec should have scaler_type='{scaler_type}', got '{spec.scaler_type}'"
            )


@pytest.mark.unit
class TestFeatureSpecPipeline:
    """Tests for FeatureSpecPipeline."""

    def test_initialization_with_specs(self, simple_feature_specs):
        """Test pipeline initialization with specs."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)

        assert pipeline.feature_specs == simple_feature_specs, (
            "Pipeline should store the provided feature specs"
        )
        assert pipeline.transformer_ is None, (
            "Transformer should be None before fitting"
        )

    def test_empty_specs_raise_value_error(self):
        """Test empty specs raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FeatureSpecPipeline([])
        
        assert "At least one feature specification" in str(exc_info.value), (
            f"Error message should mention 'At least one feature specification', "
            f"got: {exc_info.value}"
        )

    def test_fit_sets_transformer(self, simple_feature_specs, sample_classification_array):
        """Test fit sets transformer."""
        X, _ = sample_classification_array
        pipeline = FeatureSpecPipeline(simple_feature_specs)

        pipeline.fit(X)

        assert pipeline.transformer_ is not None, (
            "Transformer should be set after calling fit()"
        )
        assert len(pipeline.transformer_.transformers) == len(simple_feature_specs), (
            f"Expected {len(simple_feature_specs)} transformers, "
            f"got {len(pipeline.transformer_.transformers)}"
        )

    @pytest.mark.parametrize(
        "invalid_input,expected_type",
        [
            pytest.param("string", "str", id="string_input"),
            pytest.param(["list"], "list", id="list_input"),
            pytest.param({"dict": "value"}, "dict", id="dict_input"),
            pytest.param(42, "int", id="int_input"),
        ],
    )
    def test_fit_requires_numpy_array(
        self, simple_feature_specs, invalid_input, expected_type
    ):
        """Test fit requires numpy array and raises TypeError for invalid input."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)

        with pytest.raises(TypeError) as exc_info:
            pipeline.fit(invalid_input)
        
        assert "numpy array" in str(exc_info.value).lower(), (
            f"Error message should mention 'numpy array', got: {exc_info.value}"
        )
        assert expected_type.lower() in str(exc_info.value).lower() or "got" in str(
            exc_info.value
        ).lower(), (
            f"Error should indicate wrong type, got: {exc_info.value}"
        )

    def test_transform_returns_numpy_array(self, fitted_pipeline, sample_classification_array):
        """Test transform returns numpy array with correct shape."""
        X, _ = sample_classification_array

        transformed = fitted_pipeline.transform(X)

        assert isinstance(transformed, np.ndarray), (
            f"Transform should return numpy array, got {type(transformed)}"
        )
        assert transformed.shape[0] == X.shape[0], (
            f"Transform should preserve number of rows: expected {X.shape[0]}, "
            f"got {transformed.shape[0]}"
        )

    def test_transform_before_fit_raises(self, simple_feature_specs, sample_classification_array):
        """Test transform before fit raises ValueError."""
        X, _ = sample_classification_array
        pipeline = FeatureSpecPipeline(simple_feature_specs)

        with pytest.raises(ValueError) as exc_info:
            pipeline.transform(X)
        
        assert "fitted before transform" in str(exc_info.value).lower(), (
            f"Error message should mention 'fitted before transform', "
            f"got: {exc_info.value}"
        )

    def test_fit_transform_completes_round_trip(
        self, simple_feature_specs, sample_classification_array
    ):
        """Test fit_transform completes round trip and returns valid output."""
        X, _ = sample_classification_array
        pipeline = FeatureSpecPipeline(simple_feature_specs)

        transformed = pipeline.fit_transform(X)

        assert isinstance(transformed, np.ndarray), (
            f"fit_transform should return numpy array, got {type(transformed)}"
        )
        assert transformed.shape[0] == X.shape[0], (
            f"fit_transform should preserve number of rows: expected {X.shape[0]}, "
            f"got {transformed.shape[0]}"
        )
        assert not np.isnan(transformed).any(), (
            "Transformed data should not contain NaN values"
        )

    @pytest.mark.parametrize(
        "column_index",
        [
            pytest.param(999, id="too_large_index"),
            pytest.param(-1, id="negative_index"),
            pytest.param(100, id="out_of_bounds"),
        ],
    )
    def test_fit_with_invalid_feature_index_raises(
        self, sample_classification_array, column_index
    ):
        """Test fit with invalid feature index raises ValueError."""
        X, _ = sample_classification_array
        spec = NumericFeatureSpec(
            feature_name="missing", metadata={"column_index": column_index}
        )
        pipeline = FeatureSpecPipeline([spec])

        with pytest.raises(ValueError) as exc_info:
            pipeline.fit(X)
        
        assert "No valid feature specifications" in str(exc_info.value), (
            f"Error message should mention 'No valid feature specifications', "
            f"got: {exc_info.value}"
        )

    def test_fit_with_invalid_feature_index_logs_warning(
        self, sample_classification_array
    ):
        """Test fit logs warning when feature index is invalid."""
        X, _ = sample_classification_array
        spec = NumericFeatureSpec(
            feature_name="missing", metadata={"column_index": 999}
        )
        pipeline = FeatureSpecPipeline([spec])

        with patch("module.pre_processing.logger") as mock_logger:
            try:
                pipeline.fit(X)
            except ValueError:
                pass  
        
        warning_calls = [
            call for call in mock_logger.warning.call_args_list
            if "invalid index" in str(call).lower() or "missing" in str(call).lower()
        ]
        assert len(warning_calls) > 0, (
            "Logger should log warning about invalid feature index"
        )

    def test_pipeline_idempotent_transform(self, fitted_pipeline, sample_classification_array):
        """Test that transform is idempotent (same input gives same output)."""
        X, _ = sample_classification_array

        transformed1 = fitted_pipeline.transform(X)
        transformed2 = fitted_pipeline.transform(X)

        np.testing.assert_array_equal(
            transformed1,
            transformed2,
            err_msg="Transform should be idempotent: same input should produce same output",
        )

    def test_transform_requires_numpy_array(self, fitted_pipeline):
        """Test transform requires numpy array and raises TypeError for invalid input."""
        invalid_input = "not an array"

        with pytest.raises(TypeError) as exc_info:
            fitted_pipeline.transform(invalid_input)
        
        assert "numpy array" in str(exc_info.value).lower(), (
            f"Error message should mention 'numpy array', got: {exc_info.value}"
        )


@pytest.mark.unit
class TestSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_get_params_returns_feature_specs(self, simple_feature_specs):
        """Test get_params returns feature_specs."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)

        params = pipeline.get_params()

        assert "feature_specs" in params, (
            "get_params() should return 'feature_specs' key"
        )
        assert params["feature_specs"] == simple_feature_specs, (
            "get_params() should return the same feature specs used in initialization"
        )

    @pytest.mark.parametrize(
        "deep",
        [
            pytest.param(True, id="deep_true"),
            pytest.param(False, id="deep_false"),
        ],
    )
    def test_get_params_with_deep_flag(self, simple_feature_specs, deep):
        """Test get_params with different deep flag values."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)

        params = pipeline.get_params(deep=deep)

        assert "feature_specs" in params, (
            f"get_params(deep={deep}) should return 'feature_specs' key"
        )
        assert isinstance(params["feature_specs"], list), (
            f"feature_specs should be a list, got {type(params['feature_specs'])}"
        )

    def test_set_params_updates_specs(self, simple_feature_specs):
        """Test set_params updates specs."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)
        new_specs = [NumericFeatureSpec(feature_name="alt_feature")]

        result = pipeline.set_params(feature_specs=new_specs)

        assert pipeline.feature_specs == new_specs, (
            "set_params should update the feature_specs attribute"
        )
        assert result is pipeline, (
            "set_params should return self for method chaining"
        )

    def test_sklearn_tags_mark_transformer(self, simple_feature_specs):
        """Test sklearn tags mark as transformer."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)

        tags = pipeline.__sklearn_tags__()

        assert tags.estimator_type == "transformer", (
            f"Pipeline should be marked as transformer, got '{tags.estimator_type}'"
        )

    def test_estimator_type_attribute(self, simple_feature_specs):
        """Test _estimator_type attribute is set correctly."""
        pipeline = FeatureSpecPipeline(simple_feature_specs)

        assert hasattr(pipeline, "_estimator_type"), (
            "Pipeline should have _estimator_type attribute"
        )
        assert pipeline._estimator_type == "transformer", (
            f"Pipeline _estimator_type should be 'transformer', "
            f"got '{pipeline._estimator_type}'"
        )


@pytest.mark.unit
class TestCreatePreprocessingPipeline:
    """Tests for create_preprocessing_pipeline factory function."""

    def test_factory_returns_pipeline_instance(self, simple_feature_specs):
        """Test factory returns pipeline instance."""
        pipeline = create_preprocessing_pipeline(simple_feature_specs)

        assert isinstance(pipeline, FeatureSpecPipeline), (
            f"Factory should return FeatureSpecPipeline instance, "
            f"got {type(pipeline)}"
        )
        assert pipeline.feature_specs == simple_feature_specs, (
            "Factory should create pipeline with provided feature specs"
        )

    def test_factory_with_empty_specs_raises(self):
        """Test factory with empty specs raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            create_preprocessing_pipeline([])
        
        assert "At least one feature specification" in str(exc_info.value), (
            f"Error message should mention 'At least one feature specification', "
            f"got: {exc_info.value}"
        )

    @pytest.mark.parametrize(
        "spec_type,spec_kwargs",
        [
            pytest.param(
                "numeric",
                {"feature_name": "num_feature"},
                id="numeric_spec",
            ),
            pytest.param(
                "categorical",
                {"feature_name": "cat_feature"},
                id="categorical_spec",
            ),
        ],
    )
    def test_factory_with_different_spec_types(self, spec_type, spec_kwargs):
        """Test factory with different spec types."""
        if spec_type == "numeric":
            spec = NumericFeatureSpec(**spec_kwargs)
        else:
            spec = CategoricalFeatureSpec(**spec_kwargs)

        pipeline = create_preprocessing_pipeline([spec])

        assert isinstance(pipeline, FeatureSpecPipeline), (
            f"Factory should return FeatureSpecPipeline for {spec_type} spec"
        )
        assert len(pipeline.feature_specs) == 1, (
            f"Pipeline should have 1 feature spec, got {len(pipeline.feature_specs)}"
        )
        assert pipeline.feature_specs[0].feature_name == spec_kwargs["feature_name"], (
            f"Pipeline should use provided feature name '{spec_kwargs['feature_name']}'"
        )
