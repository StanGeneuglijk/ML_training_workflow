"""
Unit tests for model specifications.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from specs.model_spec import (
    ModelSpec,
    ClassifierModelSpec,
    ModelSpecBuilder,
)


@pytest.mark.unit
class TestModelSpec:
    """Tests for ModelSpec base class."""

    def test_abstract_class(self):
        """Test that ModelSpec is abstract and cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            ModelSpec(model_name="test")

        assert "abstract" in str(exc_info.value).lower() or "instantiate" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention abstract class or instantiation, got: {exc_info.value}"
        )

    def test_model_name_validation(self):
        """Test model name validation."""
        spec = ClassifierModelSpec(model_name="valid_name")
        assert spec.model_name == "valid_name", (
            f"Model name should be 'valid_name', got '{spec.model_name}'"
        )

        with pytest.raises(ValidationError) as exc_info:
            ClassifierModelSpec(model_name="")

        assert "empty" in str(exc_info.value).lower() or "whitespace" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention empty or whitespace, got: {exc_info.value}"
        )

        with pytest.raises(ValidationError) as exc_info:
            ClassifierModelSpec(model_name=123)  

        assert "string" in str(exc_info.value).lower() or "str" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention string type, got: {exc_info.value}"
        )

    def test_metadata(self):
        """Test metadata handling."""
        spec = ClassifierModelSpec(
            model_name="test",
            metadata={"key": "value"},
        )

        assert spec.metadata["key"] == "value", (
            f"Metadata should contain key='value', got '{spec.metadata.get('key')}'"
        )

    @pytest.mark.parametrize(
        "random_state",
        [
            pytest.param(42, id="set_random_state"),
            pytest.param(None, id="no_random_state"),
        ],
    )
    def test_random_state(self, random_state):
        """Test random state handling."""
        spec = ClassifierModelSpec(model_name="test", random_state=random_state)

        assert spec.random_state == random_state, (
            f"random_state should be {random_state}, got {spec.random_state}"
        )


@pytest.mark.unit
class TestClassifierModelSpec:
    """Tests for ClassifierModelSpec."""

    def test_default_algorithm(self):
        """Test default algorithm is gradient_boosting."""
        spec = ClassifierModelSpec(model_name="test")

        assert spec.algorithm == "gradient_boosting", (
            f"Default algorithm should be 'gradient_boosting', got '{spec.algorithm}'"
        )

    def test_default_hyperparameters(self):
        """Test default hyperparameters are empty dict."""
        spec = ClassifierModelSpec(model_name="test")

        assert isinstance(spec.hyperparameters, dict), (
            f"hyperparameters should be a dict, got {type(spec.hyperparameters)}"
        )
        assert spec.hyperparameters == {}, (
            f"Default hyperparameters should be empty dict, got {spec.hyperparameters}"
        )

    def test_custom_hyperparameters(self):
        """Test custom hyperparameters are stored correctly."""
        spec = ClassifierModelSpec(
            model_name="test",
            hyperparameters={"n_estimators": 50, "learning_rate": 0.05},
        )

        assert spec.hyperparameters["n_estimators"] == 50, (
            f"n_estimators should be 50, got {spec.hyperparameters.get('n_estimators')}"
        )
        assert spec.hyperparameters["learning_rate"] == 0.05, (
            f"learning_rate should be 0.05, got {spec.hyperparameters.get('learning_rate')}"
        )

    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError) as exc_info:
            ClassifierModelSpec(
                model_name="test",
                algorithm="invalid_algorithm",  
            )

        assert "algorithm" in str(exc_info.value).lower() or "gradient_boosting" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention algorithm or gradient_boosting, got: {exc_info.value}"
        )

    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises error."""
        with pytest.raises(ValueError) as exc_info:
            ClassifierModelSpec(
                model_name="test",
                hyperparameters={"learning_rate": 1.5}, 
            )

        assert "learning_rate" in str(exc_info.value).lower(), (
            f"Error should mention 'learning_rate', got: {exc_info.value}"
        )

    def test_invalid_n_estimators(self):
        """Test that invalid n_estimators raises error."""
        with pytest.raises(ValueError) as exc_info:
            ClassifierModelSpec(
                model_name="test",
                hyperparameters={"n_estimators": -1},
            )

        assert "n_estimators" in str(exc_info.value).lower(), (
            f"Error should mention 'n_estimators', got: {exc_info.value}"
        )

    def test_evaluation_metrics(self):
        """Test evaluation metrics defaults."""
        spec = ClassifierModelSpec(model_name="test")

        assert "accuracy" in spec.evaluation_metrics, (
            "Default evaluation_metrics should include 'accuracy'"
        )
        assert "roc_auc" in spec.evaluation_metrics, (
            "Default evaluation_metrics should include 'roc_auc'"
        )

    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError) as exc_info:
            ClassifierModelSpec(
                model_name="test",
                evaluation_metrics=["invalid_metric"],
            )

        assert "metric" in str(exc_info.value).lower() or "invalid" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention metric or invalid, got: {exc_info.value}"
        )

    def test_get_model_type(self):
        """Test get_model_type method returns correct type."""
        spec = ClassifierModelSpec(model_name="test")

        assert spec.get_model_type() == "classifier", (
            f"get_model_type() should return 'classifier', got '{spec.get_model_type()}'"
        )


@pytest.mark.unit
class TestModelSpecBuilder:
    """Tests for ModelSpecBuilder."""

    def test_build_empty(self):
        """Test building empty spec list."""
        builder = ModelSpecBuilder()
        specs = builder.build()

        assert len(specs) == 0, (
            f"Empty builder should return empty list, got {len(specs)} specs"
        )
        assert isinstance(specs, list), (
            f"build() should return a list, got {type(specs)}"
        )

    def test_add_classifier(self):
        """Test adding classifier specification."""
        builder = ModelSpecBuilder()
        builder.add_classifier("test_classifier")
        specs = builder.build()

        assert len(specs) == 1, (
            f"Should have 1 spec, got {len(specs)}"
        )
        assert isinstance(specs[0], ClassifierModelSpec), (
            f"First spec should be ClassifierModelSpec, got {type(specs[0])}"
        )
        assert specs[0].model_name == "test_classifier", (
            f"Model name should be 'test_classifier', got '{specs[0].model_name}'"
        )

    def test_add_multiple_classifiers(self):
        """Test adding multiple classifiers."""
        builder = ModelSpecBuilder()
        builder.add_classifier("classifier1")
        builder.add_classifier("classifier2", hyperparameters={"n_estimators": 50})
        specs = builder.build()

        assert len(specs) == 2, (
            f"Should have 2 specs, got {len(specs)}"
        )
        assert specs[0].model_name == "classifier1", (
            f"First model name should be 'classifier1', got '{specs[0].model_name}'"
        )
        assert specs[1].model_name == "classifier2", (
            f"Second model name should be 'classifier2', got '{specs[1].model_name}'"
        )
        assert specs[1].hyperparameters["n_estimators"] == 50, (
            f"Second model n_estimators should be 50, "
            f"got {specs[1].hyperparameters.get('n_estimators')}"
        )

    def test_fluent_interface(self):
        """Test builder fluent interface with chaining."""
        specs = (
            ModelSpecBuilder()
            .add_classifier("model1")
            .add_classifier("model2")
            .build()
        )

        assert len(specs) == 2, (
            f"Should have 2 specs, got {len(specs)}"
        )
        assert all(
            isinstance(spec, ClassifierModelSpec) for spec in specs
        ), (
            "All specs should be ClassifierModelSpec instances"
        )
