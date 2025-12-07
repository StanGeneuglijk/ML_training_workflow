"""
Unit tests for MLflow specifications.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from specs.mlflow_spec import (
    MLflowSpec,
    MLflowSpecBuilder,
    create_default_mlflow_spec,
    create_production_mlflow_spec,
)


@pytest.mark.unit
class TestMLflowSpec:
    """Tests for MLflowSpec core functionality."""

    def test_default_values(self):
        """Test default values for MLflowSpec."""
        spec = MLflowSpec()

        assert spec.enabled is True, (
            "Default enabled should be True"
        )
        assert spec.experiment_name == "ml_workflow_experiments", (
            f"Default experiment_name should be 'ml_workflow_experiments', "
            f"got '{spec.experiment_name}'"
        )
        assert spec.register_model is False, (
            "Default register_model should be False"
        )
        assert spec.model_stage == "None", (
            f"Default model_stage should be 'None', got '{spec.model_stage}'"
        )
        assert spec.tags == {}, (
            f"Default tags should be empty dict, got {spec.tags}"
        )

    def test_custom_values(self):
        """Test custom configuration for MLflowSpec."""
        spec = MLflowSpec(
            enabled=True,
            experiment_name="custom_experiment",
            register_model=True,
            model_stage="Staging",
            tags={"team": "ml"},
        )

        assert spec.experiment_name == "custom_experiment", (
            f"experiment_name should be 'custom_experiment', got '{spec.experiment_name}'"
        )
        assert spec.register_model is True, (
            "register_model should be True"
        )
        assert spec.model_stage == "Staging", (
            f"model_stage should be 'Staging', got '{spec.model_stage}'"
        )
        assert spec.tags == {"team": "ml"}, (
            f"tags should be {{'team': 'ml'}}, got {spec.tags}"
        )

    def test_invalid_model_stage(self):
        """Test that invalid model stage raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MLflowSpec(model_stage="InvalidStage")  

        assert "model_stage" in str(exc_info.value).lower() or "invalid" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention model_stage or invalid, got: {exc_info.value}"
        )

    def test_empty_experiment_name_when_enabled(self):
        """Test that empty experiment name raises ValidationError when enabled."""
        with pytest.raises(ValidationError) as exc_info:
            MLflowSpec(enabled=True, experiment_name="")

        assert "experiment_name" in str(exc_info.value).lower() or "empty" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention experiment_name or empty, got: {exc_info.value}"
        )

    def test_invalid_tags_type(self):
        """Test that non-dict tags raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            MLflowSpec(tags="not_a_dict")  

        assert "tags" in str(exc_info.value).lower() or "dict" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention tags or dict, got: {exc_info.value}"
        )


@pytest.mark.unit
class TestMLflowSpecHelperMethods:
    """Tests for MLflowSpec helper methods."""

    def test_get_run_name(self):
        """Test get_run_name method."""
        spec = MLflowSpec(run_name="custom_run")

        assert spec.get_run_name(default="default") == "custom_run", (
            f"get_run_name() should return 'custom_run', "
            f"got '{spec.get_run_name(default='default')}'"
        )

        spec_no_name = MLflowSpec(run_name=None)

        assert spec_no_name.get_run_name(default="default") == "default", (
            f"get_run_name() should return default when run_name is None, "
            f"got '{spec_no_name.get_run_name(default='default')}'"
        )

    def test_get_registered_model_name(self):
        """Test get_registered_model_name method."""
        spec = MLflowSpec(register_model=True, registered_model_name="custom")

        assert spec.get_registered_model_name(default="base") == "custom", (
            f"get_registered_model_name() should return 'custom', "
            f"got '{spec.get_registered_model_name(default='base')}'"
        )

        spec_auto = MLflowSpec(register_model=True)

        assert spec_auto.get_registered_model_name(default="base") == "base_registered", (
            f"get_registered_model_name() should auto-generate name, "
            f"got '{spec_auto.get_registered_model_name(default='base')}'"
        )

    def test_get_run_tags(self):
        """Test get_run_tags with merging."""
        spec = MLflowSpec(tags={"team": "ml"})
        tags = spec.get_run_tags(additional_tags={"version": "v1"})

        assert tags == {"team": "ml", "version": "v1"}, (
            f"Tags should be merged correctly, got {tags}"
        )

    @pytest.mark.parametrize(
        "enabled,register_model,expected",
        [
            pytest.param(True, True, True, id="enabled_and_register"),
            pytest.param(True, False, False, id="enabled_but_no_register"),
            pytest.param(False, True, False, id="disabled"),
        ],
    )
    def test_should_register_model(self, enabled, register_model, expected):
        """Test should_register_model method."""
        spec = MLflowSpec(enabled=enabled, register_model=register_model)

        assert spec.should_register_model() == expected, (
            f"should_register_model() should return {expected} for "
            f"enabled={enabled}, register_model={register_model}, "
            f"got {spec.should_register_model()}"
        )


@pytest.mark.unit
class TestMLflowSpecSerialization:
    """Tests for MLflowSpec serialization methods."""

    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original = MLflowSpec(
            enabled=True,
            experiment_name="roundtrip_test",
            register_model=True,
            model_stage="Production",
            tags={"test": "roundtrip"},
        )

        spec_dict = original.to_dict()
        restored = MLflowSpec.from_dict(spec_dict)

        assert restored.enabled == original.enabled, (
            f"restored.enabled should be {original.enabled}, got {restored.enabled}"
        )
        assert restored.experiment_name == original.experiment_name, (
            f"restored.experiment_name should be '{original.experiment_name}', "
            f"got '{restored.experiment_name}'"
        )
        assert restored.register_model == original.register_model, (
            f"restored.register_model should be {original.register_model}, "
            f"got {restored.register_model}"
        )
        assert restored.tags == original.tags, (
            f"restored.tags should be {original.tags}, got {restored.tags}"
        )


@pytest.mark.unit
class TestMLflowSpecBuilder:
    """Tests for MLflowSpecBuilder."""

    def test_default_build(self):
        """Test building with defaults."""
        spec = MLflowSpecBuilder().build()

        assert spec.enabled is True, (
            "Default enabled should be True"
        )
        assert spec.experiment_name == "ml_workflow_experiments", (
            f"Default experiment_name should be 'ml_workflow_experiments', "
            f"got '{spec.experiment_name}'"
        )
        assert spec.register_model is False, (
            "Default register_model should be False"
        )

    @pytest.mark.parametrize(
        "method,expected_enabled",
        [
            pytest.param("enable", True, id="enable_method"),
            pytest.param("disable", False, id="disable_method"),
        ],
    )
    def test_enable_disable(self, method, expected_enabled):
        """Test enable and disable methods."""
        if method == "enable":
            spec = MLflowSpecBuilder().enable().build()
        else:
            spec = MLflowSpecBuilder().disable().build()

        assert spec.enabled is expected_enabled, (
            f"enabled should be {expected_enabled} after {method}(), got {spec.enabled}"
        )

    def test_set_experiment_and_run_name(self):
        """Test setting experiment and run names."""
        spec = (
            MLflowSpecBuilder()
            .set_experiment("custom_experiment")
            .set_run_name("custom_run")
            .build()
        )

        assert spec.experiment_name == "custom_experiment", (
            f"experiment_name should be 'custom_experiment', got '{spec.experiment_name}'"
        )
        assert spec.run_name == "custom_run", (
            f"run_name should be 'custom_run', got '{spec.run_name}'"
        )

    def test_enable_model_registry(self):
        """Test enable_model_registry method."""
        spec = (
            MLflowSpecBuilder()
            .enable_model_registry(model_name="custom_model", stage="Staging")
            .build()
        )

        assert spec.register_model is True, (
            "register_model should be True"
        )
        assert spec.registered_model_name == "custom_model", (
            f"registered_model_name should be 'custom_model', "
            f"got '{spec.registered_model_name}'"
        )
        assert spec.model_stage == "Staging", (
            f"model_stage should be 'Staging', got '{spec.model_stage}'"
        )

    def test_add_tags(self):
        """Test add_tag and add_tags methods."""
        spec = (
            MLflowSpecBuilder()
            .add_tag("team", "ml")
            .add_tags({"version": "v1", "env": "prod"})
            .build()
        )

        assert spec.tags == {"team": "ml", "version": "v1", "env": "prod"}, (
            f"tags should be {{'team': 'ml', 'version': 'v1', 'env': 'prod'}}, got {spec.tags}"
        )

    def test_fluent_interface(self):
        """Test comprehensive fluent interface."""
        spec = (
            MLflowSpecBuilder()
            .enable()
            .set_experiment("production")
            .set_run_name("deployment")
            .enable_model_registry("classifier", stage="Staging")
            .add_tags({"team": "ml_ops", "version": "v1.0"})
            .build()
        )

        assert spec.enabled is True, (
            "enabled should be True"
        )
        assert spec.experiment_name == "production", (
            f"experiment_name should be 'production', got '{spec.experiment_name}'"
        )
        assert spec.register_model is True, (
            "register_model should be True"
        )
        assert spec.tags["team"] == "ml_ops", (
            f"tags['team'] should be 'ml_ops', got '{spec.tags.get('team')}'"
        )


@pytest.mark.unit
class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_create_default_mlflow_spec(self):
        """Test create_default_mlflow_spec factory function."""
        spec = create_default_mlflow_spec(experiment_name="custom_exp")

        assert spec.enabled is True, (
            "Default spec should be enabled"
        )
        assert spec.experiment_name == "custom_exp", (
            f"experiment_name should be 'custom_exp', got '{spec.experiment_name}'"
        )
        assert spec.register_model is False, (
            "Default spec should not register model"
        )

    def test_create_production_mlflow_spec(self):
        """Test create_production_mlflow_spec factory function."""
        spec = create_production_mlflow_spec(
            experiment_name="production",
            model_name="classifier_v1",
        )

        assert spec.enabled is True, (
            "Production spec should be enabled"
        )
        assert spec.experiment_name == "production", (
            f"experiment_name should be 'production', got '{spec.experiment_name}'"
        )
        assert spec.register_model is True, (
            "Production spec should register model"
        )
        assert spec.registered_model_name == "classifier_v1", (
            f"registered_model_name should be 'classifier_v1', "
            f"got '{spec.registered_model_name}'"
        )
        assert spec.model_stage == "Staging", (
            f"model_stage should be 'Staging', got '{spec.model_stage}'"
        )
