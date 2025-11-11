"""
Tests for MLflow specifications.
"""
import pytest
from pydantic import ValidationError

from specs.mlflow_spec import (
    MLflowSpec,
    MLflowSpecBuilder,
    create_default_mlflow_spec,
    create_production_mlflow_spec,
)


class TestMLflowSpec:
    """Tests for MLflowSpec core functionality."""
    
    def test_default_values(self):
        """Test default values for MLflowSpec."""
        spec = MLflowSpec()
        
        assert spec.enabled is True
        assert spec.experiment_name == "ml_workflow_experiments"
        assert spec.register_model is False
        assert spec.model_stage == "None"
        assert spec.tags == {}
    
    def test_custom_values(self):
        """Test custom configuration."""
        spec = MLflowSpec(
            enabled=True,
            experiment_name="custom_experiment",
            register_model=True,
            model_stage="Staging",
            tags={"team": "ml"}
        )
        
        assert spec.experiment_name == "custom_experiment"
        assert spec.register_model is True
        assert spec.model_stage == "Staging"
        assert spec.tags == {"team": "ml"}
    
    def test_invalid_model_stage(self):
        """Test that invalid model stage raises ValidationError."""
        with pytest.raises(ValidationError):
            MLflowSpec(model_stage="InvalidStage")
    
    def test_empty_experiment_name_when_enabled(self):
        """Test that empty experiment name raises ValidationError when enabled."""
        with pytest.raises(ValidationError):
            MLflowSpec(enabled=True, experiment_name="")
    
    def test_invalid_tags_type(self):
        """Test that non-dict tags raise ValidationError."""
        with pytest.raises(ValidationError):
            MLflowSpec(tags="not_a_dict")


class TestMLflowSpecHelperMethods:
    """Tests for MLflowSpec helper methods."""
    
    def test_get_run_name(self):
        """Test get_run_name method."""
        spec = MLflowSpec(run_name="custom_run")
        assert spec.get_run_name(default="default") == "custom_run"
        
        spec_no_name = MLflowSpec(run_name=None)
        assert spec_no_name.get_run_name(default="default") == "default"
    
    def test_get_registered_model_name(self):
        """Test get_registered_model_name method."""
        spec = MLflowSpec(register_model=True, registered_model_name="custom")
        assert spec.get_registered_model_name(default="base") == "custom"
        
        spec_auto = MLflowSpec(register_model=True)
        assert spec_auto.get_registered_model_name(default="base") == "base_registered"
    
    def test_get_run_tags(self):
        """Test get_run_tags with merging."""
        spec = MLflowSpec(tags={"team": "ml"})
        tags = spec.get_run_tags(additional_tags={"version": "v1"})
        assert tags == {"team": "ml", "version": "v1"}
    
    def test_should_register_model(self):
        """Test should_register_model method."""
        spec = MLflowSpec(enabled=True, register_model=True)
        assert spec.should_register_model() is True
        
        spec_disabled = MLflowSpec(enabled=False, register_model=True)
        assert spec_disabled.should_register_model() is False


class TestMLflowSpecSerialization:
    """Tests for MLflowSpec serialization methods."""
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict and from_dict are inverses."""
        original = MLflowSpec(
            enabled=True,
            experiment_name="roundtrip_test",
            register_model=True,
            model_stage="Production",
            tags={"test": "roundtrip"}
        )
        
        spec_dict = original.to_dict()
        restored = MLflowSpec.from_dict(spec_dict)
        
        assert restored.enabled == original.enabled
        assert restored.experiment_name == original.experiment_name
        assert restored.register_model == original.register_model
        assert restored.tags == original.tags


class TestMLflowSpecBuilder:
    """Tests for MLflowSpecBuilder."""
    
    def test_default_build(self):
        """Test building with defaults."""
        spec = MLflowSpecBuilder().build()
        
        assert spec.enabled is True
        assert spec.experiment_name == "ml_workflow_experiments"
        assert spec.register_model is False
    
    def test_enable_disable(self):
        """Test enable and disable methods."""
        spec = MLflowSpecBuilder().enable().build()
        assert spec.enabled is True
        
        spec_disabled = MLflowSpecBuilder().disable().build()
        assert spec_disabled.enabled is False
    
    def test_set_experiment_and_run_name(self):
        """Test setting experiment and run names."""
        spec = (MLflowSpecBuilder()
                .set_experiment("custom_experiment")
                .set_run_name("custom_run")
                .build())
        
        assert spec.experiment_name == "custom_experiment"
        assert spec.run_name == "custom_run"
    
    def test_enable_model_registry(self):
        """Test enable_model_registry method."""
        spec = (MLflowSpecBuilder()
                .enable_model_registry(model_name="custom_model", stage="Staging")
                .build())
        
        assert spec.register_model is True
        assert spec.registered_model_name == "custom_model"
        assert spec.model_stage == "Staging"
    
    def test_add_tags(self):
        """Test add_tag and add_tags methods."""
        spec = (MLflowSpecBuilder()
                .add_tag("team", "ml")
                .add_tags({"version": "v1", "env": "prod"})
                .build())
        
        assert spec.tags == {"team": "ml", "version": "v1", "env": "prod"}
    
    def test_fluent_interface(self):
        """Test comprehensive fluent interface."""
        spec = (MLflowSpecBuilder()
                .enable()
                .set_experiment("production")
                .set_run_name("deployment")
                .enable_model_registry("classifier", stage="Staging")
                .add_tags({"team": "ml_ops", "version": "v1.0"})
                .build())
        
        assert spec.enabled is True
        assert spec.experiment_name == "production"
        assert spec.register_model is True
        assert spec.tags["team"] == "ml_ops"


class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_default_mlflow_spec(self):
        """Test create_default_mlflow_spec."""
        spec = create_default_mlflow_spec(experiment_name="custom_exp")
        
        assert spec.enabled is True
        assert spec.experiment_name == "custom_exp"
        assert spec.register_model is False
    
    def test_create_production_mlflow_spec(self):
        """Test create_production_mlflow_spec."""
        spec = create_production_mlflow_spec(
            experiment_name="production",
            model_name="classifier_v1"
        )
        
        assert spec.enabled is True
        assert spec.experiment_name == "production"
        assert spec.register_model is True
        assert spec.registered_model_name == "classifier_v1"
        assert spec.model_stage == "Staging"

