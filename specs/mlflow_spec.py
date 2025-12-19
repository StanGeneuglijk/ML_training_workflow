"""
MLflow specifications
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
import logging

import utils
logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)

class MLflowSpec(BaseModel):
    """
    MLflow specification with Pydantic validation.
    """
    model_config = ConfigDict(
        extra='forbid', 
        validate_assignment=True
    )
    
    enabled: bool = Field(
        default=True,
        description="Whether MLflow tracking is enabled"
    )
    experiment_name: str = Field(
        default="ml_workflow_experiments",
        description="Name of the MLflow experiment"
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Custom name for the MLflow run"
    )
    tracking_uri: Optional[str] = Field(
        default=None,
        description="MLflow tracking server URI"
    )
    registry_uri: Optional[str] = Field(
        default=None,
        description="Model registry URI"
    )
    artifact_location: Optional[str] = Field(
        default=None,
        description="Custom artifact storage location"
    )
    register_model: bool = Field(
        default=False,
        description="Whether to register model in Model Registry"
    )
    registered_model_name: Optional[str] = Field(
        default=None,
        description="Custom name for registered model"
    )
    model_stage: Literal["None", "Staging", "Production", "Archived"] = Field(
        default="None",
        description="Initial stage for registered model"
    )
    log_model_signature: bool = Field(
        default=True,
        description="Whether to infer and log model signature"
    )
    log_input_example: bool = Field(
        default=True,
        description="Whether to log input example with model"
    )
    log_artifacts: bool = Field(
        default=True,
        description="Whether to log additional artifacts"
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional tags to attach to the MLflow run"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description for the run"
    )
    
    @model_validator(mode='after')
    def validate_experiment_name_when_enabled(
        self
    ) -> 'MLflowSpec':
        """
        Validate experiment name is not empty when MLflow is enabled.
        
        Args:
            None
        """
        if self.enabled and not self.experiment_name:
            raise ValueError("experiment_name cannot be empty when MLflow is enabled")
        return self
    
    @field_validator('tags')
    @classmethod
    def validate_tags_are_strings(
        cls, 
        v: Dict[str, str]
        ) -> Dict[str, str]:
        """
        Validate that all tags are string key-value pairs.
        
        Args:
            v: Dictionary of tags
        """
        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("All tags must be string key-value pairs")
        return v
    
    def get_run_name(
        self, 
        default: str = "mlflow_run"
    ) -> str:
        """
        Get the run name, using default if not specified.
        
        Args:
            default: Default run name to use if run_name is None
            
        Returns:
            Run name string
        """
        return self.run_name if self.run_name else default
    
    def get_registered_model_name(
        self, 
        default: str
    ) -> str:
        """
        Get the registered model name, using default if not specified.
        
        Args:
            default: Default model name (typically from model_spec.model_name)
            
        Returns:
            Registered model name string
        """
        if self.registered_model_name:
            return self.registered_model_name
        return f"{default}_registered" if self.register_model else default
    
    def get_tracking_config(
        self
    ) -> Dict[str, Any]:
        """
        Get tracking configuration as dictionary.
        
        Returns:
            Dictionary with tracking configuration
        """
        return {
            'tracking_uri': self.tracking_uri,
            'registry_uri': self.registry_uri,
            'experiment_name': self.experiment_name,
            'artifact_location': self.artifact_location
        }
    
    def get_run_tags(
        self, 
        additional_tags: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Get run tags, merging with additional tags.
        
        Args:
            additional_tags: Additional tags to merge with spec tags
            
        Returns:
            Merged tags dictionary
        """
        tags = self.tags.copy()
        if additional_tags:
            tags.update(additional_tags)
        return tags
    
    def should_register_model(
        self
    ) -> bool:
        """
        Check if model should be registered.

        Args:
            None

        Returns:
            True if model registration is enabled
        """
        return self.enabled and self.register_model
    
    def to_dict(
        self
    ) -> Dict[str, Any]:
        """
        Convert spec to dictionary.

        Args:
            None
        
        Returns:
            Dictionary representation of the spec
        """
        return {
            'enabled': self.enabled,
            'experiment_name': self.experiment_name,
            'run_name': self.run_name,
            'tracking_uri': self.tracking_uri,
            'registry_uri': self.registry_uri,
            'artifact_location': self.artifact_location,
            'register_model': self.register_model,
            'registered_model_name': self.registered_model_name,
            'model_stage': self.model_stage,
            'log_model_signature': self.log_model_signature,
            'log_input_example': self.log_input_example,
            'log_artifacts': self.log_artifacts,
            'tags': self.tags,
            'description': self.description
        }
    
    @classmethod
    def from_dict(
        cls, 
        data: Dict[str, Any]
    ) -> MLflowSpec:
        """
        Create MLflowSpec from dictionary.
        
        Args:
            data: Dictionary with spec data
            
        Returns:
            MLflowSpec instance
        """
        return cls(**data)
    
    def __repr__(
        self
    ) -> str:
        """
        String representation of MLflowSpec.
        
        Args:
            None
        
        Returns:
            String representation of MLflowSpec
        """
        status = "enabled" if self.enabled else "disabled"
        registry = "with registry" if self.register_model else "no registry"
        return f"MLflowSpec(experiment='{self.experiment_name}', {status}, {registry})"


class MLflowSpecBuilder:
    """
    Builder for creating MLflowSpec configurations.
    """
    
    def __init__(self) -> None:
        """
        Initialize builder with default values.
        
        Args:
            None
        
        Returns:
            None
        """
        self._enabled: bool = True
        self._experiment_name: str = "ml_workflow_experiments"
        self._run_name: Optional[str] = None
        self._tracking_uri: Optional[str] = None
        self._registry_uri: Optional[str] = None
        self._artifact_location: Optional[str] = None
        self._register_model: bool = False
        self._registered_model_name: Optional[str] = None
        self._model_stage: Literal["None", "Staging", "Production", "Archived"] = "None"
        self._log_model_signature: bool = True
        self._log_input_example: bool = True
        self._log_artifacts: bool = True
        self._tags: Dict[str, str] = {}
        self._description: Optional[str] = None
    
    def enable(
        self, 
        enabled: bool = True
    ) -> 'MLflowSpecBuilder':
        """
        Enable or disable MLflow tracking.
        
        Args:
            enabled: Whether to enable MLflow tracking

        Returns:
            self
        """
        self._enabled = enabled
        return self
    
    def disable(
        self
    ) -> 'MLflowSpecBuilder':
        """
        Disable MLflow tracking.
        
        Args:
            None
        
        Returns:
            self
        """
        self._enabled = False
        return self
    
    def set_experiment(
        self, 
        name: str
    ) -> 'MLflowSpecBuilder':
        """
        Set experiment name.
        
        Args: 
            name: Name of the experiment
        
        Returns:
            Set a MLflowSpecBuilder with the experiment name
        """
        self._experiment_name = name
        return self
    
    def set_run_name(
        self, 
        name: str
    ) -> 'MLflowSpecBuilder':
        """
        Set run name.
        
        Args:
            name: Name of the run
        
        Returns:
            Set a MLflowSpecBuilder with the run name
        """
        self._run_name = name
        return self
    
    def set_tracking_uri(
        self, 
        uri: str
    ) -> 'MLflowSpecBuilder':
        """
        Set tracking server URI.
        
        Args:
            uri: Tracking server URI
        
        Returns:
            Set a MLflowSpecBuilder with the tracking server URI
        """
        self._tracking_uri = uri
        return self
    
    def set_registry_uri(
        self, 
        uri: str
    ) -> 'MLflowSpecBuilder':
        """
        Set model registry URI.
        
        Args:
            uri: Model registry URI
        
        Returns:
            Set a MLflowSpecBuilder with the model registry URI
        """
        self._registry_uri = uri
        return self
    
    def set_artifact_location(
        self, 
        location: str
    ) -> 'MLflowSpecBuilder':
        """
        Set custom artifact storage location.
        
        Args:
            location: Custom artifact storage location
        
        Returns:
            Set a MLflowSpecBuilder with the artifact storage location
        """
        self._artifact_location = location
        return self
    
    def enable_model_registry(
        self,
        model_name: Optional[str] = None,
        stage: Literal["None", "Staging", "Production", "Archived"] = "None"
    ) -> 'MLflowSpecBuilder':
        """
        Enable model registry with optional custom name and stage.
        
        Args:
            model_name: Custom registered model name
            stage: Initial stage for the model

        Returns:
            Set a MLflowSpecBuilder with the model registry enabled
        """
        self._register_model = True
        self._registered_model_name = model_name
        self._model_stage = stage
        return self
    
    def disable_model_registry(
        self
    ) -> 'MLflowSpecBuilder':
        """
        Disable model registry.
        
        Args: 
            None

        Returns:
            Set a MLflowSpecBuilder with the model registry disabled
        """
        self._register_model = False
        return self
    
    def set_logging_options(
        self,
        signature: bool = True,
        input_example: bool = True,
        artifacts: bool = True
    ) -> 'MLflowSpecBuilder':
        """
        Set logging options for model and artifacts.
        
        Args:
            signature: Whether to log model signature
            input_example: Whether to log input example
            artifacts: Whether to log artifacts
        
        Returns:
            Set a MLflowSpecBuilder with the logging options
        """
        self._log_model_signature = signature
        self._log_input_example = input_example
        self._log_artifacts = artifacts
        return self
    
    def add_tag(
        self, 
        key: str, 
        value: str
    ) -> 'MLflowSpecBuilder':
        """
        Add a single tag.
        
        Args:
            key: Tag key
            value: Tag value
        
        Returns:
            Set a MLflowSpecBuilder with the tag
        """
        self._tags[key] = value
        return self
    
    def add_tags(
        self, 
        tags: Dict[str, str]
    ) -> 'MLflowSpecBuilder':
        """
        Add multiple tags.
        
        Args:
            tags: Dictionary of tags
        
        Returns:
            Set a MLflowSpecBuilder with the tags
        """
        self._tags.update(tags)
        return self
    
    def set_description(
        self, 
        description: str
    ) -> 'MLflowSpecBuilder':
        """
        Set run description.
        
        Args:
            description: Description of the run
        
        Returns:
            Set a MLflowSpecBuilder with the description
        """
        self._description = description
        return self
    
    def build(
        self
    ) -> MLflowSpec:
        """
        Build and return the MLflowSpec.
        
        Args:
            None

        Returns:
            Configured MLflowSpec instance
        """
        return MLflowSpec(
            enabled=self._enabled,
            experiment_name=self._experiment_name,
            run_name=self._run_name,
            tracking_uri=self._tracking_uri,
            registry_uri=self._registry_uri,
            artifact_location=self._artifact_location,
            register_model=self._register_model,
            registered_model_name=self._registered_model_name,
            model_stage=self._model_stage,
            log_model_signature=self._log_model_signature,
            log_input_example=self._log_input_example,
            log_artifacts=self._log_artifacts,
            tags=self._tags,
            description=self._description
        )


def create_default_mlflow_spec(
    experiment_name: str = "ml_workflow_experiments",
    register_model: bool = False
) -> MLflowSpec:
    """
    Create a default MLflowSpec with common settings.
    
    Args:
        experiment_name: Name of the experiment
        register_model: Whether to enable model registry
        
    Returns:
        Configured MLflowSpec with defaults
    """
    return MLflowSpec(
        enabled=True,
        experiment_name=experiment_name,
        register_model=register_model,
        log_model_signature=True,
        log_input_example=True,
        log_artifacts=True
    )


def create_production_mlflow_spec(
    experiment_name: str,
    model_name: str,
    run_name: Optional[str] = None
) -> MLflowSpec:
    """
    Create MLflowSpec configured for production use.
    
    Args:
        experiment_name: Name of the experiment
        model_name: Name for the registered model
        run_name: Optional custom run name
        
    Returns:
        MLflowSpec configured for production deployment
    """
    return MLflowSpec(
        enabled=True,
        experiment_name=experiment_name,
        run_name=run_name,
        register_model=True,
        registered_model_name=model_name,
        model_stage="Staging", 
        log_model_signature=True,
        log_input_example=True,
        log_artifacts=True,
        tags={"environment": "production", "auto_registered": "true"}
    )


__all__ = [
    "MLflowSpec",
    "MLflowSpecBuilder",
    "create_default_mlflow_spec",
    "create_production_mlflow_spec",
]
