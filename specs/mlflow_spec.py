"""
MLflow specification for ML workflow version 1.

Defines configuration specifications for MLflow experiment tracking,
model logging, and model registry integration.
"""
from __future__ import annotations

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


class MLflowSpec(BaseModel):
    """
    Specification for MLflow tracking and model registry configuration with Pydantic validation.
    
    This spec encapsulates all MLflow-related settings for experiment tracking,
    model logging, and model registry operations.
    
    Attributes:
        enabled: Whether MLflow tracking is enabled
        experiment_name: Name of the MLflow experiment
        run_name: Optional custom name for the MLflow run (auto-generated if None)
        tracking_uri: MLflow tracking server URI (None uses local ./mlruns)
        registry_uri: Model registry URI (None uses tracking_uri)
        artifact_location: Custom artifact storage location
        register_model: Whether to register the trained model in Model Registry
        registered_model_name: Custom name for registered model (uses model_spec.model_name if None)
        model_stage: Initial stage for registered model ('None', 'Staging', 'Production')
        log_model_signature: Whether to infer and log model signature
        log_input_example: Whether to log input example with model
        log_artifacts: Whether to log additional artifacts (architecture, specs, etc.)
        tags: Additional tags to attach to the MLflow run
        description: Optional description for the run
    
    Example:
        >>> mlflow_spec = MLflowSpec(
        ...     enabled=True,
        ...     experiment_name="production_models",
        ...     register_model=True,
        ...     tags={"team": "ml_team", "version": "v1.0"}
        ... )
    """
    
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    # Core settings
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
    
    # Server configuration
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
    
    # Model registry settings
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
    
    # Logging configuration
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
    
    # Metadata
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional tags to attach to the MLflow run"
    )
    description: Optional[str] = Field(
        default=None,
        description="Description for the run"
    )
    
    @model_validator(mode='after')
    def validate_experiment_name_when_enabled(self) -> 'MLflowSpec':
        """Validate experiment name is not empty when MLflow is enabled."""
        if self.enabled and not self.experiment_name:
            raise ValueError("experiment_name cannot be empty when MLflow is enabled")
        return self
    
    @field_validator('tags')
    @classmethod
    def validate_tags_are_strings(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Validate that all tags are string key-value pairs."""
        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError("All tags must be string key-value pairs")
        return v
    
    def get_run_name(self, default: str = "mlflow_run") -> str:
        """
        Get the run name, using default if not specified.
        
        Args:
            default: Default run name to use if run_name is None
            
        Returns:
            Run name string
        """
        return self.run_name if self.run_name else default
    
    def get_registered_model_name(self, default: str) -> str:
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
    
    def get_tracking_config(self) -> Dict[str, Any]:
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
    
    def get_run_tags(self, additional_tags: Optional[Dict[str, str]] = None) -> Dict[str, str]:
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
    
    def should_register_model(self) -> bool:
        """
        Check if model should be registered.
        
        Returns:
            True if model registration is enabled
        """
        return self.enabled and self.register_model
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert spec to dictionary.
        
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
    def from_dict(cls, data: Dict[str, Any]) -> MLflowSpec:
        """
        Create MLflowSpec from dictionary.
        
        Args:
            data: Dictionary with spec data
            
        Returns:
            MLflowSpec instance
        """
        return cls(**data)
    
    def __repr__(self) -> str:
        """String representation of MLflowSpec."""
        status = "enabled" if self.enabled else "disabled"
        registry = "with registry" if self.register_model else "no registry"
        return f"MLflowSpec(experiment='{self.experiment_name}', {status}, {registry})"


class MLflowSpecBuilder:
    """
    Builder for creating MLflowSpec configurations.
    
    Provides a fluent interface for constructing MLflow specifications.
    
    Example:
        >>> spec = (MLflowSpecBuilder()
        ...     .set_experiment("my_experiment")
        ...     .enable_model_registry("my_model", stage="Staging")
        ...     .add_tags({"team": "ml", "version": "v1"})
        ...     .build())
    """
    
    def __init__(self) -> None:
        """Initialize builder with default values."""
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
    
    def enable(self, enabled: bool = True) -> 'MLflowSpecBuilder':
        """Enable or disable MLflow tracking."""
        self._enabled = enabled
        return self
    
    def disable(self) -> 'MLflowSpecBuilder':
        """Disable MLflow tracking."""
        self._enabled = False
        return self
    
    def set_experiment(self, name: str) -> 'MLflowSpecBuilder':
        """Set experiment name."""
        self._experiment_name = name
        return self
    
    def set_run_name(self, name: str) -> 'MLflowSpecBuilder':
        """Set run name."""
        self._run_name = name
        return self
    
    def set_tracking_uri(self, uri: str) -> 'MLflowSpecBuilder':
        """Set tracking server URI."""
        self._tracking_uri = uri
        return self
    
    def set_registry_uri(self, uri: str) -> 'MLflowSpecBuilder':
        """Set model registry URI."""
        self._registry_uri = uri
        return self
    
    def set_artifact_location(self, location: str) -> 'MLflowSpecBuilder':
        """Set custom artifact storage location."""
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
        """
        self._register_model = True
        self._registered_model_name = model_name
        self._model_stage = stage
        return self
    
    def disable_model_registry(self) -> 'MLflowSpecBuilder':
        """Disable model registry."""
        self._register_model = False
        return self
    
    def set_logging_options(
        self,
        signature: bool = True,
        input_example: bool = True,
        artifacts: bool = True
    ) -> 'MLflowSpecBuilder':
        """Set logging options for model and artifacts."""
        self._log_model_signature = signature
        self._log_input_example = input_example
        self._log_artifacts = artifacts
        return self
    
    def add_tag(self, key: str, value: str) -> 'MLflowSpecBuilder':
        """Add a single tag."""
        self._tags[key] = value
        return self
    
    def add_tags(self, tags: Dict[str, str]) -> 'MLflowSpecBuilder':
        """Add multiple tags."""
        self._tags.update(tags)
        return self
    
    def set_description(self, description: str) -> 'MLflowSpecBuilder':
        """Set run description."""
        self._description = description
        return self
    
    def build(self) -> MLflowSpec:
        """
        Build and return the MLflowSpec.
        
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
        model_stage="Staging",  # Start in staging, promote to production later
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
