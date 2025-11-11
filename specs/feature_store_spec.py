"""
Feature Store specification module for ML workflow version 1.

Defines Pydantic specifications for feature store configuration.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Literal
from datetime import datetime

from pydantic import BaseModel, Field, field_validator, ConfigDict


logger = logging.getLogger(__name__)


class FeatureStoreSpec(BaseModel):
    """
    Configuration specification for Feast feature store integration.
    
    Attributes:
        enabled: Whether to use feature store for feature retrieval
        repo_path: Path to Feast repository directory
        n_features: Number of features in the dataset
        offline_store_type: Type of offline store (file, bigquery, snowflake, etc.)
        online_store_type: Type of online store (sqlite, redis, dynamodb, etc.)
        feature_view_name: Name of the feature view to use
        feature_service_name: Name of the feature service to use
        use_full_feature_names: Whether to use full feature names (view:feature)
        timestamp: Optional timestamp for point-in-time feature retrieval
        initialize_on_start: Whether to initialize feature store on workflow start
        force_recreate: Whether to force recreate the feature store
        materialize_online: Whether to materialize features to online store
    """
    
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    enabled: bool = Field(
        default=False,
        description="Whether to use feature store for feature retrieval"
    )
    repo_path: str = Field(
        default="feature_repo",
        description="Path to Feast repository directory"
    )
    n_features: int = Field(
        default=20,
        ge=1,
        le=1000,
        description="Number of features in the dataset"
    )
    offline_store_type: Literal["file", "spark", "bigquery", "snowflake", "redshift"] = Field(
        default="file",
        description="Type of offline store"
    )
    online_store_type: Optional[Literal["sqlite", "redis", "dynamodb", "datastore"]] = Field(
        default=None,
        description="Type of online store (None for offline-only)"
    )
    feature_view_name: str = Field(
        default="classification_features",
        description="Name of the feature view to use"
    )
    feature_service_name: Optional[str] = Field(
        default="classification_service",
        description="Name of the feature service to use"
    )
    use_full_feature_names: bool = Field(
        default=False,
        description="Whether to use full feature names (view:feature)"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp for point-in-time feature retrieval"
    )
    initialize_on_start: bool = Field(
        default=True,
        description="Whether to initialize feature store on workflow start"
    )
    force_recreate: bool = Field(
        default=False,
        description="Whether to force recreate the feature store"
    )
    materialize_online: bool = Field(
        default=False,
        description="Whether to materialize features to online store"
    )
    sample_indices: Optional[List[int]] = Field(
        default=None,
        description="Specific sample indices to retrieve (None for all)"
    )
    
    @field_validator('repo_path')
    @classmethod
    def validate_repo_path(cls, v: str) -> str:
        """Validate repo path is not empty."""
        if not v.strip():
            raise ValueError("repo_path cannot be empty")
        return v.strip()
    
    @field_validator('feature_view_name')
    @classmethod
    def validate_feature_view_name(cls, v: str) -> str:
        """Validate feature view name is not empty."""
        if not v.strip():
            raise ValueError("feature_view_name cannot be empty")
        return v.strip()
    
    def get_feature_references(self) -> List[str]:
        """
        Get list of feature references for retrieval.
        
        Returns:
            List of feature references (e.g., ["classification_features:feature_0", ...])
        """
        feature_refs = [
            f"{self.feature_view_name}:feature_{i}" 
            for i in range(self.n_features)
        ]
        # Add target feature
        feature_refs.append(f"{self.feature_view_name}:target")
        return feature_refs
    
    def should_initialize(self) -> bool:
        """Check if feature store should be initialized."""
        return self.enabled and self.initialize_on_start
    
    def should_materialize(self) -> bool:
        """Check if features should be materialized to online store."""
        return self.enabled and self.materialize_online and self.online_store_type is not None


class FeatureStoreSpecBuilder:
    """
    Builder for creating FeatureStoreSpec instances.
    
    Provides a fluent interface for configuring feature store specifications.
    """
    
    def __init__(self) -> None:
        """Initialize builder with default values."""
        self._enabled = False
        self._repo_path = "feature_repo"
        self._n_features = 20
        self._offline_store_type = "file"
        self._online_store_type = None
        self._feature_view_name = "classification_features"
        self._feature_service_name = "classification_service"
        self._use_full_feature_names = False
        self._timestamp = None
        self._initialize_on_start = True
        self._force_recreate = False
        self._materialize_online = False
        self._sample_indices = None
    
    def enable(self) -> 'FeatureStoreSpecBuilder':
        """Enable feature store integration."""
        self._enabled = True
        return self
    
    def disable(self) -> 'FeatureStoreSpecBuilder':
        """Disable feature store integration."""
        self._enabled = False
        return self
    
    def set_repo_path(self, repo_path: str) -> 'FeatureStoreSpecBuilder':
        """Set Feast repository path."""
        self._repo_path = repo_path
        return self
    
    def set_n_features(self, n_features: int) -> 'FeatureStoreSpecBuilder':
        """Set number of features."""
        self._n_features = n_features
        return self
    
    def set_offline_store(
        self, 
        store_type: Literal["file", "spark", "bigquery", "snowflake", "redshift"]
    ) -> 'FeatureStoreSpecBuilder':
        """Set offline store type."""
        self._offline_store_type = store_type
        return self
    
    def set_online_store(
        self,
        store_type: Optional[Literal["sqlite", "redis", "dynamodb", "datastore"]]
    ) -> 'FeatureStoreSpecBuilder':
        """Set online store type."""
        self._online_store_type = store_type
        return self
    
    def set_feature_view(self, view_name: str) -> 'FeatureStoreSpecBuilder':
        """Set feature view name."""
        self._feature_view_name = view_name
        return self
    
    def set_feature_service(self, service_name: str) -> 'FeatureStoreSpecBuilder':
        """Set feature service name."""
        self._feature_service_name = service_name
        return self
    
    def use_full_names(self, use_full: bool = True) -> 'FeatureStoreSpecBuilder':
        """Set whether to use full feature names."""
        self._use_full_feature_names = use_full
        return self
    
    def set_timestamp(self, timestamp: Optional[datetime]) -> 'FeatureStoreSpecBuilder':
        """Set timestamp for point-in-time retrieval."""
        self._timestamp = timestamp
        return self
    
    def set_initialize_on_start(self, initialize: bool) -> 'FeatureStoreSpecBuilder':
        """Set whether to initialize on workflow start."""
        self._initialize_on_start = initialize
        return self
    
    def set_force_recreate(self, force: bool) -> 'FeatureStoreSpecBuilder':
        """Set whether to force recreate feature store."""
        self._force_recreate = force
        return self
    
    def enable_materialization(self) -> 'FeatureStoreSpecBuilder':
        """Enable online store materialization."""
        self._materialize_online = True
        return self
    
    def set_sample_indices(self, indices: Optional[List[int]]) -> 'FeatureStoreSpecBuilder':
        """Set specific sample indices to retrieve."""
        self._sample_indices = indices
        return self
    
    def build(self) -> FeatureStoreSpec:
        """
        Build and return FeatureStoreSpec instance.
        
        Returns:
            Configured FeatureStoreSpec
        """
        return FeatureStoreSpec(
            enabled=self._enabled,
            repo_path=self._repo_path,
            n_features=self._n_features,
            offline_store_type=self._offline_store_type,
            online_store_type=self._online_store_type,
            feature_view_name=self._feature_view_name,
            feature_service_name=self._feature_service_name,
            use_full_feature_names=self._use_full_feature_names,
            timestamp=self._timestamp,
            initialize_on_start=self._initialize_on_start,
            force_recreate=self._force_recreate,
            materialize_online=self._materialize_online,
            sample_indices=self._sample_indices
        )


__all__ = [
    "FeatureStoreSpec",
    "FeatureStoreSpecBuilder",
]

