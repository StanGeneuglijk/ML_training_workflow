"""
Feature Store specification module.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Literal, Any
from datetime import datetime

from data.delta_lake import get_delta_path
from deltalake import DeltaTable

from pydantic import BaseModel, Field, field_validator, ConfigDict


logger = logging.getLogger(__name__)


class FeatureStoreSpec(BaseModel):
    """
    Configuration specification for feature store integration.
    
    Attributes:
        enabled: Whether to use feature store 
        repo_path: Path to repository directory
        project_name: Name of the project
        dataset_name: Name of the dataset
        n_features: Optional number of features 
        offline_store_type: Type of offline store 
        online_store_type: Type of online store 
        feature_view_name: Name of the feature view to use
        use_full_feature_names: Whether to use full feature names 
        timestamp: Optional timestamp for point-in-time feature retrieval
        initialize_on_start: Whether to initialize feature store on workflow start
        materialize_online: Whether to materialize features to online store
    """
    # Pydantic configuration
    model_config = ConfigDict(
        extra='forbid', 
        validate_assignment=True
    )
    
    # Required fields
    enabled: bool = Field(
        default=False,
        description="Whether to enable the feature store"
    )
    repo_path: str = Field(
        default="feature_repo",
        description="Path to repository directory"
    )
    project_name: str = Field(
        default="ml_workflow",
        description="Name of the project"
    )
    dataset_name: str = Field(
        default="classification_data",
        description="Name of the dataset"
    )
    n_features: Optional[int] = Field(
        default=None,
        ge=1,
        le=1000,
        description="Number of features in the dataset"
    )
    offline_store_type: Literal["file", "spark"] = Field(
        default="file",
        description="Type of offline store"
    )
    online_store_type: Optional[Any] = Field(
        default=None,
        description="Type of online store"
    )
    feature_view_name: str = Field(
        default="classification_features",
        description="Name of the feature view to use"
    )
    use_full_feature_names: bool = Field(
        default=False,
        description="Whether to use full feature names"
    )
    timestamp: Optional[datetime] = Field(
        default=None,
        description="Timestamp for point-in-time feature retrieval"
    )
    initialize_on_start: bool = Field(
        default=True,
        description="Whether to initialize feature store on workflow start"
    )
    materialize_online: bool = Field(
        default=False,
        description="Whether to materialize features to online store"
    )
    sample_indices: Optional[List[int]] = Field(
        default=None,
        description="Specific sample indices to retrieve"
    )
    
    # Validators
    @field_validator('repo_path')
    @classmethod
    def validate_repo_path(
        cls, 
        v: str
    ) -> str:
        """
        Validate repository path is not empty.
        
        Args:
            v: Repository path to validate
        """
        if not v.strip():
            raise ValueError("repo_path cannot be empty")
        return v.strip()
    
    @field_validator('feature_view_name')
    @classmethod
    def validate_feature_view_name(
        cls, 
        v: str
    ) -> str:
        """
        Validate feature view name is not empty.
        
        Args:
            v: Feature view name to validate
        """
        if not v.strip():
            raise ValueError("feature_view_name cannot be empty")
        return v.strip()
    
    @field_validator('dataset_name')
    @classmethod
    def validate_dataset_name(
        cls, 
        v: str
    ) -> str:
        """
        Validate dataset name is not empty.
        
        Args:
            v: Dataset name to validate
        """
        if not v.strip():
            raise ValueError("dataset_name cannot be empty")
        return v.strip()
    
    @field_validator('project_name')
    @classmethod
    def validate_project_name(
        cls, 
        v: str
    ) -> str:
        """
        Validate project name is not empty.
        
        Args:
            v: Project name to validate
        """
        if not v.strip():
            raise ValueError("project_name cannot be empty")
        return v.strip()
    
    # Checks
    def should_initialize(
        self
    ) -> bool:
        """
        Check if feature store should be initialized.
        """
        return self.enabled and self.initialize_on_start
    
    def should_materialize(
        self
    ) -> bool:
        """
        Check if features should be materialized to online store.
        """
        return self.enabled and self.materialize_online and self.online_store_type is not None
    
    def detect_n_features(
        self
    ) -> int:
        """
        Detect number of features from  table.
        """
        try:
            delta_path = get_delta_path(self.dataset_name)
            dt = DeltaTable(str(delta_path))

            data_df = dt.to_pandas()
            exclude_cols = {
                'sample_index', 'target', 'ingested_at', 'ingested_date'
                }
            feature_cols = [col for col in data_df.columns if col not in exclude_cols]
            
            return len(feature_cols)

        except Exception as e:
            logger.warning(f"Failed to detect n_features: {e}")
            raise ValueError(f"Cannot auto-detect n_features, please set it explicitly. Error: {e}")
    
    # Getters
    def get_feature_references(
        self, 
        n_features: Optional[int] = None
    ) -> List[str]:
        """
        Get list of feature references for retrieval.
        
        Args:
            n_features: Number of features (uses self.n_features if None)
        
        Returns:
            List of feature references 
        """
        num_features = n_features or self.n_features
        if num_features is None:
            raise ValueError("n_features must be set")
        
        feature_refs = [
            f"{self.feature_view_name}:feature_{i}" 
            for i in range(num_features)
        ]
        feature_refs.append(f"{self.feature_view_name}:target")

        return feature_refs
        
    def get_n_features(
        self
    ) -> int:
        """
        Get number of features, auto-detecting if not set.
        
        Args:
            None 
            
        Returns:
            Number of features 
        """
        if self.n_features is not None:
            num_features = self.n_features
        else:
            num_features = self.detect_n_features()

        return num_features


class FeatureStoreSpecBuilder:
    """
    Builder for creating FeatureStoreSpec instances.
    """
    
    def __init__(self) -> None:
        """
        Initialize builder with default values.
        """
        self._enabled = False
        self._repo_path = "feature_repo"
        self._project_name = "ml_workflow"
        self._dataset_name = "classification_data"
        self._n_features = None
        self._offline_store_type = "file"
        self._online_store_type = None
        self._feature_view_name = "classification_features"
        self._use_full_feature_names = False
        self._timestamp = None
        self._initialize_on_start = True
        self._materialize_online = False
        self._sample_indices = None
    
    def enable(self) -> 'FeatureStoreSpecBuilder':
        """
        Enable feature store integration.
        
        Returns:
            Builder instance for method chaining
        """
        self._enabled = True
        return self
    
    def disable(self) -> 'FeatureStoreSpecBuilder':
        """
        Disable feature store integration.
        
        Returns:
            Builder instance for method chaining
        """
        self._enabled = False
        return self
    
    def set_repo_path(self, repo_path: str) -> 'FeatureStoreSpecBuilder':
        """
        Set repository path.
        
        Args:
            repo_path: Path to repository directory
            
        Returns:
            Builder instance for method chaining
        """
        self._repo_path = repo_path
        return self
    
    def set_project_name(self, project_name: str) -> 'FeatureStoreSpecBuilder':
        """
        Set project name.
        
        Args:
            project_name: Name of the project
            
        Returns:
            Builder instance for method chaining
        """
        self._project_name = project_name
        return self
    
    def set_dataset_name(self, dataset_name: str) -> 'FeatureStoreSpecBuilder':
        """
        Set dataset name.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Builder instance for method chaining
        """
        self._dataset_name = dataset_name
        return self
    
    def set_n_features(self, n_features: int) -> 'FeatureStoreSpecBuilder':
        """
        Set number of features.
        
        Args:
            n_features: Number of features in the dataset
            
        Returns:
            Builder instance for method chaining
        """
        self._n_features = n_features
        return self
    
    def set_offline_store(
        self, 
        store_type: Literal["file", "spark"]
    ) -> 'FeatureStoreSpecBuilder':
        """
        Set offline store type.
        
        Args:
            store_type: Type of offline store ("file" or "spark")
            
        Returns:
            Builder instance for method chaining
        """
        self._offline_store_type = store_type
        return self
    
    def set_online_store(
        self,
        store_type: Optional[Literal["sqlite", "redis"]]
    ) -> 'FeatureStoreSpecBuilder':
        """
        Set online store type.
        
        Args:
            store_type: Type of online store ("sqlite", "redis", or None)
            
        Returns:
            Builder instance for method chaining
        """
        self._online_store_type = store_type
        return self
    
    def set_feature_view(self, view_name: str) -> 'FeatureStoreSpecBuilder':
        """
        Set feature view name.
        
        Args:
            view_name: Name of the feature view to use
            
        Returns:
            Builder instance for method chaining
        """
        self._feature_view_name = view_name
        return self
    
    def use_full_names(self, use_full: bool = True) -> 'FeatureStoreSpecBuilder':
        """
        Set whether to use full feature names.
        
        Args:
            use_full: Whether to use full feature names (default: True)
            
        Returns:
            Builder instance for method chaining
        """
        self._use_full_feature_names = use_full
        return self
    
    def set_timestamp(self, timestamp: Optional[datetime]) -> 'FeatureStoreSpecBuilder':
        """
        Set timestamp for point-in-time retrieval.
        
        Args:
            timestamp: Timestamp for point-in-time feature retrieval (None for latest)
            
        Returns:
            Builder instance for method chaining
        """
        self._timestamp = timestamp
        return self
    
    def set_initialize_on_start(self, initialize: bool) -> 'FeatureStoreSpecBuilder':
        """
        Set whether to initialize on workflow start.
        
        Args:
            initialize: Whether to initialize feature store on workflow start
            
        Returns:
            Builder instance for method chaining
        """
        self._initialize_on_start = initialize
        return self
    
    def enable_materialization(self) -> 'FeatureStoreSpecBuilder':
        """
        Enable online store materialization.
        
        Returns:
            Builder instance for method chaining
        """
        self._materialize_online = True
        return self
    
    def disable_materialization(self) -> 'FeatureStoreSpecBuilder':
        """
        Disable online store materialization.
        
        Returns:
            Builder instance for method chaining
        """
        self._materialize_online = False
        return self
    
    def set_sample_indices(self, indices: Optional[List[int]]) -> 'FeatureStoreSpecBuilder':
        """
        Set specific sample indices to retrieve.
        
        Args:
            indices: List of sample indices to retrieve (None for all samples)
            
        Returns:
            Builder instance for method chaining
        """
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
            project_name=self._project_name,
            dataset_name=self._dataset_name,
            n_features=self._n_features,
            offline_store_type=self._offline_store_type,
            online_store_type=self._online_store_type,
            feature_view_name=self._feature_view_name,
            use_full_feature_names=self._use_full_feature_names,
            timestamp=self._timestamp,
            initialize_on_start=self._initialize_on_start,
            materialize_online=self._materialize_online,
            sample_indices=self._sample_indices
        )


__all__ = [
    "FeatureStoreSpec",
    "FeatureStoreSpecBuilder",
]

