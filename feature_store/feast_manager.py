"""
Feast manager module.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Union, Dict, Any

import numpy as np
import pandas as pd
from feast import FeatureStore, FeatureView

import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent #CAUTIONS: this might change due architecture change!

REGISTRY_PATH = Path(__file__).resolve().parent.parent / "data" / "registry.db" #CAUTIONS: this might change due architecture change!

class FeastManager:
    """
    Generic manager for feature store operations.
    """
    
    def __init__(
        self,
        repo_path: str = "feature_repo",
        project_name: str = "ml_workflow",
    ):
        """
        Initialize Feast manager.
        
        Args:
            repo_path: Path to Feast repository directory
            project_name: Name of the Feast project (default: "ml_workflow")
        """
        self.repo_path = self._get_absolute_path(repo_path)
        self.project_name = project_name

        self.store: FeatureStore | None = None     
        self.initialized = False 
        self.feature_views: dict[str, FeatureView] = {}
        self._last_provider: Optional[str] = None 
        self._last_offline_store_type: Optional[str] = None
        self._last_online_store_type: Optional[str] = None
        self._last_online_store_config: Optional[Dict[str, Any]] = None
        
        logger.info("FeastManager created: repo_path=%s, project_name=%s", str(self.repo_path), project_name)
    
    def _get_absolute_path(
        self, 
        path: str
    ) -> Path:
        """Convert relative path to absolute based on project root."""
        input_path = Path(path)
        if input_path.is_absolute():
            absolute_path = input_path
        else:
            absolute_path = PROJECT_ROOT / path

        return absolute_path
    
    def initialize(
        self,
        project_name: Optional[str] = None,
        provider: str = "local",
        offline_store_type: str = "file",
        online_store_type: Optional[str] = None,
        online_store_config: Optional[Dict[str, Any]] = None,
    ) -> FeatureStore:
        """
        Initialize feature store.
                
        Args:
            project_name: Name of the project 
            provider: Provider type
            offline_store_type: Type of offline store
            online_store_type: Type of online store ("sqlite" or "redis")
            online_store_config: Optional configuration for online store
            
        Returns:
            Initialized FeatureStore instance
            
        """
        if self.initialized and self.store is not None:
            logger.info("Feature store already initialized")
            return self.store
        
        self.repo_path.mkdir(parents=True, exist_ok=True)

        self._create_feature_store_yaml(
            project_name=project_name,
            provider=provider,
            offline_store_type=offline_store_type,
            online_store_type=online_store_type,
            online_store_config=online_store_config,
        )
        logger.info("Initializing feature store at: %s", self.repo_path)

        self.store = FeatureStore(repo_path=str(self.repo_path)) 
        self.initialized = True
        self._last_provider = provider
        self._last_offline_store_type = offline_store_type
        self._last_online_store_type = online_store_type
        self._last_online_store_config = online_store_config

        logger.info("Feature store initialized successfully (offline=%s, online=%s)", 
                   offline_store_type, online_store_type or "None")

        return self.store
    
    def _create_feature_store_yaml(
        self,
        project_name: Optional[str] = None,
        provider: str = "local",
        offline_store_type: str = "file",
        online_store_type: Optional[str] = None,
        online_store_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Create feature_store.yaml configuration file.
        
        Args:
            project_name: Name of the project 
            provider:  Provider type 
            offline_store_type: Type of offline store
            online_store_type: Type of online store ("sqlite" or "redis")
            online_store_config: Optional configuration for online store
        """
        project = project_name or self.project_name

        registry_path = REGISTRY_PATH
        registry_path.parent.mkdir(parents=True, exist_ok=True)
                
        yaml_path = self.repo_path / "feature_store.yaml"
        
        yaml_lines = [
            f"project: {project}",
            f"registry: {str(registry_path)}",
            f"provider: {provider}",
            "offline_store:",
            f"    type: {offline_store_type}",
            "entity_key_serialization_version: 2"
        ]
        
        if online_store_type:
            yaml_lines.append("online_store:")
            if online_store_type == "sqlite":
                sqlite_path = online_store_config.get("path") if online_store_config else None
                if sqlite_path is None:
                    sqlite_path = str(self.repo_path / "online_store.db")
                yaml_lines.append("    type: sqlite")
                yaml_lines.append(f"    path: {sqlite_path}")
            elif online_store_type == "redis":
                yaml_lines.append("    type: redis")
                if online_store_config:
                    if "host" in online_store_config:
                        yaml_lines.append(f"    host: {online_store_config['host']}")
                    if "port" in online_store_config:
                        yaml_lines.append(f"    port: {online_store_config['port']}")
                    if "db" in online_store_config:
                        yaml_lines.append(f"    db: {online_store_config['db']}")
        
        yaml_content = "\n".join(yaml_lines) + "\n"
        yaml_path.write_text(yaml_content)
        
        logger.info("Created configuration: %s (online_store=%s)", yaml_path, online_store_type or "None")
    
    def register_feature_view(
        self,
        feature_view: Union[FeatureView, list[FeatureView]],
    ) -> None:
        """
        Register feature views.
        
        Args:
            feature_view: FeatureView or list of FeatureViews to register
        """
        if self.store is None:
            raise ValueError("Feature store must be initialized first")
        
        if isinstance(feature_view, FeatureView):
            feature_views = [feature_view]
        else:
            feature_views = feature_view
        
        self.store.apply(feature_views)
        
        for fv in feature_views:
            self.feature_views[fv.name] = fv
        
        logger.info("Registered %d feature views", len(feature_views))
    
    def get_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
        feature_view_name: Optional[str] = None,
        full_feature_names: bool = False,
    ) -> pd.DataFrame:
        """
        Get features from feature store.
        
        Args:
            entity_df: DataFrame with entity keys 
            features: List of feature names (e.g., ["feature_0", "feature_1"]
            feature_view_name: Feature view name to prefix features
            full_feature_names: If True, use full feature names 
            
        Returns:
            DataFrame with features joined to entities
            
        """
        if self.store is None:
            raise ValueError("Feature store must be initialized first")
        
        if feature_view_name and not full_feature_names:
            features = [f"{feature_view_name}:{feat}" for feat in features]
        
        logger.info("Retrieving %s features for %s entities", len(features), len(entity_df))
    
        
        features_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=features,
            full_feature_names=full_feature_names,
        ).to_df()
        
        logger.info("Retrieved features: shape %s", features_df.shape)

        return features_df
    
    def get_training_data(
        self,
        entity_df: pd.DataFrame,
        feature_names: list[str],
        target_name: str,
        feature_view_name: Optional[str] = None,
        full_feature_names: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Get training data.
        
        Args:
            entity_df: DataFrame with entity keys 
            feature_names: List of feature column names to retrieve
            target_name: Name of target column
            feature_view_name: Feature view name to prefix features
            full_feature_names: If True, use full feature names 
            
        Returns:
            Tuple of (X, y) as numpy arrays
        """
        all_features = feature_names + [target_name]

        features_df = self.get_features(
            entity_df=entity_df,
            features=all_features,
            feature_view_name=feature_view_name,
            full_feature_names=full_feature_names,
        )
        X = features_df[feature_names].values
        y = features_df[target_name].values
        
        logger.info("Retrieved training data: X shape %s, y shape %s", X.shape, y.shape)

        return X, y
    

    def get_entity_values(
        self,
        entity_column: str,
        dataset_name: Optional[str] = None,
        data_path: Optional[Union[str, Path]] = None,
    ) -> list:
        """
        Get all unique entity values from a table.
        
        Args:
            entity_column: Name of the entity column
            dataset_name: Dataset name 
            data_path: Optional path to Delta Lake table 
            
        Returns:
            List of unique entity values
            
        Raises:
            ValueError: If neither dataset_name nor data_path is provided
            FileNotFoundError: If Delta Lake table doesn't exist
        """
        from data.delta_lake import get_delta_path
        from deltalake import DeltaTable
        
        if dataset_name:
            delta_path = get_delta_path(dataset_name)
        elif data_path is not None:
            delta_path = Path(data_path)
        else:
            raise ValueError("Must provide either dataset_name or data_path")
        
        if not delta_path.exists():
            raise FileNotFoundError(
                f"Delta Lake table not found at {delta_path}. "
            )
        dt = DeltaTable(str(delta_path))
        df = dt.to_pandas()
        
        if entity_column not in df.columns:
            raise ValueError(f"Entity column '{entity_column}' not found in table")
        
        entity_values = sorted(df[entity_column].unique().tolist())
        
        logger.info("Found %s unique entity values in Delta Lake table", len(entity_values))

        return entity_values
    
    def get_online_features(
        self,
        entity_df: pd.DataFrame,
        features: list[str],
        feature_view_name: Optional[str] = None,
        full_feature_names: bool = False,
    ) -> pd.DataFrame:
        """
        Get features from online store (low-latency for inference).
        
        Args:
            entity_df: DataFrame with entity keys and event_timestamp
            features: List of feature names (e.g., ["feature_0", "feature_1"])
            feature_view_name: Feature view name to prefix features
            full_feature_names: If True, use full feature names
            
        Returns:
            DataFrame with features joined to entities
            
        Raises:
            ValueError: If feature store is not initialized
            RuntimeError: If online store is not configured
        """
        if self.store is None:
            raise ValueError("Feature store must be initialized first")
        
        if self._last_online_store_type is None:
            raise RuntimeError(
                "Online store not configured. Initialize with online_store_type parameter."
            )
        
        if feature_view_name and not full_feature_names:
            features = [f"{feature_view_name}:{feat}" for feat in features]
        
        logger.info("Retrieving %s features from online store for %s entities", 
                   len(features), len(entity_df))
        
        try:
            features_df = self.store.get_online_features(
                entity_rows=[dict(row) for _, row in entity_df.iterrows()],
                features=features,
                full_feature_names=full_feature_names,
            ).to_df()
            
            logger.info("Retrieved features from online store: shape %s", features_df.shape)
            return features_df
        except Exception as e:
            logger.error("Failed to retrieve features from online store: %s", e)
            raise RuntimeError(f"Online feature retrieval failed: {e}") from e
    
    def materialize_features(
        self,
        feature_view_name: str,
        start_date: datetime,
        end_date: datetime,
    ) -> None:
        """
        Materialize features from offline store to online store.
        
        This syncs historical features to the online store for fast inference.
        
        Args:
            feature_view_name: Name of the feature view to materialize
            start_date: Start date for materialization window
            end_date: End date for materialization window
            
        Raises:
            ValueError: If feature store is not initialized
            RuntimeError: If online store is not configured
        """
        if self.store is None:
            raise ValueError("Feature store must be initialized first")
        
        if self._last_online_store_type is None:
            raise RuntimeError(
                "Online store not configured. Initialize with online_store_type parameter."
            )
        
        logger.info(
            "Materializing features for '%s' from %s to %s",
            feature_view_name, start_date, end_date
        )
        
        try:
            self.store.materialize(
                start_date=start_date,
                end_date=end_date,
                feature_views=[feature_view_name]
            )
            logger.info("Materialization completed successfully")
        except Exception as e:
            logger.error("Materialization failed: %s", e)
            raise RuntimeError(f"Materialization failed: {e}") from e
    
    def materialize_incremental(
        self,
        feature_view_name: str,
        end_date: Optional[datetime] = None,
    ) -> None:
        """
        Incrementally materialize features (from last materialized date to now).
        
        Args:
            feature_view_name: Name of the feature view to materialize
            end_date: End date (defaults to now)
            
        Raises:
            ValueError: If feature store is not initialized
            RuntimeError: If online store is not configured
        """
        if self.store is None:
            raise ValueError("Feature store must be initialized first")
        
        if self._last_online_store_type is None:
            raise RuntimeError(
                "Online store not configured. Initialize with online_store_type parameter."
            )
        
        if end_date is None:
            end_date = datetime.utcnow()
        
        logger.info("Incremental materialization for '%s' until %s", feature_view_name, end_date)
        
        try:
            self.store.materialize_incremental(
                end_date=end_date,
                feature_views=[feature_view_name]
            )
            logger.info("Incremental materialization completed successfully")
        except Exception as e:
            logger.error("Incremental materialization failed: %s", e)
            raise RuntimeError(f"Incremental materialization failed: {e}") from e
    
    def cleanup(
        self
    ) -> None:
        """Remove feature store repository and reset state."""
        if self.repo_path.exists():
            logger.info("Cleaning up feature store: %s", self.repo_path)
            shutil.rmtree(self.repo_path)
            self.initialized = False
            self.store = None
            logger.info("Feature store cleaned up")

def create_feast_manager(
    repo_path: str = "feature_repo",
    project_name: str = "ml_workflow",
    initialize: bool = True,
    provider: str = "local",
    offline_store_type: str = "file",
    online_store_type: Optional[str] = None,
    online_store_config: Optional[Dict[str, Any]] = None,
) -> FeastManager:
    """
    Create a FeastManager.
    
    Args:
        repo_path: Path to repository 
        project_name: Name of the project 
        initialize: Auto-initialize feature store 
        provider: Provider type
        offline_store_type: Type of offline store
        online_store_type: Type of online store ("sqlite" or "redis")
        online_store_config: Optional configuration for online store
        
    Returns:
        FeastManager instance
        
    Example:
        >>> # Offline only (training)
        >>> manager = create_feast_manager(initialize=True)
        >>> 
        >>> # With online store (inference)
        >>> manager = create_feast_manager(
        ...     initialize=True,
        ...     online_store_type="sqlite"
        ... )
        >>> 
        >>> # With Redis online store
        >>> manager = create_feast_manager(
        ...     initialize=True,
        ...     online_store_type="redis",
        ...     online_store_config={"host": "localhost", "port": 6379}
        ... )
    """
    manager = FeastManager(repo_path=repo_path, project_name=project_name)
    
    if initialize:
        manager.initialize(
            provider=provider,
            offline_store_type=offline_store_type,
            online_store_type=online_store_type,
            online_store_config=online_store_config,
        )
    
    return manager


__all__ = [
    "FeastManager",
    "create_feast_manager",
]
