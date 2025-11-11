"""
Feast manager module
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from feast import FeatureStore

from feature_store.feature_definitions import create_classification_features
import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


class FeastManager:
    """
    Manager for Feast feature store operations.
    
    Handles initialization and feature retrieval from Delta Lake.
    """
    
    def __init__(
        self,
        repo_path: str = "feature_repo",
        n_features: int = 20
    ):
        """
        Initialize Feast manager.
        
        Args:
            repo_path: Path to Feast repository directory
            n_features: Number of features in dataset (default: 20)
        """
        self.repo_path = self._get_absolute_path(repo_path)
        self.n_features = n_features
        self.store: FeatureStore | None = None
        self.initialized = False
        
        logger.info("FeastManager created: repo_path=%s, n_features=%s", self.repo_path, n_features)
    
    def _get_absolute_path(self, path: str) -> str:
        """Convert relative path to absolute based on project root."""
        if os.path.isabs(path):
            return path
        
        current_dir = Path(__file__).resolve().parent
        project_root = current_dir.parent
        return str(project_root / path)
    
    def initialize(self, force_recreate: bool = False) -> FeatureStore:
        """
        Initialize Feast feature store.
        
        Creates repository, configuration, and registers feature views.
        
        Args:
            force_recreate: If True, delete existing repo and recreate
            
        Returns:
            Initialized FeatureStore instance
            
        Raises:
            ValueError: If initialization fails
        """
        if self.initialized and self.store is not None and not force_recreate:
            logger.info("Feature store already initialized")
            return self.store
        
        if force_recreate and os.path.exists(self.repo_path):
            logger.info("Removing existing feature repo: %s", self.repo_path)
            shutil.rmtree(self.repo_path)
        
        os.makedirs(self.repo_path, exist_ok=True)
        
        self._create_feature_store_yaml()
        
        logger.info("Initializing Feast feature store at: %s", self.repo_path)
        self.store = FeatureStore(repo_path=self.repo_path)
        
        self._apply_feature_definitions()
        
        self.initialized = True
        logger.info("Feature store initialized successfully")
        return self.store
    
    def _create_feature_store_yaml(self) -> None:
        """Create feature_store.yaml configuration file."""
        registry_path = os.path.join(self.repo_path, "data", "registry.db")
        os.makedirs(os.path.dirname(registry_path), exist_ok=True)
        
        yaml_content = f"""project: ml_workflow
registry: {registry_path}
provider: local
offline_store:
    type: file
entity_key_serialization_version: 2
"""
        
        yaml_path = os.path.join(self.repo_path, "feature_store.yaml")
        with open(yaml_path, "w") as f:
            f.write(yaml_content)
        
        logger.info("Created configuration: %s", yaml_path)
    
    def _apply_feature_definitions(self) -> None:
        """Register feature views with Feast."""
        if self.store is None:
            raise ValueError("Feature store must be initialized first")
        
        logger.info("Registering feature views...")
        
        feature_view = create_classification_features(n_features=self.n_features)
        self.store.apply([feature_view])
        
        logger.info("Feature view registered: classification_features")
    
    def get_training_data(
        self,
        sample_indices: list[int] | None = None,
        timestamp = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get training data as numpy arrays.
        
        Retrieves features from Feast offline store (Delta Lake).
        
        Args:
            sample_indices: List of sample indices to retrieve
                          If None, retrieves all available samples
            timestamp: Timestamp for point-in-time retrieval
                      If None, uses current time (ignored for offline store)
            
        Returns:
            Tuple of (X, y) as numpy arrays where:
            - X: Feature matrix
            - y: Target vector
            
        Raises:
            ValueError: If feature store not initialized
            
        Example:
            >>> manager = FeastManager(n_features=20)
            >>> manager.initialize()
            >>> X, y = manager.get_training_data(sample_indices=[0, 1, 2])
            >>> print(X.shape)  # (3, 20)
        """
        if self.store is None:
            raise ValueError("Feature store must be initialized first")
        
        if sample_indices is None:
            sample_indices = self._get_all_sample_indices()
        
        # Use provided timestamp or current time
        event_timestamp = timestamp if timestamp is not None else pd.Timestamp.now()
        
        entity_df = pd.DataFrame({
            "sample_index": sample_indices,
            "event_timestamp": [event_timestamp] * len(sample_indices)
        })
        
        features = [
            f"classification_features:feature_{i}" 
            for i in range(self.n_features)
        ]
        features.append("classification_features:target")
        
        logger.info("Retrieving %s features for %s samples", len(features), len(sample_indices))
        
        features_df = self.store.get_historical_features(
            entity_df=entity_df,
            features=features,
            full_feature_names=False
        ).to_df()
        
        feature_cols = [f"feature_{i}" for i in range(self.n_features)]
        X = features_df[feature_cols].values
        y = features_df["target"].values
        
        logger.info("Retrieved training data: X shape %s, y shape %s", X.shape, y.shape)
        return X, y
    
    def _get_all_sample_indices(self) -> list[int]:
        """Get all available sample indices from SQLite database."""
        import sqlite3
        from data import get_database_path
        
        db_path = get_database_path()
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get dataset_id for classification_data
        cursor.execute("SELECT id FROM datasets WHERE name = ?", ("classification_data",))
        result = cursor.fetchone()
        if not result:
            conn.close()
            raise ValueError("Dataset 'classification_data' not found")
        dataset_id = result[0]
        
        # Get all sample indices
        cursor.execute(
            "SELECT DISTINCT sample_index FROM features WHERE dataset_id = ? ORDER BY sample_index",
            (dataset_id,)
        )
        sample_indices = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        logger.info("Found %s samples in database", len(sample_indices))
        return sample_indices
    
    def cleanup(self) -> None:
        """Remove feature store repository and reset state."""
        if os.path.exists(self.repo_path):
            logger.info("Cleaning up feature store: %s", self.repo_path)
            shutil.rmtree(self.repo_path)
            self.initialized = False
            self.store = None
            logger.info("Feature store cleaned up")


def create_feast_manager(
    repo_path: str = "feature_repo",
    n_features: int = 20,
    initialize: bool = True,
    force_recreate: bool = False
) -> FeastManager:
    """
    Create and optionally initialize a FeastManager.
    
    Args:
        repo_path: Path to Feast repository (default: "feature_repo")
        n_features: Number of features (default: 20)
        initialize: Auto-initialize feature store (default: True)
        force_recreate: Recreate from scratch (default: False)
        
    Returns:
        FeastManager instance
        
    Example:
        >>> manager = create_feast_manager(n_features=20, initialize=True)
        >>> X, y = manager.get_training_data(sample_indices=[0, 1, 2])
    """
    manager = FeastManager(repo_path=repo_path, n_features=n_features)
    
    if initialize:
        manager.initialize(force_recreate=force_recreate)
    
    return manager
