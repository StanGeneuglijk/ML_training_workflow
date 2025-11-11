"""
Unit tests for Feast manager.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import os
import shutil

from feature_store.feast_manager import FeastManager, create_feast_manager


@pytest.fixture
def temp_repo_path(tmp_path):
    """Create a temporary repository path."""
    repo_path = tmp_path / "test_feast_repo"
    yield str(repo_path)
    # Cleanup
    if repo_path.exists():
        shutil.rmtree(repo_path)


class TestFeastManager:
    """Tests for FeastManager."""
    
    def test_feast_manager_init(self, temp_repo_path):
        """Test FeastManager initialization."""
        manager = FeastManager(
            repo_path=temp_repo_path,
            n_features=10
        )
        
        assert manager.n_features == 10
        assert manager.initialized is False
        assert manager.store is None
    
    def test_feast_manager_get_absolute_path_relative(self, temp_repo_path):
        """Test _get_absolute_path with relative path."""
        manager = FeastManager(repo_path="relative/path")
        
        abs_path = manager._get_absolute_path("test_path")
        assert os.path.isabs(abs_path)
    
    def test_feast_manager_get_absolute_path_absolute(self, temp_repo_path):
        """Test _get_absolute_path with absolute path."""
        manager = FeastManager(repo_path=temp_repo_path)
        
        abs_path = manager._get_absolute_path("/absolute/path")
        assert abs_path == "/absolute/path"
    
    @patch('feature_store.feast_manager.FeatureStore')
    def test_feast_manager_initialize(self, mock_feature_store, temp_repo_path):
        """Test feature store initialization."""
        manager = FeastManager(repo_path=temp_repo_path, n_features=5)
        
        # Mock FeatureStore
        mock_store = Mock()
        mock_feature_store.return_value = mock_store
        
        store = manager.initialize()
        
        assert manager.initialized is True
        assert manager.store is not None
        assert os.path.exists(temp_repo_path)
    
    @patch('feature_store.feast_manager.FeatureStore')
    def test_feast_manager_initialize_idempotent(self, mock_feature_store, temp_repo_path):
        """Test that initialize is idempotent."""
        manager = FeastManager(repo_path=temp_repo_path)
        
        mock_store = Mock()
        mock_feature_store.return_value = mock_store
        
        # Initialize twice
        store1 = manager.initialize()
        store2 = manager.initialize()
        
        # Should return same store
        assert store1 == store2
        # Should only call FeatureStore constructor once
        assert mock_feature_store.call_count == 1
    
    @patch('feature_store.feast_manager.FeatureStore')
    def test_feast_manager_initialize_force_recreate(self, mock_feature_store, temp_repo_path):
        """Test force recreate feature store."""
        manager = FeastManager(repo_path=temp_repo_path)
        
        # Create the directory first
        os.makedirs(temp_repo_path, exist_ok=True)
        test_file = os.path.join(temp_repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        mock_store = Mock()
        mock_feature_store.return_value = mock_store
        
        # Initialize with force_recreate
        manager.initialize(force_recreate=True)
        
        # Old file should be gone
        assert not os.path.exists(test_file)
    
    def test_feast_manager_create_feature_store_yaml(self, temp_repo_path):
        """Test creation of feature_store.yaml."""
        manager = FeastManager(repo_path=temp_repo_path)
        
        os.makedirs(temp_repo_path, exist_ok=True)
        manager._create_feature_store_yaml()
        
        yaml_path = os.path.join(temp_repo_path, "feature_store.yaml")
        assert os.path.exists(yaml_path)
        
        # Check content
        with open(yaml_path, "r") as f:
            content = f.read()
            assert "project: ml_workflow" in content
            assert "provider: local" in content
            assert "offline_store:" in content
    
    def test_feast_manager_get_training_data_error_not_initialized(self):
        """Test get_training_data raises error if not initialized."""
        manager = FeastManager(repo_path="test_repo")
        
        with pytest.raises(ValueError, match="must be initialized"):
            manager.get_training_data(sample_indices=[0, 1])
    
    def test_feast_manager_cleanup(self, temp_repo_path):
        """Test cleanup method."""
        manager = FeastManager(repo_path=temp_repo_path)
        
        # Create repo directory
        os.makedirs(temp_repo_path, exist_ok=True)
        test_file = os.path.join(temp_repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        manager.initialized = True
        manager.cleanup()
        
        assert not os.path.exists(temp_repo_path)
        assert manager.initialized is False
        assert manager.store is None


class TestCreateFeastManager:
    """Tests for create_feast_manager factory function."""
    
    @patch('feature_store.feast_manager.FeastManager.initialize')
    def test_create_feast_manager_with_initialize(self, mock_init, temp_repo_path):
        """Test create_feast_manager with initialize=True."""
        manager = create_feast_manager(
            repo_path=temp_repo_path,
            n_features=10,
            initialize=True
        )
        
        assert manager.n_features == 10
        mock_init.assert_called_once()
    
    @patch('feature_store.feast_manager.FeastManager.initialize')
    def test_create_feast_manager_without_initialize(self, mock_init, temp_repo_path):
        """Test create_feast_manager with initialize=False."""
        manager = create_feast_manager(
            repo_path=temp_repo_path,
            n_features=10,
            initialize=False
        )
        
        assert manager.n_features == 10
        mock_init.assert_not_called()
    
    @patch('feature_store.feast_manager.FeastManager.initialize')
    def test_create_feast_manager_force_recreate(self, mock_init, temp_repo_path):
        """Test create_feast_manager with force_recreate."""
        manager = create_feast_manager(
            repo_path=temp_repo_path,
            initialize=True,
            force_recreate=True
        )
        
        mock_init.assert_called_once_with(force_recreate=True)

