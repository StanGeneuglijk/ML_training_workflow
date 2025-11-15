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
    if repo_path.exists():
        shutil.rmtree(repo_path)


class TestFeastManager:
    """Tests for FeastManager."""
    
    def test_feast_manager_init(self, temp_repo_path):
        """Test FeastManager initialization."""
        manager = FeastManager(
            repo_path=temp_repo_path,
            project_name="test_project"
        )
        
        assert manager.project_name == "test_project"
        assert manager.initialized is False
        assert manager.store is None
        assert manager.feature_views == {}
    
    def test_feast_manager_get_absolute_path_relative(self, temp_repo_path):
        """Test _get_absolute_path with relative path."""
        manager = FeastManager(repo_path="relative/path")
        
        abs_path = manager._get_absolute_path("test_path")
        assert os.path.isabs(abs_path)
    
    def test_feast_manager_get_absolute_path_absolute(self, temp_repo_path):
        """Test _get_absolute_path with absolute path."""
        from pathlib import Path
        
        manager = FeastManager(repo_path=temp_repo_path)
        
        abs_path = manager._get_absolute_path("/absolute/path")
        assert abs_path == Path("/absolute/path")
    
    @patch('feature_store.feast_manager.FeatureStore')
    def test_feast_manager_initialize(self, mock_feature_store, temp_repo_path):
        """Test feature store initialization."""
        manager = FeastManager(repo_path=temp_repo_path)
        
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
        
        store1 = manager.initialize()
        store2 = manager.initialize()
        
        assert store1 == store2
        assert mock_feature_store.call_count == 1
    
    def test_feast_manager_cleanup_before_reinitialize(self, temp_repo_path):
        """Test cleanup before reinitializing feature store."""
        manager = FeastManager(repo_path=temp_repo_path)
        
        os.makedirs(temp_repo_path, exist_ok=True)
        test_file = os.path.join(temp_repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        manager.cleanup()
        
        assert not os.path.exists(test_file)
    
    def test_feast_manager_create_feature_store_yaml(self, temp_repo_path):
        """Test creation of feature_store.yaml."""
        from feature_store.feast_manager import REGISTRY_PATH
        
        manager = FeastManager(repo_path=temp_repo_path)
        
        os.makedirs(temp_repo_path, exist_ok=True)
        manager._create_feature_store_yaml()
        
        yaml_path = os.path.join(temp_repo_path, "feature_store.yaml")
        assert os.path.exists(yaml_path)
        
        with open(yaml_path, "r") as f:
            content = f.read()
            assert "project: ml_workflow" in content
            assert "provider: local" in content
            assert "offline_store:" in content
            assert str(REGISTRY_PATH) in content
    
    def test_feast_manager_get_training_data_error_not_initialized(self):
        """Test get_training_data raises error if not initialized."""
        manager = FeastManager(repo_path="test_repo")
        
        entity_df = pd.DataFrame({
            "sample_index": [0, 1],
            "event_timestamp": [pd.Timestamp.now()] * 2
        })
        
        with pytest.raises(ValueError, match="must be initialized"):
            manager.get_training_data(
                entity_df=entity_df,
                feature_names=["feature_0", "feature_1"],
                target_name="target"
            )
    
    def test_feast_manager_cleanup(self, temp_repo_path):
        """Test cleanup method."""
        manager = FeastManager(repo_path=temp_repo_path)
        
        os.makedirs(temp_repo_path, exist_ok=True)
        test_file = os.path.join(temp_repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")
        
        manager.initialized = True
        manager.cleanup()
        
        assert not os.path.exists(temp_repo_path)
        assert manager.initialized is False
        assert manager.store is None
    
    @patch('feature_store.feast_manager.FeatureStore')
    def test_feast_manager_register_feature_view_single(self, mock_feature_store, temp_repo_path):
        """Test registering a single feature view."""
        from feast import FeatureView, Entity
        from feature_store.feature_definitions import create_entity, create_schema
        from feature_store.data_sources import create_file_source
        from data.delta_lake import get_delta_path
        from feast.types import Float64
        
        manager = FeastManager(repo_path=temp_repo_path)
        mock_store = Mock()
        mock_store.apply = Mock()
        mock_feature_store.return_value = mock_store
        manager.initialize()
        
        entity = create_entity()
        schema = create_schema(["feature_0", "feature_1"], default_type=Float64)
        source = create_file_source(
            path=get_delta_path("test_data"),
            timestamp_field="ingested_at"
        )
        feature_view = FeatureView(
            name="test_features",
            entities=[entity],
            schema=schema,
            source=source
        )
        
        manager.register_feature_view(feature_view)
        
        mock_store.apply.assert_called_once()
        assert "test_features" in manager.feature_views
        assert manager.feature_views["test_features"] == feature_view
    
    @patch('feature_store.feast_manager.FeatureStore')
    def test_feast_manager_register_feature_view_list(self, mock_feature_store, temp_repo_path):
        """Test registering multiple feature views."""
        from feast import FeatureView, Entity
        from feature_store.feature_definitions import create_entity, create_schema
        from feature_store.data_sources import create_file_source
        from data.delta_lake import get_delta_path
        from feast.types import Float64
        
        manager = FeastManager(repo_path=temp_repo_path)
        mock_store = Mock()
        mock_store.apply = Mock()
        mock_feature_store.return_value = mock_store
        manager.initialize()
        
        entity = create_entity()
        schema1 = create_schema(["feature_0"], default_type=Float64)
        schema2 = create_schema(["feature_1"], default_type=Float64)
        source = create_file_source(
            path=get_delta_path("test_data"),
            timestamp_field="ingested_at"
        )
        fv1 = FeatureView(name="features_1", entities=[entity], schema=schema1, source=source)
        fv2 = FeatureView(name="features_2", entities=[entity], schema=schema2, source=source)
        
        manager.register_feature_view([fv1, fv2])
        
        mock_store.apply.assert_called_once()
        assert len(manager.feature_views) == 2
        assert "features_1" in manager.feature_views
        assert "features_2" in manager.feature_views
    
    def test_feast_manager_register_feature_view_error_not_initialized(self):
        """Test register_feature_view raises error if not initialized."""
        from feast import FeatureView
        
        manager = FeastManager(repo_path="test_repo")
        mock_fv = Mock(spec=FeatureView)
        
        with pytest.raises(ValueError, match="must be initialized"):
            manager.register_feature_view(mock_fv)
    
    @patch('feature_store.feast_manager.FeatureStore')
    def test_feast_manager_get_features(self, mock_feature_store, temp_repo_path):
        """Test get_features method."""
        manager = FeastManager(repo_path=temp_repo_path)
        mock_store = Mock()
        mock_historical_features = Mock()
        mock_df = pd.DataFrame({
            "sample_index": [0, 1],
            "feature_0": [0.5, 0.6],
            "feature_1": [0.7, 0.8]
        })
        mock_historical_features.to_df.return_value = mock_df
        mock_store.get_historical_features.return_value = mock_historical_features
        mock_feature_store.return_value = mock_store
        manager.initialize()
        
        entity_df = pd.DataFrame({
            "sample_index": [0, 1],
            "event_timestamp": [pd.Timestamp.now()] * 2
        })
        
        features_df = manager.get_features(
            entity_df=entity_df,
            features=["feature_0", "feature_1"]
        )
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == 2
        assert "feature_0" in features_df.columns
        assert "feature_1" in features_df.columns
        mock_store.get_historical_features.assert_called_once()
    
    @patch('feature_store.feast_manager.FeatureStore')
    def test_feast_manager_get_training_data_success(self, mock_feature_store, temp_repo_path):
        """Test get_training_data success case."""
        manager = FeastManager(repo_path=temp_repo_path)
        mock_store = Mock()
        mock_historical_features = Mock()
        mock_df = pd.DataFrame({
            "sample_index": [0, 1],
            "feature_0": [0.5, 0.6],
            "feature_1": [0.7, 0.8],
            "target": [0, 1]
        })
        mock_historical_features.to_df.return_value = mock_df
        mock_store.get_historical_features.return_value = mock_historical_features
        mock_feature_store.return_value = mock_store
        manager.initialize()
        
        entity_df = pd.DataFrame({
            "sample_index": [0, 1],
            "event_timestamp": [pd.Timestamp.now()] * 2
        })
        
        X, y = manager.get_training_data(
            entity_df=entity_df,
            feature_names=["feature_0", "feature_1"],
            target_name="target"
        )
        
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert X.shape == (2, 2)
        assert y.shape == (2,)
        assert np.array_equal(y, np.array([0, 1]))
    
    @patch('deltalake.DeltaTable')
    @patch('data.delta_lake.get_delta_path')
    def test_feast_manager_get_entity_values_with_dataset_name(
        self, mock_get_delta_path, mock_delta_table, temp_repo_path
    ):
        """Test get_entity_values with dataset_name."""
        from pathlib import Path
        
        manager = FeastManager(repo_path=temp_repo_path)
        
        mock_path = Path(temp_repo_path) / "delta_table"
        mock_path.mkdir(parents=True, exist_ok=True)
        mock_get_delta_path.return_value = mock_path
        
        mock_dt = Mock()
        mock_df = pd.DataFrame({
            "sample_index": [0, 1, 2, 3, 4],
            "feature_0": [0.1, 0.2, 0.3, 0.4, 0.5]
        })
        mock_dt.to_pandas.return_value = mock_df
        mock_delta_table.return_value = mock_dt
        
        entity_values = manager.get_entity_values(
            entity_column="sample_index",
            dataset_name="test_dataset"
        )
        
        assert entity_values == [0, 1, 2, 3, 4]
        mock_get_delta_path.assert_called_once_with("test_dataset")
    
    @patch('deltalake.DeltaTable')
    def test_feast_manager_get_entity_values_with_data_path(
        self, mock_delta_table, temp_repo_path
    ):
        """Test get_entity_values with data_path."""
        from pathlib import Path
        
        manager = FeastManager(repo_path=temp_repo_path)
        
        mock_path = Path(temp_repo_path) / "delta_table"
        mock_path.mkdir(parents=True, exist_ok=True)
        
        mock_dt = Mock()
        mock_df = pd.DataFrame({
            "sample_index": [10, 20, 30],
            "feature_0": [0.1, 0.2, 0.3]
        })
        mock_dt.to_pandas.return_value = mock_df
        mock_delta_table.return_value = mock_dt
        
        entity_values = manager.get_entity_values(
            entity_column="sample_index",
            data_path=str(mock_path)
        )
        
        assert entity_values == [10, 20, 30]
    
    def test_feast_manager_get_entity_values_error_no_params(self):
        """Test get_entity_values raises error when neither dataset_name nor data_path provided."""
        manager = FeastManager(repo_path="test_repo")
        
        with pytest.raises(ValueError, match="Must provide either dataset_name or data_path"):
            manager.get_entity_values(entity_column="sample_index")
    
    @patch('deltalake.DeltaTable')
    @patch('data.delta_lake.get_delta_path')
    def test_feast_manager_get_entity_values_error_column_not_found(
        self, mock_get_delta_path, mock_delta_table, temp_repo_path
    ):
        """Test get_entity_values raises error when column not found."""
        from pathlib import Path
        
        manager = FeastManager(repo_path=temp_repo_path)
        
        mock_path = Path(temp_repo_path) / "delta_table"
        mock_path.mkdir(parents=True, exist_ok=True)
        mock_get_delta_path.return_value = mock_path
        
        mock_dt = Mock()
        mock_df = pd.DataFrame({
            "feature_0": [0.1, 0.2],
            "feature_1": [0.3, 0.4]
        })
        mock_dt.to_pandas.return_value = mock_df
        mock_delta_table.return_value = mock_dt
        
        with pytest.raises(ValueError, match="Entity column 'sample_index' not found"):
            manager.get_entity_values(
                entity_column="sample_index",
                dataset_name="test_dataset"
            )


class TestCreateFeastManager:
    """Tests for create_feast_manager factory function."""
    
    @patch('feature_store.feast_manager.FeastManager.initialize')
    def test_create_feast_manager_with_initialize(self, mock_init, temp_repo_path):
        """Test create_feast_manager with initialize=True."""
        manager = create_feast_manager(
            repo_path=temp_repo_path,
            project_name="test_project",
            initialize=True
        )
        
        assert manager.project_name == "test_project"
        mock_init.assert_called_once()
    
    @patch('feature_store.feast_manager.FeastManager.initialize')
    def test_create_feast_manager_without_initialize(self, mock_init, temp_repo_path):
        """Test create_feast_manager with initialize=False."""
        manager = create_feast_manager(
            repo_path=temp_repo_path,
            project_name="test_project",
            initialize=False
        )
        
        assert manager.project_name == "test_project"
        mock_init.assert_not_called()

