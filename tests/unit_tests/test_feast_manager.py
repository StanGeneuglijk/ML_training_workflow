"""
Unit tests for Feast manager.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock, patch

from feature_store.feast_manager import FeastManager, create_feast_manager


@pytest.mark.unit
class TestFeastManager:
    """Tests for FeastManager."""

    def test_feast_manager_init(self, temp_repo_path):
        """Test FeastManager initialization."""
        manager = FeastManager(
            repo_path=temp_repo_path,
            project_name="test_project",
        )

        assert manager.project_name == "test_project", (
            f"Project name should be 'test_project', got '{manager.project_name}'"
        )
        assert manager.initialized is False, (
            "Manager should not be initialized by default"
        )
        assert manager.store is None, (
            "Store should be None before initialization"
        )
        assert manager.feature_views == {}, (
            f"Feature views should be empty dict, got {manager.feature_views}"
        )

    @pytest.mark.parametrize(
        "input_path,expected_absolute",
        [
            pytest.param("relative/path", True, id="relative_path"),
            pytest.param("/absolute/path", "/absolute/path", id="absolute_path"),
        ],
    )
    def test_feast_manager_get_absolute_path(self, input_path, expected_absolute):
        """Test _get_absolute_path with relative and absolute paths."""
        manager = FeastManager(repo_path="test_repo")

        abs_path = manager._get_absolute_path(input_path)

        if expected_absolute is True:
            assert os.path.isabs(abs_path), (
                f"Path should be absolute, got '{abs_path}'"
            )
        else:
            assert abs_path == Path(expected_absolute), (
                f"Path should be '{expected_absolute}', got '{abs_path}'"
            )

    @patch("feature_store.feast_manager.FeatureStore")
    def test_feast_manager_initialize(self, mock_feature_store, temp_repo_path):
        """Test feature store initialization."""
        manager = FeastManager(repo_path=temp_repo_path)

        mock_store = Mock()
        mock_feature_store.return_value = mock_store

        store = manager.initialize()

        assert manager.initialized is True, (
            "Manager should be initialized after calling initialize()"
        )
        assert manager.store is not None, (
            "Store should not be None after initialization"
        )
        assert os.path.exists(temp_repo_path), (
            f"Repository path should exist, got '{temp_repo_path}'"
        )

    @patch("feature_store.feast_manager.FeatureStore")
    def test_feast_manager_initialize_idempotent(
        self, mock_feature_store, temp_repo_path
    ):
        """Test that initialize is idempotent."""
        manager = FeastManager(repo_path=temp_repo_path)

        mock_store = Mock()
        mock_feature_store.return_value = mock_store

        store1 = manager.initialize()
        store2 = manager.initialize()

        assert store1 == store2, (
            "Multiple initialize() calls should return the same store instance"
        )
        assert mock_feature_store.call_count == 1, (
            f"FeatureStore should be instantiated only once, got {mock_feature_store.call_count} calls"
        )

    def test_feast_manager_cleanup_before_reinitialize(self, temp_repo_path):
        """Test cleanup before reinitializing feature store."""
        manager = FeastManager(repo_path=temp_repo_path)

        os.makedirs(temp_repo_path, exist_ok=True)
        test_file = os.path.join(temp_repo_path, "test.txt")
        with open(test_file, "w") as f:
            f.write("test")

        manager.cleanup()

        assert not os.path.exists(test_file), (
            f"Test file should be removed during cleanup, but still exists at '{test_file}'"
        )

    def test_feast_manager_create_feature_store_yaml(self, temp_repo_path):
        """Test creation of feature_store.yaml."""
        from feature_store.feast_manager import REGISTRY_PATH

        manager = FeastManager(repo_path=temp_repo_path)

        os.makedirs(temp_repo_path, exist_ok=True)
        manager._create_feature_store_yaml()

        yaml_path = os.path.join(temp_repo_path, "feature_store.yaml")
        assert os.path.exists(yaml_path), (
            f"feature_store.yaml should exist at '{yaml_path}'"
        )

        with open(yaml_path, "r") as f:
            content = f.read()
            assert "project: ml_workflow" in content, (
                "YAML should contain project name 'ml_workflow'"
            )
            assert "provider: local" in content, (
                "YAML should contain provider 'local'"
            )
            assert "offline_store:" in content, (
                "YAML should contain offline_store configuration"
            )
            assert str(REGISTRY_PATH) in content, (
                f"YAML should contain registry path '{REGISTRY_PATH}'"
            )

    def test_feast_manager_get_training_data_error_not_initialized(self):
        """Test get_training_data raises error if not initialized."""
        manager = FeastManager(repo_path="test_repo")

        entity_df = pd.DataFrame(
            {
                "sample_index": [0, 1],
                "event_timestamp": [pd.Timestamp.now()] * 2,
            }
        )

        with pytest.raises(ValueError, match="must be initialized") as exc_info:
            manager.get_training_data(
                entity_df=entity_df,
                feature_names=["feature_0", "feature_1"],
                target_name="target",
            )

        assert "must be initialized" in str(exc_info.value).lower(), (
            f"Error message should mention initialization, got: {exc_info.value}"
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

        assert not os.path.exists(temp_repo_path), (
            f"Repository path should be removed after cleanup, but still exists at '{temp_repo_path}'"
        )
        assert manager.initialized is False, (
            "Manager should be marked as not initialized after cleanup"
        )
        assert manager.store is None, (
            "Store should be None after cleanup"
        )

    @patch("feature_store.feast_manager.FeatureStore")
    def test_feast_manager_register_feature_view_single(
        self, mock_feature_store, temp_repo_path
    ):
        """Test registering a single feature view."""
        from feast import FeatureView
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
            timestamp_field="ingested_at",
        )
        feature_view = FeatureView(
            name="test_features",
            entities=[entity],
            schema=schema,
            source=source,
        )

        manager.register_feature_view(feature_view)

        mock_store.apply.assert_called_once(), (
            "store.apply() should be called once"
        )
        assert "test_features" in manager.feature_views, (
            "Feature view 'test_features' should be registered"
        )
        assert manager.feature_views["test_features"] == feature_view, (
            "Registered feature view should match the input"
        )

    @patch("feature_store.feast_manager.FeatureStore")
    def test_feast_manager_register_feature_view_list(
        self, mock_feature_store, temp_repo_path
    ):
        """Test registering multiple feature views."""
        from feast import FeatureView
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
            timestamp_field="ingested_at",
        )
        fv1 = FeatureView(
            name="features_1", entities=[entity], schema=schema1, source=source
        )
        fv2 = FeatureView(
            name="features_2", entities=[entity], schema=schema2, source=source
        )

        manager.register_feature_view([fv1, fv2])

        mock_store.apply.assert_called_once(), (
            "store.apply() should be called once"
        )
        assert len(manager.feature_views) == 2, (
            f"Should have 2 feature views registered, got {len(manager.feature_views)}"
        )
        assert "features_1" in manager.feature_views, (
            "Feature view 'features_1' should be registered"
        )
        assert "features_2" in manager.feature_views, (
            "Feature view 'features_2' should be registered"
        )

    def test_feast_manager_register_feature_view_error_not_initialized(self):
        """Test register_feature_view raises error if not initialized."""
        from feast import FeatureView

        manager = FeastManager(repo_path="test_repo")
        mock_fv = Mock(spec=FeatureView)

        with pytest.raises(ValueError, match="must be initialized") as exc_info:
            manager.register_feature_view(mock_fv)

        assert "must be initialized" in str(exc_info.value).lower(), (
            f"Error message should mention initialization, got: {exc_info.value}"
        )

    @patch("feature_store.feast_manager.FeatureStore")
    def test_feast_manager_get_features(self, mock_feature_store, temp_repo_path):
        """Test get_features method."""
        manager = FeastManager(repo_path=temp_repo_path)
        mock_store = Mock()
        mock_historical_features = Mock()
        mock_df = pd.DataFrame(
            {
                "sample_index": [0, 1],
                "feature_0": [0.5, 0.6],
                "feature_1": [0.7, 0.8],
            }
        )
        mock_historical_features.to_df.return_value = mock_df
        mock_store.get_historical_features.return_value = mock_historical_features
        mock_feature_store.return_value = mock_store
        manager.initialize()

        entity_df = pd.DataFrame(
            {
                "sample_index": [0, 1],
                "event_timestamp": [pd.Timestamp.now()] * 2,
            }
        )

        features_df = manager.get_features(
            entity_df=entity_df,
            features=["feature_0", "feature_1"],
        )

        assert isinstance(features_df, pd.DataFrame), (
            f"Should return DataFrame, got {type(features_df)}"
        )
        assert len(features_df) == 2, (
            f"Features DataFrame should have 2 rows, got {len(features_df)}"
        )
        assert "feature_0" in features_df.columns, (
            "Features DataFrame should contain 'feature_0' column"
        )
        assert "feature_1" in features_df.columns, (
            "Features DataFrame should contain 'feature_1' column"
        )
        mock_store.get_historical_features.assert_called_once(), (
            "get_historical_features() should be called once"
        )

    @patch("feature_store.feast_manager.FeatureStore")
    def test_feast_manager_get_training_data_success(
        self, mock_feature_store, temp_repo_path
    ):
        """Test get_training_data success case."""
        manager = FeastManager(repo_path=temp_repo_path)
        mock_store = Mock()
        mock_historical_features = Mock()
        mock_df = pd.DataFrame(
            {
                "sample_index": [0, 1],
                "feature_0": [0.5, 0.6],
                "feature_1": [0.7, 0.8],
                "target": [0, 1],
            }
        )
        mock_historical_features.to_df.return_value = mock_df
        mock_store.get_historical_features.return_value = mock_historical_features
        mock_feature_store.return_value = mock_store
        manager.initialize()

        entity_df = pd.DataFrame(
            {
                "sample_index": [0, 1],
                "event_timestamp": [pd.Timestamp.now()] * 2,
            }
        )

        X, y = manager.get_training_data(
            entity_df=entity_df,
            feature_names=["feature_0", "feature_1"],
            target_name="target",
        )

        assert isinstance(X, np.ndarray), (
            f"X should be numpy array, got {type(X)}"
        )
        assert isinstance(y, np.ndarray), (
            f"y should be numpy array, got {type(y)}"
        )
        assert X.shape == (2, 2), (
            f"X should have shape (2, 2), got {X.shape}"
        )
        assert y.shape == (2,), (
            f"y should have shape (2,), got {y.shape}"
        )
        assert np.array_equal(y, np.array([0, 1])), (
            f"y values should be [0, 1], got {y.tolist()}"
        )

    @patch("deltalake.DeltaTable")
    @patch("data.delta_lake.get_delta_path")
    def test_feast_manager_get_entity_values_with_dataset_name(
        self, mock_get_delta_path, mock_delta_table, temp_repo_path
    ):
        """Test get_entity_values with dataset_name."""
        manager = FeastManager(repo_path=temp_repo_path)

        mock_path = Path(temp_repo_path) / "delta_table"
        mock_path.mkdir(parents=True, exist_ok=True)
        mock_get_delta_path.return_value = mock_path

        mock_dt = Mock()
        mock_df = pd.DataFrame(
            {
                "sample_index": [0, 1, 2, 3, 4],
                "feature_0": [0.1, 0.2, 0.3, 0.4, 0.5],
            }
        )
        mock_dt.to_pandas.return_value = mock_df
        mock_delta_table.return_value = mock_dt

        entity_values = manager.get_entity_values(
            entity_column="sample_index",
            dataset_name="test_dataset",
        )

        assert entity_values == [0, 1, 2, 3, 4], (
            f"Entity values should be [0, 1, 2, 3, 4], got {entity_values}"
        )
        mock_get_delta_path.assert_called_once_with("test_dataset"), (
            "get_delta_path should be called with 'test_dataset'"
        )

    @patch("deltalake.DeltaTable")
    def test_feast_manager_get_entity_values_with_data_path(
        self, mock_delta_table, temp_repo_path
    ):
        """Test get_entity_values with data_path."""
        manager = FeastManager(repo_path=temp_repo_path)

        mock_path = Path(temp_repo_path) / "delta_table"
        mock_path.mkdir(parents=True, exist_ok=True)

        mock_dt = Mock()
        mock_df = pd.DataFrame(
            {
                "sample_index": [10, 20, 30],
                "feature_0": [0.1, 0.2, 0.3],
            }
        )
        mock_dt.to_pandas.return_value = mock_df
        mock_delta_table.return_value = mock_dt

        entity_values = manager.get_entity_values(
            entity_column="sample_index",
            data_path=str(mock_path),
        )

        assert entity_values == [10, 20, 30], (
            f"Entity values should be [10, 20, 30], got {entity_values}"
        )

    def test_feast_manager_get_entity_values_error_no_params(self):
        """Test get_entity_values raises error when neither dataset_name nor data_path provided."""
        manager = FeastManager(repo_path="test_repo")

        with pytest.raises(ValueError, match="Must provide either") as exc_info:
            manager.get_entity_values(entity_column="sample_index")

        assert "Must provide either" in str(exc_info.value), (
            f"Error message should mention providing parameters, got: {exc_info.value}"
        )

    @patch("deltalake.DeltaTable")
    @patch("data.delta_lake.get_delta_path")
    def test_feast_manager_get_entity_values_error_column_not_found(
        self, mock_get_delta_path, mock_delta_table, temp_repo_path
    ):
        """Test get_entity_values raises error when column not found."""
        manager = FeastManager(repo_path=temp_repo_path)

        mock_path = Path(temp_repo_path) / "delta_table"
        mock_path.mkdir(parents=True, exist_ok=True)
        mock_get_delta_path.return_value = mock_path

        mock_dt = Mock()
        mock_df = pd.DataFrame(
            {
                "feature_0": [0.1, 0.2],
                "feature_1": [0.3, 0.4],
            }
        )
        mock_dt.to_pandas.return_value = mock_df
        mock_delta_table.return_value = mock_dt

        with pytest.raises(ValueError, match="Entity column 'sample_index' not found") as exc_info:
            manager.get_entity_values(
                entity_column="sample_index",
                dataset_name="test_dataset",
            )

        assert "not found" in str(exc_info.value).lower(), (
            f"Error message should mention column not found, got: {exc_info.value}"
        )


@pytest.mark.unit
class TestCreateFeastManager:
    """Tests for create_feast_manager factory function."""

    @patch("feature_store.feast_manager.FeastManager.initialize")
    def test_create_feast_manager_with_initialize(self, mock_init, temp_repo_path):
        """Test create_feast_manager with initialize=True."""
        manager = create_feast_manager(
            repo_path=temp_repo_path,
            project_name="test_project",
            initialize=True,
        )

        assert manager.project_name == "test_project", (
            f"Project name should be 'test_project', got '{manager.project_name}'"
        )
        mock_init.assert_called_once(), (
            "initialize() should be called when initialize=True"
        )

    @patch("feature_store.feast_manager.FeastManager.initialize")
    def test_create_feast_manager_without_initialize(self, mock_init, temp_repo_path):
        """Test create_feast_manager with initialize=False."""
        manager = create_feast_manager(
            repo_path=temp_repo_path,
            project_name="test_project",
            initialize=False,
        )

        assert manager.project_name == "test_project", (
            f"Project name should be 'test_project', got '{manager.project_name}'"
        )
        mock_init.assert_not_called(), (
            "initialize() should not be called when initialize=False"
        )
