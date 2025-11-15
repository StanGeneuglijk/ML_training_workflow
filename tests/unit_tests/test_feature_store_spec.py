"""
Unit tests for feature store specification.
"""
from __future__ import annotations

import pytest
from datetime import datetime

from specs import FeatureStoreSpec, FeatureStoreSpecBuilder


class TestFeatureStoreSpec:
    """Tests for FeatureStoreSpec."""
    
    def test_feature_store_spec_defaults(self):
        """Test default values of FeatureStoreSpec."""
        spec = FeatureStoreSpec()
        
        assert spec.enabled is False
        assert spec.repo_path == "feature_repo"
        assert spec.project_name == "ml_workflow"
        assert spec.dataset_name == "classification_data"
        assert spec.n_features is None
        assert spec.offline_store_type == "file"
        assert spec.online_store_type is None
        assert spec.feature_view_name == "classification_features"
        assert spec.use_full_feature_names is False
        assert spec.initialize_on_start is True
    
    def test_feature_store_spec_custom_values(self):
        """Test FeatureStoreSpec with custom values."""
        spec = FeatureStoreSpec(
            enabled=True,
            repo_path="custom_repo",
            project_name="custom_project",
            dataset_name="custom_dataset",
            n_features=30,
            offline_store_type="spark",
            online_store_type="redis",
            feature_view_name="custom_features",
            use_full_feature_names=True
        )
        
        assert spec.enabled is True
        assert spec.repo_path == "custom_repo"
        assert spec.project_name == "custom_project"
        assert spec.dataset_name == "custom_dataset"
        assert spec.n_features == 30
        assert spec.offline_store_type == "spark"
        assert spec.online_store_type == "redis"
        assert spec.feature_view_name == "custom_features"
        assert spec.use_full_feature_names is True
    
    def test_feature_store_spec_validation_n_features(self):
        """Test n_features validation."""
        # Test minimum
        with pytest.raises(ValueError):
            FeatureStoreSpec(n_features=0)
        
        # Test maximum
        with pytest.raises(ValueError):
            FeatureStoreSpec(n_features=1001)
        
        # Test valid values
        spec = FeatureStoreSpec(n_features=1)
        assert spec.n_features == 1
        
        spec = FeatureStoreSpec(n_features=1000)
        assert spec.n_features == 1000
    
    def test_feature_store_spec_validation_repo_path(self):
        """Test repo_path validation."""
        with pytest.raises(ValueError):
            FeatureStoreSpec(repo_path="")
        
        with pytest.raises(ValueError):
            FeatureStoreSpec(repo_path="   ")
    
    def test_feature_store_spec_validation_feature_view_name(self):
        """Test feature_view_name validation."""
        with pytest.raises(ValueError):
            FeatureStoreSpec(feature_view_name="")
        
        with pytest.raises(ValueError):
            FeatureStoreSpec(feature_view_name="   ")
    
    def test_feature_store_spec_validation_dataset_name(self):
        """Test dataset_name validation."""
        with pytest.raises(ValueError):
            FeatureStoreSpec(dataset_name="")
        
        with pytest.raises(ValueError):
            FeatureStoreSpec(dataset_name="   ")
    
    def test_feature_store_spec_validation_project_name(self):
        """Test project_name validation."""
        with pytest.raises(ValueError):
            FeatureStoreSpec(project_name="")
        
        with pytest.raises(ValueError):
            FeatureStoreSpec(project_name="   ")
    
    def test_get_feature_references(self):
        """Test get_feature_references method."""
        spec = FeatureStoreSpec(n_features=5, feature_view_name="test_features")
        
        # Test with n_features set
        refs = spec.get_feature_references()
        
        assert len(refs) == 6  # 5 features + 1 target
        assert refs[0] == "test_features:feature_0"
        assert refs[4] == "test_features:feature_4"
        assert refs[5] == "test_features:target"
        
        # Test with n_features as argument
        refs2 = spec.get_feature_references(n_features=3)
        assert len(refs2) == 4  # 3 features + 1 target
        
        # Test with None n_features should raise error
        spec_no_features = FeatureStoreSpec(feature_view_name="test_features")
        with pytest.raises(ValueError, match="n_features must be set"):
            spec_no_features.get_feature_references()
    
    def test_should_initialize(self):
        """Test should_initialize method."""
        # Enabled and initialize_on_start=True
        spec = FeatureStoreSpec(enabled=True, initialize_on_start=True)
        assert spec.should_initialize() is True
        
        # Enabled but initialize_on_start=False
        spec = FeatureStoreSpec(enabled=True, initialize_on_start=False)
        assert spec.should_initialize() is False
        
        # Disabled
        spec = FeatureStoreSpec(enabled=False, initialize_on_start=True)
        assert spec.should_initialize() is False
    
    def test_should_materialize(self):
        """Test should_materialize method."""
        # All conditions met
        spec = FeatureStoreSpec(
            enabled=True,
            materialize_online=True,
            online_store_type="redis"
        )
        assert spec.should_materialize() is True
        
        # No online store
        spec = FeatureStoreSpec(
            enabled=True,
            materialize_online=True,
            online_store_type=None
        )
        assert spec.should_materialize() is False
        
        # Disabled
        spec = FeatureStoreSpec(
            enabled=False,
            materialize_online=True,
            online_store_type="redis"
        )
        assert spec.should_materialize() is False
    
    def test_get_n_features_with_value(self):
        """Test get_n_features when n_features is set."""
        spec = FeatureStoreSpec(n_features=42)
        assert spec.get_n_features() == 42
    
    def test_get_n_features_auto_detect(self, tmp_path, monkeypatch):
        """Test get_n_features auto-detection."""
        import pandas as pd
        from pathlib import Path
        from unittest.mock import Mock, patch
        
        # Create a mock DeltaTable
        mock_df = pd.DataFrame({
            'sample_index': [0, 1, 2],
            'target': [0, 1, 0],
            'ingested_at': pd.date_range('2024-01-01', periods=3),
            'feature_0': [1.0, 2.0, 3.0],
            'feature_1': [4.0, 5.0, 6.0],
            'feature_2': [7.0, 8.0, 9.0],
        })
        
        spec = FeatureStoreSpec(dataset_name="test_dataset")
        
        with patch('specs.feature_store_spec.get_delta_path') as mock_path, \
             patch('specs.feature_store_spec.DeltaTable') as mock_delta:
            mock_path.return_value = Path("/fake/path")
            mock_dt = Mock()
            mock_dt.to_pandas.return_value = mock_df
            mock_delta.return_value = mock_dt
            
            n_features = spec.get_n_features()
            assert n_features == 3  # feature_0, feature_1, feature_2
    
    def test_detect_n_features_error(self):
        """Test detect_n_features raises error when detection fails."""
        spec = FeatureStoreSpec(dataset_name="nonexistent")
        
        with pytest.raises(ValueError, match="Cannot auto-detect n_features"):
            spec.detect_n_features()


class TestFeatureStoreSpecBuilder:
    """Tests for FeatureStoreSpecBuilder."""
    
    def test_builder_defaults(self):
        """Test builder with default values."""
        spec = FeatureStoreSpecBuilder().build()
        
        assert spec.enabled is False
        assert spec.repo_path == "feature_repo"
        assert spec.n_features is None
    
    def test_builder_enable(self):
        """Test builder enable method."""
        spec = FeatureStoreSpecBuilder().enable().build()
        assert spec.enabled is True
        
        spec = FeatureStoreSpecBuilder().enable().disable().build()
        assert spec.enabled is False
    
    def test_builder_set_repo_path(self):
        """Test builder set_repo_path method."""
        spec = FeatureStoreSpecBuilder().set_repo_path("my_repo").build()
        assert spec.repo_path == "my_repo"
    
    def test_builder_set_project_name(self):
        """Test builder set_project_name method."""
        spec = FeatureStoreSpecBuilder().set_project_name("my_project").build()
        assert spec.project_name == "my_project"
    
    def test_builder_set_dataset_name(self):
        """Test builder set_dataset_name method."""
        spec = FeatureStoreSpecBuilder().set_dataset_name("my_dataset").build()
        assert spec.dataset_name == "my_dataset"
    
    def test_builder_set_n_features(self):
        """Test builder set_n_features method."""
        spec = FeatureStoreSpecBuilder().set_n_features(50).build()
        assert spec.n_features == 50
    
    def test_builder_set_offline_store(self):
        """Test builder set_offline_store method."""
        spec = FeatureStoreSpecBuilder().set_offline_store("spark").build()
        assert spec.offline_store_type == "spark"
    
    def test_builder_set_online_store(self):
        """Test builder set_online_store method."""
        spec = FeatureStoreSpecBuilder().set_online_store("redis").build()
        assert spec.online_store_type == "redis"
    
    def test_builder_set_feature_view(self):
        """Test builder set_feature_view method."""
        spec = FeatureStoreSpecBuilder().set_feature_view("my_features").build()
        assert spec.feature_view_name == "my_features"
    
    def test_builder_use_full_names(self):
        """Test builder use_full_names method."""
        spec = FeatureStoreSpecBuilder().use_full_names(True).build()
        assert spec.use_full_feature_names is True
    
    def test_builder_set_timestamp(self):
        """Test builder set_timestamp method."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        spec = FeatureStoreSpecBuilder().set_timestamp(ts).build()
        assert spec.timestamp == ts
    
    def test_builder_set_initialize_on_start(self):
        """Test builder set_initialize_on_start method."""
        spec = FeatureStoreSpecBuilder().set_initialize_on_start(False).build()
        assert spec.initialize_on_start is False
        
        spec = FeatureStoreSpecBuilder().set_initialize_on_start(True).build()
        assert spec.initialize_on_start is True
    
    def test_builder_enable_materialization(self):
        """Test builder enable_materialization method."""
        spec = FeatureStoreSpecBuilder().enable_materialization().build()
        assert spec.materialize_online is True
    
    def test_builder_disable_materialization(self):
        """Test builder disable_materialization method."""
        spec = FeatureStoreSpecBuilder().disable_materialization().build()
        assert spec.materialize_online is False
        
        # Test chaining
        spec = (FeatureStoreSpecBuilder()
            .enable_materialization()
            .disable_materialization()
            .build())
        assert spec.materialize_online is False
    
    def test_builder_set_sample_indices(self):
        """Test builder set_sample_indices method."""
        indices = [0, 1, 2, 3]
        spec = FeatureStoreSpecBuilder().set_sample_indices(indices).build()
        assert spec.sample_indices == indices
    
    def test_builder_chaining(self):
        """Test builder method chaining."""
        spec = (FeatureStoreSpecBuilder()
            .enable()
            .set_repo_path("test_repo")
            .set_project_name("test_project")
            .set_dataset_name("test_dataset")
            .set_n_features(15)
            .set_offline_store("spark")
            .set_online_store("redis")
            .use_full_names(True)
            .enable_materialization()
            .build()
        )
        
        assert spec.enabled is True
        assert spec.repo_path == "test_repo"
        assert spec.project_name == "test_project"
        assert spec.dataset_name == "test_dataset"
        assert spec.n_features == 15
        assert spec.offline_store_type == "spark"
        assert spec.online_store_type == "redis"
        assert spec.use_full_feature_names is True
        assert spec.materialize_online is True

