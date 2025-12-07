"""
Unit tests for feature store specification.
"""
from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
from datetime import datetime

from specs import FeatureStoreSpec, FeatureStoreSpecBuilder


@pytest.mark.unit
class TestFeatureStoreSpec:
    """Tests for FeatureStoreSpec."""

    def test_feature_store_spec_defaults(self):
        """Test default values of FeatureStoreSpec."""
        spec = FeatureStoreSpec()

        assert spec.enabled is False, (
            "Default enabled should be False"
        )
        assert spec.repo_path == "feature_repo", (
            f"Default repo_path should be 'feature_repo', got '{spec.repo_path}'"
        )
        assert spec.project_name == "ml_workflow", (
            f"Default project_name should be 'ml_workflow', got '{spec.project_name}'"
        )
        assert spec.dataset_name == "classification_data", (
            f"Default dataset_name should be 'classification_data', got '{spec.dataset_name}'"
        )
        assert spec.n_features is None, (
            "Default n_features should be None"
        )
        assert spec.offline_store_type == "file", (
            f"Default offline_store_type should be 'file', got '{spec.offline_store_type}'"
        )
        assert spec.online_store_type is None, (
            "Default online_store_type should be None"
        )
        assert spec.feature_view_name == "classification_features", (
            f"Default feature_view_name should be 'classification_features', "
            f"got '{spec.feature_view_name}'"
        )
        assert spec.use_full_feature_names is False, (
            "Default use_full_feature_names should be False"
        )
        assert spec.initialize_on_start is True, (
            "Default initialize_on_start should be True"
        )

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
            use_full_feature_names=True,
        )

        assert spec.enabled is True, (
            "enabled should be True"
        )
        assert spec.repo_path == "custom_repo", (
            f"repo_path should be 'custom_repo', got '{spec.repo_path}'"
        )
        assert spec.project_name == "custom_project", (
            f"project_name should be 'custom_project', got '{spec.project_name}'"
        )
        assert spec.dataset_name == "custom_dataset", (
            f"dataset_name should be 'custom_dataset', got '{spec.dataset_name}'"
        )
        assert spec.n_features == 30, (
            f"n_features should be 30, got {spec.n_features}"
        )
        assert spec.offline_store_type == "spark", (
            f"offline_store_type should be 'spark', got '{spec.offline_store_type}'"
        )
        assert spec.online_store_type == "redis", (
            f"online_store_type should be 'redis', got '{spec.online_store_type}'"
        )
        assert spec.feature_view_name == "custom_features", (
            f"feature_view_name should be 'custom_features', "
            f"got '{spec.feature_view_name}'"
        )
        assert spec.use_full_feature_names is True, (
            "use_full_feature_names should be True"
        )

    @pytest.mark.parametrize(
        "n_features,should_raise",
        [
            pytest.param(0, True, id="zero_features"),
            pytest.param(1, False, id="min_valid"),
            pytest.param(1000, False, id="max_valid"),
            pytest.param(1001, True, id="too_many_features"),
        ],
    )
    def test_feature_store_spec_validation_n_features(self, n_features, should_raise):
        """Test n_features validation."""
        if should_raise:
            with pytest.raises(ValueError) as exc_info:
                FeatureStoreSpec(n_features=n_features)

            assert "n_features" in str(exc_info.value).lower() or "features" in str(
                exc_info.value
            ).lower(), (
                f"Error should mention n_features or features, got: {exc_info.value}"
            )
        else:
            spec = FeatureStoreSpec(n_features=n_features)
            assert spec.n_features == n_features, (
                f"n_features should be {n_features}, got {spec.n_features}"
            )

    @pytest.mark.parametrize(
        "invalid_repo_path",
        [
            pytest.param("", id="empty_string"),
            pytest.param("   ", id="whitespace_only"),
        ],
    )
    def test_feature_store_spec_validation_repo_path(self, invalid_repo_path):
        """Test repo_path validation."""
        with pytest.raises(ValueError) as exc_info:
            FeatureStoreSpec(repo_path=invalid_repo_path)

        assert "repo_path" in str(exc_info.value).lower() or "empty" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention repo_path or empty, got: {exc_info.value}"
        )

    @pytest.mark.parametrize(
        "invalid_feature_view_name",
        [
            pytest.param("", id="empty_string"),
            pytest.param("   ", id="whitespace_only"),
        ],
    )
    def test_feature_store_spec_validation_feature_view_name(
        self, invalid_feature_view_name
    ):
        """Test feature_view_name validation."""
        with pytest.raises(ValueError) as exc_info:
            FeatureStoreSpec(feature_view_name=invalid_feature_view_name)

        assert "feature_view_name" in str(exc_info.value).lower() or "empty" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention feature_view_name or empty, got: {exc_info.value}"
        )

    @pytest.mark.parametrize(
        "invalid_dataset_name",
        [
            pytest.param("", id="empty_string"),
            pytest.param("   ", id="whitespace_only"),
        ],
    )
    def test_feature_store_spec_validation_dataset_name(self, invalid_dataset_name):
        """Test dataset_name validation."""
        with pytest.raises(ValueError) as exc_info:
            FeatureStoreSpec(dataset_name=invalid_dataset_name)

        assert "dataset_name" in str(exc_info.value).lower() or "empty" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention dataset_name or empty, got: {exc_info.value}"
        )

    @pytest.mark.parametrize(
        "invalid_project_name",
        [
            pytest.param("", id="empty_string"),
            pytest.param("   ", id="whitespace_only"),
        ],
    )
    def test_feature_store_spec_validation_project_name(self, invalid_project_name):
        """Test project_name validation."""
        with pytest.raises(ValueError) as exc_info:
            FeatureStoreSpec(project_name=invalid_project_name)

        assert "project_name" in str(exc_info.value).lower() or "empty" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention project_name or empty, got: {exc_info.value}"
        )

    def test_get_feature_references(self):
        """Test get_feature_references method."""
        spec = FeatureStoreSpec(
            n_features=5, feature_view_name="test_features"
        )

        refs = spec.get_feature_references()

        assert len(refs) == 6, (
            f"Should have 6 references (5 features + 1 target), got {len(refs)}"
        )
        assert refs[0] == "test_features:feature_0", (
            f"First reference should be 'test_features:feature_0', got '{refs[0]}'"
        )
        assert refs[4] == "test_features:feature_4", (
            f"Fifth reference should be 'test_features:feature_4', got '{refs[4]}'"
        )
        assert refs[5] == "test_features:target", (
            f"Last reference should be 'test_features:target', got '{refs[5]}'"
        )

        refs2 = spec.get_feature_references(n_features=3)
        assert len(refs2) == 4, (
            f"Should have 4 references (3 features + 1 target), got {len(refs2)}"
        )

        spec_no_features = FeatureStoreSpec(feature_view_name="test_features")
        with pytest.raises(ValueError) as exc_info:
            spec_no_features.get_feature_references()

        assert "n_features" in str(exc_info.value).lower(), (
            f"Error should mention 'n_features', got: {exc_info.value}"
        )

    @pytest.mark.parametrize(
        "enabled,initialize_on_start,expected",
        [
            pytest.param(True, True, True, id="enabled_and_initialize"),
            pytest.param(True, False, False, id="enabled_but_no_initialize"),
            pytest.param(False, True, False, id="disabled"),
        ],
    )
    def test_should_initialize(self, enabled, initialize_on_start, expected):
        """Test should_initialize method."""
        spec = FeatureStoreSpec(
            enabled=enabled, initialize_on_start=initialize_on_start
        )

        assert spec.should_initialize() == expected, (
            f"should_initialize() should return {expected} for "
            f"enabled={enabled}, initialize_on_start={initialize_on_start}, "
            f"got {spec.should_initialize()}"
        )

    @pytest.mark.parametrize(
        "enabled,materialize_online,online_store_type,expected",
        [
            pytest.param(True, True, "redis", True, id="all_conditions_met"),
            pytest.param(True, True, None, False, id="no_online_store"),
            pytest.param(False, True, "redis", False, id="disabled"),
            pytest.param(True, False, "redis", False, id="materialization_disabled"),
        ],
    )
    def test_should_materialize(
        self, enabled, materialize_online, online_store_type, expected
    ):
        """Test should_materialize method."""
        spec = FeatureStoreSpec(
            enabled=enabled,
            materialize_online=materialize_online,
            online_store_type=online_store_type,
        )

        assert spec.should_materialize() == expected, (
            f"should_materialize() should return {expected} for "
            f"enabled={enabled}, materialize_online={materialize_online}, "
            f"online_store_type={online_store_type}, got {spec.should_materialize()}"
        )

    def test_get_n_features_with_value(self):
        """Test get_n_features when n_features is set."""
        spec = FeatureStoreSpec(n_features=42)

        assert spec.get_n_features() == 42, (
            f"get_n_features() should return 42, got {spec.get_n_features()}"
        )

    def test_get_n_features_auto_detect(self):
        """Test get_n_features auto-detection."""
        import pandas as pd
        from pathlib import Path

        mock_df = pd.DataFrame(
            {
                "sample_index": [0, 1, 2],
                "target": [0, 1, 0],
                "ingested_at": pd.date_range("2024-01-01", periods=3),
                "feature_0": [1.0, 2.0, 3.0],
                "feature_1": [4.0, 5.0, 6.0],
                "feature_2": [7.0, 8.0, 9.0],
            }
        )

        spec = FeatureStoreSpec(dataset_name="test_dataset")

        with patch("specs.feature_store_spec.get_delta_path") as mock_path, patch(
            "specs.feature_store_spec.DeltaTable"
        ) as mock_delta:
            mock_path.return_value = Path("/fake/path")
            mock_dt = Mock()
            mock_dt.to_pandas.return_value = mock_df
            mock_delta.return_value = mock_dt

            n_features = spec.get_n_features()

            assert n_features == 3, (
                f"Should detect 3 features (feature_0, feature_1, feature_2), "
                f"got {n_features}"
            )

    def test_detect_n_features_error(self):
        """Test detect_n_features raises error when detection fails."""
        spec = FeatureStoreSpec(dataset_name="nonexistent")

        with pytest.raises(ValueError) as exc_info:
            spec.detect_n_features()

        assert "Cannot auto-detect n_features" in str(exc_info.value) or "auto-detect" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention 'Cannot auto-detect n_features', got: {exc_info.value}"
        )


@pytest.mark.unit
class TestFeatureStoreSpecBuilder:
    """Tests for FeatureStoreSpecBuilder."""

    def test_builder_defaults(self):
        """Test builder with default values."""
        spec = FeatureStoreSpecBuilder().build()

        assert spec.enabled is False, (
            "Default enabled should be False"
        )
        assert spec.repo_path == "feature_repo", (
            f"Default repo_path should be 'feature_repo', got '{spec.repo_path}'"
        )
        assert spec.n_features is None, (
            "Default n_features should be None"
        )

    def test_builder_enable(self):
        """Test builder enable and disable methods."""
        spec = FeatureStoreSpecBuilder().enable().build()

        assert spec.enabled is True, (
            "enable() should set enabled to True"
        )

        spec = FeatureStoreSpecBuilder().enable().disable().build()

        assert spec.enabled is False, (
            "disable() should set enabled to False"
        )

    def test_builder_set_repo_path(self):
        """Test builder set_repo_path method."""
        spec = FeatureStoreSpecBuilder().set_repo_path("my_repo").build()

        assert spec.repo_path == "my_repo", (
            f"repo_path should be 'my_repo', got '{spec.repo_path}'"
        )

    def test_builder_set_project_name(self):
        """Test builder set_project_name method."""
        spec = FeatureStoreSpecBuilder().set_project_name("my_project").build()

        assert spec.project_name == "my_project", (
            f"project_name should be 'my_project', got '{spec.project_name}'"
        )

    def test_builder_set_dataset_name(self):
        """Test builder set_dataset_name method."""
        spec = FeatureStoreSpecBuilder().set_dataset_name("my_dataset").build()

        assert spec.dataset_name == "my_dataset", (
            f"dataset_name should be 'my_dataset', got '{spec.dataset_name}'"
        )

    def test_builder_set_n_features(self):
        """Test builder set_n_features method."""
        spec = FeatureStoreSpecBuilder().set_n_features(50).build()

        assert spec.n_features == 50, (
            f"n_features should be 50, got {spec.n_features}"
        )

    def test_builder_set_offline_store(self):
        """Test builder set_offline_store method."""
        spec = FeatureStoreSpecBuilder().set_offline_store("spark").build()

        assert spec.offline_store_type == "spark", (
            f"offline_store_type should be 'spark', got '{spec.offline_store_type}'"
        )

    def test_builder_set_online_store(self):
        """Test builder set_online_store method."""
        spec = FeatureStoreSpecBuilder().set_online_store("redis").build()

        assert spec.online_store_type == "redis", (
            f"online_store_type should be 'redis', got '{spec.online_store_type}'"
        )

    def test_builder_set_feature_view(self):
        """Test builder set_feature_view method."""
        spec = FeatureStoreSpecBuilder().set_feature_view("my_features").build()

        assert spec.feature_view_name == "my_features", (
            f"feature_view_name should be 'my_features', got '{spec.feature_view_name}'"
        )

    def test_builder_use_full_names(self):
        """Test builder use_full_names method."""
        spec = FeatureStoreSpecBuilder().use_full_names(True).build()

        assert spec.use_full_feature_names is True, (
            "use_full_feature_names should be True"
        )

    def test_builder_set_timestamp(self):
        """Test builder set_timestamp method."""
        ts = datetime(2024, 1, 1, 12, 0, 0)
        spec = FeatureStoreSpecBuilder().set_timestamp(ts).build()

        assert spec.timestamp == ts, (
            f"timestamp should be {ts}, got {spec.timestamp}"
        )

    def test_builder_set_initialize_on_start(self):
        """Test builder set_initialize_on_start method."""
        spec = FeatureStoreSpecBuilder().set_initialize_on_start(False).build()

        assert spec.initialize_on_start is False, (
            "initialize_on_start should be False"
        )

        spec = FeatureStoreSpecBuilder().set_initialize_on_start(True).build()

        assert spec.initialize_on_start is True, (
            "initialize_on_start should be True"
        )

    def test_builder_enable_materialization(self):
        """Test builder enable_materialization method."""
        spec = FeatureStoreSpecBuilder().enable_materialization().build()

        assert spec.materialize_online is True, (
            "materialize_online should be True"
        )

    def test_builder_disable_materialization(self):
        """Test builder disable_materialization method."""
        spec = FeatureStoreSpecBuilder().disable_materialization().build()

        assert spec.materialize_online is False, (
            "materialize_online should be False"
        )

    def test_builder_set_sample_indices(self):
        """Test builder set_sample_indices method."""
        indices = [0, 1, 2, 3]
        spec = FeatureStoreSpecBuilder().set_sample_indices(indices).build()

        assert spec.sample_indices == indices, (
            f"sample_indices should be {indices}, got {spec.sample_indices}"
        )

    def test_builder_chaining(self):
        """Test builder method chaining."""
        spec = (
            FeatureStoreSpecBuilder()
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

        assert spec.enabled is True, (
            "enabled should be True"
        )
        assert spec.repo_path == "test_repo", (
            f"repo_path should be 'test_repo', got '{spec.repo_path}'"
        )
        assert spec.project_name == "test_project", (
            f"project_name should be 'test_project', got '{spec.project_name}'"
        )
        assert spec.dataset_name == "test_dataset", (
            f"dataset_name should be 'test_dataset', got '{spec.dataset_name}'"
        )
        assert spec.n_features == 15, (
            f"n_features should be 15, got {spec.n_features}"
        )
        assert spec.offline_store_type == "spark", (
            f"offline_store_type should be 'spark', got '{spec.offline_store_type}'"
        )
        assert spec.online_store_type == "redis", (
            f"online_store_type should be 'redis', got '{spec.online_store_type}'"
        )
        assert spec.use_full_feature_names is True, (
            "use_full_feature_names should be True"
        )
        assert spec.materialize_online is True, (
            "materialize_online should be True"
        )
