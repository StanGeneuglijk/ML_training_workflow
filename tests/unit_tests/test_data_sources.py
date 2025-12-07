"""
Unit tests for feature store data sources.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from feast import FileSource
from feast.data_format import DeltaFormat, ParquetFormat

from feature_store.data_sources import create_file_source
from data.delta_lake import get_delta_path


@pytest.mark.unit
class TestCreateFileSource:
    """Tests for create_file_source."""

    def test_create_file_source_minimal(self):
        """Test creating file source with minimal args."""
        delta_path = get_delta_path("classification_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
        )

        assert isinstance(source, FileSource), (
            f"Should return FileSource instance, got {type(source)}"
        )
        assert source.timestamp_field == "ingested_at", (
            f"timestamp_field should be 'ingested_at', got '{source.timestamp_field}'"
        )
        assert "classification_data" in source.name, (
            f"Source name should contain 'classification_data', got '{source.name}'"
        )

    def test_create_file_source_custom_path(self):
        """Test creating file source with custom path."""
        custom_path = "/data/my_dataset.parquet"
        source = create_file_source(
            path=custom_path,
            timestamp_field="event_time",
        )

        assert custom_path in source.path, (
            f"Source path should contain '{custom_path}', got '{source.path}'"
        )
        assert source.timestamp_field == "event_time", (
            f"timestamp_field should be 'event_time', got '{source.timestamp_field}'"
        )

    @pytest.mark.parametrize(
        "timestamp_field",
        [
            pytest.param("ingested_at", id="default_timestamp"),
            pytest.param("updated_at", id="custom_timestamp"),
            pytest.param("event_time", id="event_timestamp"),
        ],
    )
    def test_create_file_source_custom_timestamp(self, timestamp_field):
        """Test creating file source with custom timestamp field."""
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field=timestamp_field,
        )

        assert source.timestamp_field == timestamp_field, (
            f"timestamp_field should be '{timestamp_field}', got '{source.timestamp_field}'"
        )

    def test_create_file_source_custom_source_name(self):
        """Test creating file source with custom source name."""
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
            source_name="my_custom_source",
        )

        assert source.name == "my_custom_source", (
            f"Source name should be 'my_custom_source', got '{source.name}'"
        )

    def test_create_file_source_custom_description(self):
        """Test creating file source with custom description."""
        delta_path = get_delta_path("test_data")
        custom_desc = "My custom data source"
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
            description=custom_desc,
        )

        assert source.description == custom_desc, (
            f"Source description should be '{custom_desc}', got '{source.description}'"
        )

    def test_create_file_source_with_delta_lake(self):
        """Test creating file source pointing to Delta Lake."""
        delta_path = get_delta_path("classification_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
            file_format=DeltaFormat(),
        )

        assert isinstance(source, FileSource), (
            f"Should return FileSource instance, got {type(source)}"
        )
        assert str(delta_path) in source.path, (
            f"Source path should contain delta path, got '{source.path}'"
        )
        assert isinstance(source.file_format, DeltaFormat), (
            f"File format should be DeltaFormat, got {type(source.file_format)}"
        )

    def test_create_file_source_default_format(self):
        """Test creating file source uses ParquetFormat by default."""
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
        )

        assert isinstance(source.file_format, ParquetFormat), (
            f"Default file format should be ParquetFormat, got {type(source.file_format)}"
        )

    def test_create_file_source_path_as_pathlib(self):
        """Test creating file source with Path object."""
        delta_path = Path(get_delta_path("test_data"))
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
        )

        assert isinstance(source, FileSource), (
            f"Should return FileSource instance, got {type(source)}"
        )
        assert str(delta_path) in source.path, (
            f"Source path should contain path string, got '{source.path}'"
        )
