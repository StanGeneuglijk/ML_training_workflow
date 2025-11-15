"""
Unit tests for feature store data sources.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from feast import FileSource

from feature_store.data_sources import create_file_source
from data.delta_lake import get_delta_path


class TestCreateFileSource:
    """Tests for create_file_source."""
    
    def test_create_file_source_minimal(self):
        """Test creating file source with minimal args."""
        delta_path = get_delta_path("classification_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        assert isinstance(source, FileSource)
        assert source.timestamp_field == "ingested_at"
        assert "classification_data" in source.name
    
    def test_create_file_source_custom_path(self):
        """Test creating file source with custom path."""
        custom_path = "/data/my_dataset.parquet"
        source = create_file_source(
            path=custom_path,
            timestamp_field="event_time"
        )
        
        assert custom_path in source.path
        assert source.timestamp_field == "event_time"
    
    def test_create_file_source_custom_timestamp(self):
        """Test creating file source with custom timestamp field."""
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="updated_at"
        )
        
        assert source.timestamp_field == "updated_at"
    
    def test_create_file_source_custom_source_name(self):
        """Test creating file source with custom source name."""
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
            source_name="my_custom_source"
        )
        
        assert source.name == "my_custom_source"
    
    def test_create_file_source_custom_description(self):
        """Test creating file source with custom description."""
        delta_path = get_delta_path("test_data")
        custom_desc = "My custom data source"
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
            description=custom_desc
        )
        
        assert source.description == custom_desc
    
    def test_create_file_source_with_delta_lake(self):
        """Test creating file source pointing to Delta Lake."""
        from feast.data_format import DeltaFormat
        
        delta_path = get_delta_path("classification_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
            file_format=DeltaFormat()
        )
        
        assert isinstance(source, FileSource)
        assert str(delta_path) in source.path
        assert isinstance(source.file_format, DeltaFormat)

