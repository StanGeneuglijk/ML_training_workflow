"""
Unit tests for feature store data sources.
"""
from __future__ import annotations

import pytest
import os
from feast import FileSource

from feature_store.data_sources import (
    get_data_path,
    create_classification_data_source
)


class TestGetDataPath:
    """Tests for get_data_path."""
    
    def test_get_data_path_default(self):
        """Test getting data path with default dataset name."""
        path = get_data_path()
        
        assert "classification_data.parquet" in path
        assert os.path.isabs(path)
    
    def test_get_data_path_custom_dataset(self):
        """Test getting data path with custom dataset name."""
        path = get_data_path(dataset_name="custom_dataset")
        
        assert "custom_dataset.parquet" in path
        assert os.path.isabs(path)


class TestCreateClassificationDataSource:
    """Tests for create_classification_data_source."""
    
    def test_create_classification_data_source_default(self):
        """Test creating classification data source with defaults."""
        source = create_classification_data_source()
        
        assert isinstance(source, FileSource)
        assert source.name == "classification_data_source"
        assert source.timestamp_field == "ingested_at"
        assert "Parquet" in source.description
    
    def test_create_classification_data_source_custom_dataset(self):
        """Test creating data source with custom dataset name."""
        source = create_classification_data_source(dataset_name="test_data")
        
        assert source.name == "test_data_source"
        assert "test_data" in source.path
    
    def test_create_classification_data_source_custom_timestamp(self):
        """Test creating data source with custom timestamp field."""
        source = create_classification_data_source(timestamp_field="updated_at")
        
        assert source.timestamp_field == "updated_at"

