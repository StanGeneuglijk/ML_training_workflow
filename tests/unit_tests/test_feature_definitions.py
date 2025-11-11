"""
Unit tests for feature definitions.
"""
from __future__ import annotations

import pytest
from feast import Entity, FeatureView
from feast.types import Float64, Int64

from feature_store.feature_definitions import (
    create_sample_entity,
    create_classification_features
)


class TestCreateSampleEntity:
    """Tests for create_sample_entity."""
    
    def test_create_sample_entity(self):
        """Test creating sample entity."""
        entity = create_sample_entity()
        
        assert isinstance(entity, Entity)
        assert entity.name == "sample"
        assert entity.join_key == "sample_index"
        assert "Sample identifier" in entity.description


class TestCreateClassificationFeatures:
    """Tests for create_classification_features."""
    
    def test_create_classification_features_default(self):
        """Test creating classification features with defaults."""
        fv = create_classification_features()
        
        assert isinstance(fv, FeatureView)
        assert fv.name == "classification_features"
        assert len(fv.schema) == 21  # 20 features + 1 target
        assert len(fv.entities) == 1
        assert fv.entities[0] == "sample"
    
    def test_create_classification_features_custom_n_features(self):
        """Test creating classification features with custom n_features."""
        fv = create_classification_features(n_features=10)
        
        assert len(fv.schema) == 11  # 10 features + 1 target
        
        # Check feature names
        field_names = [field.name for field in fv.schema]
        for i in range(10):
            assert f"feature_{i}" in field_names
        assert "target" in field_names
    
    def test_create_classification_features_schema_types(self):
        """Test schema field types."""
        fv = create_classification_features(n_features=3)
        
        # Check data types
        for field in fv.schema:
            if field.name == "target":
                assert field.dtype == Int64
            else:
                assert field.dtype == Float64
    
    def test_create_classification_features_custom_dataset(self):
        """Test creating classification features with custom dataset."""
        fv = create_classification_features(n_features=10, dataset_name="custom_data")
        
        assert fv.name == "classification_features"
        assert len(fv.schema) == 11  # 10 features + 1 target

