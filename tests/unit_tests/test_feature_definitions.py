"""
Unit tests for feature definitions.
"""
from __future__ import annotations

import pytest
from feast import Entity, FeatureView, Field
from feast.types import Float64, Int64

from feature_store.feature_definitions import (
    create_entity,
    create_feature_view,
    create_schema,
)
from feature_store.data_sources import create_file_source
from data.delta_lake import get_delta_path


class TestCreateEntity:
    """Tests for create_entity."""
    
    def test_create_entity_defaults(self):
        """Test creating entity with defaults."""
        entity = create_entity()
        
        assert isinstance(entity, Entity)
        assert entity.name == "sample"
        assert entity.join_key == "sample_index"  # Feast uses join_key (singular)
        assert "Sample identifier" in entity.description
    
    def test_create_entity_custom(self):
        """Test creating custom entity."""
        entity = create_entity(
            name="customer",
            join_keys=["customer_id"],
            description="Customer identifier"
        )
        
        assert entity.name == "customer"
        assert entity.join_key == "customer_id"  # Feast uses join_key (singular)
        assert entity.description == "Customer identifier"


class TestCreateFeatureView:
    """Tests for create_feature_view."""
    
    def test_create_feature_view_minimal(self):
        """Test creating feature view with minimal args."""
        schema = [Field(name=f"feature_{i}", dtype=Float64) for i in range(5)]
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        fv = create_feature_view(
            view_name="test_features",
            source=source,
            schema=schema
        )
        
        assert isinstance(fv, FeatureView)
        assert fv.name == "test_features"
        assert len(fv.schema) == 5
        assert len(fv.entities) == 1
        assert fv.entities[0] == "sample"
    
    def test_create_feature_view_custom_entity(self):
        """Test creating feature view with custom entity."""
        entity = create_entity(name="customer", join_keys=["customer_id"])
        schema = [Field(name="purchase_count", dtype=Int64)]
        delta_path = get_delta_path("customer_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        fv = create_feature_view(
            view_name="customer_features",
            source=source,
            schema=schema,
            entity=entity
        )
        
        assert fv.entities[0] == "customer"


class TestCreateSchema:
    """Tests for create_schema and create_feature_view integration."""
    
    def test_create_schema_with_feature_view(self):
        """Test creating feature view with schema builder."""
        delta_path = get_delta_path("classification_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        # Build schema using create_schema
        feature_names = [f"feature_{i}" for i in range(20)]
        schema = create_schema(feature_names, default_type=Float64)
        schema.append(Field(name="target", dtype=Int64))
        
        fv = create_feature_view(
            view_name="classification_features",
            source=source,
            schema=schema
        )
        
        assert isinstance(fv, FeatureView)
        assert fv.name == "classification_features"
        assert len(fv.schema) == 21  # 20 features + 1 target
        assert len(fv.entities) == 1
        assert fv.entities[0] == "sample"
    
    def test_create_schema_custom_n_features(self):
        """Test creating schema with custom number of features."""
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        feature_names = [f"feature_{i}" for i in range(10)]
        schema = create_schema(feature_names)
        schema.append(Field(name="target", dtype=Int64))
        
        fv = create_feature_view(
            view_name="test_features",
            source=source,
            schema=schema
        )
        
        assert len(fv.schema) == 11  # 10 features + 1 target
        
        # Check feature names
        field_names = [field.name for field in fv.schema]
        for i in range(10):
            assert f"feature_{i}" in field_names
        assert "target" in field_names
    
    def test_create_schema_types(self):
        """Test schema with mixed types."""
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        feature_names = ["age", "name", "active"]
        feature_types = [Int64, Float64, Float64]
        schema = create_schema(feature_names, feature_types)
        
        fv = create_feature_view(
            view_name="test_features",
            source=source,
            schema=schema
        )
        
        # Check data types by field name (Feast may reorder fields)
        field_dict = {field.name: field.dtype for field in fv.schema}
        assert field_dict["age"] == Int64
        assert field_dict["name"] == Float64
        assert field_dict["active"] == Float64
    
    def test_create_schema_regression(self):
        """Test creating regression feature schema."""
        delta_path = get_delta_path("regression_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        feature_names = [f"feature_{i}" for i in range(15)]
        schema = create_schema(feature_names)
        schema.append(Field(name="price", dtype=Float64))
        
        fv = create_feature_view(
            view_name="regression_features",
            source=source,
            schema=schema
        )
        
        assert len(fv.schema) == 16  # 15 features + 1 target
        field_names = [field.name for field in fv.schema]
        assert "price" in field_names
    
    def test_create_schema_no_target(self):
        """Test creating schema without targets."""
        delta_path = get_delta_path("inference_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        feature_names = [f"feature_{i}" for i in range(10)]
        schema = create_schema(feature_names)
        
        fv = create_feature_view(
            view_name="inference_features",
            source=source,
            schema=schema
        )
        
        assert len(fv.schema) == 10  # Only features, no target
    
    def test_create_schema_multi_target(self):
        """Test creating schema with multiple targets."""
        delta_path = get_delta_path("multi_task_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        feature_names = [f"feature_{i}" for i in range(5)]
        schema = create_schema(feature_names)
        schema.append(Field(name="target_a", dtype=Int64))
        schema.append(Field(name="target_b", dtype=Float64))
        
        fv = create_feature_view(
            view_name="multi_task_features",
            source=source,
            schema=schema
        )
        
        assert len(fv.schema) == 7  # 5 features + 2 targets
        field_names = [field.name for field in fv.schema]
        assert "target_a" in field_names
        assert "target_b" in field_names
    
    def test_create_schema_custom_prefix(self):
        """Test creating schema with custom prefix."""
        delta_path = get_delta_path("custom_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at"
        )
        
        feature_names = [f"col_{i}" for i in range(5)]
        schema = create_schema(feature_names)
        
        fv = create_feature_view(
            view_name="custom_features",
            source=source,
            schema=schema
        )
        
        field_names = [field.name for field in fv.schema]
        for i in range(5):
            assert f"col_{i}" in field_names

