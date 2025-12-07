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


@pytest.mark.unit
class TestCreateEntity:
    """Tests for create_entity."""

    def test_create_entity_defaults(self):
        """Test creating entity with defaults."""
        entity = create_entity()

        assert isinstance(entity, Entity), (
            f"Should return Entity instance, got {type(entity)}"
        )
        assert entity.name == "sample", (
            f"Default entity name should be 'sample', got '{entity.name}'"
        )
        assert entity.join_key == "sample_index", (
            f"Default join_key should be 'sample_index', got '{entity.join_key}'"
        )
        assert "Sample identifier" in entity.description, (
            f"Description should contain 'Sample identifier', got '{entity.description}'"
        )

    def test_create_entity_custom(self):
        """Test creating custom entity."""
        entity = create_entity(
            name="customer",
            join_keys=["customer_id"],
            description="Customer identifier",
        )

        assert entity.name == "customer", (
            f"Entity name should be 'customer', got '{entity.name}'"
        )
        assert entity.join_key == "customer_id", (
            f"join_key should be 'customer_id', got '{entity.join_key}'"
        )
        assert entity.description == "Customer identifier", (
            f"Description should be 'Customer identifier', got '{entity.description}'"
        )

    def test_create_entity_auto_description(self):
        """Test entity gets auto-generated description when not provided."""
        entity = create_entity(name="product", join_keys=["product_id"])

        assert "Product identifier" in entity.description, (
            f"Description should contain 'Product identifier', got '{entity.description}'"
        )


@pytest.mark.unit
class TestCreateSchema:
    """Tests for create_schema."""

    def test_create_schema_with_default_type(self):
        """Test creating schema with default type."""
        feature_names = [f"feature_{i}" for i in range(5)]
        schema = create_schema(feature_names, default_type=Float64)

        assert len(schema) == 5, (
            f"Schema should have 5 fields, got {len(schema)}"
        )
        assert all(isinstance(field, Field) for field in schema), (
            "All schema items should be Field instances"
        )
        assert all(field.dtype == Float64 for field in schema), (
            "All fields should have Float64 dtype"
        )

    def test_create_schema_length_mismatch_raises(self):
        """Test creating schema with mismatched lengths raises ValueError."""
        feature_names = ["feature_0", "feature_1"]
        feature_types = [Float64]  

        with pytest.raises(ValueError) as exc_info:
            create_schema(feature_names, feature_types=feature_types)

        assert "Length mismatch" in str(exc_info.value) or "length" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention length mismatch, got: {exc_info.value}"
        )

    @pytest.mark.parametrize(
        "n_features",
        [
            pytest.param(5, id="small"),
            pytest.param(10, id="medium"),
            pytest.param(20, id="large"),
        ],
    )
    def test_create_schema_variable_features(self, n_features):
        """Test creating schema with variable number of features."""
        feature_names = [f"feature_{i}" for i in range(n_features)]
        schema = create_schema(feature_names)

        assert len(schema) == n_features, (
            f"Schema should have {n_features} fields, got {len(schema)}"
        )
        assert all(
            schema[i].name == f"feature_{i}" for i in range(n_features)
        ), (
            "Field names should match feature names"
        )

    def test_create_schema_mixed_types(self):
        """Test creating schema with mixed types."""
        feature_names = ["age", "name", "active"]
        feature_types = [Int64, Float64, Float64]
        schema = create_schema(feature_names, feature_types=feature_types)

        assert len(schema) == 3, (
            f"Schema should have 3 fields, got {len(schema)}"
        )
        field_dict = {field.name: field.dtype for field in schema}
        assert field_dict["age"] == Int64, (
            f"'age' field should be Int64, got {field_dict.get('age')}"
        )
        assert field_dict["name"] == Float64, (
            f"'name' field should be Float64, got {field_dict.get('name')}"
        )
        assert field_dict["active"] == Float64, (
            f"'active' field should be Float64, got {field_dict.get('active')}"
        )


@pytest.mark.unit
class TestCreateFeatureView:
    """Tests for create_feature_view."""

    def test_create_feature_view_minimal(self):
        """Test creating feature view with minimal args."""
        schema = [Field(name=f"feature_{i}", dtype=Float64) for i in range(5)]
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
        )

        fv = create_feature_view(
            view_name="test_features",
            source=source,
            schema=schema,
        )

        assert isinstance(fv, FeatureView), (
            f"Should return FeatureView instance, got {type(fv)}"
        )
        assert fv.name == "test_features", (
            f"Feature view name should be 'test_features', got '{fv.name}'"
        )
        assert len(fv.schema) == 5, (
            f"Feature view should have 5 schema fields, got {len(fv.schema)}"
        )
        assert len(fv.entities) == 1, (
            f"Feature view should have 1 entity, got {len(fv.entities)}"
        )
        assert fv.entities[0] == "sample", (
            f"Default entity should be 'sample', got '{fv.entities[0]}'"
        )

    def test_create_feature_view_custom_entity(self):
        """Test creating feature view with custom entity."""
        entity = create_entity(name="customer", join_keys=["customer_id"])
        schema = [Field(name="purchase_count", dtype=Int64)]
        delta_path = get_delta_path("customer_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
        )

        fv = create_feature_view(
            view_name="customer_features",
            source=source,
            schema=schema,
            entity=entity,
        )

        assert fv.entities[0] == "customer", (
            f"Entity should be 'customer', got '{fv.entities[0]}'"
        )

    def test_create_feature_view_with_target(self):
        """Test creating feature view with target included."""
        delta_path = get_delta_path("classification_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
        )

        feature_names = [f"feature_{i}" for i in range(20)]
        schema = create_schema(feature_names, default_type=Float64)
        schema.append(Field(name="target", dtype=Int64))

        fv = create_feature_view(
            view_name="classification_features",
            source=source,
            schema=schema,
        )

        assert isinstance(fv, FeatureView), (
            f"Should return FeatureView instance, got {type(fv)}"
        )
        assert fv.name == "classification_features", (
            f"Feature view name should be 'classification_features', got '{fv.name}'"
        )
        assert len(fv.schema) == 21, (
            f"Feature view should have 21 schema fields (20 features + 1 target), "
            f"got {len(fv.schema)}"
        )
        assert len(fv.entities) == 1, (
            f"Feature view should have 1 entity, got {len(fv.entities)}"
        )
        assert fv.entities[0] == "sample", (
            f"Default entity should be 'sample', got '{fv.entities[0]}'"
        )

    def test_create_feature_view_custom_ttl(self):
        """Test creating feature view with custom TTL."""
        schema = [Field(name="feature_0", dtype=Float64)]
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
        )

        fv = create_feature_view(
            view_name="test_features",
            source=source,
            schema=schema,
            ttl_days=30,
        )

        assert fv.ttl.days == 30, (
            f"TTL should be 30 days, got {fv.ttl.days} days"
        )

    def test_create_feature_view_with_tags(self):
        """Test creating feature view with tags."""
        schema = [Field(name="feature_0", dtype=Float64)]
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
        )
        tags = {"team": "data-science", "version": "v1"}

        fv = create_feature_view(
            view_name="test_features",
            source=source,
            schema=schema,
            tags=tags,
        )

        assert fv.tags == tags, (
            f"Tags should be {tags}, got {fv.tags}"
        )

    def test_create_feature_view_online_false(self):
        """Test creating feature view with online=False."""
        schema = [Field(name="feature_0", dtype=Float64)]
        delta_path = get_delta_path("test_data")
        source = create_file_source(
            path=delta_path,
            timestamp_field="ingested_at",
        )

        fv = create_feature_view(
            view_name="test_features",
            source=source,
            schema=schema,
            online=False,
        )

        assert fv.online is False, (
            "Feature view should have online=False"
        )
