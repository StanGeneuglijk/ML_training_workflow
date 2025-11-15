"""
Feature definitions module.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Callable, Optional, Union

import pandas as pd
from feast import Entity, FeatureView, Field, on_demand_feature_view
from feast.data_source import DataSource
from feast.types import Float64, Int64, String, Bool, PrimitiveFeastType

import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


def create_entity(
    name: str = "sample",
    join_keys: Optional[list[str]] = None,
    description: Optional[str] = None
) -> Entity:
    """
    Create a generic Entity for feature store.
    
    *Entity defines the primary key used to join features.
    
    Args:
        name: Entity name 
        join_keys: List of join key column names
        description: Entity description
        
    Returns:
        Feast Entity for data identification
        
    Example:
        >>> # Custom entity
        >>> entity = create_entity(
        ...     name="customer",
        ...     join_keys=["customer_id"],
        ...     description="Customer identifier"
        ... )
    """
    if join_keys is None:
        join_keys = ["sample_index"]
    if description is None:
        description = f"{name.capitalize()} identifier"
    
    entity = Entity(
        name=name,
        join_keys=join_keys,
        description=description
    )
    
    logger.info("Created entity: %s (join_keys: %s)", name, join_keys)

    return entity


def create_schema(
    feature_names: list[str],
    feature_types: Optional[list[PrimitiveFeastType]] = None,
    default_type: PrimitiveFeastType = Float64
) -> list[Field]:
    """
    Create a schema from feature names and types.
        
    Args:
        feature_names: List of feature column names
        feature_types: List of Feast types 
        default_type: Default type if feature_types not specified
        
    Returns:
        List of Feast Field objects
        
    Examples:
        >>> schema = create_schema(
        ...     feature_names=["age", "name", "active"],
        ...     feature_types=[Int64, String, Bool]
        ... )
    """
    if feature_types is None:
        feature_types = [default_type] * len(feature_names)
    
    if len(feature_names) != len(feature_types):
        raise ValueError(
            f"Length mismatch: {len(feature_names)} names vs {len(feature_types)} types"
        )
    
    schema = [
        Field(name=name, dtype=dtype)
        for name, dtype in zip(feature_names, feature_types)
    ]
    
    logger.info("Created schema with %d fields", len(schema))

    return schema

def create_feature_view(
    view_name: str,
    source: DataSource,
    schema: list[Field],
    entity: Optional[Entity] = None,
    ttl_days: int = 365,
    description: Optional[str] = None,
    tags: Optional[dict[str, str]] = None,
    online: bool = True,
) -> FeatureView:
    """
    Create a generic FeatureView from any data source.
    
    *FeatureView defines features and how to retrieve them from data source.
    
    Args:
        view_name: Name of the feature view
        source: Feast DataSource 
        schema: List of Feast Field objects defining the schema
        entity: Entity to associate with 
        ttl_days: Time-to-live in days for feature freshness 
        description: Custom description 
        tags: Optional metadata tags for the feature view
        online: Whether to materialize to online store (default: True)
        
    Returns:
        Feast FeatureView configured with specified parameters
        
    Examples:
        >>> # With custom entity
        >>> customer_entity = create_entity("customer", ["customer_id"])
        >>> fv = create_feature_view(
        ...     view_name="customer_features",
        ...     source=source,
        ...     schema=schema,
        ...     entity=customer_entity,
        ...     tags={"team": "data-science", "version": "v2"}
        ... )
    """
    if entity is None:
        entity = create_entity(name="sample", join_keys=["sample_index"])
    
    if description is None:
        description = f"FeatureView '{view_name}' with {len(schema)} fields"
    
    feature_view = FeatureView(
        name=view_name,
        entities=[entity],
        schema=schema,
        source=source,
        ttl=timedelta(days=ttl_days),
        description=description,
        tags=tags or {},
        online=online,
    )
    
    logger.info(
        "Created feature view '%s' (%d fields, ttl=%d days, online=%s)",
        view_name, len(schema), ttl_days, online
    )
    
    return feature_view


__all__ = [
    "create_entity",
    "create_schema",
    "create_feature_view"

]
