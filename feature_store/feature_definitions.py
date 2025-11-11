"""
Feature definitions module.
"""
from __future__ import annotations

import logging
from datetime import timedelta

from feast import Entity, FeatureView, Field
from feast.types import Float64, Int64

from feature_store.data_sources import create_classification_data_source
import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


def create_sample_entity() -> Entity:
    """
    Create entity for feature store.
    
    Entity defines the primary key used to join features.
    
    Returns:
        Feast Entity for sample identification
        
    Example:
        >>> entity = create_sample_entity()
        >>> print(entity.join_key)  # 'sample_index'
    """
    entity = Entity(
        name="sample",
        join_keys=["sample_index"],
        description="Sample identifier for classification data"
    )
    
    logger.info("Created entity: sample (join_key: sample_index)")
    return entity


def create_classification_features(
    n_features: int = 20,
    dataset_name: str = "classification_data"
) -> FeatureView:
    """
    Create feature view for classification data.
    
    FeatureView defines features and how to retrieve them from data source.
    
    Args:
        n_features: Number of features (default: 20)
        dataset_name: Dataset name in Delta Lake (default: "classification_data")
        
    Returns:
        Feast FeatureView with features and target
        
    Example:
        >>> fv = create_classification_features(n_features=20)
        >>> print(fv.name)  # 'classification_features'
    """
    sample_entity = create_sample_entity()
    data_source = create_classification_data_source(dataset_name)
    
    schema = [
        Field(name=f"feature_{i}", dtype=Float64) 
        for i in range(n_features)
    ]
    schema.append(Field(name="target", dtype=Int64))
    
    feature_view = FeatureView(
        name="classification_features",
        entities=[sample_entity],
        schema=schema,
        source=data_source,
        ttl=timedelta(days=365),
        description=f"Classification features with {n_features} numeric features and target"
    )
    
    logger.info("Created feature view: classification_features (%s features)", n_features)
    return feature_view
