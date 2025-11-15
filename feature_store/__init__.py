"""
Feature store module
"""
from __future__ import annotations

from feature_store.feast_manager import FeastManager, create_feast_manager
from feature_store.feature_definitions import (
    create_entity,
    create_schema,
    create_feature_view,
)

__all__ = [
    "FeastManager",
    "create_feast_manager",
    "create_entity",
    "create_schema",
    "create_feature_view",
]
