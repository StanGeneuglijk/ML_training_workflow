"""
Feature store module
"""
from __future__ import annotations

from feature_store.feast_manager import FeastManager, create_feast_manager
from feature_store.feature_definitions import create_classification_features

__all__ = [
    "FeastManager",
    "create_feast_manager",
    "create_classification_features",
]
