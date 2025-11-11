"""
Data sources module
"""
from __future__ import annotations

import logging
from pathlib import Path

from feast import FileSource
from feast.data_format import ParquetFormat

import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


def get_data_path(dataset_name: str = "classification_data") -> str:
    """
    Get filesystem path to Parquet data for Feast.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Absolute path to Parquet file
    """
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent
    # Parquet file exported from SQLite database
    parquet_path = project_root / "data" / f"{dataset_name}.parquet"
    
    return str(parquet_path)


def create_classification_data_source(
    dataset_name: str = "classification_data",
    timestamp_field: str = "ingested_at"
) -> FileSource:
    """
    Create Feast data source pointing to Parquet file.
    
    Reads Parquet file exported from SQLite database.
    
    Args:
        dataset_name: Name of dataset
        timestamp_field: Timestamp column for point-in-time correctness
        
    Returns:
        Feast FileSource configured for Parquet file
        
    Example:
        >>> source = create_classification_data_source("classification_data")
        >>> print(source.path)
    """
    data_path = get_data_path(dataset_name)
    
    source = FileSource(
        name=f"{dataset_name}_source",
        path=data_path,
        timestamp_field=timestamp_field,
        file_format=ParquetFormat(),
        description=f"Parquet source for {dataset_name}"
    )
    
    logger.info("Created data source for Parquet file: %s", data_path)
    return source
