"""
Data sources module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from feast import FileSource
from feast.data_format import ParquetFormat, DeltaFormat, FileFormat

import utils

logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


def create_file_source(
    path: Union[str, Path],
    timestamp_field: str,
    source_name: Optional[str] = None,
    description: Optional[str] = None,
    file_format: Optional[FileFormat] = None,
    created_timestamp_column: Optional[str] = None,
) -> FileSource:
    """
    Create a generic Feast FileSource from any file path.
    
    *FileSource defines how to retrieve data from a file-based data source.
    
    Args:
        path: Path to data file or directory 
        timestamp_field: Column name for point-in-time lookups
        source_name: Name for the Feast source 
        description: Description of the source 
        file_format: File format 
        created_timestamp_column: Column for creation timestamp 
        
    Returns:
        Feast FileSource configured for the specified path
        
    Examples:
        >>> # Local parquet file
        >>> source = create_file_source(
        ...     path="/data/features.parquet",
        ...     timestamp_field="event_time"
        ... )
    """
    path_str = str(path)
    
    if source_name is None:
        path_obj = Path(path_str)
        source_name = f"{path_obj.stem}_source"
    
    if description is None:
        description = f"File source from {path_str}"
    
    if file_format is None:
        file_format = ParquetFormat()
    
    source = FileSource(
        name=source_name,
        path=path_str,
        timestamp_field=timestamp_field,
        file_format=file_format,
        description=description,
        **({"created_timestamp_column": created_timestamp_column} if created_timestamp_column else {})
    )
    
    logger.info(
        "Created file source '%s' (format=%s, path=%s)",
        source_name, type(file_format).__name__, path_str
    )
    return source


__all__ = [
    "create_file_source",
]
