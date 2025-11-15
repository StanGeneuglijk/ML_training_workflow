"""
Delta Lake module.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from deltalake import DeltaTable, write_deltalake
from pyspark.sql import SparkSession, DataFrame

DELTA_BASE_PATH = Path(__file__).resolve().parent / "delta_lake" #CAUTIONS: this might change due architecture change!

DATASET_NAME = "classification_data"

logger = logging.getLogger(__name__)

def get_spark_session() -> SparkSession:
    """
    Get or create a Spark session configured with Delta Lake support.
    
    Returns:
        SparkSession compatible with Delta Lake support
    """
    spark = (
        SparkSession.builder
        .appName("DeltaLakeOperations")
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
        .getOrCreate()
    )
    
    spark.sparkContext.setLogLevel("WARN")

    return spark

def get_delta_path(
    dataset_name: str = DATASET_NAME
) -> Path:
    """
    Get path to Delta Lake table directory.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Path to Delta Lake table directory
    """
    path = DELTA_BASE_PATH / dataset_name

    return path

def read_delta_table(
    dataset_name: str = DATASET_NAME,
    version: Optional[int] = None,
) -> DataFrame:
    """
    Read data from Delta Lake table.
    
    Args:
        dataset_name: Name of the dataset
        version: Optional version number of the table 
        
    Returns:
        DataFrame with the data
    """
    delta_path = get_delta_path(dataset_name)
    
    if not delta_path.exists():
        raise FileNotFoundError(
            f"Delta Lake table not found at {delta_path}. "
        )
    
    spark = get_spark_session()
    
    if version is not None:
        df = spark.read.format("delta").option("versionAsOf", version).load(str(delta_path))
    else:
        df = spark.read.format("delta").load(str(delta_path))
    
    return df

def write_to_delta_lake(
    df: pd.DataFrame,
    dataset_name: str = DATASET_NAME,
    mode: str = "overwrite",
    partition_by: Optional[list[str]] = None,
) -> Path:
    """
    Write DataFrame to Delta Lake table.
    
    Args:
        df: DataFrame to write
        dataset_name:  Name of dataset
        mode: Write mode - "overwrite" or "append".
        partition_by: Optional list of columns to partition
        
    Returns:
        Path to Delta Lake table
    """
    delta_path = get_delta_path(dataset_name)
    
    delta_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Writing to Delta Lake: %s (mode=%s)", delta_path, mode)
    
    write_deltalake(
        table_or_uri=str(delta_path),
        data=df,
        mode=mode,
        partition_by=partition_by,
    )
    
    dt = DeltaTable(str(delta_path))

    version = dt.version()
    
    logger.info(
        "Successfully wrote to Delta Lake. Version: %d, Rows: %d",
        version, len(df)
    )
    
    return delta_path

def get_table_info(dataset_name: str = DATASET_NAME) -> dict:
    """
    Get information about a Delta Lake table.
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        Dictionary with table metadata
    """
    delta_path = get_delta_path(dataset_name)
    
    if not delta_path.exists():
        raise FileNotFoundError(f"Delta Lake table not found at {delta_path}")
    
    spark = get_spark_session()
    
    df = spark.read.format("delta").load(str(delta_path))
    
    history_df = spark.sql(f"DESCRIBE HISTORY delta.`{delta_path}`")
    
    latest_version = history_df.select("version").first()[0]
    
    info = {
        "path": str(delta_path),
        "version": int(latest_version),
        "num_rows": df.count(),
        "num_columns": len(df.columns),
        "columns": df.columns,
        "schema": df.schema.simpleString(),
    }
    
    return info

def load_data(
    dataset_name: str = DATASET_NAME,
    version: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from Delta Lake as numpy arrays.
    
    Args:
        dataset_name: Name of dataset
        version: Optional version number of the table 
        
    Returns:
        Tuple of (X, y)
    """
    spark_df = read_delta_table(dataset_name, version)
    
    pandas_df = spark_df.toPandas()
    
    exclude_cols = {
        'sample_index', 'target', 'ingested_at', 'ingested_date'
        }
    feature_cols = [col for col in pandas_df.columns if col not in exclude_cols]
    feature_cols = sorted(feature_cols) 
    
    X = pandas_df[feature_cols].values

    y = pandas_df['target'].values
    
    return X, y

__all__ = [
    "get_spark_session",
    "get_delta_path",
    "read_delta_table",
    "write_to_delta_lake",
    "load_data",
    "get_table_info",
]

