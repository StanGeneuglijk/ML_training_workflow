"""
Data package for ML workflow version 1.

Simple data loading from SQLite database.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def get_database_path() -> Path:
    """
    Get path to SQLite database.
    
    Returns:
        Path to sample_data.db
    """
    return Path(__file__).parent / "database" / "sample_data.db"


def load_data(dataset_name: str = "classification_data") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from SQLite database.
    
    Args:
        dataset_name: Name of dataset to load (default: "classification_data")
        
    Returns:
        Tuple of (X, y) where X is feature matrix and y is target vector
        
    Example:
        >>> X, y = load_data()
        >>> print(X.shape, y.shape)
        (1000, 20) (1000,)
    """
    db_path = get_database_path()
    
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run 'python data/generate_database.py' first."
        )
    
    conn = sqlite3.connect(db_path)
    
    # Get dataset_id
    cursor = conn.cursor()
    cursor.execute("SELECT dataset_id FROM datasets WHERE dataset_name = ?", (dataset_name,))
    result = cursor.fetchone()
    
    if result is None:
        conn.close()
        raise ValueError(f"Dataset '{dataset_name}' not found in database")
    
    dataset_id = result[0]
    
    # Load features
    query = f"""
        SELECT 
            {', '.join([f'feature_{i}' for i in range(20)])},
            target
        FROM features
        WHERE dataset_id = ?
        ORDER BY sample_index
    """
    
    df = pd.read_sql_query(query, conn, params=(dataset_id,))
    conn.close()
    
    # Split into X and y
    feature_cols = [f'feature_{i}' for i in range(20)]
    X = df[feature_cols].values
    y = df['target'].values
    
    return X, y


def get_data_path(dataset_name: str = "classification_data") -> Path:
    """
    Get path to data storage (SQLite database).
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Path to database file
    """
    return get_database_path()


def export_to_parquet(dataset_name: str = "classification_data") -> Path:
    """
    Export SQLite data to Parquet format for Feast.
    
    Args:
        dataset_name: Name of dataset to export
        
    Returns:
        Path to exported Parquet file
        
    Example:
        >>> export_to_parquet("classification_data")
    """
    db_path = get_database_path()
    
    if not db_path.exists():
        raise FileNotFoundError(
            f"Database not found at {db_path}. "
            "Run 'python data/generate_database.py' first."
        )
    
    conn = sqlite3.connect(db_path)
    
    # Get dataset_id
    cursor = conn.cursor()
    cursor.execute("SELECT dataset_id FROM datasets WHERE dataset_name = ?", (dataset_name,))
    result = cursor.fetchone()
    
    if result is None:
        conn.close()
        raise ValueError(f"Dataset '{dataset_name}' not found in database")
    
    dataset_id = result[0]
    
    # Load all data
    query = f"""
        SELECT 
            sample_index,
            {', '.join([f'feature_{i}' for i in range(20)])},
            target
        FROM features
        WHERE dataset_id = ?
        ORDER BY sample_index
    """
    
    df = pd.read_sql_query(query, conn, params=(dataset_id,))
    conn.close()
    
    # Add timestamp for Feast
    df['ingested_at'] = pd.Timestamp.now()
    
    # Export to Parquet
    output_path = Path(__file__).parent / f"{dataset_name}.parquet"
    df.to_parquet(output_path, index=False)
    
    return output_path


__all__ = [
    "load_data",
    "get_database_path",
    "get_data_path",
    "export_to_parquet",
]
