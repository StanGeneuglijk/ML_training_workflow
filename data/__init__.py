"""
Data package.

Provides data loading from Delta Lake tables using PySpark.
"""
from __future__ import annotations

from data.delta_lake import (
    get_spark_session,
    get_delta_path,
    get_table_info,
    load_data,
    read_delta_table,
    write_to_delta_lake,
)

__all__ = [
    "load_data",
    "read_delta_table",
    "write_to_delta_lake",
    "get_delta_path",
    "get_spark_session",
    "get_table_info",
]
