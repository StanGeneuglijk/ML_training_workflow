"""
Delta Lake data generation module.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from deltalake import DeltaTable
from sklearn.datasets import make_classification

import utils
from data.delta_lake import get_delta_path, write_to_delta_lake

logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)

DATASET_NAME = "classification_data"
N_SAMPLES = 1000
N_FEATURES = 20
N_CLASSES = 2
RANDOM_STATE = 42


def create_delta_table(
    dataset_name: str = DATASET_NAME,
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
    random_state: int = RANDOM_STATE,
    mode: str = "overwrite",
    partition_by: Optional[list[str]] = None,
) -> Path:
    """
    Generate data and create Delta Lake table (entry point).
    
    Args:
        dataset_name: Name of the dataset
        n_samples: Number of samples to generate
        n_features: Number of features
        n_classes: Number of classes
        random_state: Random seed
        mode: Write mode for Delta Lake
        partition_by: Optional partitioning columns
        
    Returns:
        Path to created Delta Lake table
        
    """
    logger.info("=" * 80)
    logger.info(
        "Generating classification data: n_samples=%d, n_features=%d, n_classes=%d",
        n_samples, n_features, n_classes
    )
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=random_state,
        flip_y=0.01, 
    )
    
    df = pd.DataFrame(
        X,
        columns=[f"feature_{i}" for i in range(n_features)]
    )
    df["sample_index"] = range(n_samples)
    df["target"] = y.astype("int64")
    df["ingested_at"] = pd.Timestamp.now()
    df["ingested_date"] = pd.Timestamp.now().date()
    
    columns_order = (
        ["sample_index"] +
        [f"feature_{i}" for i in range(n_features)] +
        ["target", "ingested_at", "ingested_date"]
    )
    df = df[columns_order]
    
    
    logger.info("Creating Delta Lake table: %s", dataset_name)
    logger.info("=" * 80)

    delta_path = write_to_delta_lake(
        df=df,
        dataset_name=dataset_name,
        mode=mode,
        partition_by=partition_by,
    )
    
    logger.info("=" * 80)
    logger.info("Delta Lake table created successfully at: %s", delta_path)
    logger.info("=" * 80)
    
    return delta_path


def main() -> None:
    """Entry point for Delta Lake table generation."""
    try:
        create_delta_table(
            dataset_name=DATASET_NAME,
            n_samples=N_SAMPLES,
            n_features=N_FEATURES,
            n_classes=N_CLASSES,
            random_state=RANDOM_STATE,
            mode="overwrite",
        )
    except Exception as e:
        logger.error("Failed to create Delta Lake table: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()

