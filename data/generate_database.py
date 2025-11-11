"""
Database generation module.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

from sklearn.datasets import make_classification

DB_PATH = Path(__file__).resolve().parent / "database" / "sample_data.db"
DATASET_NAME = "classification_data"
N_SAMPLES = 1000
N_FEATURES = 20
N_CLASSES = 2
RANDOM_STATE = 42


def ensure_schema(
    db_path: Path
) -> None:
    """
    Create database schema with datasets and features tables.
    
    Args:
        db_path: Path to SQLite database file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_name TEXT UNIQUE NOT NULL,
                n_samples INTEGER NOT NULL,
                n_features INTEGER NOT NULL,
                n_classes INTEGER NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                sample_index INTEGER NOT NULL,
                feature_0 REAL, feature_1 REAL, feature_2 REAL, feature_3 REAL, feature_4 REAL,
                feature_5 REAL, feature_6 REAL, feature_7 REAL, feature_8 REAL, feature_9 REAL,
                feature_10 REAL, feature_11 REAL, feature_12 REAL, feature_13 REAL, feature_14 REAL,
                feature_15 REAL, feature_16 REAL, feature_17 REAL, feature_18 REAL, feature_19 REAL,
                target INTEGER NOT NULL,
                FOREIGN KEY (dataset_id) REFERENCES datasets(dataset_id)
            )
            """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_dataset_sample
            ON features(dataset_id, sample_index)
            """
        )


def populate_dataset(
    db_path: Path,
    dataset_name: str,
    n_samples: int,
    n_features: int,
    n_classes: int,
    random_state: int,
) -> None:
    """
    Generate synthetic classification data and populate database.
    
    Args:
        db_path: Path to SQLite database file
        dataset_name: Unique identifier for the dataset
        n_samples: Number of samples to generate
        n_features: Number of features per sample (must be 20)
        n_classes: Number of target classes
        random_state: Random seed for reproducibility
        
    Raises:
        ValueError: If n_features does not equal 20
    """
    if n_features != 20:
        raise ValueError("20 features are required to match schema")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=random_state
        )

    rows = [(idx, *features, int(y[idx])) for idx, features in enumerate(X)]

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT dataset_id FROM datasets WHERE dataset_name = ?", (dataset_name,))
        record = cursor.fetchone()
        dataset_id = record[0] if record else None

        if dataset_id:
            cursor.execute(
                """
                UPDATE datasets
                SET n_samples = ?, n_features = ?, n_classes = ?, description = ?, created_at = ?
                WHERE dataset_id = ?
                """,
                (
                    n_samples,
                    n_features,
                    n_classes,
                    f"Classification dataset with {n_samples} samples",
                    datetime.now(),
                    dataset_id,
                ),
            )
            cursor.execute("DELETE FROM features WHERE dataset_id = ?", (dataset_id,))
        else:
            cursor.execute(
                """
                INSERT INTO datasets (dataset_name, n_samples, n_features, n_classes, description, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_name,
                    n_samples,
                    n_features,
                    n_classes,
                    f"Classification dataset with {n_samples} samples",
                    datetime.now(),
                ),
            )
            dataset_id = cursor.lastrowid

        cursor.executemany(
            """
            INSERT INTO features (
                dataset_id, sample_index,
                feature_0, feature_1, feature_2, feature_3, feature_4,
                feature_5, feature_6, feature_7, feature_8, feature_9,
                feature_10, feature_11, feature_12, feature_13, feature_14,
                feature_15, feature_16, feature_17, feature_18, feature_19,
                target
            )
            VALUES (
                ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?
            )
            """,
            [(dataset_id, *row) for row in rows],
        )


def main() -> None:
    """Entry point for database generation."""
    ensure_schema(DB_PATH)
    populate_dataset(
        db_path=DB_PATH,
        dataset_name=DATASET_NAME,
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        random_state=RANDOM_STATE,
    )
    print("Sample database created.")
    print(f"Location: {DB_PATH}")


if __name__ == "__main__":
    main()

