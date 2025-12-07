"""
Utils
"""

from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
from typing import Any, Optional, Tuple, Union


def validate_array_input(
    data: Any,
    name: str = "data",
    min_dim: int = 1,
    check_finite: bool = True
) -> np.ndarray:
    """
    Validate and convert input to numpy array.
    
    Args:
        data: Input data to validate 
        name: Name of the variable for debugging
        min_dim: Minimum number of dimensions
        check_finite: Whether to check for finite values
        
    Returns:
        Validated numpy array
    """
    if data is None:
        raise ValueError(f"{name} cannot be None")
    
    if isinstance(data, np.ndarray):
        array = data
    elif isinstance(data, (pd.DataFrame, pd.Series)):
        array = data.values
    else:
        try:
            array = np.asarray(data)
        except Exception as e:
            raise TypeError(f"{name} must be convertible to numpy array: {e}") from e
    
    if array.ndim < min_dim:
        raise ValueError(
            f"{name} must have at least {min_dim} dimension(s), got {array.ndim}"
        )
    
    if check_finite:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} contains non-finite values (NaN or Inf)")
    
    return array


def validate_training_data(
    X: Any,
    y: Any
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate training data inputs.
    
    Args:
        X: Feature matrix
        y: Target vector
        
    Returns:
        Tuple of validated (X, y) arrays
    """
    X_array = validate_array_input(X, "X")
    y_array = validate_array_input(y, "y")

    if X_array.ndim != 2:
        raise ValueError(f"X must be 2D array, got {X_array.ndim}D")
    
    if y_array.ndim != 1:
        raise ValueError(f"y must be 1D array, got {y_array.ndim}D")
    
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError(
            f"X and y must have same number of samples: "
            f"{X_array.shape[0]} vs {y_array.shape[0]}"
        )
    
    if X_array.shape[0] == 0:
        raise ValueError("Cannot train on empty dataset")
    
    return X_array, y_array


def validate_prediction_data(
    X: Any,
    expected_n_features: Optional[int] = None
) -> np.ndarray:
    """
    Validate prediction data input.
    
    Args:
        X: Feature matrix for prediction
        expected_n_features: Expected number of features (if known)
        
    Returns:
        Validated feature array
    """
    X_array = validate_array_input(X, "X")
    
    if X_array.ndim != 2:
        raise ValueError(f"X must be 2D array, got {X_array.ndim}D")
    
    if X_array.shape[0] == 0:
        raise ValueError("Cannot predict on empty dataset")
    
    if expected_n_features is not None:
        if X_array.shape[1] != expected_n_features:
            raise ValueError(
                f"X must have {expected_n_features} features, "
                f"got {X_array.shape[1]}"
            )
    
    return X_array


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    logger_name: str = __name__
) -> logging.Logger:
    """
    Configure logging for the application.
    
    Args:
        level: Logging level 
        log_file: Optional path to log file
        logger_name: Name for the logger
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(logger_name)
    
    if logger.handlers:  
        return logger
    
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    handlers = [logging.StreamHandler()]
    
    if log_file:
        ensure_directory(os.path.dirname(log_file) if os.path.dirname(log_file) else ".")
        handlers.append(logging.FileHandler(log_file))
    
    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    logger.setLevel(level)
    
    return logger


def ensure_directory(directory_path: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory_path: Path to the directory
    """
    if directory_path and not os.path.exists(directory_path):
        os.makedirs(directory_path, exist_ok=True)


__all__ = [
    "validate_array_input",
    "validate_training_data",
    "validate_prediction_data",
    "setup_logging",
    "ensure_directory",
]

