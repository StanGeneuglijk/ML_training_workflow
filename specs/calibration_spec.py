"""
Calibration specifications
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union, Literal, List
from pydantic import BaseModel, Field, field_validator, ConfigDict
import logging

import utils
logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


class CalibrationSpec(BaseModel):
    """
    Base calibration specification with Pydantic validation.
    """
    model_config = ConfigDict(
        extra='forbid', 
        validate_assignment=True
    )
    
    calibration_name: str = Field(
        ..., 
        min_length=1, 
        description="Name of the calibration method"
    )
    enabled: bool = Field(
        default=True,
        description="Whether calibration is enabled"
    )
    random_state: Optional[int] = Field(
        default=None, 
        description="Random state for reproducibility"
    )
    description: Optional[str] = Field(
        default=None, 
        description="Description of the calibration"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Additional metadata"
    )

    @field_validator('calibration_name')
    @classmethod
    def validate_name(
        cls, 
        v: str
    ) -> str:
        """
        Validate calibration name is not empty or whitespace.
        
        Args:
            v: The value to validate
        """
        if not v.strip():
            raise ValueError("calibration_name cannot be empty or whitespace")
        return v.strip()

    def get_calibration_type(
        self
    ) -> str:
        """
        Return the calibration type identifier.
        
        Args:
            None

        Returns:
            The calibration type identifier
        """
        return "classifier_calibration"

    def to_dict(
        self
    ) -> Dict[str, Any]:
        """
        Convert specification to dictionary.
        
        Args:
            None

        Returns:
            Dictionary representation of the specification
        """
        return {
            "calibration_name": self.calibration_name,
            "calibration_type": self.get_calibration_type(),
            "enabled": self.enabled,
            "random_state": self.random_state,
            "description": self.description,
            "metadata": self.metadata.copy(),
        }


class ClassifierCalibrationSpec(CalibrationSpec):
    """
    Classifier calibration specification with Pydantic validation.
    """
    method: Literal["sigmoid", "isotonic"] = Field(
        default="sigmoid",
        description="Calibration method: 'sigmoid' or 'isotonic'"
    )
    cv_strategy: Union[Literal["prefit", "kfold", "stratified_kfold"], int] = Field(
        default="prefit",
        description="Cross-validation strategy name or 'prefit' for pre-fitted estimator"
    )
    n_splits: int = Field(
        default=5,
        ge=2,
        description="Number of cross-validation splits when using CV strategies"
    )
    ensemble: bool = Field(
        default=True,
        description="Whether to use ensemble of calibrators"
    )

    @field_validator('cv_strategy')
    @classmethod
    def validate_cv_strategy(
        cls, 
        v: Union[str, int]
    ) -> Union[str, int]:
        """
        Validate cv_strategy string values or positive integers.
        
        Args:
            v: The value to validate
        """
        if isinstance(v, str):
            allowed = {"prefit", "kfold", "stratified_kfold"}
            if v not in allowed:
                raise ValueError(f"cv_strategy must be one of {allowed}")
        elif isinstance(v, int):
            if v < 2:
                raise ValueError("cv_strategy must be at least 2 when specified as integer")
        else:
            raise TypeError("cv_strategy must be a string or integer")
        return v


class CalibrationSpecBuilder:
    """Builder for creating calibration specifications."""
    
    def __init__(
        self
    ) -> None:
        """
        Initialize builder.
        
        Args:
            None
        
        Returns:
            None
        """
        self._specs: List[CalibrationSpec] = []

    def add_platt_scaling(
        self,
        name: str,
        cv_strategy: Union[str, int] = "prefit",
        ensemble: bool = True,
        **kwargs: Any,
    ) -> "CalibrationSpecBuilder":
        """
        Add sigmoid scaling calibration specification.
        
        Args:
            name: Name of the calibration
            cv_strategy: Cross-validation strategy
            ensemble: Whether to use ensemble
            **kwargs: Additional parameters
            
        Returns:
            Self for chaining
        """
        spec = ClassifierCalibrationSpec(
            calibration_name=name,
            method="sigmoid",
            cv_strategy=cv_strategy,
            ensemble=ensemble,
            **kwargs,
        )
        self._specs.append(spec)
        return self

    def add_isotonic_regression(
        self,
        name: str,
        cv_strategy: Union[str, int] = "prefit",
        ensemble: bool = True,
        **kwargs: Any,
    ) -> "CalibrationSpecBuilder":
        """
        Add isotonic regression calibration specification.
        
        Args:
            name: Name of the calibration
            cv_strategy: Cross-validation strategy
            ensemble: Whether to use ensemble
            **kwargs: Additional parameters
            
        Returns:
            Self for chaining
        """
        spec = ClassifierCalibrationSpec(
            calibration_name=name,
            method="isotonic",
            cv_strategy=cv_strategy,
            ensemble=ensemble,
            **kwargs,
        )
        self._specs.append(spec)
        return self

    def build(self) -> List[CalibrationSpec]:
        """
        Build and return list of calibration specifications.
        
        Returns:
            List of calibration specs
        """
        return self._specs.copy()


__all__ = [
    "CalibrationSpec",
    "ClassifierCalibrationSpec",
    "CalibrationSpecBuilder",
]
