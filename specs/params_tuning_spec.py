"""
Core parameter tuning specifications for ML workflow v1.

Provides minimal yet robust specs to drive grid and random search.
"""
from __future__ import annotations

import abc
import logging
from typing import Optional, Literal, Dict, Any, List, Union
from pydantic import BaseModel, Field, field_validator, ConfigDict

logger = logging.getLogger(__name__)


class ParamTuningSpec(BaseModel, abc.ABC):
    """Abstract base class for parameter tuning specifications with Pydantic validation."""
    
    model_config = ConfigDict(extra='forbid', validate_assignment=True)
    
    tuning_name: str = Field(..., min_length=1, description="Name of the tuning specification")
    enabled: bool = Field(default=True, description="Whether tuning is enabled")
    random_state: Optional[int] = Field(default=None, description="Random state for reproducibility")
    description: Optional[str] = Field(default=None, description="Description of the tuning")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator('tuning_name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate tuning name is not empty or whitespace."""
        if not v.strip():
            raise ValueError("tuning_name cannot be empty or whitespace")
        return v.strip()

    @abc.abstractmethod
    def get_tuning_type(self) -> str:
        """Return the tuning type identifier."""
        raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        """Convert specification to dictionary."""
        return {
            "tuning_name": self.tuning_name,
            "tuning_type": self.get_tuning_type(),
            "enabled": self.enabled,
            "random_state": self.random_state,
            "description": self.description,
            "metadata": self.metadata.copy(),
        }


class GridSearchSpec(ParamTuningSpec):
    """Grid search parameter tuning specification with Pydantic validation."""
    
    param_grid: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Grid of parameters to search over"
    )
    cv_strategy: Literal["stratified_kfold", "kfold"] = Field(
        default="stratified_kfold",
        description="Cross-validation strategy"
    )
    n_splits: int = Field(
        default=5,
        ge=2,
        description="Number of cross-validation splits"
    )
    scoring: Union[str, List[str], Dict[str, str]] = Field(
        default="accuracy",
        description="Scoring metric(s) to use"
    )
    refit_score: Optional[str] = Field(
        default="accuracy",
        description="Score to use for refitting the best estimator"
    )
    n_jobs: int = Field(
        default=-1,
        description="Number of jobs to run in parallel (-1 means all processors)"
    )
    return_train_score: bool = Field(
        default=True,
        description="Whether to return training scores"
    )
    verbose: int = Field(
        default=0,
        ge=0,
        description="Verbosity level"
    )

    def get_tuning_type(self) -> str:
        """Return 'grid_search' as the tuning type."""
        return "grid_search"


class RandomSearchSpec(ParamTuningSpec):
    """Random search parameter tuning specification with Pydantic validation."""
    
    param_distributions: Dict[str, List[Any]] = Field(
        default_factory=dict,
        description="Distributions of parameters to sample from"
    )
    n_iter: int = Field(
        default=10,
        ge=1,
        description="Number of parameter settings that are sampled"
    )
    cv_strategy: Literal["stratified_kfold", "kfold"] = Field(
        default="stratified_kfold",
        description="Cross-validation strategy"
    )
    n_splits: int = Field(
        default=5,
        ge=2,
        description="Number of cross-validation splits"
    )
    scoring: Union[str, List[str], Dict[str, str]] = Field(
        default="accuracy",
        description="Scoring metric(s) to use"
    )
    refit_score: Optional[str] = Field(
        default="accuracy",
        description="Score to use for refitting the best estimator"
    )
    n_jobs: int = Field(
        default=-1,
        description="Number of jobs to run in parallel (-1 means all processors)"
    )
    return_train_score: bool = Field(
        default=True,
        description="Whether to return training scores"
    )
    verbose: int = Field(
        default=0,
        ge=0,
        description="Verbosity level"
    )

    def get_tuning_type(self) -> str:
        """Return 'random_search' as the tuning type."""
        return "random_search"


class ParamTuningSpecBuilder:
    """Builder for creating parameter tuning specifications."""
    
    def __init__(self) -> None:
        """Initialize builder."""
        self._specs: List[ParamTuningSpec] = []

    def add_grid_search(
        self,
        name: str,
        param_grid: Dict[str, List[Any]],
        cv_strategy: Literal["stratified_kfold", "kfold"] = "stratified_kfold",
        n_splits: int = 5,
        scoring: Union[str, List[str], Dict[str, str]] = "accuracy",
        refit_score: Optional[str] = "accuracy",
        **kwargs: Any,
    ) -> "ParamTuningSpecBuilder":
        """
        Add a grid search specification.
        
        Args:
            name: Name of the tuning specification
            param_grid: Grid of parameters to search
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV splits
            scoring: Scoring metric(s)
            refit_score: Score to use for refitting
            **kwargs: Additional parameters
            
        Returns:
            Self for chaining
        """
        spec = GridSearchSpec(
            tuning_name=name,
            param_grid=param_grid,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            scoring=scoring,
            refit_score=refit_score,
            random_state=kwargs.get("random_state"),
            n_jobs=kwargs.get("n_jobs", -1),
            return_train_score=kwargs.get("return_train_score", True),
            verbose=kwargs.get("verbose", 0),
        )
        self._specs.append(spec)
        return self

    def add_random_search(
        self,
        name: str,
        param_distributions: Dict[str, List[Any]],
        n_iter: int = 10,
        cv_strategy: Literal["stratified_kfold", "kfold"] = "stratified_kfold",
        n_splits: int = 5,
        scoring: Union[str, List[str], Dict[str, str]] = "accuracy",
        refit_score: Optional[str] = "accuracy",
        **kwargs: Any,
    ) -> "ParamTuningSpecBuilder":
        """
        Add a random search specification.
        
        Args:
            name: Name of the tuning specification
            param_distributions: Distributions to sample from
            n_iter: Number of parameter settings to sample
            cv_strategy: Cross-validation strategy
            n_splits: Number of CV splits
            scoring: Scoring metric(s)
            refit_score: Score to use for refitting
            **kwargs: Additional parameters
            
        Returns:
            Self for chaining
        """
        spec = RandomSearchSpec(
            tuning_name=name,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv_strategy=cv_strategy,
            n_splits=n_splits,
            scoring=scoring,
            refit_score=refit_score,
            random_state=kwargs.get("random_state"),
            n_jobs=kwargs.get("n_jobs", -1),
            return_train_score=kwargs.get("return_train_score", True),
            verbose=kwargs.get("verbose", 0),
        )
        self._specs.append(spec)
        return self

    def build(self) -> List[ParamTuningSpec]:
        """
        Build and return list of tuning specifications.
        
        Returns:
            List of tuning specs
        """
        return self._specs.copy()


__all__ = [
    "ParamTuningSpec",
    "GridSearchSpec",
    "RandomSearchSpec",
    "ParamTuningSpecBuilder",
]
