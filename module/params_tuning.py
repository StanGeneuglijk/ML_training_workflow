"""
Parameter tuning module./yes!
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from specs.params_tuning_spec import ParamTuningSpec, GridSearchSpec, RandomSearchSpec

import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


def _cv_splitter(
    spec: ParamTuningSpec
) -> Union[KFold, StratifiedKFold]:
    """
    Create a cross-validation splitter.

    Args:
        spec: The specification for the parameter tuning.

    Returns:
        A cross-validation splitter.
    """
    cv_strategy = spec.cv_strategy
    n_splits = spec.n_splits
    random_state = spec.random_state

    if cv_strategy == "kfold":
        cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state) #see sklearn.model_selection.KFold!
    elif cv_strategy == "stratified_kfold":
        cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) #see sklearn.model_selection.StratifiedKFold!
    else:
        raise ValueError(f"invalid cv strategy: {cv_strategy}")

    return cv_splitter


class BaseTuning:
    """
    Base class for parameter tuning.
    """
    def __init__(
        self, 
        spec: ParamTuningSpec
    ) -> None:
        """
        Initialize the base tuning class.

        Args:
            spec: The specification for the parameter tuning.

        Returns:
            None
        """
        if not isinstance(spec, ParamTuningSpec):
            raise TypeError(f"spec must be a ParamTuningSpec, got {type(spec).__name__}")
        self.spec = spec
        self.search_cv: Optional[Union[GridSearchCV, RandomizedSearchCV]] = None
        self.is_fitted: bool = False

    def fit(
        self, 
        pipeline: Pipeline, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> "BaseTuning":
        """
        Fit the parameter tuning.

        Args:
            pipeline: The pipeline to tune.
            X: The training features.
            y: The training labels.

        Returns:
            self
        """
        X_arr, y_arr = utils.validate_training_data(X, y)

        self.search_cv = self._create_search_cv(pipeline)
        self.search_cv.fit(X_arr, y_arr)
        self.is_fitted = True

        logger.info(f"Parameter tuning {self.spec.tuning_name} fitted successfully")

        return self

    def _create_search_cv(
        self, 
        pipeline: Pipeline
    ) -> Union[GridSearchCV, RandomizedSearchCV]:  
        """
        Create a search cross-validator.

        Args:
            pipeline: The pipeline to tune.

        Returns:
            A search cross-validator.
        """
        raise NotImplementedError

    def get_best_params(
        self
    ) -> Dict[str, Any]:
        """
        Get the best parameters.

        Args:
            None

        Returns:
            The best parameters.
        """
        if not self.is_fitted or self.search_cv is None:
            raise ValueError("tuning must be fitted first")

        return self.search_cv.best_params_

    def get_best_score(
        self
    ) -> float:
        """
        Get the best score.

        Args:
            None

        Returns:
            The best score.
        """
        if not self.is_fitted or self.search_cv is None:
            raise ValueError("tuning must be fitted first")

        return float(self.search_cv.best_score_)

    def get_best_estimator(
        self
    ) -> Pipeline:
        """
        Get the best estimator.

        Args:
            None

        Returns:
            The best estimator.
        """
        if not self.is_fitted or self.search_cv is None:
            raise ValueError("tuning must be fitted first")
        
        return self.search_cv.best_estimator_  

    def get_cv_results(
        self
    ) -> Dict[str, Any]:
        """
        Get the cross-validation results.

        Args:
            None

        Returns:
            The cross-validation results.
        """
        if not self.is_fitted or self.search_cv is None:
            raise ValueError("tuning must be fitted first")
            
        return dict(self.search_cv.cv_results_)

    def get_results_summary(
        self
    ) -> Dict[str, Any]:
        """
        Get the results summary.

        Args:
            None

        Returns:
            The results summary.
        """
        return {
            "tuning_name": self.spec.tuning_name,
            "tuning_type": self.spec.get_tuning_type(),
            "best_score": self.get_best_score(),
            "best_params": self.get_best_params(),
            "n_splits": self.spec.n_splits,
            "cv_strategy": self.spec.cv_strategy,
        }


class GridSearch(BaseTuning):
    """
    Grid search parameter tuning.
    """

    def __init__(
        self, 
        spec: GridSearchSpec
    ) -> None:
        """
        Initialize the grid search parameter tuning.

        Args:
            spec: The specification for the grid search parameter tuning.

        Returns:
            None
        """
        if not isinstance(spec, GridSearchSpec):
            raise TypeError(f"spec must be a GridSearchSpec, got {type(spec).__name__}")
        super().__init__(spec)
        self.grid_spec = spec

    def _create_search_cv(
        self, 
        pipeline: Pipeline
    ) -> GridSearchCV:
        """
        Create a grid search cross-validator.

        Args:
            pipeline: The pipeline to tune.

        Returns:
            A grid search cross-validator.
        """
        if not isinstance(self.grid_spec.param_grid, dict) or not self.grid_spec.param_grid:
            raise ValueError("param_grid must be a non-empty dict")
        
        logger.info(f"Creating grid search cross-validator with {len(self.grid_spec.param_grid)} parameter combinations")

        return GridSearchCV(
            estimator=pipeline,
            param_grid=self.grid_spec.param_grid,
            scoring=self.grid_spec.scoring,
            refit=self.grid_spec.refit_score,
            cv=_cv_splitter(self.grid_spec),
            n_jobs=self.grid_spec.n_jobs,
            return_train_score=self.grid_spec.return_train_score,
            verbose=self.grid_spec.verbose,
        )


class RandomSearch(BaseTuning):
    """
    Random search parameter tuning.
    """

    def __init__(
        self, 
        spec: RandomSearchSpec
    ) -> None:
        """
        Initialize the random search parameter tuning.

        Args:
            spec: The specification for the random search parameter tuning.

        Returns:
            None
        """
        if not isinstance(spec, RandomSearchSpec):
            raise TypeError(f"spec must be a RandomSearchSpec, got {type(spec).__name__}")
        super().__init__(spec)
        self.random_spec = spec

    def _create_search_cv(
        self, 
        pipeline: Pipeline
    ) -> RandomizedSearchCV:
        """
        Create a random search cross-validator.

        Args:
            pipeline: The pipeline to tune.

        Returns:
            A random search cross-validator.
        """
        if not isinstance(self.random_spec.param_distributions, dict) or not self.random_spec.param_distributions:
            raise ValueError(f"param_distributions must be a non-empty dict, got {type(self.random_spec.param_distributions).__name__}")
        
        logger.info(f"Creating random search cross-validator with {self.random_spec.n_iter} iterations")

        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=self.random_spec.param_distributions,
            n_iter=self.random_spec.n_iter,
            scoring=self.random_spec.scoring,
            refit=self.random_spec.refit_score,
            cv=_cv_splitter(self.random_spec),
            random_state=self.random_spec.random_state,
            n_jobs=self.random_spec.n_jobs,
            return_train_score=self.random_spec.return_train_score,
            verbose=self.random_spec.verbose,
        )


def create_tuning(
    spec: ParamTuningSpec
    ) -> BaseTuning:
    """
    Create a tuning.

    Args:
        spec: The specification for the tuning.

    Returns:
        A tuning.
    """
    if not isinstance(spec, ParamTuningSpec):
        raise TypeError(f"spec must be a ParamTuningSpec, got {type(spec).__name__}")

    tuning_type = spec.get_tuning_type()
    if tuning_type == "grid_search":
        tuning = GridSearch(spec)  
    elif tuning_type == "random_search":
        tuning = RandomSearch(spec)  
    else:
        raise ValueError(f"unsupported tuning type: {tuning_type}")

    logger.info(f"Created {type(tuning).__name__} tuning")
    
    return tuning


__all__ = [
    "BaseTuning",
    "GridSearch",
    "RandomSearch",
    "create_tuning"
]


