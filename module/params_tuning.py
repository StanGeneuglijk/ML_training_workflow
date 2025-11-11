"""
Core parameter tuning utilities for ML workflow v1.

Minimal implementations of grid and random search with sklearn.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union
import logging

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from specs.params_tuning_spec import (
    ParamTuningSpec,
    GridSearchSpec,
    RandomSearchSpec,
)
import utils


logger = logging.getLogger(__name__)


def _cv_splitter(spec: ParamTuningSpec):
    if getattr(spec, "cv_strategy", "stratified_kfold") == "kfold":
        return KFold(n_splits=getattr(spec, "n_splits", 5), shuffle=True, random_state=spec.random_state)
    return StratifiedKFold(n_splits=getattr(spec, "n_splits", 5), shuffle=True, random_state=spec.random_state)


class BaseTuning:
    def __init__(self, spec: ParamTuningSpec) -> None:
        if not isinstance(spec, ParamTuningSpec):
            raise TypeError("spec must be a ParamTuningSpec")
        self.spec = spec
        self.search_cv: Optional[Union[GridSearchCV, RandomizedSearchCV]] = None
        self.is_fitted: bool = False

    def fit(self, pipeline: Pipeline, X: Any, y: Any) -> "BaseTuning":
        # Use shared utilities for validation, consistent with other modules
        X_arr, y_arr = utils.validate_training_data(X, y)
        self.search_cv = self._create_search_cv(pipeline)
        self.search_cv.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def _create_search_cv(self, pipeline: Pipeline) -> Union[GridSearchCV, RandomizedSearchCV]:  # noqa: D401
        raise NotImplementedError

    def get_best_params(self) -> Dict[str, Any]:
        if not self.is_fitted or self.search_cv is None:
            raise ValueError("tuning must be fitted first")
        return self.search_cv.best_params_

    def get_best_score(self) -> float:
        if not self.is_fitted or self.search_cv is None:
            raise ValueError("tuning must be fitted first")
        return float(self.search_cv.best_score_)

    def get_best_estimator(self) -> Pipeline:
        if not self.is_fitted or self.search_cv is None:
            raise ValueError("tuning must be fitted first")
        return self.search_cv.best_estimator_  # type: ignore[return-value]

    def get_cv_results(self) -> Dict[str, Any]:
        if not self.is_fitted or self.search_cv is None:
            raise ValueError("tuning must be fitted first")
        return dict(self.search_cv.cv_results_)

    def get_results_summary(self) -> Dict[str, Any]:
        return {
            "tuning_name": self.spec.tuning_name,
            "tuning_type": self.spec.get_tuning_type(),
            "best_score": self.get_best_score(),
            "best_params": self.get_best_params(),
            "n_splits": getattr(self.spec, "n_splits", None),
            "cv_strategy": getattr(self.spec, "cv_strategy", None),
        }


class GridSearch(BaseTuning):
    def __init__(self, spec: GridSearchSpec) -> None:
        if not isinstance(spec, GridSearchSpec):
            raise TypeError("spec must be a GridSearchSpec")
        super().__init__(spec)
        self.grid_spec = spec

    def _create_search_cv(self, pipeline: Pipeline) -> GridSearchCV:
        if not isinstance(self.grid_spec.param_grid, dict) or not self.grid_spec.param_grid:
            raise ValueError("param_grid must be a non-empty dict")
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
    def __init__(self, spec: RandomSearchSpec) -> None:
        if not isinstance(spec, RandomSearchSpec):
            raise TypeError("spec must be a RandomSearchSpec")
        super().__init__(spec)
        self.random_spec = spec

    def _create_search_cv(self, pipeline: Pipeline) -> RandomizedSearchCV:
        if not isinstance(self.random_spec.param_distributions, dict) or not self.random_spec.param_distributions:
            raise ValueError("param_distributions must be a non-empty dict")
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


def create_tuning(spec: ParamTuningSpec) -> BaseTuning:
    t = spec.get_tuning_type()
    if t == "grid_search":
        return GridSearch(spec)  # type: ignore[arg-type]
    if t == "random_search":
        return RandomSearch(spec)  # type: ignore[arg-type]
    raise ValueError(f"unsupported tuning type: {t}")


def run_grid_search(
    pipeline: Pipeline,
    X: Any,
    y: Any,
    param_grid: Dict[str, List[Any]],
    cv_strategy: str = "stratified_kfold",
    n_splits: int = 5,
    scoring: Union[str, List[str], Dict[str, str]] = "accuracy",
    refit_score: str = "accuracy",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    spec = GridSearchSpec(
        tuning_name="grid_search",
        param_grid=param_grid,
        cv_strategy=cv_strategy,  # type: ignore[arg-type]
        n_splits=n_splits,
        scoring=scoring,
        refit_score=refit_score,
        n_jobs=n_jobs,
        random_state=random_state,
        return_train_score=kwargs.get("return_train_score", True),
        verbose=kwargs.get("verbose", 0),
    )
    strategy = GridSearch(spec).fit(pipeline, X, y)
    return {
        "strategy": strategy,
        "best_params": strategy.get_best_params(),
        "best_score": strategy.get_best_score(),
        "best_estimator": strategy.get_best_estimator(),
        "cv_results": strategy.get_cv_results(),
        "summary": strategy.get_results_summary(),
    }


def run_random_search(
    pipeline: Pipeline,
    X: Any,
    y: Any,
    param_distributions: Dict[str, List[Any]],
    n_iter: int = 10,
    cv_strategy: str = "stratified_kfold",
    n_splits: int = 5,
    scoring: Union[str, List[str], Dict[str, str]] = "accuracy",
    refit_score: str = "accuracy",
    n_jobs: int = -1,
    random_state: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    spec = RandomSearchSpec(
        tuning_name="random_search",
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv_strategy=cv_strategy,  # type: ignore[arg-type]
        n_splits=n_splits,
        scoring=scoring,
        refit_score=refit_score,
        n_jobs=n_jobs,
        random_state=random_state,
        return_train_score=kwargs.get("return_train_score", True),
        verbose=kwargs.get("verbose", 0),
    )
    strategy = RandomSearch(spec).fit(pipeline, X, y)
    return {
        "strategy": strategy,
        "best_params": strategy.get_best_params(),
        "best_score": strategy.get_best_score(),
        "best_estimator": strategy.get_best_estimator(),
        "cv_results": strategy.get_cv_results(),
        "summary": strategy.get_results_summary(),
    }


__all__ = [
    "BaseTuning",
    "GridSearch",
    "RandomSearch",
    "create_tuning",
    "run_grid_search",
    "run_random_search",
]


