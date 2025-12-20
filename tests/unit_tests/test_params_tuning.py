"""
Unit tests for the parameter tuning module.
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from module.params_tuning import GridSearch, RandomSearch, create_tuning
from specs_training import GridSearchSpec, RandomSearchSpec


def _make_data(n_samples: int = 120):
    """Create test data for tuning."""
    rng = np.random.RandomState(7)
    X = rng.normal(size=(n_samples, 4))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.3, size=n_samples) > 0).astype(int)
    return X, y


def _make_pipeline() -> Pipeline:
    """Create a test pipeline with scaler and classifier."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ]
    )


@pytest.mark.unit
class TestGridSearch:
    """Tests for GridSearch tuning strategy."""

    def test_grid_search_runs_and_produces_summary(self):
        """Test grid search runs successfully and produces summary."""
        X, y = _make_data()
        pipeline = _make_pipeline()
        spec = GridSearchSpec(
            tuning_name="grid",
            param_grid={"classifier__n_estimators": [20, 30], "classifier__max_depth": [2, 3]},
            scoring="accuracy",
            refit_score="accuracy",
            n_jobs=1,
            verbose=0,
        )

        grid = GridSearch(spec).fit(pipeline, X, y)

        assert grid.get_best_score() > 0, (
            f"Best score should be positive, got {grid.get_best_score()}"
        )
        assert "classifier__n_estimators" in grid.get_best_params(), (
            "Best parameters should include 'classifier__n_estimators'"
        )
        
        summary = grid.get_results_summary()
        
        assert summary["tuning_type"] == "grid_search", (
            f"Summary should indicate 'grid_search', got '{summary.get('tuning_type')}'"
        )
        assert summary["n_splits"] == spec.n_splits, (
            f"Summary n_splits should be {spec.n_splits}, got {summary.get('n_splits')}"
        )
        assert grid.get_best_estimator() is not None, (
            "Best estimator should not be None after fitting"
        )

    def test_accessors_require_fitted_state(self):
        """Test accessor methods require fitted tuning object."""
        pipeline = _make_pipeline()
        spec = GridSearchSpec(
            tuning_name="grid",
            param_grid={"classifier__n_estimators": [10]},
        )
        grid = GridSearch(spec)

        with pytest.raises(ValueError) as exc_info:
            grid.get_best_params()
        
        assert "tuning must be fitted" in str(exc_info.value).lower(), (
            f"Error message should mention 'tuning must be fitted', got: {exc_info.value}"
        )

        with pytest.raises(ValueError) as exc_info:
            grid.get_best_score()
        
        assert "tuning must be fitted" in str(exc_info.value).lower(), (
            f"Error message should mention 'tuning must be fitted', got: {exc_info.value}"
        )

    def test_get_best_params_returns_valid_params(self):
        """Test get_best_params returns valid parameter dictionary."""
        X, y = _make_data()
        pipeline = _make_pipeline()
        spec = GridSearchSpec(
            tuning_name="grid",
            param_grid={
                "classifier__n_estimators": [20, 30],
                "classifier__max_depth": [2, 3],
            },
            n_jobs=1,
        )

        grid = GridSearch(spec).fit(pipeline, X, y)
        best_params = grid.get_best_params()

        assert isinstance(best_params, dict), (
            f"Best params should be a dict, got {type(best_params)}"
        )
        assert len(best_params) > 0, (
            "Best params should contain at least one parameter"
        )
        assert all(key.startswith("classifier__") for key in best_params.keys()), (
            "All parameter keys should start with 'classifier__'"
        )


@pytest.mark.unit
class TestRandomSearch:
    """Tests for RandomSearch tuning strategy."""

    def test_random_search_runs_and_returns_results(self):
        """Test random search runs successfully and returns results."""
        X, y = _make_data()
        pipeline = _make_pipeline()
        spec = RandomSearchSpec(
            tuning_name="random",
            param_distributions={
                "classifier__n_estimators": [10, 20, 30],
                "classifier__max_depth": [2, 3],
            },
            n_iter=2,
            scoring="accuracy",
            refit_score="accuracy",
            n_jobs=1,
        )

        random = RandomSearch(spec).fit(pipeline, X, y)

        assert random.get_best_score() > 0, (
            f"Best score should be positive, got {random.get_best_score()}"
        )
        assert len(random.get_best_params().keys()) > 0, (
            "Best params should contain parameters"
        )
        
        results = random.get_cv_results()
        
        assert "mean_test_score" in results, (
            "CV results should include 'mean_test_score'"
        )
        assert len(results["mean_test_score"]) == spec.n_iter, (
            f"Should have {spec.n_iter} test scores, got {len(results['mean_test_score'])}"
        )

    def test_random_search_respects_n_iter(self):
        """Test random search respects n_iter parameter."""
        X, y = _make_data()
        pipeline = _make_pipeline()
        n_iter = 5
        spec = RandomSearchSpec(
            tuning_name="random",
            param_distributions={"classifier__n_estimators": [10, 20, 30]},
            n_iter=n_iter,
            n_jobs=1,
        )

        random = RandomSearch(spec).fit(pipeline, X, y)
        results = random.get_cv_results()

        assert len(results["mean_test_score"]) == n_iter, (
            f"Should have {n_iter} iterations, got {len(results['mean_test_score'])}"
        )


@pytest.mark.unit
class TestCreateTuning:
    """Tests for create_tuning factory function."""

    def test_factory_creates_correct_strategy(self):
        """Test factory creates correct tuning strategy based on spec."""
        grid_spec = GridSearchSpec(
            tuning_name="grid",
            param_grid={"classifier__n_estimators": [10]},
        )
        random_spec = RandomSearchSpec(
            tuning_name="random",
            param_distributions={"classifier__n_estimators": [10, 20]},
        )

        grid_tuner = create_tuning(grid_spec)
        random_tuner = create_tuning(random_spec)

        assert isinstance(grid_tuner, GridSearch), (
            f"Factory should return GridSearch for GridSearchSpec, got {type(grid_tuner)}"
        )
        assert isinstance(random_tuner, RandomSearch), (
            f"Factory should return RandomSearch for RandomSearchSpec, got {type(random_tuner)}"
        )

    def test_factory_requires_param_tuning_spec(self):
        """Test factory requires ParamTuningSpec instance."""
        with pytest.raises(TypeError) as exc_info:
            create_tuning("not a spec")
        
        assert "TypeError" in type(exc_info.value).__name__ or "spec" in str(
            exc_info.value
        ).lower(), (
            f"Error should be TypeError or mention 'spec', got: {exc_info.value}"
        )
