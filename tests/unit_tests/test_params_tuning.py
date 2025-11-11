"""
Tests for parameter tuning module.
"""

import numpy as np
import pandas as pd

from specs import FeatureSpecBuilder, ModelSpecBuilder
from specs import GridSearchSpec
from module.pre_processing import FeatureSpecPipeline
from module.classifier import GradientBoostingClassifierImpl
from module.params_tuning import GridSearch, run_grid_search
from sklearn.pipeline import Pipeline as SklearnPipeline


def _make_data(n: int = 80):
    rng = np.random.RandomState(0)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.normal(size=n),
        "x3": rng.normal(size=n),
    })
    y = ((X["x1"] + X["x2"] * 0.5 + rng.normal(scale=0.5, size=n)) > 0).astype(int)
    return X, y


def _make_pipeline():
    X, _ = _make_data(10)
    fb = FeatureSpecBuilder()
    feature_specs = fb.add_numeric_group(X.columns.tolist(), imputer_strategy="mean", scaler_type="standard").build()
    mb = ModelSpecBuilder()
    model_spec = mb.add_classifier("gb", hyperparameters={"n_estimators": 30, "max_depth": 2}).build()[0]
    pre = FeatureSpecPipeline(feature_specs)
    clf = GradientBoostingClassifierImpl(model_spec)
    return SklearnPipeline([
        ("preprocessor", pre),
        ("classifier", clf),
    ])


def test_grid_search_strategy_fits_and_returns_best():
    """Test GridSearch with tunable hyperparameters via model_spec."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    X, y = _make_data(60)
    
    # Use raw sklearn pipeline for testing GridSearch
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])
    
    spec = GridSearchSpec(
        tuning_name="gs",
        param_grid={
            "classifier__n_estimators": [20, 40],
            "classifier__max_depth": [2, 3],
        },
        scoring="accuracy",
        refit_score="accuracy",
        n_jobs=1,
        verbose=0,
    )
    gs = GridSearch(spec).fit(pipe, X, y)
    summary = gs.get_results_summary()
    assert summary["tuning_type"] == "grid_search"
    assert "classifier__n_estimators" in gs.get_best_params()
    assert gs.get_best_score() > 0
    assert gs.get_best_estimator() is not None


def test_run_grid_search_convenience():
    """Test run_grid_search convenience function."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    X, y = _make_data(50)
    
    # Use raw sklearn pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(random_state=42))
    ])
    
    res = run_grid_search(
        pipeline=pipe,
        X=X,
        y=y,
        param_grid={
            "classifier__n_estimators": [10, 20],
        },
        scoring="accuracy",
        refit_score="accuracy",
        n_jobs=1,
        verbose=0,
    )
    assert "best_params" in res and "best_estimator" in res
    assert res["best_score"] > 0


