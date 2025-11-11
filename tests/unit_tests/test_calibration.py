"""
Tests for calibration module.
"""

import numpy as np
import pandas as pd

from specs import FeatureSpecBuilder, ModelSpecBuilder
from specs import ClassifierCalibrationSpec
from module.pre_processing import FeatureSpecPipeline
from module.classifier import GradientBoostingClassifierImpl
from module.calibration import ClassifierCalibration
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


def _make_fitted_pipeline():
    X, y = _make_data(100)
    fb = FeatureSpecBuilder()
    feature_specs = fb.add_numeric_group(X.columns.tolist(), imputer_strategy="mean", scaler_type="standard").build()
    mb = ModelSpecBuilder()
    model_spec = mb.add_classifier("gb", hyperparameters={"n_estimators": 30, "max_depth": 2}).build()[0]
    pre = FeatureSpecPipeline(feature_specs)
    clf = GradientBoostingClassifierImpl(model_spec)
    pipe = SklearnPipeline([
        ("preprocessor", pre),
        ("classifier", clf),
    ])
    pipe.fit(X, y)
    return pipe, X, y


def test_classifier_calibration_prefit_sigmoid():
    """Test classifier calibration with CV strategy."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    
    X, y = _make_data(100)
    X_array = X.values  # Convert to numpy for simpler pipeline
    
    # Use simple sklearn pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GradientBoostingClassifier(n_estimators=30, max_depth=2, random_state=42))
    ])
    pipe.fit(X_array, y)
    
    spec = ClassifierCalibrationSpec(
        calibration_name="calib",
        method="sigmoid",
        cv_strategy=3,  # Use 3-fold CV
        ensemble=True,
    )
    calib = ClassifierCalibration(spec).fit(pipe, X_array, y)
    proba = calib.predict_proba(X_array)
    assert proba.shape[0] == X_array.shape[0]
    assert proba.shape[1] == 2  # Binary classification
    summary = calib.get_results_summary()
    assert summary["method"] == "sigmoid"
    assert summary["cv_strategy"] == 3


