"""
Calibration utilities for ML workflow v1.

Minimal classifier probability calibration using sklearn CalibratedClassifierCV.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union
import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from specs.calibration_spec import ClassifierCalibrationSpec, CalibrationSpec
import utils


logger = logging.getLogger(__name__)


class ClassifierCalibration:
    def __init__(self, spec: ClassifierCalibrationSpec) -> None:
        if not isinstance(spec, ClassifierCalibrationSpec):
            raise TypeError("spec must be a ClassifierCalibrationSpec")
        self.spec = spec
        self.calibrated_: Optional[CalibratedClassifierCV] = None
        self.is_fitted: bool = False

    def _cv(self):
        if self.spec.cv_strategy == "prefit":
            return "prefit"
        if isinstance(self.spec.cv_strategy, int):
            return StratifiedKFold(
                n_splits=self.spec.cv_strategy,
                shuffle=True,
                random_state=self.spec.random_state,
            )
        return self.spec.cv_strategy

    def fit(self, pipeline: Pipeline, X_cal: Any, y_cal: Any) -> "ClassifierCalibration":
        X_arr, y_arr = utils.validate_training_data(X_cal, y_cal)
        self.calibrated_ = CalibratedClassifierCV(
            estimator=pipeline,
            method=self.spec.method,
            cv=self._cv(),
            ensemble=self.spec.ensemble,
        )
        self.calibrated_.fit(X_arr, y_arr)
        self.is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not self.is_fitted or self.calibrated_ is None:
            raise ValueError("calibration must be fitted first")
        X_arr = utils.validate_prediction_data(X)
        return self.calibrated_.predict(X_arr)

    def predict_proba(self, X: Any) -> np.ndarray:
        if not self.is_fitted or self.calibrated_ is None:
            raise ValueError("calibration must be fitted first")
        X_arr = utils.validate_prediction_data(X)
        return self.calibrated_.predict_proba(X_arr)

    def get_results_summary(self) -> Dict[str, Any]:
        if not self.is_fitted or self.calibrated_ is None:
            raise ValueError("calibration must be fitted first")
        return {
            "calibration_name": self.spec.calibration_name,
            "calibration_type": self.spec.get_calibration_type(),
            "method": self.spec.method,
            "cv_strategy": self.spec.cv_strategy,
            "ensemble": self.spec.ensemble,
            "n_calibrated_models": len(self.calibrated_.calibrated_classifiers_),
        }


def create_calibration(spec: CalibrationSpec) -> ClassifierCalibration:
    if not isinstance(spec, CalibrationSpec):
        raise TypeError("spec must be a CalibrationSpec")
    if isinstance(spec, ClassifierCalibrationSpec):
        return ClassifierCalibration(spec)
    raise ValueError(f"unsupported calibration type: {spec.get_calibration_type()}")


__all__ = [
    "ClassifierCalibration",
    "create_calibration",
]


