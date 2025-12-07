"""
Calibration module./yes!
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Union
import logging

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

from specs.calibration_spec import ClassifierCalibrationSpec, CalibrationSpec
import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


def _cv_splitter(
    spec: ClassifierCalibrationSpec,
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

    if isinstance(cv_strategy, int):
        if cv_strategy < 2:
            raise ValueError("cv_strategy integer value must be >= 2")
        cv_strategy = "kfold"
        n_splits = cv_strategy

    if cv_strategy == "kfold":
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    elif cv_strategy == "stratified_kfold":
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
    elif cv_strategy == "prefit":
        raise ValueError("prefit strategy should be handled before calling _cv_splitter")
    else:
        raise ValueError(f"invalid cv strategy: {cv_strategy}")

    return cv_splitter


class BaseCalibration:
    """
    Base calibration class, analogous to BaseTuning.
    """

    def __init__(
        self, 
        spec: CalibrationSpec
    ) -> None:
        """
        Initialize base calibration.

        Args:
            spec: Calibration specification.
        """
        if not isinstance(spec, CalibrationSpec):
            raise TypeError(f"spec must be a CalibrationSpec, got {type(spec).__name__}")
        self.spec = spec
        self.calibrator: Optional[Any] = None
        self.is_fitted: bool = False

    def fit(
        self,
        pipeline: Pipeline,
        X_cal: Any,
        y_cal: Any
    ) -> "BaseCalibration":
        """
        Fit the calibration strategy.

        Args:
            pipeline: The pipeline to calibrate.
            X_cal: The training features.
            y_cal: The training labels.

        Returns:
            self
        """
        X_arr, y_arr = utils.validate_training_data(X_cal, y_cal)

        self.calibrator = self._create_calibrator(pipeline)
        self.calibrator.fit(X_arr, y_arr)
        self.is_fitted = True

        logger.info("Calibration %s fitted successfully", self.spec.calibration_name)

        return self

    def _create_calibrator(
        self, 
        pipeline: Pipeline
    ) -> Any:
        """
        Create the calibration estimator.
        
        Args:
            pipeline: The pipeline to calibrate.

        Returns:
            The calibration estimator.
        """
        raise NotImplementedError

    def _ensure_fitted(
        self
    ) -> None:
        """
        Ensure the calibration is fitted.
  
        Args:
            None

        Returns:
            None
        """
        if not self.is_fitted or self.calibrator is None:
            raise ValueError("calibration must be fitted first")

    def predict(
        self, 
        X: Any
    ) -> np.ndarray:
        """
        Predict calibrated labels.

        Args:
            X: The features to predict.

        Returns:
            The predicted labels.
        """
        self._ensure_fitted()

        X_arr = utils.validate_prediction_data(X)
        predict = self.calibrator.predict(X_arr)

        logger.info(f"Predicted {predict.shape[0]} samples")

        return predict

    def predict_proba(
        self, 
        X: Any
    ) -> np.ndarray:
        """
        Predict calibrated probabilities.

        Args:
            X: The features to predict.

        Returns:
            The predicted probabilities.
        """
        self._ensure_fitted()

        X_arr = utils.validate_prediction_data(X)
        predict_proba = self.calibrator.predict_proba(X_arr)

        logger.info(f"Predicted {predict_proba.shape[0]} samples")

        return predict_proba


    def get_results_summary(
        self
    ) -> Dict[str, Any]:
        """
        Summarize calibration configuration.

        Args:
            None

        Returns:
            Dictionary of calibration results
        """
        self._ensure_fitted()
        return {
            "calibration_name": self.spec.calibration_name,
            "calibration_type": self.spec.get_calibration_type(),
        }

class ClassifierCalibration(BaseCalibration):
    """
    Classifier calibration implementation.
    """

    def __init__(
        self, 
        spec: ClassifierCalibrationSpec
    ) -> None:
        """
        Initialize classifier calibration.

        Args:
            spec: Classifier calibration specification.

        Returns:
            None
        """
        if not isinstance(spec, ClassifierCalibrationSpec):
            raise TypeError(f"spec must be a ClassifierCalibrationSpec, got {type(spec).__name__}")
        
        super().__init__(spec)
        self.classifier_spec = spec

    def _create_calibrator(
        self, 
        pipeline: Pipeline
    ) -> CalibratedClassifierCV:
        """
        Instantiate the calibrated classifier.

        Args:
            pipeline: The pipeline to calibrate.

        Returns:
            The calibrated classifier.
        """
        cv_strategy = self.classifier_spec.cv_strategy

        if cv_strategy == "prefit":
            cv = "prefit"
        else:
            cv = _cv_splitter(self.classifier_spec)

        return CalibratedClassifierCV(
            estimator=pipeline,
            method=self.classifier_spec.method,
            cv=cv,
            ensemble=self.classifier_spec.ensemble,
        )

    def get_results_summary(
        self
    ) -> Dict[str, Any]:
        """
        Include classifier-specific summary information.

        Args:
            None

        Returns:
            Dictionary of calibration results
        """
        summary = super().get_results_summary()
        self._ensure_fitted()
        
        summary.update(
            {
                "method": self.classifier_spec.method,
                "cv_strategy": self.classifier_spec.cv_strategy,
                "ensemble": self.classifier_spec.ensemble,
                "n_calibrated_models": len(self.calibrator.calibrated_classifiers_),
            }
        )
        return summary


def create_calibration(
    spec: CalibrationSpec
) -> BaseCalibration:
    """
    Create a calibration.

    Args:
        spec: The calibration specification.

    Returns:
        The calibration.
    """
    if not isinstance(spec, CalibrationSpec):
        raise TypeError(f"spec must be a CalibrationSpec, got {type(spec).__name__}")
    if isinstance(spec, ClassifierCalibrationSpec):
        return ClassifierCalibration(spec)
    raise ValueError(f"unsupported calibration type: {spec.get_calibration_type()}")


__all__ = [
    "BaseCalibration",
    "ClassifierCalibration",
    "create_calibration",
]

