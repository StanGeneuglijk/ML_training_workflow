"""
Training application entry point
"""
from __future__ import annotations

from typing import Any, Dict

from specs_training import (
    FeatureSpec,
    FeatureSpecBuilder,
    FeatureStoreSpec,
    FeatureStoreSpecBuilder,
    GridSearchSpec,
    MLflowSpec,
    MLflowSpecBuilder,
    ClassifierModelSpec,
    ModelSpecBuilder,
    ClassifierCalibrationSpec,
)
from src_training.orchestrator import run_ml_workflow, get_workflow_summary
from data import load_data

import logging
import utils
logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


DEFAULT_CONFIG: Dict[str, Any] = {
    "n_features": 20,
    "cv_folds": 5,
    "random_state": 42,
    "tuning_name": "gb_grid",
    "tuning_grid": {"classifier__n_estimators": [50, 100]},
    "calibration_name": "gb_platt",
    "model_name": "gradient_boosting_classifier",
    "mlflow_experiment": "classification_experiments",
    "feature_repo": "feature_repo",
    "mlflow_tags": {
        "model_type": "gradient_boosting",
        "dataset": "classification_data",
        "tuning": "enabled",
        "calibration": "enabled",
    },
    "calibration_params": {
        "method": "sigmoid",
        "cv_strategy": 3,
        "ensemble": True,
    },
}


def _build_config(
    overrides: Dict[str, Any] | None = None
) -> Dict[str, Any]:
    """
    Build configuration dictionary with defaults and overrides.
    """
    config = DEFAULT_CONFIG.copy()
    if overrides:
        config.update(overrides)
    return config


def _build_feature_store_spec(
    config: Dict[str, Any]
) -> FeatureStoreSpec:
    """
    Build feature store specification.
    """
    return (
        FeatureStoreSpecBuilder()
        .enable()
        .set_repo_path(config["feature_repo"])
        .set_n_features(config["n_features"])
        .set_initialize_on_start(True)
        .build()
    )


def _build_feature_specs(
    config: Dict[str, Any]
) -> list[FeatureSpec]:
    """
    Build feature specifications.
    """
    feature_names = [f"feature_{i}" for i in range(config["n_features"])]
    feature_builder = FeatureSpecBuilder()
    return feature_builder.add_numeric_group(
        feature_names,
        imputer_strategy="mean",
        scaler_type="standard",
    ).build()


def _build_model_spec(
    config: Dict[str, Any]
) -> ClassifierModelSpec:
    """
    Build model specification.
    """
    return ModelSpecBuilder().add_classifier(
        name=config["model_name"],
        hyperparameters={"n_estimators": 100, "learning_rate": 0.1},
        evaluation_metrics=["accuracy", "roc_auc"],
    ).build()[0]


def _build_tuning_spec(
    config: Dict[str, Any]
) -> GridSearchSpec:
    """
    Build tuning specification.
    """
    return GridSearchSpec(
        tuning_name=config["tuning_name"],
        param_grid=config["tuning_grid"],
        scoring="accuracy",
        refit_score="accuracy",
        n_jobs=-1,
        verbose=0,
    )


def _build_calibration_spec(
    config: Dict[str, Any]
) -> ClassifierCalibrationSpec:
    """
    Build calibration specification.
    """
    calibration_params = config["calibration_params"]
    return ClassifierCalibrationSpec(
        calibration_name=config["calibration_name"],
        method=calibration_params["method"],
        cv_strategy=calibration_params["cv_strategy"],
        ensemble=calibration_params["ensemble"],
    )


def _build_mlflow_spec(
    config: Dict[str, Any],
    model_spec: ClassifierModelSpec,
) -> MLflowSpec:
    """
    Build MLflow specification.
    """
    return (
        MLflowSpecBuilder()
        .enable()
        .set_experiment(config["mlflow_experiment"])
        .set_run_name(f"gradient_boosting_{model_spec.model_name}")
        .enable_model_registry(stage="Staging")
        .add_tags(config["mlflow_tags"])
        .set_description("Classification workflow with tuning and calibration")
        .build()
    )


def _print_mlflow_info(
    results: Dict[str, Any],
    model_spec: ClassifierModelSpec,
    mlflow_spec: MLflowSpec,
) -> None:
    """
    Print MLflow tracking information.
    """
    if 'mlflow_run_id' not in results:
        return

    print("\nMLflow Tracking:")
    print("-" * 60)
    print(f"  MLflow Run ID: {results['mlflow_run_id']}")
    print(f"  MLflow Experiment ID: {results['mlflow_experiment_id']}")
    print(f"  View in MLflow UI: mlflow ui --port 5000")

    if mlflow_spec.register_model:
        print("\nMLflow Model Registry:")
        print("-" * 60)
        registered_name = f"{model_spec.model_name}_registered"
        print(f"  Model Name: {registered_name}")
        print(f"  Stage: {mlflow_spec.model_stage}")


def _print_results_summary(
    summary: Dict[str, Any]
) -> None:
    """
    Print training results summary.
    """
    print("\n" + "=" * 60)
    print("Training Results Summary")
    print("=" * 60)
    print(f"Model: {summary['model_name']}")
    print(f"Algorithm: {summary['algorithm']}")
    print(f"Validation: {summary['validation_strategy']}")
    if 'cv_score' in summary:
        print(f"CV Accuracy: {summary['cv_score']:.4f} ± {summary['cv_std']:.4f}")
    if 'test_score' in summary:
        print(f"Test Accuracy: {summary['test_score']:.4f}")
    if 'accuracy' in summary:
        print(f"Train Accuracy: {summary['accuracy']:.4f}")
    if 'roc_auc' in summary:
        print(f"Train ROC AUC: {summary['roc_auc']:.4f}")


def _print_optional_sections(
    results: Dict[str, Any]
) -> None:
    """
    Print optional sections.
    """
    print("=" * 60)
    if 'tuning_summary' in results:
        ts = results['tuning_summary']
        print("\nTuning Summary")
        print("-" * 60)
        print(f"Name: {ts['tuning_name']}")
        print(f"Type: {ts['tuning_type']}")
        print(f"Best score: {ts['best_score']:.4f}")

    if 'calibration_summary' in results:
        cs = results['calibration_summary']
        print("\nCalibration Summary")
        print("-" * 60)
        print(f"Name: {cs['calibration_name']}")
        print(f"Method: {cs['method']}")
        print(f"CV: {cs['cv_strategy']}")


def main():
    """
    Main training application entry point.
    """
    try:
        logger.info("=" * 60)
        logger.info("ML Workflow - Training Application")
        logger.info("=" * 60)

        config = _build_config()

        # --- Specification assembly ---
        feature_store_spec = _build_feature_store_spec(config)

        X, y = load_data()
        logger.info(
            "Successfully loaded training data (fallback) with shapes X=%s, y=%s",
            getattr(X, "shape", None),
            getattr(y, "shape", None),
        )


        feature_specs = _build_feature_specs(config)
        logger.info(f"Created {len(feature_specs)} feature specifications")
        
        model_spec = _build_model_spec(config)
        logger.info(f"Created model specification: {model_spec.model_name}")
        
        tuning_spec = _build_tuning_spec(config)
        
        calibration_spec = _build_calibration_spec(config)
        
        mlflow_spec = _build_mlflow_spec(config, model_spec)
        logger.info(f"MLflow configuration: {mlflow_spec}")
                
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": config["cv_folds"]},
            random_state=config["random_state"],
            tuning_spec=tuning_spec,
            calibration_spec=calibration_spec,
            mlflow_spec=mlflow_spec,
            feature_store_spec=feature_store_spec,
        )
        
        summary = get_workflow_summary(results)
        
        _print_mlflow_info(results, model_spec, mlflow_spec)
        _print_results_summary(summary)
        _print_optional_sections(results)
        
        logger.info("Training application completed successfully")
        
    except Exception as e:
        logger.error(f"Training application failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    print("\n ML Workflow Training Application\n")
    main()

