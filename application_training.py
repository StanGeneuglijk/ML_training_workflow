"""
Training application entry point.
"""
from __future__ import annotations

import logging
import numpy as np

from specs import FeatureSpecBuilder, ModelSpecBuilder, MLflowSpecBuilder, FeatureStoreSpecBuilder
from src.orchestrator import run_ml_workflow, get_workflow_summary
from data import load_data
import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


def main():
    """
    Main training application entry point.
    """
    try:
        logger.info("=" * 60)
        logger.info("ML Workflow - Training Application")
        logger.info("=" * 60)
        
        # Configure feature store specification
        logger.info("Loading training data from feature store...")
        feature_store_builder = FeatureStoreSpecBuilder()
        feature_store_spec = (
            feature_store_builder
            .enable()
            .set_repo_path("feature_repo")
            .set_n_features(20)
            .set_initialize_on_start(True)
            .build()
        )
        X, y = None, None 

        # Configure feature specification
        n_features = 20
        feature_names = [f"feature_{i}" for i in range(n_features)]
        feature_builder = FeatureSpecBuilder()
        feature_specs = feature_builder.add_numeric_group(
            feature_names,
            imputer_strategy="mean",
            scaler_type="standard"
        ).build()
        logger.info(f"Created {len(feature_specs)} feature specifications")
        
        # Configure model specification
        model_builder = ModelSpecBuilder()
        model_spec = model_builder.add_classifier(
            name="gradient_boosting_classifier",
            hyperparameters={
                "n_estimators": 100,
                "learning_rate": 0.1            },
            evaluation_metrics=["accuracy", "roc_auc"]
        ).build()[0]
        logger.info(f"Created model specification: {model_spec.model_name}")
        
        # Configure grid search specification
        from specs import GridSearchSpec, ClassifierCalibrationSpec
        tuning_spec = GridSearchSpec(
            tuning_name="gb_grid",
            param_grid={
                'classifier__n_estimators': [50, 100]},
            scoring='accuracy',
            refit_score='accuracy',
            n_jobs=-1,
            verbose=0,
        )
        
        # Configure calibration specification
        calibration_spec = ClassifierCalibrationSpec(
            calibration_name="gb_platt",
            method="sigmoid",
            cv_strategy=3,
            ensemble=True,
        )
        
        # Configure MLflow tracking and registry
        mlflow_spec = (MLflowSpecBuilder()
            .enable()
            .set_experiment("classification_experiments")
            .set_run_name(f"gradient_boosting_{model_spec.model_name}")
            .enable_model_registry(stage="Staging")
            .add_tags({
                "model_type": "gradient_boosting",
                "dataset": "classification_data",
                "tuning": "enabled",
                "calibration": "enabled"
            })
            .set_description("Classification workflow with tuning and calibration")
            .build()
        )
        logger.info(f"MLflow configuration: {mlflow_spec}")
                
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 5},
            random_state=42,
            tuning_spec=tuning_spec,
            calibration_spec=calibration_spec,
            mlflow_spec=mlflow_spec,
            feature_store_spec=feature_store_spec,
        )
        
        summary = get_workflow_summary(results)
        
        # Display MLflow information if enabled
        if 'mlflow_run_id' in results:
            print("\nMLflow Tracking:")
            print("-" * 60)
            print(f"  MLflow Run ID: {results['mlflow_run_id']}")
            print(f"  MLflow Experiment ID: {results['mlflow_experiment_id']}")
            print(f"  View in MLflow UI: mlflow ui --port 5000")
            
            # Display model registry information if enabled
            if mlflow_spec.register_model:
                print("\nMLflow Model Registry:")
                print("-" * 60)
                registered_name = f"{model_spec.model_name}_registered"
                print(f"  Model Name: {registered_name}")
                print(f"  Stage: {mlflow_spec.model_stage}")
                
        # Display training results summary
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
        
        # Display tuning information if enabled
        print("=" * 60)
        if 'tuning_summary' in results:
            ts = results['tuning_summary']
            print("\nTuning Summary")
            print("-" * 60)
            print(f"Name: {ts['tuning_name']}")
            print(f"Type: {ts['tuning_type']}")
            print(f"Best score: {ts['best_score']:.4f}")

        # Display calibration information if enabled
        if 'calibration_summary' in results:
            cs = results['calibration_summary']
            print("\nCalibration Summary")
            print("-" * 60)
            print(f"Name: {cs['calibration_name']}")
            print(f"Method: {cs['method']}")
            print(f"CV: {cs['cv_strategy']}")
        
        logger.info("Training application completed successfully")
        
    except Exception as e:
        logger.error(f"Training application failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    print("\n ML Workflow Training Application\n")
    main()

