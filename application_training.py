"""
Training application entry point.
"""
from __future__ import annotations

import logging
import numpy as np

from specs import FeatureSpecBuilder, ModelSpecBuilder, MLflowSpecBuilder
from src.orchestrator import run_ml_workflow, get_workflow_summary
from data import load_data
import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


def main():
    """
    Main training application entry point.
    
    Loads data from SQLite database and runs ML workflow with MLflow tracking.
    """
    try:
        logger.info("=" * 60)
        logger.info("ML Workflow Version 1 - Training Application")
        logger.info("Data Source: SQLite Database")
        logger.info("=" * 60)
        
        # Load data from SQLite
        logger.info("Loading training data from SQLite database...")
        try:
            X, y = load_data(dataset_name="classification_data")
            logger.info(f"Loaded data: X shape {X.shape}, y shape {y.shape}")
        except (FileNotFoundError, ValueError) as e:
            logger.error(f"Data not found: {e}")
            logger.error("\n" + "=" * 70)
            logger.error("DATA NOT FOUND - Please run data generation first:")
            logger.error("  python data/generate_database.py")
            logger.error("=" * 70)
            raise FileNotFoundError(
                "Data not found in SQLite. "
                "Run 'python data/generate_database.py' first."
            )
        
        # Create feature specifications
        feature_builder = FeatureSpecBuilder()
        n_features = 20
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        feature_specs = feature_builder.add_numeric_group(
            feature_names,
            imputer_strategy="mean",
            scaler_type="standard"
        ).build()
        
        logger.info(f"Created {len(feature_specs)} feature specifications")
        
        # Create model specification
        model_builder = ModelSpecBuilder()
        model_spec = model_builder.add_classifier(
            name="gradient_boosting_classifier",
            hyperparameters={
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3
            },
            evaluation_metrics=["accuracy", "roc_auc", "f1_score"]
        ).build()[0]
        
        logger.info(f"Created model specification: {model_spec.model_name}")
        
        # Optional: simple grid search over classifier hyperparameters
        # Classifier exposes sklearn model parameters directly
        from specs import GridSearchSpec, ClassifierCalibrationSpec
        tuning_spec = GridSearchSpec(
            tuning_name="gb_grid",
            param_grid={
                'classifier__n_estimators': [50, 100],
                'classifier__max_depth': [2, 3],
            },
            scoring='accuracy',
            refit_score='accuracy',
            n_jobs=-1,
            verbose=0,
        )
        
        # Optional calibration on the fitted pipeline
        calibration_spec = ClassifierCalibrationSpec(
            calibration_name="gb_platt",
            method="sigmoid",
            cv_strategy=3,  # Use 3-fold CV instead of deprecated 'prefit'
            ensemble=True,
        )
        
        # Configure MLflow tracking and model registry
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
        
        # Disable Feast feature store temporarily (entity join issues to resolve)
        logger.info("Feature store: Disabled (using direct SQLite loading)")
        feature_store_spec = None
        
        # Run ML workflow with MLflow tracking and optional feature store
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
        
        # Get summary
        summary = get_workflow_summary(results)
        
        # Display Feature Store information
        if 'feature_store_enabled' in results and results['feature_store_enabled']:
            print("\nFeature Store (Feast):")
            print("-" * 60)
            print(f"  Repository: {results['feature_store_repo']}")
            print(f"  Feature Views: {', '.join(results['feature_views'])}")
            print(f"  Status: ✓ Features retrieved successfully")
        
        # Display MLflow information
        if 'mlflow_run_id' in results:
            print("\nMLflow Tracking:")
            print("-" * 60)
            print(f"  MLflow Run ID: {results['mlflow_run_id']}")
            print(f"  MLflow Experiment ID: {results['mlflow_experiment_id']}")
            print(f"  View in MLflow UI: mlflow ui --port 5000")
            print(f"  Then navigate to: http://localhost:5000")
            
            # Show model registry information if enabled
            if mlflow_spec.register_model:
                print("\nMLflow Model Registry:")
                print("-" * 60)
                registered_name = f"{model_spec.model_name}_registered"
                print(f"  Model Name: {registered_name}")
                print(f"  Stage: {mlflow_spec.model_stage}")
                print(f"  Status: ✓ Registered successfully")
                
                # Offer to promote to Production
                if mlflow_spec.model_stage != "Production":
                    print(f"\n  💡 To promote to Production, run:")
                    print(f"    poetry run python -m module.mlflow_tracker --promote {registered_name} --stage Production")
        
        # Display results
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
        if 'f1_score' in summary:
            print(f"Train F1 Score: {summary['f1_score']:.4f}")
        
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
        
        print("\n" + "=" * 60)
        print("Training completed successfully!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. View results in MLflow UI:")
        print("      poetry run mlflow ui --port 5000")
        
        if mlflow_spec.register_model:
            registered_name = f"{model_spec.model_name}_registered"
            print(f"\n  2. Promote model '{registered_name}' to Production:")
            print(f"      poetry run python -m module.mlflow_tracker --promote {registered_name} --stage Production")
            print(f"\n  3. Test inference with the registered model:")
            print(f"      cd ../ML_WORKFLOW_INFERENCE_SERVING")
            print(f"      poetry run python mlflow_bridge.py --model-name {registered_name} --stage {mlflow_spec.model_stage}")
        else:
            print("\n  2. Enable model registry in MLflow spec to register models")
            print("  3. Use application_inference_serving.py for predictions")
        
        logger.info("Training application completed successfully")
        
    except Exception as e:
        logger.error(f"Training application failed: {e}", exc_info=True)
        print(f"\nERROR: {e}")
        raise


if __name__ == "__main__":
    print("\n🚀 ML Workflow Training Application\n")
    main()

