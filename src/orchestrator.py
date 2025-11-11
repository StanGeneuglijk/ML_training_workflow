"""
Orchestrator module for ML workflow version 1.

Simplified workflow orchestration for building and running ML pipelines.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from typing import Optional, Any
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.model_selection import cross_val_score, train_test_split

from specs import FeatureSpec, ClassifierModelSpec
from specs import ParamTuningSpec
from module.params_tuning import create_tuning
from specs import CalibrationSpec
from module.calibration import create_calibration
from specs import MLflowSpec
from specs import FeatureStoreSpec
from module.pre_processing import FeatureSpecPipeline
from module.classifier import GradientBoostingClassifierImpl
from module.mlflow_tracker import MLflowTracker, create_mlflow_tracker
from feature_store import FeastManager, create_feast_manager
import utils


logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


def build_ml_pipeline(
    feature_specs: list[FeatureSpec],
    model_spec: ClassifierModelSpec
) -> SklearnPipeline:
    """
    Build sklearn-compatible ML pipeline from specifications.
    
    Args:
        feature_specs: List of feature processing specifications
        model_spec: Model specification
        
    Returns:
        Sklearn pipeline with preprocessing and model
        
    Example:
        >>> from specs import FeatureSpecBuilder, ModelSpecBuilder
        >>> feature_specs = FeatureSpecBuilder().add_numeric_group(['age', 'income']).build()
        >>> model_spec = ModelSpecBuilder().add_classifier('gb_classifier').build()[0]
        >>> pipeline = build_ml_pipeline(feature_specs, model_spec)
        >>> pipeline.fit(X_train, y_train)
    """
    # Create preprocessing pipeline
    preprocessor = FeatureSpecPipeline(feature_specs)
    
    # Create classifier
    classifier = GradientBoostingClassifierImpl(model_spec)
    
    # Combine into sklearn pipeline
    pipeline = SklearnPipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    logger.info(f"Built ML pipeline: {model_spec.model_name}")
    return pipeline


def run_ml_workflow(
    feature_specs: list[FeatureSpec],
    model_spec: ClassifierModelSpec,
    X,
    y,
    validation_strategy: str = "cross_validation",
    validation_params: dict | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    X_test: Optional[Any] = None,
    y_test: Optional[Any] = None,
    tuning_spec: Optional[ParamTuningSpec] = None,
    calibration_spec: Optional[CalibrationSpec] = None,
    mlflow_spec: Optional[MLflowSpec] = None,
    feature_store_spec: Optional[FeatureStoreSpec] = None,
) -> dict:
    """
    Run complete ML workflow: build pipeline, train, validate, and return results.
    
    Args:
        feature_specs: List of feature specifications
        model_spec: Model specification
        X: Feature matrix (can be None if using feature_store_spec)
        y: Target vector (can be None if using feature_store_spec)
        validation_strategy: Validation approach ('cross_validation', 'train_test', 'none')
        validation_params: Parameters for validation (e.g., {'cv_folds': 5})
        test_size: Test set size for train_test split (if used)
        random_state: Random state for reproducibility
        X_test: Optional test features for generating predictions
        y_test: Optional test labels for evaluation
        tuning_spec: Optional parameter tuning specification
        calibration_spec: Optional calibration specification
        mlflow_spec: Optional MLflow tracking and registry specification
        feature_store_spec: Optional feature store specification for retrieving features
        
    Returns:
        Dictionary containing pipeline, metrics, and results
        
    Example:
        >>> from specs import MLflowSpec
        >>> mlflow_spec = MLflowSpec(enabled=True, experiment_name="my_exp", register_model=True)
        >>> results = run_ml_workflow(
        ...     feature_specs=feature_specs,
        ...     model_spec=model_spec,
        ...     X=X_train,
        ...     y=y_train,
        ...     validation_strategy='cross_validation',
        ...     validation_params={'cv_folds': 5},
        ...     mlflow_spec=mlflow_spec
        ... )
        >>> print(f"CV Score: {results['cv_score']:.3f}")
    """
    logger.info("Starting ML workflow")
    
    # Initialize Feature Store if FeatureStoreSpec provided and enabled
    feast_manager = None
    if feature_store_spec is not None and feature_store_spec.enabled:
        try:
            logger.info("Initializing Feast feature store")
            feast_manager = create_feast_manager(
                repo_path=feature_store_spec.repo_path,
                n_features=feature_store_spec.n_features,
                initialize=feature_store_spec.should_initialize(),
                force_recreate=feature_store_spec.force_recreate
            )
            
            # Retrieve features from feature store
            logger.info("Retrieving features from feature store")
            X, y = feast_manager.get_training_data(
                sample_indices=feature_store_spec.sample_indices,
                timestamp=feature_store_spec.timestamp
            )
            logger.info(f"Retrieved features from feature store: X shape {X.shape}, y shape {y.shape}")
        except Exception as e:
            logger.warning(f"Failed to initialize feature store: {e}. Falling back to provided X, y.")
            feast_manager = None
            if X is None or y is None:
                raise ValueError(
                    "Feature store initialization failed and no fallback X, y provided"
                ) from e
    
    # Initialize MLflow tracker if MLflowSpec provided and enabled
    mlflow_tracker = None
    if mlflow_spec is not None and mlflow_spec.enabled:
        try:
            mlflow_tracker = create_mlflow_tracker(
                experiment_name=mlflow_spec.experiment_name,
                tracking_uri=mlflow_spec.tracking_uri
            )
            
            # Get run name (use spec or auto-generate)
            run_name = mlflow_spec.get_run_name(
                default=f"{model_spec.model_name}_{validation_strategy}"
            )
            
            # Prepare tags (merge spec tags with workflow tags)
            workflow_tags = {
                'validation_strategy': validation_strategy,
                'model_type': model_spec.get_model_type() if hasattr(model_spec, 'get_model_type') else 'unknown',
                'algorithm': model_spec.algorithm
            }
            run_tags = mlflow_spec.get_run_tags(workflow_tags)
            
            # Start MLflow run
            mlflow_run_id = mlflow_tracker.start_run(
                run_name=run_name,
                tags=run_tags
            )
            logger.info(f"Started MLflow run: {run_name} (ID: {mlflow_run_id})")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}. Continuing without MLflow tracking.")
            mlflow_tracker = None
    
    # Validate inputs
    X_array, y_array = utils.validate_training_data(X, y)
    
    # Build pipeline
    pipeline = build_ml_pipeline(feature_specs, model_spec)
    
    results = {
        'pipeline': pipeline,
        'model_spec': model_spec,
        'feature_specs': feature_specs,
    }
    
    # Store MLflow info in results
    if mlflow_tracker:
        results['mlflow_run_id'] = mlflow_tracker.run_id
        results['mlflow_experiment_id'] = mlflow_tracker.experiment_id
    
    # Store feature store info in results
    if feast_manager:
        results['feature_store_enabled'] = True
        results['feature_store_repo'] = feast_manager.repo_path
    
    # Optional parameter tuning (replaces pipeline with best estimator)
    if tuning_spec is not None and tuning_spec.enabled:
        try:
            logger.info("Running parameter tuning: %s", tuning_spec.tuning_name)
            tuning = create_tuning(tuning_spec)
            tuning.fit(pipeline, X_array, y_array)
            results['tuning_summary'] = tuning.get_results_summary()
            pipeline = tuning.get_best_estimator()
            results['pipeline'] = pipeline
            logger.info("Parameter tuning completed with best score: %.4f", results['tuning_summary']['best_score'])
        except Exception as e:
            logger.warning("Parameter tuning failed (%s). Proceeding without tuning.", e)

    # Fit pipeline (best or original)
    logger.info("Fitting pipeline on training data")
    pipeline.fit(X_array, y_array)

    # Optional calibration on held-out validation if provided via params
    if calibration_spec is not None and calibration_spec.enabled:
        try:
            logger.info("Running calibration: %s", calibration_spec.calibration_name)
            # If cv_strategy == 'prefit', estimator must be fitted already (we fitted above)
            calibration = create_calibration(calibration_spec)
            # Use full training data for calibration if user did not provide split; in practice, a proper hold-out is recommended
            calibration.fit(pipeline, X_array, y_array)
            results['calibration_summary'] = calibration.get_results_summary()
            pipeline = calibration.calibrated_  # type: ignore[assignment]
            results['pipeline'] = pipeline
            logger.info("Calibration completed. Calibrated models: %s", results['calibration_summary']['n_calibrated_models'])
        except Exception as e:
            logger.warning("Calibration failed (%s). Proceeding without calibration.", e)
    
    # Validation
    if validation_strategy == "cross_validation":
        cv_folds = validation_params.get('cv_folds', 5) if validation_params else 5
        logger.info(f"Running {cv_folds}-fold cross-validation")
        
        cv_scores = cross_val_score(
            pipeline, X_array, y_array,
            cv=cv_folds,
            scoring='accuracy'
        )
        
        results['cv_score'] = cv_scores.mean()
        results['cv_std'] = cv_scores.std()
        results['cv_scores'] = cv_scores
        
        logger.info(f"CV Accuracy: {results['cv_score']:.4f} ± {results['cv_std']:.4f}")
    
    elif validation_strategy == "train_test":
        logger.info("Splitting data into train/test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array,
            test_size=test_size,
            random_state=random_state,
            stratify=y_array
        )
        
        # Refit on train set
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        test_score = pipeline.score(X_test, y_test)
        
        results['test_score'] = test_score
        results['train_size'] = len(X_train)
        results['test_size'] = len(X_test)
        
        logger.info(f"Test Accuracy: {test_score:.4f}")
    
    # Get final model metrics
    # Handle both regular pipeline and calibrated pipeline
    if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
        if hasattr(pipeline.named_steps['classifier'], 'evaluation_metrics_'):
            results['train_metrics'] = pipeline.named_steps['classifier'].evaluation_metrics_
    elif hasattr(pipeline, 'estimator') and hasattr(pipeline.estimator, 'named_steps'):
        # Calibrated pipeline: pipeline.estimator is the original pipeline
        if hasattr(pipeline.estimator.named_steps['classifier'], 'evaluation_metrics_'):
            results['train_metrics'] = pipeline.estimator.named_steps['classifier'].evaluation_metrics_
    
    # Log to MLflow if enabled
    if mlflow_tracker and mlflow_spec:
        try:
            # Prepare sample data for model signature (if enabled)
            X_sample = None
            if mlflow_spec.log_model_signature or mlflow_spec.log_input_example:
                X_sample = X_array[:min(100, len(X_array))]
            
            # Log workflow results to MLflow
            mlflow_tracker.log_workflow_results(
                results=results,
                X_sample=X_sample,
                register_model=mlflow_spec.should_register_model()
            )
            
            # Store MLflow spec in results for reference
            results['mlflow_spec'] = mlflow_spec
            
            # End MLflow run with success status
            mlflow_tracker.end_run(status="FINISHED")
            logger.info("MLflow tracking completed successfully")
        except Exception as e:
            logger.warning(f"MLflow logging failed: {e}")
            if mlflow_tracker and mlflow_tracker.run_id:
                mlflow_tracker.end_run(status="FAILED")
    
    logger.info("ML workflow completed successfully")
    return results


def get_workflow_summary(results: dict) -> dict:
    """
    Extract summary information from workflow results.
    
    Args:
        results: Workflow results dictionary
        
    Returns:
        Summary dictionary with key metrics and information
    """
    summary = {
        'model_name': results['model_spec'].model_name,
        'algorithm': results['model_spec'].algorithm,
    }
    
    if 'cv_score' in results:
        summary['validation_strategy'] = 'cross_validation'
        summary['cv_score'] = results['cv_score']
        summary['cv_std'] = results['cv_std']
    elif 'test_score' in results:
        summary['validation_strategy'] = 'train_test'
        summary['test_score'] = results['test_score']
    else:
        summary['validation_strategy'] = 'none'
    
    if 'train_metrics' in results:
        summary.update(results['train_metrics'])
    
    return summary

