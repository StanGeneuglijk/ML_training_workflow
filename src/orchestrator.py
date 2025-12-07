"""
Orchestrator module
"""
from __future__ import annotations

import logging
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass, field
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

from feature_store.data_sources import create_file_source
from feature_store.feature_definitions import create_entity, create_feature_view, create_schema
from data.delta_lake import get_delta_path
from feast import Field
from feast.types import Float64, Int64
from feast.data_format import DeltaFormat

logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


@dataclass
class WorkflowResults:
    pipeline: Any
    model_spec: ClassifierModelSpec
    feature_specs: list[FeatureSpec]
    mlflow: dict[str, Any] = field(default_factory=dict)
    feature_store: dict[str, Any] = field(default_factory=dict)
    tuning_summary: Optional[dict] = None
    calibration_summary: Optional[dict] = None
    validation: dict[str, Any] = field(default_factory=dict)
    train_metrics: Optional[dict] = None

    def to_dict(self) -> dict:
        """
        Convert the workflow results dataclass into the legacy dictionary shape.

        Args:
            None
        Returns:
            dict: Aggregated workflow outputs in dictionary form.
        """
        data = {
            'pipeline': self.pipeline,
            'model_spec': self.model_spec,
            'feature_specs': self.feature_specs,
        }
        data.update(self.feature_store)
        data.update(self.mlflow)
        data.update(self.validation)

        if self.tuning_summary:
            data['tuning_summary'] = self.tuning_summary
        if self.calibration_summary:
            data['calibration_summary'] = self.calibration_summary
        if self.train_metrics:
            data['train_metrics'] = self.train_metrics

        return data


def _prepare_feature_store_data(
    feature_store_spec: Optional[FeatureStoreSpec],
    X,
    y
) -> tuple[Optional[FeastManager], Any, Any, dict[str, Any]]:
    """
    Initialize the feature store and optionally hydrate X, y from it.

    Args:
        feature_store_spec: Feature store configuration, possibly disabled.
        X: Fallback feature matrix if Feast is unavailable.
        y: Fallback target vector if Feast is unavailable.

    Returns:
        Tuple containing the manager, feature matrix, target vector, and metadata about the feature store.
    """
    if feature_store_spec is None or not feature_store_spec.enabled:
        return None, X, y, {}

    feast_manager: Optional[FeastManager] = None
    feature_store_info: dict[str, Any] = {}

    try:
        logger.info("Initializing feature store")
        feast_manager = create_feast_manager(
            repo_path=feature_store_spec.repo_path,
            project_name=feature_store_spec.project_name,
            initialize=feature_store_spec.should_initialize(),
            offline_store_type=feature_store_spec.offline_store_type,
        )
        logger.info("Feature store initialized successfully at %s", feast_manager.repo_path)

        if feature_store_spec.should_initialize():
            logger.info("Registering feature view: %s", feature_store_spec.feature_view_name)
            entity = create_entity(
                name="sample",
                join_keys=["sample_index"]
            )
            feast_manager.store.apply([entity])

            delta_path = get_delta_path(feature_store_spec.dataset_name)
            data_source = create_file_source(
                path=delta_path,
                timestamp_field="ingested_at",
                file_format=DeltaFormat()
            )

            n_features = feature_store_spec.get_n_features()
            feature_names = [f"feature_{i}" for i in range(n_features)]

            schema = create_schema(
                feature_names,
                default_type=Float64
            )
            schema.append(
                Field(name="target", dtype=Int64)
            )

            feature_view = create_feature_view(
                view_name=feature_store_spec.feature_view_name,
                source=data_source,
                schema=schema,
                entity=entity
            )
            feast_manager.register_feature_view(feature_view)

        if feature_store_spec.sample_indices is None:
            sample_indices = feast_manager.get_entity_values(
                entity_column="sample_index",
                dataset_name=feature_store_spec.dataset_name
            )
        else:
            sample_indices = feature_store_spec.sample_indices

        if feature_store_spec.timestamp:
            event_timestamp = feature_store_spec.timestamp
        else:
            from deltalake import DeltaTable
            delta_path = get_delta_path(feature_store_spec.dataset_name)
            dt = DeltaTable(str(delta_path))
            data_df = dt.to_pandas()
            max_timestamp = data_df['ingested_at'].max()
            event_timestamp = max_timestamp + pd.Timedelta(seconds=1)

        entity_df = pd.DataFrame({
            "sample_index": sample_indices,
            "event_timestamp": [event_timestamp] * len(sample_indices)
        })

        n_features = feature_store_spec.get_n_features()
        feature_names = [f"feature_{i}" for i in range(n_features)]

        logger.info("Retrieving features from feature store")
        X, y = feast_manager.get_training_data(
            entity_df=entity_df,
            feature_names=feature_names,
            target_name="target",
            feature_view_name=feature_store_spec.feature_view_name,
            full_feature_names=feature_store_spec.use_full_feature_names
        )
        logger.info("Retrieved features from feature store: X shape %s, y shape %s", X.shape, y.shape)

        feature_store_info = {
            'feature_store_enabled': True,
            'feature_store_repo': str(feast_manager.repo_path)
        }
    except Exception as e:
        logger.warning("Failed to initialize feature store: %s. Falling back to provided X, y.", e)
        feast_manager = None
        if X is None or y is None:
            raise ValueError(
                "Feature store initialization failed and no fallback X, y provided"
            ) from e

    return feast_manager, X, y, feature_store_info


def _start_mlflow_tracking(
    mlflow_spec: Optional[MLflowSpec],
    model_spec: ClassifierModelSpec,
    validation_strategy: str
) -> tuple[Optional[MLflowTracker], dict[str, Any]]:
    """
    Initialize an MLflow tracker if tracking is enabled.

    Args:
        mlflow_spec: MLflow configuration that determines enablement and metadata.
        model_spec: Model specification used to build default run names/tags.
        validation_strategy: Validation strategy name for tagging.

    Returns:
        Tuple of optional MLflowTracker and metadata dict describing the run.
    """
    if mlflow_spec is None or not mlflow_spec.enabled:
        return None, {}

    try:
        mlflow_tracker = create_mlflow_tracker(
            experiment_name=mlflow_spec.experiment_name,
            tracking_uri=mlflow_spec.tracking_uri
        )

        run_name = mlflow_spec.get_run_name(
            default=f"{model_spec.model_name}_{validation_strategy}"
        )

        workflow_tags = {
            'validation_strategy': validation_strategy,
            'model_type': model_spec.get_model_type() if hasattr(model_spec, 'get_model_type') else 'unknown',
            'algorithm': model_spec.algorithm
        }
        run_tags = mlflow_spec.get_run_tags(workflow_tags)

        mlflow_run_id = mlflow_tracker.start_run(
            run_name=run_name,
            tags=run_tags
        )
        logger.info("Started MLflow run: %s (ID: %s)", run_name, mlflow_run_id)
        return mlflow_tracker, {
            'mlflow_run_id': mlflow_tracker.run_id,
            'mlflow_experiment_id': mlflow_tracker.experiment_id
        }
    except Exception as e:
        logger.warning("Failed to initialize MLflow: %s. Continuing without MLflow tracking.", e)
        return None, {}


def _run_parameter_tuning(
    tuning_spec: Optional[ParamTuningSpec],
    pipeline: SklearnPipeline,
    X_array,
    y_array
) -> tuple[SklearnPipeline, Optional[dict]]:
    """
    Execute the parameter tuning stage, returning the best estimator and summary.

    Args:
        tuning_spec: Hyperparameter tuning configuration, may be disabled.
        pipeline: Current sklearn pipeline instance targeted for tuning.
        X_array: Validated feature matrix.
        y_array: Validated target vector.

    Returns:
        Tuple of possibly pipeline and optional tuning summary dict.
    """
    if tuning_spec is None or not tuning_spec.enabled:
        return pipeline, None

    try:
        logger.info("Running parameter tuning: %s", tuning_spec.tuning_name)
        tuning = create_tuning(tuning_spec)
        tuning.fit(pipeline, X_array, y_array)
        summary = tuning.get_results_summary()
        pipeline = tuning.get_best_estimator()
        logger.info("Parameter tuning completed with best score: %.4f", summary.get('best_score', float('nan')))
        return pipeline, summary
    except Exception as e:
        logger.warning("Parameter tuning failed (%s). Proceeding without tuning.", e)
        return pipeline, None


def _run_calibration(
    calibration_spec: Optional[CalibrationSpec],
    pipeline: SklearnPipeline,
    X_array,
    y_array
) -> tuple[SklearnPipeline, Optional[dict]]:
    """
    Perform probability calibration if requested.

    Args:
        calibration_spec: Calibration configuration, may be disabled.
        pipeline: Fitted pipeline ready for calibration.
        X_array: Feature matrix for calibration fitting.
        y_array: Target vector for calibration fitting.

    Returns:
        Tuple of calibrated pipeline and optional calibration summary.
    """
    if calibration_spec is None or not calibration_spec.enabled:
        return pipeline, None

    try:
        logger.info("Running calibration: %s", calibration_spec.calibration_name)
        calibration = create_calibration(calibration_spec)
        calibration.fit(pipeline, X_array, y_array)
        summary = calibration.get_results_summary()
        logger.info("Calibration completed. Calibrated models: %s", summary.get('n_calibrated_models'))
        return calibration.calibrator, summary
    except Exception as e:
        logger.warning("Calibration failed (%s). Proceeding without calibration.", e)
        return pipeline, None


def _run_validation(
    validation_strategy: str,
    pipeline: SklearnPipeline,
    X_array,
    y_array,
    validation_params: Optional[dict],
    test_size: float,
    random_state: int
) -> tuple[SklearnPipeline, dict[str, Any]]:
    """
    Execute the requested validation strategy and collect metrics.

    Args:
        validation_strategy: Validation strategy name.
        pipeline: Pipeline to evaluate.
        X_array: Feature matrix for evaluation.
        y_array: Target vector for evaluation.
        validation_params: Optional strategy-specific parameters.
        test_size: Fractional test size for train/test splits.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of pipeline and a dict of validation metrics/results.
    """
    if validation_strategy == "cross_validation":
        cv_folds = validation_params.get('cv_folds', 5) if validation_params else 5
        logger.info("Running %s-fold cross-validation", cv_folds)
        cv_scores = cross_val_score(
            pipeline, X_array, y_array,
            cv=cv_folds,
            scoring='accuracy'
        )
        return pipeline, {
            'validation_strategy': 'cross_validation',
            'cv_score': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'cv_scores': cv_scores
        }

    if validation_strategy == "train_test":
        logger.info("Splitting data into train/test sets")
        X_train, X_test, y_train, y_test = train_test_split(
            X_array, y_array,
            test_size=test_size,
            random_state=random_state,
            stratify=y_array
        )
        pipeline.fit(X_train, y_train)
        test_score = pipeline.score(X_test, y_test)
        return pipeline, {
            'validation_strategy': 'train_test',
            'test_score': test_score,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }

    logger.info("No validation strategy matched. Skipping validation.")
    return pipeline, {'validation_strategy': 'none'}


def _extract_training_metrics(
    pipeline: SklearnPipeline
) -> Optional[dict]:
    """
    Retrieve training metrics stored on the classifier if they exist.

    Args:
        pipeline: Pipeline or wrapped estimator potentially holding metrics.

    Returns:
        Optional dict containing metric names/values collected during training.
    """
    if hasattr(pipeline, 'named_steps') and 'classifier' in pipeline.named_steps:
        classifier = pipeline.named_steps['classifier']
        if hasattr(classifier, 'evaluation_metrics_'):
            return classifier.evaluation_metrics_
    if hasattr(pipeline, 'estimator') and hasattr(pipeline.estimator, 'named_steps'):
        classifier = pipeline.estimator.named_steps.get('classifier')
        if classifier and hasattr(classifier, 'evaluation_metrics_'):
            return classifier.evaluation_metrics_
    return None


def _log_mlflow_artifacts(
    mlflow_tracker: MLflowTracker,
    mlflow_spec: MLflowSpec,
    results: dict,
    X_array
) -> None:
    """
    Send metrics, model artifacts, and signature samples to MLflow.

    Args:
        mlflow_tracker: Active MLflow tracker/run handle.
        mlflow_spec: MLflow configuration controlling logging behavior.
        results: Workflow results dictionary to be logged.
        X_array: Training data used for optional signature/input examples.

    Returns:
        None
    """
    try:
        X_sample = None
        if mlflow_spec.log_model_signature or mlflow_spec.log_input_example:
            X_sample = X_array[:min(100, len(X_array))]

        mlflow_tracker.log_workflow_results(
            results=results,
            X_sample=X_sample,
            register_model=mlflow_spec.should_register_model()
        )

        mlflow_tracker.end_run(status="FINISHED")
        logger.info("MLflow tracking completed successfully")
    except Exception as e:
        logger.warning("MLflow logging failed: %s", e)
        if mlflow_tracker and mlflow_tracker.run_id:
            mlflow_tracker.end_run(status="FAILED")


def build_ml_pipeline(
    feature_specs: list[FeatureSpec],
    model_spec: ClassifierModelSpec
) -> SklearnPipeline:
    """
    Build sklearn-compatible pipeline from specifications.
    
    Args:
        feature_specs: List of feature specifications
        model_spec: Model specification
        
    Returns:
        Sklearn pipeline with preprocessing and model
        
    Eg.:
        >>> from specs import FeatureSpecBuilder, ModelSpecBuilder
        >>> feature_specs = FeatureSpecBuilder().add_numeric_group(['age', 'income']).build()
        >>> model_spec = ModelSpecBuilder().add_classifier('gb_classifier').build()[0]
        >>> pipeline = build_ml_pipeline(feature_specs, model_spec)
        >>> pipeline.fit(X_train, y_train)
    """
    preprocessor = FeatureSpecPipeline(feature_specs)
    
    classifier = GradientBoostingClassifierImpl(model_spec)
    
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
    Run the complete workflow: build pipeline, train, validate.
    
    Args:
        feature_specs: List of feature specifications
        model_spec: Model specification
        X: Feature matrix 
        y: Target vector 
        validation_strategy: Validation approach
        validation_params: Parameters for validation 
        test_size: Test set size 
        random_state: Random state 
        X_test: Optional test features 
        y_test: Optional test labels 
        tuning_spec: Optional parameter tuning specification
        calibration_spec: Optional calibration specification
        mlflow_spec: Optional MLflow tracking and registry specification
        feature_store_spec: Optional feature store specification
        
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

    _feast_manager, X, y, feature_store_info = _prepare_feature_store_data(
        feature_store_spec=feature_store_spec,
        X=X,
        y=y
    )

    mlflow_tracker, mlflow_info = _start_mlflow_tracking(
        mlflow_spec=mlflow_spec,
        model_spec=model_spec,
        validation_strategy=validation_strategy
    )

    # --- Data validation & preprocessing ---
    X_array, y_array = utils.validate_training_data(X, y)
    logger.info("Validated training data: X shape %s, y shape %s", X_array.shape, y_array.shape)

    # --- Model training pipeline construction ---
    pipeline = build_ml_pipeline(feature_specs, model_spec)
    logger.info("Built ML pipeline: %s", model_spec.model_name)

    results = WorkflowResults(
        pipeline=pipeline,
        model_spec=model_spec,
        feature_specs=feature_specs,
        mlflow=mlflow_info,
        feature_store=feature_store_info
    )

    # --- Parameter tuning ---
    pipeline, tuning_summary = _run_parameter_tuning(
        tuning_spec=tuning_spec,
        pipeline=pipeline,
        X_array=X_array,
        y_array=y_array
    )
    results.pipeline = pipeline
    results.tuning_summary = tuning_summary

    logger.info("Fitting pipeline on training data")
    pipeline.fit(X_array, y_array)

    # --- Calibration ---
    pipeline, calibration_summary = _run_calibration(
        calibration_spec=calibration_spec,
        pipeline=pipeline,
        X_array=X_array,
        y_array=y_array
    )
    results.pipeline = pipeline
    results.calibration_summary = calibration_summary

    # --- Validation ---
    pipeline, validation_results = _run_validation(
        validation_strategy=validation_strategy,
        pipeline=pipeline,
        X_array=X_array,
        y_array=y_array,
        validation_params=validation_params,
        test_size=test_size,
        random_state=random_state
    )
    results.pipeline = pipeline
    results.validation = validation_results

    results.train_metrics = _extract_training_metrics(pipeline)

    final_results = results.to_dict()

    # --- Artifact logging (metrics, models, params) ---
    if mlflow_tracker and mlflow_spec:
        final_results['mlflow_spec'] = mlflow_spec
        _log_mlflow_artifacts(
            mlflow_tracker=mlflow_tracker,
            mlflow_spec=mlflow_spec,
            results=final_results,
            X_array=X_array
        )

    logger.info("ML workflow completed successfully")
    return final_results


def get_workflow_summary(
    results: dict
) -> dict:
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

