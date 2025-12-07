"""
MLflow tracking and model registry module
"""
from __future__ import annotations

import os
import logging
import tempfile
from typing import Any, Dict, Optional, List
from datetime import datetime
import numpy as np
import pandas as pd

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.exceptions import MlflowException

import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import utils
logger = utils.setup_logging(level=logging.INFO, logger_name=__name__)


class MLflowTracker:
    """
    MLflow tracker for experiment tracking and model registry.
    """
    
    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "ml_workflow_experiments",
        registry_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ) -> None:
        """
        Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI 
            experiment_name: Name of the experiment
            registry_uri: Model registry server URI 
            artifact_location: Custom artifact storage server URI location

        Returns:
            None
        """
        if tracking_uri is None:
            tracking_uri = f"file://{os.path.join(parent_dir, "mlruns")}" 
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI set to: {tracking_uri}")
        
        if registry_uri is None:
            registry_uri = f"file://{os.path.join(parent_dir, 'mlruns', 'registry')}"
        mlflow.set_registry_uri(registry_uri)
        logger.info(f"MLflow registry URI set to: {registry_uri}")
        
        self.client = MlflowClient(
            tracking_uri=tracking_uri, 
            registry_uri=registry_uri
        )
        logger.info(f"MLflow client initialized")
        
        self.experiment_name = experiment_name

        try:
            self.experiment = self.client.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                if artifact_location:
                    experiment_id = self.client.create_experiment(
                        experiment_name,
                        artifact_location=artifact_location
                    )
                else:
                    experiment_id = self.client.create_experiment(experiment_name)
                self.experiment = self.client.get_experiment(experiment_id)
                logger.info(f"Created new experiment: {experiment_name}")
            else:
                logger.info(f"Using existing experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Failed to create/get experiment: {e}")
            raise
        
        self.experiment_id = self.experiment.experiment_id

        self.run_id = None

        self.run_name = None
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Start a new MLflow run under the tracker's configurated experiment.
        
        Args:
            run_name: Optional name for the run
            tags: Optional dictionary of tags
            
        Returns:
            Run ID string
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run_name = run_name

        run = self.client.create_run(
            experiment_id=self.experiment_id,
            tags=tags or {}
        )
        
        self.run_id = run.info.run_id
        
        self.client.set_tag(
            run_id=self.run_id, 
            key="mlflow.runName", 
            value=run_name
        )
        logger.info(f"Started MLflow run: {run_name} (ID: {self.run_id})")

        return self.run_id
    
    def end_run(
        self, 
        status: str = "FINISHED"
    ) -> None:
        """
        End the current MLflow run.
        
        Args:
            status: Run status ('FINISHED', 'FAILED', 'KILLED')

        Returns:
            None
        """
        if self.run_id:
            self.client.set_terminated(self.run_id, status)
            logger.info(f"Ended MLflow run: {self.run_name} with status: {status}")

            self.run_id = None

            self.run_name = None
    
    def log_params(
        self, 
        params: Dict[str, Any]
    ) -> None:
        """
        Log parameters to the current run.
        
        Args:
            params: Dictionary of parameters

        Returns:
            None
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        for key, value in params.items():
            try:
                self.client.log_param(self.run_id, key, str(value))
            except Exception as e:
                logger.warning(f"Failed to log parameter {key}: {e}")
        
        logger.debug(f"Logged {len(params)} parameters")
    
    def log_metric(
        self, 
        key: str, 
        value: float, 
        step: Optional[int] = None
    ) -> None:
        """
        Log a single metric value.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number for tracking metric evolution
        
        Returns:
            None
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            timestamp = int(datetime.now().timestamp() * 1000)
            self.client.log_metric(self.run_id, key, float(value), timestamp, step or 0)
        except Exception as e:
            logger.warning(f"Failed to log metric {key}: {e}")
    
    def log_metrics(
        self, 
        metrics: Dict[str, float], 
        step: Optional[int] = None
    ) -> None:
        """
        Log multiple metrics.
        
        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Optional step number

        Returns:
            None
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        for key, value in metrics.items():
            try:
                if isinstance(value, (np.integer, np.floating)):
                    value = float(value)
                elif isinstance(value, (list, np.ndarray)):
                    value = float(np.mean(value))
                
                self.log_metric(key, value, step)
            except Exception as e:
                logger.warning(f"Failed to log metric {key}: {e}")
        
        logger.debug(f"Logged {len(metrics)} metrics")
    
    def log_artifact(
        self, 
        local_path: str, 
        artifact_path: Optional[str] = None
    ) -> None:
        """
        Log a local file as an artifact.
        
        Args:
            local_path: Path to local file
            artifact_path: Optional subdirectory in artifact storage

        Returns:
            None
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            self.client.log_artifact(self.run_id, local_path, artifact_path)
            logger.debug(f"Logged artifact: {local_path}")
        except Exception as e:
            logger.warning(f"Failed to log artifact {local_path}: {e}")
    
    def log_dict_as_artifact(
        self, 
        data: Dict[str, Any], 
        filename: str
    ) -> None:
        """
        Log a dictionary as a JSON artifact.
        
        Args:
            data: Dictionary to log
            filename: Name for the artifact file

        Returns:
            None
        """
        import json
        
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name
            
            self.client.log_artifact(self.run_id, temp_path, artifact_path=None)
            
            os.unlink(temp_path)
            
            logger.debug(f"Logged dictionary as artifact: {filename}")
        except Exception as e:
            logger.warning(f"Failed to log dictionary artifact: {e}")
    
    def log_model_architecture(
        self, 
        pipeline: Any, 
        model_spec: Any
    ) -> None:
        """
        Log model architecture details.
        
        Args:
            pipeline: Scikit-learn pipeline
            model_spec: Model specification object

        Returns:
            None
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            architecture = {
                'model_name': getattr(model_spec, 'model_name', 'unknown'),
                'algorithm': getattr(model_spec, 'algorithm', 'unknown'),
                'model_type': model_spec.get_model_type() if hasattr(model_spec, 'get_model_type') else 'unknown',
                'pipeline_steps': []
            }
            
            if hasattr(pipeline, 'steps'):
                for step_name, estimator in pipeline.steps:
                    step_info = {
                        'name': step_name,
                        'type': estimator.__class__.__name__,
                        'module': estimator.__class__.__module__
                    }
                    
                    if hasattr(estimator, 'get_params'):
                        try:
                            params = estimator.get_params(deep=False)
                            step_info['parameters'] = {
                                k: str(v) for k, v in params.items()
                                if v is not None and not callable(v)
                            }
                        except:
                            pass
                    
                    architecture['pipeline_steps'].append(step_info)
            
            if hasattr(model_spec, 'hyperparameters'):
                architecture['hyperparameters'] = model_spec.hyperparameters
            
            if hasattr(model_spec, 'evaluation_metrics'):
                architecture['evaluation_metrics'] = model_spec.evaluation_metrics
            
            self.log_dict_as_artifact(architecture, 'model_architecture.json')
            
            self.client.set_tag(self.run_id, "model.name", architecture['model_name'])
            self.client.set_tag(self.run_id, "model.algorithm", architecture['algorithm'])
            self.client.set_tag(self.run_id, "model.type", architecture['model_type'])
            
            logger.info(f"Logged model architecture for {architecture['model_name']}")
            
        except Exception as e:
            logger.warning(f"Failed to log model architecture: {e}")

    def log_sklearn_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        X_sample: Optional[Any] = None,
        y_sample: Optional[Any] = None,
        model_spec: Optional[Any] = None,
        use_pyfunc: bool = True
    ) -> None:
        """
        Log a model pipeline with MLflow.
        
        Args:
            model: Trained pipeline
            artifact_path: Path within artifact storage
            registered_model_name: Name for model registry 
            X_sample: Sample input data for signature inference
            y_sample: Sample output data for signature inference
            model_spec: Model specification for additional metadata
            use_pyfunc: If True, use pyfunc flavor with code packaging 
        
        Returns:
            None
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            signature = None
            if X_sample is not None:
                try:
                    if hasattr(model, 'predict_proba'):
                        y_pred = model.predict_proba(X_sample)
                    else:
                        y_pred = model.predict(X_sample)
                    
                    signature = infer_signature(X_sample, y_pred)
                    logger.debug("Inferred model signature")
                except Exception as e:
                    logger.warning(f"Failed to infer signature: {e}")
            
            input_example = None
            if X_sample is not None:
                try:
                    if isinstance(X_sample, pd.DataFrame):
                        input_example = X_sample.head(3)
                    elif isinstance(X_sample, np.ndarray):
                        input_example = X_sample[:3]
                except:
                    pass
            
            if use_pyfunc:
                logger.info("Logging model with code packaging (pyfunc-compatible)")
                
                code_paths = []
                
                module_dir = os.path.join(parent_dir, "module")
                if os.path.exists(module_dir):
                    code_paths.append(module_dir)
                
                specs_dir = os.path.join(parent_dir, "specs")
                if os.path.exists(specs_dir):
                    code_paths.append(specs_dir)
                
                utils_file = os.path.join(parent_dir, "utils.py")
                if os.path.exists(utils_file):
                    code_paths.append(utils_file)
                
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name,
                    code_paths=code_paths  
                )
                
                logger.info(f"Model logged with {len(code_paths)} code path(s) included")
            else:
                logger.info("Logging model without code packaging (sklearn flavor only)")
                
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artifact_path,
                    signature=signature,
                    input_example=input_example,
                    registered_model_name=registered_model_name
                )
            
            logger.info(f"Logged sklearn model to artifact path: {artifact_path}")
            
            if registered_model_name:
                logger.info(f"Registered model: {registered_model_name}")
            
        except Exception as e:
            logger.error(f"Failed to log sklearn model: {e}")
            raise
    
    def register_model(
        self,
        model_uri: str,
        name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None
    ) -> str:
        """
        Register a model in MLflow Model Registry.
        
        Args:
            model_uri: URI to the model (e.g., runs:/<run_id>/model)
            name: Name for registered model
            tags: Optional tags for the model version
            description: Optional description
            
        Returns:
            Model version number
        """
        try:
            try:
                self.client.create_registered_model(name, tags, description)
                logger.info(f"Created registered model: {name}")
            except MlflowException:
                logger.debug(f"Registered model {name} already exists")
            
            model_version = self.client.create_model_version(
                name=name,
                source=model_uri,
                run_id=self.run_id,
                tags=tags,
                description=description
            )
            
            version = model_version.version
            logger.info(f"Registered model version {version} for {name}")
            
            return str(version)
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def transition_model_stage(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing_versions: bool = False
    ) -> None:
        """
        Transition a model version to a different stage.
        
        Args:
            name: Registered model name
            version: Model version
            stage: Target stage ('Staging', 'Production', 'Archived')
            archive_existing_versions: Whether to archive existing versions in target stage
        """
        try:
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
                archive_existing_versions=archive_existing_versions
            )
            logger.info(f"Transitioned {name} v{version} to {stage}")
        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise
    
    def get_latest_model_version(
        self, 
        name: str, 
        stage: Optional[str] = None
    ) -> Optional[str]:
        """
        Get the latest version of a registered model.
        
        Args:
            name: Registered model name
            stage: Optional stage filter ('Staging', 'Production', 'Archived')
            
        Returns:
            Latest version number or None
        """
        try:
            if stage:
                versions = self.client.get_latest_versions(name, stages=[stage])
            else:
                versions = self.client.search_model_versions(f"name='{name}'")
            
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                return latest.version
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to get latest model version: {e}")
            return None
    
    def log_workflow_results(
        self,
        results: Dict[str, Any],
        X_sample: Optional[Any] = None,
        register_model: bool = False
    ) -> None:
        """
        Log complete workflow results to MLflow.
        
        Args:
            results: Workflow results dictionary from run_ml_workflow
            X_sample: Sample input data for signature inference
            register_model: Whether to register the model
        """
        if not self.run_id:
            raise ValueError("No active run. Call start_run() first.")
        
        try:
            model_spec = results.get('model_spec')
            model_name = model_spec.model_name if model_spec else "model"
            
            if 'pipeline' in results and model_spec:
                self.log_model_architecture(results['pipeline'], model_spec)
            
            if model_spec and hasattr(model_spec, 'hyperparameters'):
                params = model_spec.hyperparameters.copy()
                params['model_name'] = model_name
                params['algorithm'] = getattr(model_spec, 'algorithm', 'unknown')
                self.log_params(params)
            
            metrics = {}
            
            if 'cv_score' in results:
                metrics['cv_mean_score'] = results['cv_score']
                metrics['cv_std_score'] = results['cv_std']
                if 'cv_scores' in results:
                    for i, score in enumerate(results['cv_scores']):
                        metrics[f'cv_fold_{i+1}_score'] = score
            
            if 'test_score' in results:
                metrics['test_score'] = results['test_score']
            
            if 'train_metrics' in results:
                for key, value in results['train_metrics'].items():
                    metrics[f'train_{key}'] = value
            
            if metrics:
                self.log_metrics(metrics)
            
            if 'pipeline' in results:
                registered_name = f"{model_name}_registered" if register_model else None
                self.log_sklearn_model(
                    model=results['pipeline'],
                    artifact_path="model",
                    registered_model_name=registered_name,
                    X_sample=X_sample,
                    model_spec=model_spec
                )
            
            if 'tuning_summary' in results and results['tuning_summary']:
                self.log_dict_as_artifact(results['tuning_summary'], 'tuning_summary.json')
                if 'best_score' in results['tuning_summary']:
                    self.log_metric('tuning_best_score', results['tuning_summary']['best_score'])
            
            if 'calibration_summary' in results and results['calibration_summary']:
                self.log_dict_as_artifact(results['calibration_summary'], 'calibration_summary.json')
            
            if 'feature_specs' in results:
                try:
                    from dataclasses import asdict, is_dataclass
                    feature_specs_data = []
                    for fs in results['feature_specs']:
                        if is_dataclass(fs):
                            feature_specs_data.append(asdict(fs))
                    
                    if feature_specs_data:
                        self.log_dict_as_artifact(
                            {'feature_specs': feature_specs_data},
                            'feature_specs.json'
                        )
                except:
                    pass
            
            logger.info("Successfully logged workflow results to MLflow")
            
        except Exception as e:
            logger.error(f"Failed to log workflow results: {e}")
            raise


def create_mlflow_tracker(
    experiment_name: str = "ml_workflow_experiments",
    tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """
    Factory function to create MLflow tracker.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: Optional tracking URI
        
    Returns:
        Configured MLflowTracker instance
    """
    return MLflowTracker(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name
    )

def promote_model(
    model_name: str,
    stage: str,
    version: Optional[str] = None,
    tracking_uri: Optional[str] = None
) -> bool:
    """
    Promote a model to a specific stage.
    
    Args:
        model_name: Name of the registered model
        stage: Target stage (Production, Staging, Archived, None)
        version: Specific version to promote 
        tracking_uri: MLflow tracking URI
        
    Returns:
        True if successful, False otherwise
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        client = MlflowClient(tracking_uri=tracking_uri)
        
        if version is None:
            versions = client.search_model_versions(f"name='{model_name}'")
            if not versions:
                logger.error(f"Model '{model_name}' not found in registry")
                return False
            
            versions = sorted(versions, key=lambda x: int(x.version), reverse=True)
            version = versions[0].version
            logger.info(f"Using latest version: {version}")
        
        model_version = client.get_model_version(model_name, version)
        current_stage = model_version.current_stage
        
        if current_stage == stage:
            logger.info(f"Model is already in {stage} stage")
            return True
        
        logger.info(f"Promoting {model_name} v{version}: {current_stage} → {stage}")
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        logger.info(f"Successfully promoted {model_name} v{version} to {stage}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        return False


def list_registered_models(
    tracking_uri: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List all registered models.
    
    Args:
        tracking_uri: MLflow tracking URI
        
    Returns:
        List of dictionaries with model information
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        client = MlflowClient(tracking_uri=tracking_uri)
        models = client.search_registered_models()
        
        model_list = []
        for model in models:
            model_info = {
                'name': model.name,
                'description': model.description
            }
            
            versions = client.get_latest_versions(model.name)
            model_info['latest_versions'] = [
                {
                    'version': v.version,
                    'stage': v.current_stage,
                    'run_id': v.run_id
                }
                for v in versions
            ]
            
            model_list.append(model_info)
        
        return model_list
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []


# CLI interface for model registry management
if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="MLflow Model Management Utilities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all models
  python -m module.mlflow_tracker --list
  
  # Promote model to Production
  python -m module.mlflow_tracker --promote my_classifier --stage Production
  
  # Promote specific version
  python -m module.mlflow_tracker --promote my_classifier --version 2 --stage Production
        """
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all registered models'
    )
    parser.add_argument(
        '--promote',
        metavar='MODEL_NAME',
        help='Promote a model to a stage'
    )
    parser.add_argument(
        '--stage',
        choices=['Production', 'Staging', 'Archived', 'None'],
        help='Target stage for promotion'
    )
    parser.add_argument(
        '--version',
        help='Model version (default: latest)'
    )
    parser.add_argument(
        '--tracking-uri',
        help='MLflow tracking URI'
    )
    
    args = parser.parse_args()
    
    if args.list:
        print("\n" + "=" * 60)
        print("Registered Models")
        print("=" * 60)
        
        models = list_registered_models(tracking_uri=args.tracking_uri)
        
        if not models:
            print("\nNo models found in registry")
        else:
            for model in models:
                print(f"\nModel: {model['name']}")
                if model.get('description'):
                    print(f"  Description: {model['description']}")
                for version in model['latest_versions']:
                    print(f"  v{version['version']}: {version['stage']}")
        
        sys.exit(0)
    
    elif args.promote:
        if not args.stage:
            print("ERROR: --stage is required for promotion")
            parser.print_help()
            sys.exit(1)
        
        print("\n" + "=" * 60)
        print("MLflow Model Promotion")
        print("=" * 60)
        print(f"\nModel: {args.promote}")
        print(f"Target Stage: {args.stage}")
        if args.version:
            print(f"Version: {args.version}")
        
        success = promote_model(
            model_name=args.promote,
            stage=args.stage,
            version=args.version,
            tracking_uri=args.tracking_uri
        )
        
        if success:
            print("\n✓ Promotion successful!")
            sys.exit(0)
        else:
            print("\n✗ Promotion failed")
            sys.exit(1)
    
    else:
        parser.print_help()
        sys.exit(1)

