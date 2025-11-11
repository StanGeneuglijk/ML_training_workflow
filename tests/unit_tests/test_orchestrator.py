"""
Tests for orchestrator module.
"""
import pytest
import numpy as np
import pandas as pd

from src.orchestrator import build_ml_pipeline, run_ml_workflow, get_workflow_summary
from specs import FeatureSpecBuilder, ModelSpecBuilder


class TestBuildMLPipeline:
    """Tests for build_ml_pipeline function."""
    
    def test_build_pipeline(self, simple_feature_specs, simple_model_spec):
        """Test building ML pipeline."""
        pipeline = build_ml_pipeline(simple_feature_specs, simple_model_spec)
        
        assert pipeline is not None
        assert 'preprocessor' in pipeline.named_steps
        assert 'classifier' in pipeline.named_steps
    
    def test_pipeline_structure(self, simple_feature_specs, simple_model_spec):
        """Test pipeline structure."""
        pipeline = build_ml_pipeline(simple_feature_specs, simple_model_spec)
        
        # Check that steps are in correct order
        steps = [name for name, _ in pipeline.steps]
        assert steps == ['preprocessor', 'classifier']


class TestRunMLWorkflow:
    """Tests for run_ml_workflow function."""
    
    def test_cross_validation(self, sample_classification_array):
        """Test workflow with cross-validation."""
        X, y = sample_classification_array
        
        # Create specs
        feature_builder = FeatureSpecBuilder()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        feature_specs = feature_builder.add_numeric_group(feature_names).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_classifier",
            hyperparameters={"n_estimators": 10}
        ).build()[0]
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3},
            random_state=42
        )
        
        assert 'pipeline' in results
        assert 'cv_score' in results
        assert 'cv_std' in results
        assert results['cv_score'] >= 0.0 and results['cv_score'] <= 1.0
    
    def test_train_test_split(self, sample_classification_array):
        """Test workflow with train/test split."""
        X, y = sample_classification_array
        
        feature_builder = FeatureSpecBuilder()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        feature_specs = feature_builder.add_numeric_group(feature_names).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_classifier",
            hyperparameters={"n_estimators": 10}
        ).build()[0]
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="train_test",
            test_size=0.3,
            random_state=42
        )
        
        assert 'pipeline' in results
        assert 'test_score' in results
        assert 'train_size' in results
        assert 'test_size' in results
        assert results['test_score'] >= 0.0 and results['test_score'] <= 1.0
    
    def test_no_validation(self, sample_classification_array):
        """Test workflow without validation."""
        X, y = sample_classification_array
        
        feature_builder = FeatureSpecBuilder()
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        feature_specs = feature_builder.add_numeric_group(feature_names).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_classifier",
            hyperparameters={"n_estimators": 10}
        ).build()[0]
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X,
            y=y,
            validation_strategy="none",
            random_state=42
        )
        
        assert 'pipeline' in results
        assert 'cv_score' not in results
        assert 'test_score' not in results
    
    def test_dataframe_input(self, sample_classification_data):
        """Test workflow with DataFrame input."""
        X_df, y_series = sample_classification_data
        
        feature_builder = FeatureSpecBuilder()
        feature_specs = feature_builder.add_numeric_group(
            X_df.columns.tolist()
        ).build()
        
        model_spec = ModelSpecBuilder().add_classifier(
            name="test_classifier",
            hyperparameters={"n_estimators": 10}
        ).build()[0]
        
        results = run_ml_workflow(
            feature_specs=feature_specs,
            model_spec=model_spec,
            X=X_df,
            y=y_series,
            validation_strategy="cross_validation",
            validation_params={"cv_folds": 3}
        )
        
        assert 'cv_score' in results


class TestGetWorkflowSummary:
    """Tests for get_workflow_summary function."""
    
    def test_summary_with_cv(self):
        """Test summary with cross-validation results."""
        results = {
            'model_spec': ModelSpecBuilder().add_classifier("test").build()[0],
            'cv_score': 0.85,
            'cv_std': 0.05,
            'train_metrics': {'accuracy': 0.87}
        }
        
        summary = get_workflow_summary(results)
        
        assert summary['model_name'] == 'test'
        assert summary['validation_strategy'] == 'cross_validation'
        assert summary['cv_score'] == 0.85
        assert summary['cv_std'] == 0.05
        assert summary['accuracy'] == 0.87
    
    def test_summary_with_test(self):
        """Test summary with test split results."""
        results = {
            'model_spec': ModelSpecBuilder().add_classifier("test").build()[0],
            'test_score': 0.83,
            'train_metrics': {}
        }
        
        summary = get_workflow_summary(results)
        
        assert summary['validation_strategy'] == 'train_test'
        assert summary['test_score'] == 0.83
    
    def test_summary_no_validation(self):
        """Test summary without validation."""
        results = {
            'model_spec': ModelSpecBuilder().add_classifier("test").build()[0],
            'train_metrics': {'accuracy': 0.90}
        }
        
        summary = get_workflow_summary(results)
        
        assert summary['validation_strategy'] == 'none'
        assert 'cv_score' not in summary
        assert 'test_score' not in summary

