"""
Tests for model specifications.
"""
import pytest
from pydantic import ValidationError

from specs.model_spec import (
    ModelSpec,
    ClassifierModelSpec,
    ModelSpecBuilder
)


class TestModelSpec:
    """Tests for ModelSpec base class."""
    
    def test_abstract_class(self):
        """Test that ModelSpec is abstract."""
        with pytest.raises(TypeError):
            ModelSpec(model_name="test")
    
    def test_model_name_validation(self):
        """Test model name validation."""
        # Valid name
        spec = ClassifierModelSpec(model_name="valid_name")
        assert spec.model_name == "valid_name"
        
        # Invalid: empty string
        with pytest.raises(ValidationError):
            ClassifierModelSpec(model_name="")
        
        # Invalid: not a string
        with pytest.raises(ValidationError):
            ClassifierModelSpec(model_name=123)
    
    def test_metadata(self):
        """Test metadata handling."""
        spec = ClassifierModelSpec(
            model_name="test",
            metadata={"key": "value"}
        )
        assert spec.metadata["key"] == "value"
    
    def test_random_state(self):
        """Test random state handling."""
        spec = ClassifierModelSpec(model_name="test", random_state=42)
        assert spec.random_state == 42
        
        spec_no_seed = ClassifierModelSpec(model_name="test2", random_state=None)
        assert spec_no_seed.random_state is None


class TestClassifierModelSpec:
    """Tests for ClassifierModelSpec."""
    
    def test_default_algorithm(self):
        """Test default algorithm."""
        spec = ClassifierModelSpec(model_name="test")
        assert spec.algorithm == "gradient_boosting"
    
    def test_default_hyperparameters(self):
        """Test default hyperparameters are empty dict."""
        spec = ClassifierModelSpec(model_name="test")
        assert isinstance(spec.hyperparameters, dict)
        # Hyperparameters are set in the classifier module, not in the spec
        assert spec.hyperparameters == {}
    
    def test_custom_hyperparameters(self):
        """Test custom hyperparameters."""
        spec = ClassifierModelSpec(
            model_name="test",
            hyperparameters={"n_estimators": 50, "learning_rate": 0.05}
        )
        assert spec.hyperparameters["n_estimators"] == 50
        assert spec.hyperparameters["learning_rate"] == 0.05
    
    def test_invalid_algorithm(self):
        """Test that invalid algorithm raises error."""
        with pytest.raises(ValueError, match="Only 'gradient_boosting'"):
            ClassifierModelSpec(
                model_name="test",
                algorithm="invalid_algorithm"
            )
    
    def test_invalid_learning_rate(self):
        """Test that invalid learning rate raises error."""
        with pytest.raises(ValueError, match="learning_rate"):
            ClassifierModelSpec(
                model_name="test",
                hyperparameters={"learning_rate": 1.5}  # > 1.0
            )
    
    def test_invalid_n_estimators(self):
        """Test that invalid n_estimators raises error."""
        with pytest.raises(ValueError, match="n_estimators"):
            ClassifierModelSpec(
                model_name="test",
                hyperparameters={"n_estimators": -1}
            )
    
    def test_evaluation_metrics(self):
        """Test evaluation metrics."""
        spec = ClassifierModelSpec(model_name="test")
        assert "accuracy" in spec.evaluation_metrics
        assert "roc_auc" in spec.evaluation_metrics
    
    def test_invalid_metric(self):
        """Test that invalid metric raises error."""
        with pytest.raises(ValueError, match="Invalid metric"):
            ClassifierModelSpec(
                model_name="test",
                evaluation_metrics=["invalid_metric"]
            )
    
    def test_get_model_type(self):
        """Test get_model_type method."""
        spec = ClassifierModelSpec(model_name="test")
        assert spec.get_model_type() == "classifier"


class TestModelSpecBuilder:
    """Tests for ModelSpecBuilder."""
    
    def test_build_empty(self):
        """Test building empty spec list."""
        builder = ModelSpecBuilder()
        specs = builder.build()
        assert len(specs) == 0
        assert isinstance(specs, list)
    
    def test_add_classifier(self):
        """Test adding classifier specification."""
        builder = ModelSpecBuilder()
        builder.add_classifier("test_classifier")
        specs = builder.build()
        
        assert len(specs) == 1
        assert isinstance(specs[0], ClassifierModelSpec)
        assert specs[0].model_name == "test_classifier"
    
    def test_add_multiple_classifiers(self):
        """Test adding multiple classifiers."""
        builder = ModelSpecBuilder()
        builder.add_classifier("classifier1")
        builder.add_classifier("classifier2", hyperparameters={"n_estimators": 50})
        specs = builder.build()
        
        assert len(specs) == 2
        assert specs[0].model_name == "classifier1"
        assert specs[1].model_name == "classifier2"
        assert specs[1].hyperparameters["n_estimators"] == 50
    
    def test_fluent_interface(self):
        """Test builder fluent interface."""
        specs = (ModelSpecBuilder()
                .add_classifier("model1")
                .add_classifier("model2")
                .build())
        
        assert len(specs) == 2

