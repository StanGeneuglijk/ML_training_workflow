"""
Tests for application module.
"""
import pytest
import os
import sys
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestApplication:
    """Tests for application main function."""
    
    @patch('application_training.load_data')
    @patch('application_training.run_ml_workflow')
    @patch('application_training.get_workflow_summary')
    def test_main_success(self, mock_summary, mock_workflow, mock_load_data):
        """Test successful application run."""
        import application_training
        
        # Mock data loading
        X = np.random.rand(100, 10)
        y = np.random.randint(0, 2, 100)
        mock_load_data.return_value = (X, y)
        
        # Mock workflow
        mock_workflow.return_value = {
            'pipeline': MagicMock(),
            'cv_score': 0.85,
            'cv_std': 0.05
        }
        
        # Mock summary
        mock_summary.return_value = {
            'model_name': 'test_classifier',
            'algorithm': 'gradient_boosting',
            'validation_strategy': 'cross_validation',
            'cv_score': 0.85,
            'cv_std': 0.05
        }
        
        # Should run without error
        try:
            application_training.main()
        except SystemExit:
            pass  # Expected if sys.exit is called
        except Exception as e:
            pytest.fail(f"main() raised {e} unexpectedly")
    
    @patch('application_training.load_data')
    def test_main_with_data_not_found(self, mock_load_data):
        """Test application when data is not found."""
        import application_training
        
        # Mock data not found
        mock_load_data.side_effect = FileNotFoundError("Database not found")
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            application_training.main()

