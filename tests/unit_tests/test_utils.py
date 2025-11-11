"""
Tests for utility functions.
"""
import pytest
import numpy as np
import pandas as pd

import utils


class TestValidateArrayInput:
    """Tests for validate_array_input function."""
    
    def test_valid_numpy_array(self):
        """Test validation of valid numpy array."""
        arr = np.array([[1, 2], [3, 4]])
        result = utils.validate_array_input(arr, "test_data")
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)
    
    def test_valid_pandas_dataframe(self):
        """Test validation of pandas DataFrame."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = utils.validate_array_input(df, "test_data")
        assert isinstance(result, np.ndarray)
    
    def test_valid_list(self):
        """Test validation of list."""
        lst = [[1, 2], [3, 4]]
        result = utils.validate_array_input(lst, "test_data")
        assert isinstance(result, np.ndarray)
    
    def test_none_input(self):
        """Test that None raises ValueError."""
        with pytest.raises(ValueError, match="cannot be None"):
            utils.validate_array_input(None, "test_data")
    
    def test_non_finite_values(self):
        """Test that non-finite values raise error."""
        arr = np.array([[1, np.nan], [3, 4]])
        with pytest.raises(ValueError, match="non-finite"):
            utils.validate_array_input(arr, "test_data", check_finite=True)
    
    def test_min_dimension_check(self):
        """Test minimum dimension validation."""
        arr = np.array([1, 2, 3])  # 1D
        with pytest.raises(ValueError, match="at least 2"):
            utils.validate_array_input(arr, "test_data", min_dim=2)


class TestValidateTrainingData:
    """Tests for validate_training_data function."""
    
    def test_valid_data(self, sample_classification_array):
        """Test validation of valid training data."""
        X, y = sample_classification_array
        X_valid, y_valid = utils.validate_training_data(X, y)
        assert X_valid.shape[0] == y_valid.shape[0]
        assert X_valid.ndim == 2
        assert y_valid.ndim == 1
    
    def test_dataframe_input(self, sample_classification_data):
        """Test validation with DataFrame input."""
        X_df, y_series = sample_classification_data
        X_valid, y_valid = utils.validate_training_data(X_df, y_series)
        assert isinstance(X_valid, np.ndarray)
        assert isinstance(y_valid, np.ndarray)
    
    def test_mismatched_samples(self):
        """Test that mismatched sample counts raise error."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1, 2])  # Different length
        with pytest.raises(ValueError, match="same number of samples"):
            utils.validate_training_data(X, y)
    
    def test_empty_dataset(self):
        """Test that empty dataset raises error."""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        with pytest.raises(ValueError, match="empty dataset"):
            utils.validate_training_data(X, y)
    
    def test_wrong_dimensions(self):
        """Test that wrong dimensions raise error."""
        X = np.array([1, 2, 3])  # 1D instead of 2D
        y = np.array([0, 1, 0])
        with pytest.raises(ValueError, match="2D"):
            utils.validate_training_data(X, y)


class TestValidatePredictionData:
    """Tests for validate_prediction_data function."""
    
    def test_valid_data(self, sample_classification_array):
        """Test validation of valid prediction data."""
        X, _ = sample_classification_array
        X_valid = utils.validate_prediction_data(X)
        assert X_valid.ndim == 2
        assert X_valid.shape == X.shape
    
    def test_expected_features(self):
        """Test validation with expected feature count."""
        X = np.array([[1, 2, 3], [4, 5, 6]])
        X_valid = utils.validate_prediction_data(X, expected_n_features=3)
        assert X_valid.shape[1] == 3
    
    def test_wrong_feature_count(self):
        """Test that wrong feature count raises error."""
        X = np.array([[1, 2], [3, 4]])
        with pytest.raises(ValueError, match="3 features"):
            utils.validate_prediction_data(X, expected_n_features=3)
    
    def test_empty_dataset(self):
        """Test that empty dataset raises error."""
        X = np.array([]).reshape(0, 2)
        with pytest.raises(ValueError, match="empty dataset"):
            utils.validate_prediction_data(X)


class TestConvertToNumpy:
    """Tests for convert_to_numpy function."""
    
    def test_numpy_array(self):
        """Test conversion of numpy array."""
        arr = np.array([1, 2, 3])
        result = utils.convert_to_numpy(arr)
        assert isinstance(result, np.ndarray)
        assert np.array_equal(result, arr)
    
    def test_dataframe(self):
        """Test conversion of DataFrame."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        result = utils.convert_to_numpy(df)
        assert isinstance(result, np.ndarray)
        assert result.shape == df.shape
    
    def test_list(self):
        """Test conversion of list."""
        lst = [[1, 2], [3, 4]]
        result = utils.convert_to_numpy(lst)
        assert isinstance(result, np.ndarray)


class TestSetupLogging:
    """Tests for setup_logging function."""
    
    def test_logger_creation(self):
        """Test that logger is created."""
        logger = utils.setup_logging(level=10, logger_name="test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_logger_level(self):
        """Test that logger level is set correctly."""
        logger = utils.setup_logging(level=20, logger_name="test_logger_2")
        assert logger.level <= 20


class TestEnsureDirectory:
    """Tests for ensure_directory function."""
    
    def test_creates_directory(self, tmp_path):
        """Test that directory is created if it doesn't exist."""
        test_dir = tmp_path / "test_dir"
        assert not test_dir.exists()
        
        utils.ensure_directory(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()
    
    def test_existing_directory(self, tmp_path):
        """Test that existing directory doesn't cause error."""
        test_dir = tmp_path / "existing_dir"
        test_dir.mkdir()
        assert test_dir.exists()
        
        # Should not raise error
        utils.ensure_directory(str(test_dir))
        assert test_dir.exists()

