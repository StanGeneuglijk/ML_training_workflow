"""
Unit tests for calibration specifications.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from specs.calibration_spec import (
    CalibrationSpec,
    ClassifierCalibrationSpec,
    CalibrationSpecBuilder,
)


@pytest.mark.unit
class TestClassifierCalibrationSpec:
    """Tests for ClassifierCalibrationSpec core functionality."""

    def test_core_fields(self):
        """Test core fields of ClassifierCalibrationSpec."""
        spec = ClassifierCalibrationSpec(
            calibration_name="calib",
            method="sigmoid",
            cv_strategy="prefit",
            ensemble=True,
        )

        assert spec.calibration_name == "calib", (
            f"calibration_name should be 'calib', got '{spec.calibration_name}'"
        )
        assert spec.get_calibration_type() == "classifier_calibration", (
            f"get_calibration_type() should return 'classifier_calibration', "
            f"got '{spec.get_calibration_type()}'"
        )
        assert spec.method == "sigmoid", (
            f"method should be 'sigmoid', got '{spec.method}'"
        )

    def test_to_dict_serialization(self):
        """Test to_dict serialization."""
        spec = ClassifierCalibrationSpec(
            calibration_name="calib",
            method="sigmoid",
            cv_strategy="prefit",
        )
        d = spec.to_dict()

        assert d["calibration_name"] == "calib", (
            f"Serialized dict should contain calibration_name='calib', "
            f"got '{d.get('calibration_name')}'"
        )
        assert d["method"] == "sigmoid", (
            f"Serialized dict should contain method='sigmoid', "
            f"got '{d.get('method')}'"
        )

    @pytest.mark.parametrize(
        "method",
        [
            pytest.param("sigmoid", id="sigmoid_method"),
            pytest.param("isotonic", id="isotonic_method"),
        ],
    )
    def test_valid_methods(self, method):
        """Test valid calibration methods."""
        spec = ClassifierCalibrationSpec(
            calibration_name="calib",
            method=method,
        )

        assert spec.method == method, (
            f"method should be '{method}', got '{spec.method}'"
        )

    def test_default_values(self):
        """Test default values for ClassifierCalibrationSpec."""
        spec = ClassifierCalibrationSpec(calibration_name="calib")

        assert spec.ensemble is False, (
            "Default ensemble should be False"
        )
        assert spec.cv_strategy == "prefit", (
            f"Default cv_strategy should be 'prefit', got '{spec.cv_strategy}'"
        )


@pytest.mark.unit
class TestCalibrationSpecBuilder:
    """Tests for CalibrationSpecBuilder."""

    def test_adds_specs(self):
        """Test builder adds multiple specs correctly."""
        builder = CalibrationSpecBuilder()
        builder.add_platt_scaling("platt").add_isotonic_regression("iso")
        specs = builder.build()
        names = {s.calibration_name for s in specs}

        assert names == {"platt", "iso"}, (
            f"Builder should create specs with names {{'platt', 'iso'}}, got {names}"
        )
        assert len(specs) == 2, (
            f"Builder should create 2 specs, got {len(specs)}"
        )

    def test_build_empty(self):
        """Test building empty spec list."""
        builder = CalibrationSpecBuilder()
        specs = builder.build()

        assert len(specs) == 0, (
            f"Empty builder should return empty list, got {len(specs)} specs"
        )
        assert isinstance(specs, list), (
            f"build() should return a list, got {type(specs)}"
        )

    def test_fluent_interface(self):
        """Test builder fluent interface with chaining."""
        specs = (
            CalibrationSpecBuilder()
            .add_platt_scaling("platt1")
            .add_platt_scaling("platt2")
            .add_isotonic_regression("iso")
            .build()
        )

        assert len(specs) == 3, (
            f"Should have 3 specs, got {len(specs)}"
        )
        names = {s.calibration_name for s in specs}
        assert "platt1" in names, (
            "Should include 'platt1' spec"
        )
        assert "platt2" in names, (
            "Should include 'platt2' spec"
        )
        assert "iso" in names, (
            "Should include 'iso' spec"
        )


@pytest.mark.unit
class TestCalibrationSpecAbstract:
    """Tests for CalibrationSpec base class."""

    def test_abstract_class(self):
        """Test that CalibrationSpec is abstract and cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            CalibrationSpec(calibration_name="test")

        assert "abstract" in str(exc_info.value).lower() or "instantiate" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention abstract class or instantiation, got: {exc_info.value}"
        )
