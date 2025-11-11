"""
Tests for calibration specifications.
"""

from specs.calibration_spec import (
    CalibrationSpec,
    ClassifierCalibrationSpec,
    CalibrationSpecBuilder,
)


def test_classifier_calibration_spec_core_fields():
    spec = ClassifierCalibrationSpec(
        calibration_name="calib",
        method="sigmoid",
        cv_strategy="prefit",
        ensemble=True,
    )
    d = spec.to_dict()
    assert d["calibration_name"] == "calib"
    assert spec.get_calibration_type() == "classifier_calibration"
    assert spec.method == "sigmoid"


def test_calibration_builder_adds_specs():
    builder = CalibrationSpecBuilder()
    builder.add_platt_scaling("platt").add_isotonic_regression("iso")
    specs = builder.build()
    names = {s.calibration_name for s in specs}
    assert names == {"platt", "iso"}


