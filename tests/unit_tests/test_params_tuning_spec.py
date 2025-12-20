"""
Unit tests for parameter tuning specifications.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from specs_training.params_tuning_spec import (
    ParamTuningSpec,
    GridSearchSpec,
    RandomSearchSpec,
    ParamTuningSpecBuilder,
)


@pytest.mark.unit
class TestGridSearchSpec:
    """Tests for GridSearchSpec."""

    def test_core_fields(self):
        """Test core fields of GridSearchSpec."""
        spec = GridSearchSpec(
            tuning_name="grid",
            param_grid={"a": [1, 2]},
            scoring="accuracy",
            refit_score="accuracy",
            n_jobs=1,
        )

        assert spec.tuning_name == "grid", (
            f"tuning_name should be 'grid', got '{spec.tuning_name}'"
        )
        assert spec.get_tuning_type() == "grid_search", (
            f"get_tuning_type() should return 'grid_search', "
            f"got '{spec.get_tuning_type()}'"
        )
        assert spec.param_grid == {"a": [1, 2]}, (
            f"param_grid should be {{'a': [1, 2]}}, got {spec.param_grid}"
        )

    def test_to_dict_serialization(self):
        """Test to_dict serialization."""
        spec = GridSearchSpec(
            tuning_name="grid",
            param_grid={"a": [1, 2]},
            scoring="accuracy",
            refit_score="accuracy",
            n_jobs=1,
        )
        d = spec.to_dict()

        assert d["tuning_name"] == "grid", (
            f"Serialized dict should contain tuning_name='grid', "
            f"got '{d.get('tuning_name')}'"
        )
        assert d["tuning_type"] == "grid_search", (
            f"Serialized dict should contain tuning_type='grid_search', "
            f"got '{d.get('tuning_type')}'"
        )

    def test_default_values(self):
        """Test default values for GridSearchSpec."""
        spec = GridSearchSpec(
            tuning_name="grid",
            param_grid={"a": [1, 2]},
        )

        assert spec.scoring == "accuracy", (
            f"Default scoring should be 'accuracy', got '{spec.scoring}'"
        )
        assert spec.n_splits == 5, (
            f"Default n_splits should be 5, got {spec.n_splits}"
        )


@pytest.mark.unit
class TestRandomSearchSpec:
    """Tests for RandomSearchSpec."""

    def test_core_fields(self):
        """Test core fields of RandomSearchSpec."""
        spec = RandomSearchSpec(
            tuning_name="random",
            param_distributions={"a": [1, 2, 3]},
            n_iter=5,
            scoring="accuracy",
            refit_score="accuracy",
            n_jobs=1,
        )

        assert spec.tuning_name == "random", (
            f"tuning_name should be 'random', got '{spec.tuning_name}'"
        )
        assert spec.get_tuning_type() == "random_search", (
            f"get_tuning_type() should return 'random_search', "
            f"got '{spec.get_tuning_type()}'"
        )
        assert spec.param_distributions == {"a": [1, 2, 3]}, (
            f"param_distributions should be {{'a': [1, 2, 3]}}, "
            f"got {spec.param_distributions}"
        )

    def test_to_dict_serialization(self):
        """Test to_dict serialization."""
        spec = RandomSearchSpec(
            tuning_name="random",
            param_distributions={"a": [1, 2, 3]},
            n_iter=5,
            scoring="accuracy",
            refit_score="accuracy",
            n_jobs=1,
        )
        d = spec.to_dict()

        assert d["tuning_name"] == "random", (
            f"Serialized dict should contain tuning_name='random', "
            f"got '{d.get('tuning_name')}'"
        )
        assert d["tuning_type"] == "random_search", (
            f"Serialized dict should contain tuning_type='random_search', "
            f"got '{d.get('tuning_type')}'"
        )

    def test_n_iter_required(self):
        """Test that n_iter is required for RandomSearchSpec."""
        with pytest.raises((ValidationError, TypeError)) as exc_info:
            RandomSearchSpec(
                tuning_name="random",
                param_distributions={"a": [1, 2, 3]},
            )

        assert exc_info is not None, (
            "Should raise error when n_iter is not provided"
        )


@pytest.mark.unit
class TestParamTuningSpecBuilder:
    """Tests for ParamTuningSpecBuilder."""

    def test_adds_specs(self):
        """Test builder adds multiple specs correctly."""
        builder = ParamTuningSpecBuilder()
        builder.add_grid_search("g1", {"a": [1]})
        builder.add_random_search("r1", {"b": [1, 2]}, n_iter=3)
        specs = builder.build()
        names = {s.tuning_name for s in specs}

        assert names == {"g1", "r1"}, (
            f"Builder should create specs with names {{'g1', 'r1'}}, got {names}"
        )
        assert len(specs) == 2, (
            f"Builder should create 2 specs, got {len(specs)}"
        )

    def test_build_empty(self):
        """Test building empty spec list."""
        builder = ParamTuningSpecBuilder()
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
            ParamTuningSpecBuilder()
            .add_grid_search("grid1", {"a": [1, 2]})
            .add_random_search("random1", {"b": [1, 2, 3]}, n_iter=5)
            .build()
        )

        assert len(specs) == 2, (
            f"Should have 2 specs, got {len(specs)}"
        )
        assert isinstance(specs[0], GridSearchSpec), (
            "First spec should be GridSearchSpec"
        )
        assert isinstance(specs[1], RandomSearchSpec), (
            "Second spec should be RandomSearchSpec"
        )


@pytest.mark.unit
class TestParamTuningSpecAbstract:
    """Tests for ParamTuningSpec base class."""

    def test_abstract_class(self):
        """Test that ParamTuningSpec is abstract and cannot be instantiated."""
        with pytest.raises(TypeError) as exc_info:
            ParamTuningSpec(tuning_name="test")

        assert "abstract" in str(exc_info.value).lower() or "instantiate" in str(
            exc_info.value
        ).lower(), (
            f"Error should mention abstract class or instantiation, got: {exc_info.value}"
        )
