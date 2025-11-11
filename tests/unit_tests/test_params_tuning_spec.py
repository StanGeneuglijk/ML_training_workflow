"""
Tests for parameter tuning specifications.
"""

from specs.params_tuning_spec import (
    ParamTuningSpec,
    GridSearchSpec,
    RandomSearchSpec,
    ParamTuningSpecBuilder,
)


def test_grid_search_spec_core_fields():
    spec = GridSearchSpec(
        tuning_name="grid",
        param_grid={"a": [1, 2]},
        scoring="accuracy",
        refit_score="accuracy",
        n_jobs=1,
    )
    d = spec.to_dict()
    assert d["tuning_name"] == "grid"
    assert spec.get_tuning_type() == "grid_search"
    assert spec.param_grid == {"a": [1, 2]}


def test_random_search_spec_core_fields():
    spec = RandomSearchSpec(
        tuning_name="random",
        param_distributions={"a": [1, 2, 3]},
        n_iter=5,
        scoring="accuracy",
        refit_score="accuracy",
        n_jobs=1,
    )
    d = spec.to_dict()
    assert d["tuning_name"] == "random"
    assert spec.get_tuning_type() == "random_search"
    assert spec.param_distributions == {"a": [1, 2, 3]}


def test_builder_adds_specs():
    builder = ParamTuningSpecBuilder()
    builder.add_grid_search("g1", {"a": [1]})
    builder.add_random_search("r1", {"b": [1, 2]})
    specs = builder.build()
    names = {s.tuning_name for s in specs}
    assert names == {"g1", "r1"}


