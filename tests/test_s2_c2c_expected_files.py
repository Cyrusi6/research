from auto_research.agents.plan import _sanitize_c2c_variant_expected_files


def test_c2c_variant_expected_files_are_filtered_to_allowed_surface() -> None:
    direction = {
        "direction_id": "wrapper_heterogeneity_aware_kv_routing",
        "expected_files": ["rosetta/model/wrapper.py"],
        "implementation_surface_refs": [
            {
                "source_type": "code",
                "source_path": "rosetta/model/wrapper.py",
                "source_label": "rosetta/model/wrapper.py::RosettaModel.forward::350-585",
            }
        ],
    }
    variant = {
        "id": "wrapper_heterogeneity_aware_kv_routing",
        "expected_files": [
            "rosetta/model/wrapper.py",
            "script/train/SFT_train.py",
            "tests/test_wrapper_heterogeneity_routing.py",
        ],
        "experiment_contract": {
            "expected_files": [
                "rosetta/model/wrapper.py",
                "script/train/SFT_train.py",
                "tests/test_wrapper_heterogeneity_routing.py",
            ]
        },
    }
    config = {"c2c": {"allowed_files": ["rosetta/model/wrapper.py"], "allowed_prefixes": []}}

    _sanitize_c2c_variant_expected_files(variant, direction, config)

    assert variant["expected_files"] == ["rosetta/model/wrapper.py"]
    assert variant["experiment_contract"]["expected_files"] == ["rosetta/model/wrapper.py"]


def test_c2c_variant_expected_files_fall_back_to_direction_surface() -> None:
    direction = {
        "direction_id": "wrapper_heterogeneity_aware_kv_routing",
        "implementation_surface_refs": [
            {
                "source_type": "code",
                "source_path": "/home/user/projects/C2C/rosetta/model/wrapper.py",
                "source_label": "rosetta/model/wrapper.py::RosettaModel.forward::350-585",
            }
        ],
    }
    variant = {
        "id": "wrapper_heterogeneity_aware_kv_routing",
        "expected_files": ["script/train/SFT_train.py", "tests/test_wrapper_heterogeneity_routing.py"],
        "experiment_contract": {"expected_files": ["script/train/SFT_train.py"]},
    }
    config = {"c2c": {"allowed_files": ["rosetta/model/wrapper.py"], "allowed_prefixes": []}}

    _sanitize_c2c_variant_expected_files(variant, direction, config)

    assert variant["expected_files"] == ["rosetta/model/wrapper.py"]
    assert variant["experiment_contract"]["expected_files"] == ["rosetta/model/wrapper.py"]
