from auto_research.itr_ideas import build_laps_candidate_ideas, build_quick_screen_execution


def test_build_laps_candidate_ideas_and_quick_execution() -> None:
    theme_map = {
        "direct_retrieval": [{"title": "Multimodal Alignment and Fusion: A Survey", "year": "2024"}],
        "themes": {
            "multi_scale_fusion": [{"title": "A Multilevel Multimodal Fusion Transformer", "year": "2024"}],
            "adaptive_dynamic_attention": [{"title": "Modality-specific adaptive scaling and attention network for cross-modal retrieval", "year": "2024"}],
            "explicit_cross_modal_alignment": [{"title": "Multimodal Alignment and Fusion: A Survey", "year": "2024"}],
        },
        "weak_reference": [],
    }
    resources = {
        "codebases": {"laps_change": {"available": True, "python_ready": True, "root": "/tmp/laps", "python": "conda run -n laps python"}},
        "datasets": {"f30k": {"available": True, "image_root": "/tmp/f30k"}},
        "reusable_runs": [
            {"repo_family": "LAPS_change", "dataset": "f30k", "model_best_path": "/tmp/model_best.pth", "rsum": 497.3}
        ],
    }

    ideas = build_laps_candidate_ideas(theme_map, resources, topic="image-text retrieval")
    execution = build_quick_screen_execution(ideas, resources, project_id="proj_demo")

    assert len(ideas) == 7
    assert sum(1 for item in ideas if item["selected"]) == 1
    assert any(item["id"] == "idea_weighted_routecalib" and item["selected"] for item in ideas)
    assert execution["collector"] == "itr_quick_screen"
    assert execution["control"]["logger_name"].endswith("proj_demo_resume_screen_control")
    assert len(execution["candidates"]) == 6
    assert execution["control"]["resume"] == "/tmp/model_best.pth"
    assert execution["data_path"].endswith("/data")


def test_quick_screen_execution_filters_to_screenable_ideas() -> None:
    ideas = [
        {"id": "high_ceiling", "title": "CSIC/UARDA", "screening_recipe": None},
        {"id": "screenable", "title": "Weighted Route Calibration", "screening_recipe": {"logger_name": "screenable"}},
    ]
    resources = {
        "codebases": {"laps_change": {"available": True, "python_ready": True, "root": "/tmp/laps", "python": "python"}},
        "datasets": {"f30k": {"available": True, "image_root": "/tmp/f30k"}},
    }
    execution = build_quick_screen_execution(ideas, resources, project_id="proj_demo")
    assert len(execution["candidates"]) == 1
    assert execution["candidates"][0]["id"] == "screenable"
