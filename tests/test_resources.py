from auto_research.resources import best_itr_execution_plan, discover_local_mm_resources


def test_discover_local_mm_resources_handles_missing_root() -> None:
    resources = discover_local_mm_resources({"project": {"external_resource_roots": {"mm_root": "/path/does/not/exist"}}})
    assert resources["available"] is False


def test_best_itr_execution_plan_prefers_laps_eval_when_local_assets_exist() -> None:
    resources = {
        "available": True,
        "codebases": {"laps_change": {"available": True, "root": "/tmp/laps", "python": "python", "python_ready": True}},
        "datasets": {"coco": {"available": True, "data_path": "/tmp/data"}},
        "checkpoints": {"laps_coco_vit": "/tmp/vit.pth", "laps_coco_swin": "/tmp/swin.pth"},
    }
    plan = best_itr_execution_plan(resources)
    assert plan["mode"] == "scripted"
    assert plan["collector"] == "laps_eval"
    assert any("eval.py" in command for command in plan["commands"])
