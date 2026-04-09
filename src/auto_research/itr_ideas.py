"""Image-text retrieval specific theme mapping and quick-screen planning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .resources import best_matching_run
from .utils import compact_markdown


THEME_LABELS = {
    "multi_scale_fusion": "Multi-Scale Feature Fusion",
    "adaptive_dynamic_attention": "Adaptive Dynamic Attention",
    "explicit_cross_modal_alignment": "Explicit Cross-Modal Alignment",
}


def collect_consensus_entries(imports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in imports:
        for entry in item.get("entries", []):
            if isinstance(entry, dict) and entry.get("title"):
                entries.append(entry)
    return entries


def build_itr_theme_map(entries: list[dict[str, Any]]) -> dict[str, Any]:
    themes = {key: [] for key in THEME_LABELS}
    direct_retrieval: list[dict[str, Any]] = []
    weak_reference: list[dict[str, Any]] = []

    for entry in entries:
        title = entry.get("title", "")
        title_lc = title.lower()
        direct = _is_direct_itr_reference(title_lc)
        if direct:
            direct_retrieval.append(entry)
        matched = False
        if any(keyword in title_lc for keyword in ["multi-scale", "multiscale", "cross-scale", "multilevel"]):
            themes["multi_scale_fusion"].append(entry)
            matched = True
        if any(keyword in title_lc for keyword in ["adaptive", "dynamic", "attention", "deformable self-attention", "cross-attention"]):
            themes["adaptive_dynamic_attention"].append(entry)
            matched = True
        if any(keyword in title_lc for keyword in ["alignment", "cross-modal retrieval", "cross modal retrieval", "modality-specific"]):
            themes["explicit_cross_modal_alignment"].append(entry)
            matched = True
        if not matched:
            weak_reference.append(entry)

    return {
        "direct_retrieval": _dedupe_entries(direct_retrieval),
        "themes": {key: _dedupe_entries(value) for key, value in themes.items()},
        "weak_reference": _dedupe_entries(weak_reference),
    }


def theme_map_markdown(theme_map: dict[str, Any]) -> str:
    lines = ["# Image-Text Retrieval Theme Map", ""]
    direct = theme_map.get("direct_retrieval", [])
    lines.append("## Direct Retrieval References")
    if direct:
        for entry in direct[:10]:
            lines.append(f"- {entry.get('title')} ({entry.get('year', 'n/a')})")
    else:
        lines.append("- No directly relevant retrieval references were found in the RIS import.")
    lines.append("")
    for theme_key, label in THEME_LABELS.items():
        lines.append(f"## {label}")
        items = theme_map.get("themes", {}).get(theme_key, [])
        if items:
            for entry in items[:10]:
                relation = "direct retrieval" if _is_direct_itr_reference(entry.get("title", "").lower()) else "mechanism inspiration"
                lines.append(f"- {entry.get('title')} ({entry.get('year', 'n/a')}, {relation})")
        else:
            lines.append("- No matched references.")
        lines.append("")
    weak = theme_map.get("weak_reference", [])
    lines.append("## Weak References")
    if weak:
        for entry in weak[:10]:
            lines.append(f"- {entry.get('title')} ({entry.get('year', 'n/a')})")
    else:
        lines.append("- None")
    return compact_markdown("\n".join(lines))


def build_laps_candidate_ideas(theme_map: dict[str, Any], resources: dict[str, Any], *, topic: str) -> list[dict[str, Any]]:
    laps_anchor = best_matching_run(resources.get("reusable_runs", []), repo_family="LAPS_change", dataset="f30k")
    anchor_path = _screen_anchor_path(laps_anchor.get("model_best_path") if laps_anchor else None)
    direct_titles = [entry.get("title") for entry in theme_map.get("direct_retrieval", [])[:3] if entry.get("title")]

    baseline_refs = direct_titles or ["Modality-specific adaptive scaling and attention network for cross-modal retrieval"]
    ideas = [
        {
            "id": "idea_csic_uarda",
            "direction": "explicit_cross_modal_alignment",
            "title": "Cross-Modal Semantic Importance Consistency with Learnable Relevance Boundaries",
            "description": "Introduce token-level semantic-importance consistency and a learnable relevant/irrelevant boundary so routing and matching are trained against a sharper relevance signal.",
            "motivation": "This directly targets the relevance modeling weakness that current local routing tweaks do not fix and is the strongest candidate to challenge a 507-level ceiling.",
            "novelty_score": 9,
            "feasibility_score": 7,
            "expected_contribution": "A stronger relevance mechanism that can improve both token routing and final similarity scoring.",
            "key_baselines": baseline_refs[:2],
            "required_compute": "Needs fresh implementation and should not enter screening until reviewed.",
            "key_references": [entry.get("title") for entry in theme_map.get("themes", {}).get("explicit_cross_modal_alignment", [])[:5]],
            "selected": False,
            "primary_codebase": "LAPS_change",
            "baseline_anchor_path": anchor_path,
            "screening_recipe": {
                "logger_name": "auto_itr_csic_uarda_resume",
                "resume": anchor_path,
                "resume_strict": 0,
                "learning_rate": 6e-5,
                "batch_size": 48,
                "num_epochs": 1,
                "changes": {
                    "loss": "trip",
                    "route_mode": "sum",
                    "route_conflict_weight": 0.25,
                    "enable_csic_uarda": 1,
                    "csic_boundary_weight": 0.3,
                    "csic_consistency_weight": 0.15,
                    "csic_boundary_temp": 12.0,
                },
            },
        },
        {
            "id": "idea_tgdt_lite",
            "direction": "multi_scale_fusion",
            "title": "Token-Guided Dual-Granularity Transformer for Retrieval",
            "description": "Build a true dual-granularity training objective that couples coarse global matching with local patch-word matching, instead of only adding an auxiliary scalar loss.",
            "motivation": "A TGDT-style global-local consistency path is a more structural change than current multi-scale pooling and has higher upside.",
            "novelty_score": 8,
            "feasibility_score": 6,
            "expected_contribution": "Improved coarse-fine consistency with a better chance of lifting the ceiling than lightweight fusion.",
            "key_baselines": baseline_refs[:2],
            "required_compute": "Requires moderate model changes before screening.",
            "key_references": [entry.get("title") for entry in theme_map.get("themes", {}).get("multi_scale_fusion", [])[:5]],
            "selected": False,
            "primary_codebase": "LAPS_change",
            "baseline_anchor_path": anchor_path,
        },
        {
            "id": "idea_weighted_routecalib",
            "direction": "adaptive_dynamic_attention",
            "title": "Weighted Route Calibration with Conflict-Aware Routing",
            "description": "Use the existing weighted route mode to calibrate self-attention and text-guided attention while keeping conflict-aware routing active.",
            "motivation": "This is the strongest low-risk attention variant already hinted as positive in local LAPS_change quick screens.",
            "novelty_score": 6,
            "feasibility_score": 10,
            "expected_contribution": "A lighter calibrated routing policy with minimal extra compute.",
            "key_baselines": baseline_refs[:2],
            "required_compute": "1 resumed F30K screen on 1 GPU",
            "key_references": [entry.get("title") for entry in theme_map.get("themes", {}).get("adaptive_dynamic_attention", [])[:5]],
            "selected": True,
            "primary_codebase": "LAPS_change",
            "baseline_anchor_path": anchor_path,
            "screening_recipe": {
                "logger_name": "auto_itr_weighted_routecalib_resume",
                "resume": anchor_path,
                "resume_strict": 1,
                "learning_rate": 6e-5,
                "batch_size": 48,
                "num_epochs": 1,
                "changes": {
                    "route_mode": "weighted",
                    "route_conflict_weight": 0.25,
                    "attention_weight": 0.8,
                    "loss": "trip",
                },
            },
        },
        {
            "id": "idea_dynamic_attention",
            "direction": "adaptive_dynamic_attention",
            "title": "Adaptive Dynamic Attention Routing for Image-Text Retrieval",
            "description": "Replace the fixed routing weight with a sample-conditioned gate that balances self-attention and text-guided attention on each image-text pair.",
            "motivation": "The RIS import emphasizes adaptive attention, and local LAPS history already shows static weighted routing can help with near-zero overhead.",
            "novelty_score": 7,
            "feasibility_score": 9,
            "expected_contribution": "Better token routing on hard pairs without changing the loss or backbone.",
            "key_baselines": baseline_refs[:2],
            "required_compute": "1 quick F30K screen on 1 GPU",
            "key_references": [entry.get("title") for entry in theme_map.get("themes", {}).get("adaptive_dynamic_attention", [])[:5]],
            "selected": False,
            "primary_codebase": "LAPS_change",
            "baseline_anchor_path": anchor_path,
            "screening_recipe": {
                "logger_name": "auto_itr_dynamic_attention_resume",
                "resume": anchor_path,
                "resume_strict": 0,
                "learning_rate": 6e-5,
                "batch_size": 48,
                "num_epochs": 1,
                "changes": {
                    "route_mode": "adaptive",
                    "route_conflict_weight": 0.25,
                    "attention_weight": 0.8,
                    "loss": "trip",
                },
            },
        },
        {
            "id": "idea_multiscale_fusion",
            "direction": "multi_scale_fusion",
            "title": "Dual-Scale Token Fusion for Image-Text Retrieval",
            "description": "Add a coarse pooled visual branch alongside the sparse fine-grained branch, then fuse both scales before cross-modal matching.",
            "motivation": "The RIS import repeatedly surfaces multi-scale fusion as a stable multimodal mechanism, while current LAPS code lacks an explicit coarse branch.",
            "novelty_score": 8,
            "feasibility_score": 6,
            "expected_contribution": "Recover context missed by aggressive patch slimming on semantically diffuse captions.",
            "key_baselines": baseline_refs[:2],
            "required_compute": "1 quick F30K screen on 1 GPU",
            "key_references": [entry.get("title") for entry in theme_map.get("themes", {}).get("multi_scale_fusion", [])[:5]],
            "selected": False,
            "primary_codebase": "LAPS_change",
            "baseline_anchor_path": anchor_path,
            "screening_recipe": {
                "logger_name": "auto_itr_multiscale_resume",
                "resume": anchor_path,
                "resume_strict": 0,
                "learning_rate": 6e-5,
                "batch_size": 48,
                "num_epochs": 1,
                "changes": {
                    "enable_multiscale": 1,
                    "multiscale_grid": 4,
                    "multiscale_weight": 0.35,
                    "route_conflict_weight": 0.25,
                    "loss": "trip",
                },
            },
        },
        {
            "id": "idea_global_align_light",
            "direction": "explicit_cross_modal_alignment",
            "title": "Light Global Alignment Auxiliary Loss",
            "description": "Keep the local cross-attention path and add a small global image-text alignment loss as an auxiliary objective.",
            "motivation": "This is the closest code-level match to the explicit alignment theme and already showed a mild positive trend in local full-data screening.",
            "novelty_score": 6,
            "feasibility_score": 9,
            "expected_contribution": "Improve global semantic consistency while preserving the existing local patch-word matching pipeline.",
            "key_baselines": baseline_refs[:2],
            "required_compute": "1 resumed F30K screen on 1 GPU",
            "key_references": [entry.get("title") for entry in theme_map.get("themes", {}).get("explicit_cross_modal_alignment", [])[:5]],
            "selected": False,
            "primary_codebase": "LAPS_change",
            "baseline_anchor_path": anchor_path,
            "screening_recipe": {
                "logger_name": "auto_itr_global_align_light_resume",
                "resume": anchor_path,
                "resume_strict": 1,
                "learning_rate": 6e-5,
                "batch_size": 48,
                "num_epochs": 1,
                "changes": {
                    "global_align_weight": 0.2,
                    "route_conflict_weight": 0.25,
                    "loss": "trip",
                },
            },
        },
        {
            "id": "idea_trip_infonce",
            "direction": "explicit_cross_modal_alignment",
            "title": "Triplet-InfoNCE Hybrid Alignment",
            "description": "Add a small InfoNCE term on top of the triplet objective to strengthen global-local agreement on hard negatives.",
            "motivation": "Hybrid alignment losses are a retrieval-native extension of explicit alignment and are already supported in the local code.",
            "novelty_score": 6,
            "feasibility_score": 8,
            "expected_contribution": "More robust optimization on hard pairs while preserving the baseline router.",
            "key_baselines": baseline_refs[:2],
            "required_compute": "1 resumed F30K screen on 1 GPU",
            "key_references": [entry.get("title") for entry in theme_map.get("direct_retrieval", [])[:5]],
            "selected": False,
            "primary_codebase": "LAPS_change",
            "baseline_anchor_path": anchor_path,
            "screening_recipe": {
                "logger_name": "auto_itr_trip_infonce_resume",
                "resume": anchor_path,
                "resume_strict": 1,
                "learning_rate": 6e-5,
                "batch_size": 48,
                "num_epochs": 1,
                "changes": {
                    "loss": "trip_infonce",
                    "infonce_weight": 0.3,
                    "route_conflict_weight": 0.25,
                },
            },
        },
    ]
    return ideas


def build_quick_screen_execution(ideas: list[dict[str, Any]], resources: dict[str, Any], *, project_id: str) -> dict[str, Any]:
    laps = resources.get("codebases", {}).get("laps_change", {})
    f30k = resources.get("datasets", {}).get("f30k", {})
    if not (laps.get("available") and laps.get("python_ready") and f30k.get("available")):
        return {"mode": "manual", "commands": [], "blocked_reason": "LAPS_change with a working Python environment and F30K data is required for quick screening."}

    screenable_ideas = [idea for idea in ideas if idea.get("screening_recipe")]
    if not screenable_ideas:
        return {"mode": "manual", "commands": [], "blocked_reason": "No screenable ideas are available for the current project."}

    anchor_path = next((idea.get("baseline_anchor_path") for idea in screenable_ideas if idea.get("baseline_anchor_path")), None)
    control = {
        "logger_name": f"artifacts/runs/{project_id}_resume_screen_control",
        "resume": anchor_path,
        "resume_strict": 1,
        "learning_rate": 6e-5,
        "batch_size": 48,
        "num_epochs": 1,
        "changes": {
            "route_mode": "sum",
            "route_conflict_weight": 0.25,
            "loss": "trip",
        },
    }
    return {
        "mode": "scripted",
        "collector": "itr_quick_screen",
        "primary_codebase": "LAPS_change",
        "workdir": laps["root"],
        "python": laps["python"],
        "data_path": str(Path(laps["root"]) / "data"),
        "image_root": f30k["image_root"],
        "control": control,
        "candidates": screenable_ideas,
        "blocked_reason": None,
    }


def screening_summary_markdown(summary: dict[str, Any]) -> str:
    lines = ["# Idea Screening Summary", ""]
    baseline = summary.get("baseline")
    if baseline:
        lines.append(
            f"- Control baseline: rsum={baseline.get('rsum')}, i2t R@1={baseline.get('i2t', {}).get('R@1')}, t2i R@1={baseline.get('t2i', {}).get('R@1')}"
        )
    lines.append("")
    for result in summary.get("candidates", []):
        lines.append(f"## {result.get('title')}")
        lines.append(f"- Direction: {result.get('direction')}")
        lines.append(f"- Status: {result.get('status')}")
        if result.get("metrics"):
            metrics = result["metrics"]
            lines.append(
                f"- rsum={metrics.get('rsum')}, i2t R@1={metrics.get('i2t', {}).get('R@1')}, t2i R@1={metrics.get('t2i', {}).get('R@1')}"
            )
        lines.append(f"- Decision: {result.get('decision')}")
        lines.append("")
    viable = [item for item in summary.get("candidates", []) if item.get("decision") == "viable"]
    lines.append("## Recommended Next Step")
    if viable:
        best = viable[0]
        lines.append(f"- Continue with `{best['id']}` for a 3-epoch confirmation run.")
    else:
        lines.append("- No candidate cleared the quick-screen threshold; refine ideas before more compute.")
    return compact_markdown("\n".join(lines))


def _is_direct_itr_reference(title_lc: str) -> bool:
    direct_keywords = [
        "image-text retrieval",
        "text-image retrieval",
        "cross-modal retrieval",
        "cross modal retrieval",
        "modality-specific adaptive scaling and attention network for cross-modal retrieval",
        "multimodal alignment and fusion: a survey",
    ]
    return any(keyword in title_lc for keyword in direct_keywords)


def _screen_anchor_path(anchor_path: str | None) -> str | None:
    if not anchor_path:
        return None
    anchor = Path(anchor_path)
    parent = anchor.parent
    for candidate_name in ["model_best_reset_ep0_nooptim_best0.pth", "model_best_reset_ep0_nooptim.pth", "model_best_nooptim.pth", "model_best.pth"]:
        candidate = parent / candidate_name
        if candidate.exists():
            return str(candidate)
    return anchor_path


def _dedupe_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for entry in entries:
        title = entry.get("title", "").strip().lower()
        if not title or title in seen:
            continue
        deduped.append(entry)
        seen.add(title)
    return deduped
