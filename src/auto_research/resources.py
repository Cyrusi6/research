"""Local resource discovery."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def discover_local_mm_resources(config: dict[str, Any]) -> dict[str, Any]:
    root = Path(config.get("project", {}).get("external_resource_roots", {}).get("mm_root", ""))
    if not root.exists():
        return {"available": False, "root": str(root), "datasets": {}, "codebases": {}, "checkpoints": {}}

    dataset_root = root / "dataset"
    laps_change = root / "LAPS_change"
    laps_clean = root / "LAPS_official_clean"
    ckpt_root = root / "LAPS_official_ckpts"

    datasets = {
        "f30k": {
            "available": all(
                [
                    (dataset_root / "f30k").exists(),
                    (dataset_root / "flickr30k-images").exists(),
                ]
            ),
            "caption_root": str(dataset_root / "f30k"),
            "image_root": str(dataset_root / "flickr30k-images"),
            "data_path": str(dataset_root),
        },
        "coco": {
            "available": all(
                [
                    (dataset_root / "coco").exists(),
                    (dataset_root / "coco-images").exists(),
                ]
            ),
            "caption_root": str(dataset_root / "coco"),
            "image_root": str(dataset_root / "coco-images"),
            "data_path": str(dataset_root),
        },
    }
    codebases = {
        "laps_change": {
            "available": (laps_change / "train.py").exists() and (laps_change / "eval.py").exists(),
            "root": str(laps_change),
            "python": _best_python(
                [
                    "conda run -n laps python",
                    str(laps_change / ".venv" / "bin" / "python"),
                    shutil.which("python3") or "python3",
                    shutil.which("python") or "python",
                ]
            ),
        },
        "laps_official_clean": {
            "available": (laps_clean / "train.py").exists() and (laps_clean / "eval.py").exists(),
            "root": str(laps_clean),
            "python": _best_python(
                [
                    "conda run -n laps python",
                    str(laps_clean / ".venv" / "bin" / "python"),
                    shutil.which("python3") or "python3",
                    shutil.which("python") or "python",
                ]
            ),
        },
    }
    for item in codebases.values():
        item["python_ready"] = bool(item["python"])
        item["missing_dependencies"] = [] if item["python"] else ["torch", "transformers", "tensorboard_logger"]
    checkpoints = {
        "laps_coco_vit": str(ckpt_root / "coco_vit" / "laps_vit_coco.pth") if (ckpt_root / "coco_vit" / "laps_vit_coco.pth").exists() else None,
        "laps_coco_swin": str(ckpt_root / "coco_swin" / "laps_swin_coco.pth") if (ckpt_root / "coco_swin" / "laps_swin_coco.pth").exists() else None,
    }
    return {
        "available": True,
        "root": str(root),
        "dataset_root": str(dataset_root),
        "datasets": datasets,
        "codebases": codebases,
        "checkpoints": checkpoints,
        "reusable_runs": _scan_reusable_runs(root),
    }


def best_itr_execution_plan(resources: dict[str, Any]) -> dict[str, Any]:
    if not resources.get("available"):
        return {"mode": "manual", "commands": [], "blocked_reason": "Local MM resources not found."}

    reusable = _best_reusable_itr_runs(resources.get("reusable_runs", []))
    if reusable:
        return {
            "mode": "reuse",
            "collector": "reused_runs",
            "selected_runs": reusable,
            "commands": [],
            "blocked_reason": None,
        }

    laps = resources.get("codebases", {}).get("laps_change", {})
    coco = resources.get("datasets", {}).get("coco", {})
    if laps.get("available") and coco.get("available"):
        python_bin = laps["python"]
        workdir = laps["root"]
        if not laps.get("python_ready"):
            return {
                "mode": "manual",
                "commands": [],
                "blocked_reason": "LAPS local code was found, but no Python environment with torch, transformers, and tensorboard_logger is available.",
            }
        commands = []
        vit_ckpt = resources.get("checkpoints", {}).get("laps_coco_vit")
        swin_ckpt = resources.get("checkpoints", {}).get("laps_coco_swin")
        if vit_ckpt:
            commands.append(f"mkdir -p runs/coco_vit && cp '{vit_ckpt}' runs/coco_vit/model_best.pth")
        if swin_ckpt:
            commands.append(f"mkdir -p runs/coco_swin && cp '{swin_ckpt}' runs/coco_swin/model_best.pth")
        if commands:
            commands.append(f"{python_bin} eval.py --dataset coco --data_path '{coco['data_path']}' --gpu-id 0")
            return {
                "mode": "scripted",
                "collector": "laps_eval",
                "workdir": workdir,
                "commands": commands,
                "blocked_reason": None,
            }

    return {"mode": "manual", "commands": [], "blocked_reason": "No runnable local image-text retrieval baseline was discovered."}


def _best_python(candidates: list[str]) -> str | None:
    for candidate in candidates:
        if not candidate:
            continue
        if " " not in candidate:
            path = Path(candidate)
            if path.is_absolute() and not path.exists():
                continue
        if _python_has_laps_dependencies(candidate):
            return candidate
    return None


def _python_has_laps_dependencies(python_bin: str) -> bool:
    try:
        argv = shlex.split(python_bin)
        if not argv:
            return False
        result = subprocess.run(
            [*argv, "-c", "import torch, transformers, tensorboard_logger; print('ok')"],
            shell=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return False
    return result.returncode == 0


def _scan_reusable_runs(root: Path) -> list[dict[str, Any]]:
    records = []
    for log in _iter_eval_logs(root):
        record = _parse_eval_log(log)
        if record:
            records.append(record)
    return sorted(records, key=lambda item: item.get("rsum", 0), reverse=True)


def _iter_eval_logs(root: Path) -> Iterator[Path]:
    ignored_directories = {".git", ".pytest_cache", ".tmp", ".venv", "__pycache__"}
    for directory, directory_names, file_names in os.walk(root, topdown=True, onerror=lambda _error: None, followlinks=False):
        directory_names[:] = [name for name in directory_names if name not in ignored_directories]
        directory_path = Path(directory)
        for file_name in file_names:
            if file_name != "eval.log" and not (file_name.startswith("eval_") and file_name.endswith(".log")):
                continue
            log_path = directory_path / file_name
            try:
                if log_path.is_symlink():
                    continue
            except OSError:
                continue
            yield log_path


def _parse_eval_log(log_path: Path) -> dict[str, Any] | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    rsum_match = re.search(r"rsum:\s*([0-9.]+)", text)
    i2t_match = re.search(r"Image to text(?: \(R@1, R@5, R@10\)|:)\s*[:]?[\s]*([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)", text)
    t2i_match = re.search(r"Text to image(?: \(R@1, R@5, R@10\)|:)\s*[:]?[\s]*([0-9.]+)[,\s]+([0-9.]+)[,\s]+([0-9.]+)", text)
    if not (rsum_match or (i2t_match and t2i_match)):
        return None
    dataset_match = re.search(r"dataset='([^']+)'", text)
    vit_match = re.search(r"vit_type='([^']+)'", text)
    run_dir = log_path.parent
    model_best = run_dir / "model_best.pth"
    results_files = _safe_glob_paths(run_dir, "results_*.npy")
    if not model_best.exists():
        for parent in [run_dir.parent, run_dir.parent.parent]:
            candidate = parent / "model_best.pth"
            if candidate.exists():
                model_best = candidate
                break
    if not results_files:
        for parent in [run_dir.parent, run_dir.parent.parent]:
            results_files = _safe_glob_paths(parent, "results_*.npy")
            if results_files:
                break
    return {
        "log_path": str(log_path),
        "run_dir": str(run_dir),
        "model_best_path": str(model_best) if model_best.exists() else None,
        "results_paths": results_files,
        "dataset": dataset_match.group(1) if dataset_match else _guess_dataset_from_path(log_path),
        "encoder": vit_match.group(1) if vit_match else _guess_encoder_from_path(log_path),
        "repo_family": _repo_family(log_path),
        "rsum": float(rsum_match.group(1)) if rsum_match else None,
        "i2t": _triplet(i2t_match),
        "t2i": _triplet(t2i_match),
        "ready": bool(model_best.exists()),
    }


def _safe_glob_paths(directory: Path, pattern: str) -> list[str]:
    try:
        return [str(path) for path in directory.glob(pattern)]
    except OSError:
        return []


def _best_reusable_itr_runs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ready = [item for item in records if item.get("ready") and item.get("rsum") is not None]
    if not ready:
        return []
    selected = []
    seen_keys = set()
    for item in ready:
        dataset = item.get("dataset") or "unknown"
        key = (dataset, item.get("repo_family"))
        if key in seen_keys:
            continue
        selected.append(item)
        seen_keys.add(key)
        if len(selected) >= 3:
            break
    return selected


def best_matching_run(
    records: list[dict[str, Any]],
    *,
    repo_family: str | None = None,
    dataset: str | None = None,
    encoder: str | None = None,
) -> dict[str, Any] | None:
    for item in sorted(records, key=lambda record: record.get("rsum", 0), reverse=True):
        if repo_family and item.get("repo_family") != repo_family:
            continue
        if dataset and item.get("dataset") != dataset:
            continue
        if encoder and item.get("encoder") != encoder:
            continue
        return item
    return None


def _repo_family(path: Path) -> str:
    parts = path.parts
    for candidate in ["LAPS_change", "LAPS_official_clean", "ResiDual", "seps"]:
        if candidate in parts:
            return candidate
    return "unknown"


def _guess_dataset_from_path(path: Path) -> str:
    path_str = str(path).lower()
    if "f30k" in path_str or "flickr" in path_str:
        return "f30k"
    if "coco" in path_str:
        return "coco"
    return "unknown"


def _guess_encoder_from_path(path: Path) -> str:
    path_str = str(path).lower()
    if "swin" in path_str:
        return "swin"
    if "vit" in path_str:
        return "vit"
    return "unknown"


def _triplet(match: re.Match[str] | None) -> dict[str, float] | None:
    if not match:
        return None
    return {"R@1": float(match.group(1)), "R@5": float(match.group(2)), "R@10": float(match.group(3))}
