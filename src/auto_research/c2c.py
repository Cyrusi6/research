"""C2C-specific intake, evidence, and experiment helpers."""

from __future__ import annotations

import ast
import hashlib
import copy
import csv
import io
import json
import os
import re
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .code_intake import CodeIntakeResult, build_code_intake
from .mineru import MinerUError, MinerUPdfClient
from .shared_cache import shared_cache_root
from .utils import deep_merge, ensure_dir, now_utc, read_json, read_yaml, sanitize_filename, sha256_file, write_json, write_yaml


DEFAULT_C2C_REPO = "/home/lijunsi/projects/C2C_original_baseline_sanity"
DEFAULT_C2C_ENV_PYTHON = "/home/lijunsi/miniconda3/envs/c2c-py310-cu124/bin/python"
DEFAULT_C2C_MODEL_ROOT = "/home/lijunsi/projects/models/c2c"
DEFAULT_C2C_DATASET_ROOT = "/home/lijunsi/projects/datasets/c2c"
DEFAULT_C2C_MIN_DELTA_TO_PASS = 0.1
DEFAULT_C2C_MAX_DATASET_REGRESSION = 2.0
DEFAULT_ORIGINAL_C2C_COMMIT = "113c3a9b2538cbf096a0477e1ec99ae2a2e0d12a"
DEFAULT_BASELINE = {
    "name": "paper_original_rosetta_fuser",
    "mean": 50.06,
    "datasets": {
        "mmlu-redux": 43.06,
        "ai2-arc": 54.52,
        "openbookqa": 52.60,
    },
    "source": "paper_native_baseline/Rosetta fuser baseline",
    "source_commit": DEFAULT_ORIGINAL_C2C_COMMIT,
    "reference_role": "primary_effect_discovery_baseline",
}
DEFAULT_STRONG_REFERENCES = [
    {
        "name": "v2.2_token_mlp_entropy050",
        "mean": 50.82,
        "datasets": {
            "mmlu-redux": 47.07,
            "ai2-arc": 54.78,
            "openbookqa": 50.60,
        },
        "source": "local/final_results/route1_alignment_v22/small_loop_summary/route1_v22_small_loop_scores.csv",
        "source_commit": "32dd3aa65f1aa75242e153df3334ed5709e0a0c1",
        "reference_role": "s3_strong_reference_only",
        "visible_to_ideation": False,
        "note": "Compare only after S3 full candidate metrics exist; do not expose this reference to S1/S2 idea generation.",
    }
]
DEFAULT_DATASETS = ["mmlu-redux", "ai2-arc", "openbookqa"]
DEFAULT_C2C_PROXY_SCREEN = {
    "enabled": True,
    "mode": "replay",
    "goal": "effect_first_discovery",
    "train_samples": 128,
    "eval_limit": 64,
    "eval_datasets": DEFAULT_DATASETS,
    "commands": [],
    "command_timeout_seconds": 1800,
    "train_timeout_seconds": 1800,
    "eval_timeout_seconds": 7200,
    "preflight_timeout_seconds": 300,
    "gpu_policy": {
        "gpu_ids": "auto",
        "max_gpus": 1,
        "min_free_mb": 18000,
        "max_utilization_gpu": 40,
        "respect_resource_filters": True,
        "disable_resource_fallback": True,
        "resource_wait": {"enabled": True, "timeout_seconds": 7200, "poll_seconds": 120},
    },
    "per_device_train_batch_size": "auto",
    "gradient_accumulation_steps": 1,
    "static_hard_gate": True,
    "reject_on_command_failure": True,
    "reject_if_no_executable_change": True,
    "reject_eval_code_changes": True,
    "reject_test_only_changes": True,
    "repairable_static_risk": True,
    "require_proxy_metrics": True,
    "require_paired_baseline": True,
    "run_baseline_if_missing": True,
    "allow_configured_baseline_fallback": True,
    "min_proxy_mean_delta": -0.3,
    "repairable_proxy_mean_margin": 0.5,
    "repair_soft_proxy_fail": False,
    "soft_proxy_mean_delta": -0.1,
    "max_proxy_dataset_regression": 1.5,
    "repairable_proxy_regression_margin": 0.5,
    "soft_max_proxy_dataset_regression": 0.75,
    "proxy_score_regression_weight": 0.5,
    "risk_penalty_per_label": 0.05,
    "min_proxy_score": None,
    "repairable_proxy_score_margin": 0.25,
    "soft_min_proxy_score": -0.3,
    "allow_neutral_proxy_full_s3": True,
    "neutral_proxy_min_delta": -0.1,
    "neutral_proxy_max_dataset_regression": 0.25,
    "baseline_cache_path": "experiment/results/c2c_proxy_baseline.json",
    "eval_smoke": {
        "enabled": True,
        "max_prediction_files": 8,
        "max_prediction_rows": 512,
        "min_nonempty_prediction_rate": 0.5,
        "min_answer_parse_rate": 0.2,
    },
    "activation_smoke": {
        "enabled": True,
        "hard_gate": True,
        "require_ablation_switch": True,
        "datasets": [],
        "max_datasets": 1,
        "eval_limit": None,
        "timeout_seconds": 900,
        "min_abs_metric_delta": 0.01,
        "max_prediction_files": 8,
        "max_prediction_rows": 512,
        "min_prediction_diff_rate": 0.01,
        "min_answer_diff_rate": 0.01,
        "min_mean_output_length_delta": 1.0,
        "require_observable_difference": True,
    },
}
DEFAULT_C2C_FULL_TRAIN_OOM_RECOVERY = {
    "enabled": True,
    "per_device_train_batch_size": 1,
    "preserve_effective_batch": True,
    "gradient_accumulation_steps": None,
    "learning_rate_scale": "effective_batch_ratio",
    "max_length": None,
    "train_samples": None,
    "tag": "memory_safe",
}
DEFAULT_C2C_TRAIN_RESOURCE_POLICY = {
    "enabled": True,
    "per_device_train_batch_size": "auto",
    "gradient_accumulation_steps": "preserve_effective_batch",
    "reference_per_device_train_batch_size": 4,
    "reference_gradient_accumulation_steps": 8,
    "reference_num_gpus": 1,
    "learning_rate_scale": "effective_batch_ratio",
    "min_learning_rate": None,
    "max_learning_rate": None,
    "batch_tiers": [
        {"min_free_mb": 22000, "per_device_train_batch_size": 4},
        {"min_free_mb": 16000, "per_device_train_batch_size": 3},
        {"min_free_mb": 10000, "per_device_train_batch_size": 2},
        {"min_free_mb": 0, "per_device_train_batch_size": 1},
    ],
}
DEFAULT_ALLOWED_FILES = [
    "rosetta/model/aligner.py",
    "rosetta/model/projector.py",
    "rosetta/model/wrapper.py",
]
DEFAULT_ALLOWED_PREFIXES = [
    "recipe/",
    "local/auto_research_runs/",
]
DEFAULT_C2C_PDF_INGEST = {
    "provider": "mineru",
    "model_version": "vlm",
    "language": "en",
    "enable_formula": True,
    "enable_table": True,
    "is_ocr": False,
    "poll_interval_seconds": 5,
    "timeout_seconds": 900,
    "request_timeout_seconds": 60,
    "fallback_to_pypdf": False,
}
FULL_CODE_CHUNK_SUFFIXES = {".py", ".json", ".yaml", ".yml", ".md", ".txt", ".sh", ".cfg", ".ini", ".toml"}
FULL_CODE_CHUNK_SKIP_PARTS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "htmlcov",
    "wandb",
    "checkpoints",
    "snapshots",
    "auto_research_runs",
}
FULL_CODE_CHUNK_MAX_FILE_BYTES = 512_000
CORE_DOCS = [
    "README.md",
    "RUNBOOK.md",
    "AGENTS.md",
    "CLAUDE.md",
    "CORE_ARCHITECTURE.md",
    "C2C_Route1_v2.x迭代总结与后续安排备忘.md",
    "C2C_论文技术部分代码对照解读.md",
    "C2C_跨Tokenizer柔性对齐改进方向研究备忘.md",
    "local/final_results/EXPERIMENT_RECORD.md",
    "local/final_results/route1_alignment/EXPERIMENT_LOG.md",
]
CORE_FILES = [
    "rosetta/model/aligner.py",
    "rosetta/model/projector.py",
    "rosetta/model/wrapper.py",
    "script/train/SFT_train.py",
    "script/evaluation/unified_evaluator.py",
    "recipe/eval_recipe/unified_eval.yaml",
    "recipe/train_recipe/C2C_0.6+0.5.json",
]


def is_c2c_project(config: dict[str, Any]) -> bool:
    return bool(config.get("c2c", {}).get("enabled"))


def c2c_proxy_screen_config(config: dict[str, Any]) -> dict[str, Any]:
    proxy_cfg = ((config.get("c2c") or {}).get("small_loop") or {}).get("proxy_screen")
    if not isinstance(proxy_cfg, dict):
        proxy_cfg = {}
    return deep_merge(copy.deepcopy(DEFAULT_C2C_PROXY_SCREEN), proxy_cfg)


def build_c2c_project_config(
    *,
    topic: str,
    target_repo: Path,
    snapshot_path: str,
    ref_paper: Path,
    ref_rebuttal: Path,
    env_python: Path,
) -> dict[str, Any]:
    return {
        "project": {"research_topic": topic},
        "c2c": {
            "enabled": True,
            "workflow_goal": "effect_first_discovery",
            "target_repo": str(target_repo),
            "snapshot_path": snapshot_path,
            "ref_paper": str(ref_paper),
            "ref_rebuttal": str(ref_rebuttal),
            "env_python": str(env_python),
            "model_root": DEFAULT_C2C_MODEL_ROOT,
            "dataset_root": DEFAULT_C2C_DATASET_ROOT,
            "model_map": {
                "Qwen/Qwen3-0.6B": f"{DEFAULT_C2C_MODEL_ROOT}/Qwen3-0.6B",
                "Qwen/Qwen2.5-0.5B-Instruct": f"{DEFAULT_C2C_MODEL_ROOT}/Qwen2.5-0.5B-Instruct",
            },
            "baseline": DEFAULT_BASELINE,
            "strong_references": copy.deepcopy(DEFAULT_STRONG_REFERENCES),
            "datasets": DEFAULT_DATASETS,
            "allowed_files": DEFAULT_ALLOWED_FILES,
            "allowed_prefixes": DEFAULT_ALLOWED_PREFIXES,
            "pdf_ingest": copy.deepcopy(DEFAULT_C2C_PDF_INGEST),
            "small_loop": {
                "train_samples": 2048,
                "eval_datasets": DEFAULT_DATASETS,
                "gpu_ids": "auto",
                "max_candidates": 3,
                "min_delta_to_pass": DEFAULT_C2C_MIN_DELTA_TO_PASS,
                "max_dataset_regression": DEFAULT_C2C_MAX_DATASET_REGRESSION,
                "require_ablation_support": False,
                "paperization_after_effect": True,
                "mock_results": False,
                "strict_dataset_cache": True,
                "train_resource_policy": copy.deepcopy(DEFAULT_C2C_TRAIN_RESOURCE_POLICY),
                "full_train_oom_recovery": copy.deepcopy(DEFAULT_C2C_FULL_TRAIN_OOM_RECOVERY),
                "proxy_screen": copy.deepcopy(DEFAULT_C2C_PROXY_SCREEN),
            },
        },
        "llm": {
            "reasoning_provider": "openai",
            "execution_provider": "none",
            "json_retries": 3,
        },
        "ideation": {
            "debate": {"enabled": True, "rounds": 2},
            "max_regeneration_rounds": 3,
        },
        "experiment": {
            "disable_llm_during_execution": True,
            "self_heal": {"max_attempts": 2},
            "gpu_policy": {
                "max_gpus": 6,
                "min_free_mb": 8192,
                "max_utilization_gpu": 40,
                "respect_resource_filters": True,
            },
        },
        "orchestration": {
            "auto_mode": True,
            "stop_after_stage": "S3_experiment",
            "failure_feedback": {
                "enabled": True,
                "route_s3_failure_to_s1": True,
                "route_repairable_proxy_to_s2": True,
                "max_proxy_repair_routes_per_iteration": 3,
                "route_proxy_rejected_to_s2": True,
                "max_same_direction_proxy_failures": 5,
            },
        },
    }


def write_c2c_project_config(project_root: Path, patch: dict[str, Any]) -> None:
    config_path = project_root / "meta" / "project_config.yaml"
    existing = read_yaml(config_path, default={}) or {}
    write_yaml(config_path, deep_merge(existing, patch))


def snapshot_c2c_repo(source: Path, destination: Path) -> dict[str, Any]:
    source = source.expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"C2C target repo not found: {source}")
    ensure_dir(destination.parent)
    shutil.copytree(source, destination, ignore=_snapshot_ignore, dirs_exist_ok=True)
    return repo_snapshot_manifest(destination, source, git_commit=_git_commit(source))


def _snapshot_ignore(directory: str, names: list[str]) -> set[str]:
    ignored = {
        ".git",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "__pycache__",
        "htmlcov",
        "wandb",
    }
    if Path(directory).name == "local":
        ignored.update({"checkpoints", "snapshots"})
    heavy_suffixes = (".pt", ".pth", ".safetensors", ".bin", ".ckpt")
    for name in names:
        if name in ignored or name.endswith(heavy_suffixes):
            ignored.add(name)
    return ignored.intersection(names)


def repo_snapshot_manifest(snapshot_root: Path, source_root: Path | None = None, *, git_commit: str | None = None) -> dict[str, Any]:
    files = []
    total_bytes = 0
    for path in snapshot_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(snapshot_root).as_posix()
        total_bytes += path.stat().st_size
        if rel in CORE_DOCS or rel in CORE_FILES:
            files.append(_file_entry(snapshot_root, rel))
    external_checkpoints = []
    if source_root:
        checkpoints_root = source_root / "local" / "checkpoints"
        if checkpoints_root.exists():
            external_checkpoints = [
                str(path)
                for path in sorted(checkpoints_root.iterdir())
                if path.exists()
            ]
    return {
        "created_at": now_utc(),
        "source_root": str(source_root) if source_root else "",
        "snapshot_root": str(snapshot_root),
        "source_git_commit": git_commit or "",
        "total_files": sum(1 for path in snapshot_root.rglob("*") if path.is_file()),
        "total_bytes": total_bytes,
        "core_files": files,
        "external_checkpoints": external_checkpoints,
        "excluded": [".git", "wandb", "local/checkpoints", "local/snapshots", "cache dirs", "large checkpoint files"],
    }


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(["git", "-C", str(repo_root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30)
    return result.stdout.strip() if result.returncode == 0 else ""


def _file_entry(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    if not path.exists():
        return {"path": relative_path, "exists": False}
    return {
        "path": relative_path,
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


@dataclass
class C2CAdapter:
    project_root: Path
    config: dict[str, Any]

    @property
    def c2c_config(self) -> dict[str, Any]:
        return self.config.get("c2c", {})

    @property
    def repo_root(self) -> Path:
        configured = self.c2c_config.get("snapshot_path") or "external/c2c_snapshot"
        path = Path(configured)
        return path if path.is_absolute() else self.project_root / path

    @property
    def env_python(self) -> str:
        return self.c2c_config.get("env_python") or DEFAULT_C2C_ENV_PYTHON

    @property
    def allowed_files(self) -> list[str]:
        return list(self.c2c_config.get("allowed_files") or DEFAULT_ALLOWED_FILES)

    @property
    def allowed_prefixes(self) -> list[str]:
        return list(self.c2c_config.get("allowed_prefixes") or DEFAULT_ALLOWED_PREFIXES)

    @property
    def baseline(self) -> dict[str, Any]:
        return dict(self.c2c_config.get("baseline") or DEFAULT_BASELINE)

    @property
    def strong_references(self) -> list[dict[str, Any]]:
        configured = self.c2c_config.get("strong_references")
        if not isinstance(configured, list):
            configured = DEFAULT_STRONG_REFERENCES
        refs = []
        for item in configured:
            if isinstance(item, dict):
                refs.append(dict(item))
        return refs

    @property
    def model_map(self) -> dict[str, str]:
        return dict(self.c2c_config.get("model_map") or {})

    @property
    def dataset_root(self) -> str:
        return self.c2c_config.get("dataset_root") or DEFAULT_C2C_DATASET_ROOT

    @property
    def pdf_ingest_config(self) -> dict[str, Any]:
        configured = self.c2c_config.get("pdf_ingest")
        return deep_merge(copy.deepcopy(DEFAULT_C2C_PDF_INGEST), configured if isinstance(configured, dict) else {})

    def missing_reference_paths(self) -> list[str]:
        missing = []
        for key in ["ref_paper", "ref_rebuttal"]:
            value = self.c2c_config.get(key)
            if not value or not Path(value).expanduser().exists():
                missing.append(key)
        return missing

    def build_repo_manifest(self) -> dict[str, Any]:
        repo_root = self.repo_root
        docs = []
        for rel in CORE_DOCS:
            entry = _file_entry(repo_root, rel)
            if entry.get("exists"):
                entry["snippet"] = _read_snippet(repo_root / rel)
            docs.append(entry)
        core_files = [_file_entry(repo_root, rel) for rel in CORE_FILES]
        tests = [
            path.relative_to(repo_root).as_posix()
            for path in sorted((repo_root / "test").glob("test_*.py"))
        ] if (repo_root / "test").exists() else []
        train_recipes = sorted(path.relative_to(repo_root).as_posix() for path in repo_root.glob("recipe/train_recipe/*.json"))
        eval_recipes = sorted(path.relative_to(repo_root).as_posix() for path in repo_root.glob("recipe/eval_recipe/*.yaml"))
        return {
            "created_at": now_utc(),
            "repo_root": str(repo_root),
            "env_python": self.env_python,
            "env_python_exists": Path(self.env_python).exists(),
            "docs": docs,
            "core_files": core_files,
            "tests": tests,
            "train_recipes": train_recipes,
            "eval_recipes": eval_recipes,
        }

    def import_historical_results(self) -> dict[str, Any]:
        final_results = self.repo_root / "local" / "final_results"
        records: list[dict[str, Any]] = []
        records.extend(self._parse_small_loop_csvs(final_results))
        records.extend(self._parse_summary_jsons(final_results))
        experiment_record = final_results / "EXPERIMENT_RECORD.md"
        route1_log = final_results / "route1_alignment" / "EXPERIMENT_LOG.md"
        notes = []
        for path in [experiment_record, route1_log]:
            if path.exists():
                notes.append({"path": path.relative_to(self.repo_root).as_posix(), "snippet": _read_snippet(path, limit=4000)})
        return {
            "created_at": now_utc(),
            "records": records,
            "notes": notes,
            "counts": {
                "small_loop_rows": len([item for item in records if item["kind"] == "small_loop"]),
                "summary_jsons": len([item for item in records if item["kind"] == "summary_json"]),
            },
        }

    def build_repo_card(self, repo_manifest: dict[str, Any], historical_results: dict[str, Any]) -> dict[str, Any]:
        existing_docs = [item for item in repo_manifest.get("docs", []) if item.get("exists")]
        existing_core_files = [item for item in repo_manifest.get("core_files", []) if item.get("exists")]
        return {
            "created_at": now_utc(),
            "repo_root": repo_manifest.get("repo_root"),
            "env_python": repo_manifest.get("env_python"),
            "env_python_exists": repo_manifest.get("env_python_exists"),
            "model_root": self.c2c_config.get("model_root") or DEFAULT_C2C_MODEL_ROOT,
            "dataset_root": self.dataset_root,
            "baseline": self.baseline_evidence(historical_results),
            "editable_surface": {
                "allowed_files": self.allowed_files,
                "allowed_prefixes": self.allowed_prefixes,
            },
            "evidence_inventory": {
                "core_docs_found": [item["path"] for item in existing_docs],
                "core_files_found": [item["path"] for item in existing_core_files],
                "tests": repo_manifest.get("tests", []),
                "train_recipes": repo_manifest.get("train_recipes", [])[:20],
                "eval_recipes": repo_manifest.get("eval_recipes", [])[:20],
                "small_loop_rows": historical_results.get("counts", {}).get("small_loop_rows", 0),
                "summary_jsons": historical_results.get("counts", {}).get("summary_jsons", 0),
            },
            "protocol_constraints": [
                "Keep receiver, sharer, datasets, and small2048 protocol fixed before claiming a method gain.",
                "Treat validation loss as diagnostic; three-dataset benchmark mean is the primary S3 gate.",
                "Use guarded edits only in alignment/projector/wrapper code or generated recipe/local run files.",
            ],
        }

    def build_paper_cards(self, reference_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        paper_cards = []
        for card in reference_cards:
            if card.get("kind") != "ref_paper":
                continue
            text = card.get("text") or ""
            paper_cards.append(
                {
                    "paper_id": card.get("paper_id"),
                    "title": card.get("title"),
                    "source_path": card.get("source_path"),
                    "local_path": card.get("local_path"),
                    "topic_tags": _topic_tags(text),
                    "contribution_signals": _keyword_snippets(
                        text,
                        ["KV", "cache", "tokenizer", "communication", "multi-agent", "latency", "semantic"],
                        limit=4,
                    ),
                    "method_snippets": _keyword_snippets(
                        text,
                        ["method", "alignment", "sharing", "fusion", "attention", "selection", "training"],
                        limit=4,
                    ),
                    "evaluation_snippets": _keyword_snippets(
                        text,
                        ["experiment", "baseline", "benchmark", "accuracy", "latency", "ablation"],
                        limit=4,
                    ),
                }
            )
        return paper_cards

    def build_paper_chunks(self, reference_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for card in reference_cards:
            if card.get("kind") != "ref_paper":
                continue
            text = _strip_reference_section(card.get("text") or "")
            sections = _split_reference_sections(text)
            chunk_index = 0
            for section_name, section_text in sections:
                for local_index, chunk_text in enumerate(_chunk_text(section_text, max_chars=1800, overlap=180)):
                    chunks.append(
                        {
                            "chunk_id": f"{card.get('paper_id')}:paper:{chunk_index:04d}",
                            "paper_id": card.get("paper_id"),
                            "title": card.get("title"),
                            "source_path": card.get("source_path"),
                            "local_path": card.get("local_path"),
                            "kind": "ref_paper",
                            "section": section_name,
                            "section_chunk_index": local_index,
                            "text": chunk_text,
                            "keywords": _chunk_keywords(chunk_text, extra_terms=[section_name, card.get("title", "")]),
                            "tokens_estimate": max(1, len(chunk_text) // 4),
                        }
                    )
                    chunk_index += 1
        return chunks

    def build_bibliography_cards(self, reference_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        bibliography = []
        for card in reference_cards:
            if card.get("kind") != "ref_paper":
                continue
            entries = _extract_bibliography_entries(card.get("text") or "")
            bibliography.append(
                {
                    "paper_id": card.get("paper_id"),
                    "title": card.get("title"),
                    "source_path": card.get("source_path"),
                    "entry_count": len(entries),
                    "entries": entries[:80],
                    "note": "References are preserved for related-work expansion but excluded from ordinary method/evidence chunks.",
                }
            )
        return bibliography

    def build_rebuttal_concern_matrix(self, reference_cards: list[dict[str, Any]]) -> dict[str, Any]:
        concerns = {
            "heterogeneous_model_support": [
                "heterogeneous",
                "different architecture",
                "same architecture",
                "cross-architecture",
                "model pair",
            ],
            "training_cost_pair_specific": [
                "training cost",
                "pair-specific",
                "retrain",
                "universal fuser",
                "cost",
            ],
            "baseline_fairness": [
                "baseline",
                "weak baseline",
                "CacheBlend",
                "DroidSpeak",
                "TokenDance",
                "KVComm",
            ],
            "failure_modes_ood": [
                "failure mode",
                "OOD",
                "out-of-distribution",
                "degradation",
                "hurt",
                "generalize",
            ],
            "multi_sharer_scaling": [
                "multi",
                "multiple sharers",
                "multiple senders",
                "two-to-one",
                "2-to-1",
            ],
            "dynamic_selection": [
                "dynamic",
                "adaptive",
                "context-adaptive",
                "layer selection",
                "selective",
            ],
            "latency_memory": [
                "latency",
                "memory",
                "FLOPs",
                "communication cost",
                "throughput",
                "GPU",
            ],
            "positioning_related_work": [
                "novelty",
                "related work",
                "similar",
                "prior work",
                "comparison",
            ],
        }
        rebuttal_cards = [card for card in reference_cards if card.get("kind") == "ref_rebuttal"]
        matrix = []
        for concern_id, keywords in concerns.items():
            hits = []
            for card in rebuttal_cards:
                snippets = _keyword_snippets(card.get("text") or "", keywords, limit=3)
                if snippets:
                    hits.append(
                        {
                            "paper_id": card.get("paper_id"),
                            "title": card.get("title"),
                            "source_path": card.get("source_path"),
                            "snippets": snippets,
                        }
                    )
            priority = "high" if len(hits) >= 2 else ("medium" if hits else "low")
            matrix.append(
                {
                    "concern_id": concern_id,
                    "priority": priority,
                    "hit_count": sum(len(item["snippets"]) for item in hits),
                    "evidence": hits,
                    "experiment_implication": _concern_implication(concern_id),
                }
            )
        return {
            "created_at": now_utc(),
            "source_count": len(rebuttal_cards),
            "matrix": matrix,
            "structured_concerns": _structured_rebuttal_concerns(matrix),
            "top_concerns": [
                item["concern_id"]
                for item in sorted(matrix, key=lambda value: (value["priority"] != "high", -value["hit_count"]))[:5]
                if item["hit_count"] > 0
            ],
        }

    def build_rebuttal_chunks(self, reference_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chunks: list[dict[str, Any]] = []
        for card in reference_cards:
            if card.get("kind") != "ref_rebuttal":
                continue
            text = card.get("text") or ""
            for idx, chunk_text in enumerate(_chunk_text(text, max_chars=1600, overlap=160)):
                chunks.append(
                    {
                        "chunk_id": f"{card.get('paper_id')}:rebuttal:{idx:04d}",
                        "paper_id": card.get("paper_id"),
                        "title": card.get("title"),
                        "source_path": card.get("source_path"),
                        "local_path": card.get("local_path"),
                        "kind": "ref_rebuttal",
                        "section": "rebuttal_or_review",
                        "text": chunk_text,
                        "keywords": _chunk_keywords(chunk_text, extra_terms=["rebuttal", "review", card.get("title", "")]),
                        "tokens_estimate": max(1, len(chunk_text) // 4),
                    }
                )
        return chunks

    def build_code_cards(self, repo_manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        del repo_manifest
        cards = []
        for rel in CORE_FILES:
            path = self.repo_root / rel
            if not path.exists() or not path.is_file():
                cards.append({"path": rel, "exists": False})
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            card = {
                "path": rel,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "language": _language_for_path(path),
                "symbols": _extract_code_symbols(text, rel),
                "config_knobs": _extract_config_knobs(text),
                "imports": _extract_imports(text),
                "summary_snippet": text[:1800],
            }
            cards.append(card)
        return cards

    def build_code_intake(self) -> CodeIntakeResult:
        cache_dir = self.project_root / ".cache" / "auto_research" / "code_intake"
        return build_code_intake(self.repo_root, allowed_files=self.allowed_files, allowed_prefixes=self.allowed_prefixes, cache_dir=cache_dir)

    def build_code_chunks(self, code_cards: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        del code_cards
        return self.build_code_intake().chunks

    def build_chunk_index(
        self,
        *,
        paper_chunks: list[dict[str, Any]],
        rebuttal_chunks: list[dict[str, Any]],
        code_chunks: list[dict[str, Any]],
    ) -> dict[str, Any]:
        entries = []
        for source_type, chunks in [("paper", paper_chunks), ("rebuttal", rebuttal_chunks), ("code", code_chunks)]:
            for idx, chunk in enumerate(chunks):
                entries.append(_chunk_index_entry(chunk, source_type=source_type, ordinal=idx))
        by_source_type: dict[str, int] = {}
        for entry in entries:
            by_source_type[entry["source_type"]] = by_source_type.get(entry["source_type"], 0) + 1
        return {
            "schema_version": "c2c_full_chunk_index_v1",
            "created_at": now_utc(),
            "counts": {
                "total": len(entries),
                "paper": by_source_type.get("paper", 0),
                "rebuttal": by_source_type.get("rebuttal", 0),
                "code": by_source_type.get("code", 0),
            },
            "entries": entries,
        }

    def _chunkable_code_paths(self) -> list[str]:
        paths: set[str] = set(CORE_FILES)
        for path in sorted(self.repo_root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.repo_root).as_posix()
            if not _is_chunkable_repo_file(path, rel):
                continue
            paths.add(rel)
        return sorted(paths)

    def build_research_retrieval_plan(
        self,
        *,
        topic: str,
        repo_card: dict[str, Any],
        paper_cards: list[dict[str, Any]],
        paper_chunks: list[dict[str, Any]],
        rebuttal_matrix: dict[str, Any],
        rebuttal_chunks: list[dict[str, Any]],
        code_cards: list[dict[str, Any]],
        code_chunks: list[dict[str, Any]],
        negative_memory: dict[str, Any],
        baseline: dict[str, Any],
    ) -> dict[str, Any]:
        focus_terms = _topic_tags(topic)
        focus_terms.extend(_collect_focus_terms_from_baseline(baseline))
        focus_terms.extend(_collect_focus_terms_from_rebuttal(rebuttal_matrix))
        focus_terms.extend(_collect_focus_terms_from_code(code_cards))
        focus_terms = _deduplicate_strings(focus_terms)[:24]
        plan = {
            "topic": topic,
            "baseline": {
                "name": baseline.get("name"),
                "mean": baseline.get("mean"),
                "datasets": baseline.get("datasets"),
            },
            "focus_terms": focus_terms,
            "questions": [
                {
                    "question_id": "mechanism",
                    "question": "Which code paths and symbols implement the proposed mechanism?",
                    "priority_terms": _deduplicate_strings(
                        [
                            *focus_terms,
                            "alignment",
                            "confidence",
                            "gate",
                            "cache",
                            "projector",
                            "wrapper",
                        ]
                        + _collect_focus_terms_from_code(code_cards)
                    )[:16],
                },
                {
                    "question_id": "paper_support",
                    "question": "Which paper sections state the main contribution, method, and evaluation?",
                    "priority_terms": _deduplicate_strings(
                        [
                            *focus_terms,
                            "method",
                            "experiments",
                            "ablation",
                            "results",
                            "analysis",
                        ]
                    )[:16],
                },
                {
                    "question_id": "rebuttal_risk",
                    "question": "Which rebuttal snippets and failure memories constrain the idea?",
                    "priority_terms": _deduplicate_strings(
                        [
                            *focus_terms,
                            "baseline",
                            "regression",
                            "failure",
                            "novelty",
                            "ood",
                            "weak baseline",
                        ]
                    )[:16],
                },
            ],
            "paper_targets": _rank_chunk_targets(paper_chunks, focus_terms, max_items=16),
            "rebuttal_targets": _rank_chunk_targets(rebuttal_chunks, focus_terms, max_items=16),
            "code_targets": _rank_chunk_targets(code_chunks, focus_terms, max_items=24),
            "code_symbols": _rank_code_symbols(code_cards, focus_terms),
            "repo_evidence": {
                "editable_surface": repo_card.get("editable_surface", {}),
                "protocol_constraints": repo_card.get("protocol_constraints", []),
            },
            "negative_constraints": {
                "blocked_idea_patterns": negative_memory.get("blocked_idea_patterns", [])[:12],
                "top_concerns": rebuttal_matrix.get("top_concerns", [])[:8],
            },
        }
        return plan

    def build_research_followup_bundle(
        self,
        retrieval_plan: dict[str, Any],
        *,
        paper_chunks: list[dict[str, Any]],
        rebuttal_chunks: list[dict[str, Any]],
        code_chunks: list[dict[str, Any]],
        negative_memory: dict[str, Any],
    ) -> dict[str, Any]:
        questions = []
        for spec in retrieval_plan.get("questions", []):
            if not isinstance(spec, dict):
                continue
            terms = _deduplicate_strings(
                [
                    *(spec.get("priority_terms") or []),
                    *(spec.get("question", "").split()),
                    *(negative_memory.get("blocked_idea_patterns") or [])[:6],
                ]
            )[:20]
            question_id = str(spec.get("question_id") or "followup")
            paper_targets = _rank_chunk_targets(paper_chunks, terms, max_items=6)
            rebuttal_targets = _rank_chunk_targets(rebuttal_chunks, terms, max_items=6)
            code_targets = _rank_chunk_targets(code_chunks, terms, max_items=8)
            questions.append(
                {
                    "question_id": question_id,
                    "question": spec.get("question"),
                    "priority_terms": terms,
                    "paper_targets": paper_targets,
                    "rebuttal_targets": rebuttal_targets,
                    "code_targets": code_targets,
                    "cross_source_targets": _merge_target_groups(paper_targets, rebuttal_targets, code_targets, max_items=12),
                }
            )
        return {
            "topic": retrieval_plan.get("topic"),
            "baseline": retrieval_plan.get("baseline", {}),
            "questions": questions,
            "negative_constraints": retrieval_plan.get("negative_constraints", {}),
        }

    def build_result_ledger_csv(self, historical_results: dict[str, Any], baseline: dict[str, Any]) -> str:
        rows = self.result_ledger_rows(historical_results, baseline)
        fields = [
            "method",
            "kind",
            "route_family",
            "mean",
            "delta_vs_baseline",
            "mmlu_redux",
            "ai2_arc",
            "openbookqa",
            "verdict",
            "source",
        ]
        handle = io.StringIO()
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
        return handle.getvalue()

    def result_ledger_rows(self, historical_results: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
        baseline_mean = _safe_float(baseline.get("mean")) or DEFAULT_BASELINE["mean"]
        rows = []
        for record in historical_results.get("records", []):
            if record.get("kind") == "small_loop":
                metrics = record.get("metrics") or {}
                method = record.get("method") or Path(record.get("source", "")).stem
                mean = _safe_float(metrics.get("mean"))
                delta = round(mean - baseline_mean, 4) if mean is not None else None
                rows.append(
                    {
                        "method": method,
                        "kind": "small_loop",
                        "route_family": _route_family(record.get("source", ""), method),
                        "mean": mean,
                        "delta_vs_baseline": delta,
                        "mmlu_redux": metrics.get("mmlu-redux"),
                        "ai2_arc": metrics.get("ai2-arc"),
                        "openbookqa": metrics.get("openbookqa"),
                        "verdict": _ledger_verdict(delta),
                        "source": record.get("source"),
                    }
                )
        rows.sort(
            key=lambda item: (
                item.get("mean") is None,
                -(item.get("mean") or -999.0),
                item.get("method") or "",
            )
        )
        return rows

    def build_negative_result_memory(self, historical_results: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        baseline_mean = _safe_float(baseline.get("mean")) or DEFAULT_BASELINE["mean"]
        ledger_rows = self.result_ledger_rows(historical_results, baseline)
        failed_rows = [
            row
            for row in ledger_rows
            if row.get("mean") is not None and row.get("delta_vs_baseline") is not None and row["delta_vs_baseline"] < 0
        ]
        route_notes = [
            note
            for note in historical_results.get("notes", [])
            if "route3" in (note.get("path", "") + note.get("snippet", "")).lower()
        ]
        memory_items = []
        for row in failed_rows[:40]:
            method = row.get("method", "")
            memory_items.append(
                {
                    "method": method,
                    "route_family": row.get("route_family"),
                    "mean": row.get("mean"),
                    "delta_vs_baseline": row.get("delta_vs_baseline"),
                    "source": row.get("source"),
                    "likely_failure_reason": _failure_reason(method, row.get("source", "")),
                    "avoid_repeat_rule": _avoid_repeat_rule(method, row.get("source", "")),
                }
            )
        blocked_patterns = sorted({item["avoid_repeat_rule"] for item in memory_items if item.get("avoid_repeat_rule")})
        if route_notes:
            blocked_patterns.append("Do not re-run learned router variants without explicit anti-anchor or direct utility supervision evidence.")
        return {
            "created_at": now_utc(),
            "baseline_name": baseline.get("name"),
            "baseline_mean": baseline_mean,
            "failed_result_count": len(failed_rows),
            "items": memory_items,
            "blocked_idea_patterns": blocked_patterns,
            "route3_notes": route_notes[:3],
        }

    def baseline_evidence(self, historical_results: dict[str, Any] | None = None) -> dict[str, Any]:
        historical_results = historical_results or self.import_historical_results()
        baseline = self.baseline
        matching = [
            record
            for record in historical_results.get("records", [])
            if record.get("kind") == "small_loop" and record.get("method") == baseline.get("name")
        ]
        preferred_source = baseline.get("source", "")
        matching.sort(
            key=lambda record: (
                0 if record.get("source") == preferred_source else 1,
                0 if _has_complete_baseline_metrics(record) else 1,
                record.get("source", ""),
            )
        )
        for record in matching:
            if _has_complete_baseline_metrics(record):
                return {
                    "name": record["method"],
                    "mean": record["metrics"]["mean"],
                    "datasets": {
                        "mmlu-redux": record["metrics"].get("mmlu-redux"),
                        "ai2-arc": record["metrics"].get("ai2-arc"),
                        "openbookqa": record["metrics"].get("openbookqa"),
                    },
                    "source": record.get("source"),
                    "record": record,
                }
        return baseline

    def import_reference_materials(self) -> dict[str, Any]:
        missing = self.missing_reference_paths()
        if missing:
            return {"status": "blocked", "missing": missing, "cards": []}

        cards = []
        paper_full_manifest = []
        parse_errors = []
        for key in ["ref_paper", "ref_rebuttal"]:
            source = Path(self.c2c_config[key]).expanduser().resolve()
            for item in _iter_reference_files(source, key):
                paper_id = f"c2c_{key}_{sanitize_filename(item.stem, max_length=48)}"
                target, parse_result = self._import_reference_file(item, key=key, paper_id=paper_id)
                if parse_result.get("status") == "failed" and not parse_result.get("fallback_used"):
                    parse_errors.append(parse_result)
                    continue
                card = {
                    "paper_id": paper_id,
                    "title": _reference_title(item, key),
                    "source_path": str(item),
                    "local_path": target.relative_to(self.project_root).as_posix(),
                    "kind": key,
                    "sha256": sha256_file(target),
                    "text": parse_result.get("text") or extract_reference_text(target),
                    "parser": parse_result.get("parser") or _parser_for_path(target),
                    "parser_status": parse_result.get("status") or "ok",
                    "parser_artifacts": parse_result.get("artifacts") or [],
                }
                if parse_result.get("model_version"):
                    card["parser_model_version"] = parse_result["model_version"]
                if parse_result.get("language"):
                    card["parser_language"] = parse_result["language"]
                if parse_result.get("parser_config_hash"):
                    card["parser_config_hash"] = parse_result["parser_config_hash"]
                if parse_result.get("prompt_schema_version"):
                    card["parser_prompt_schema_version"] = parse_result["prompt_schema_version"]
                if parse_result.get("paper_full_md_path"):
                    card["paper_full_md_path"] = parse_result["paper_full_md_path"]
                if parse_result.get("mineru_result_path"):
                    card["mineru_result_path"] = parse_result["mineru_result_path"]
                if parse_result.get("cache_status"):
                    card["parser_cache_status"] = parse_result["cache_status"]
                if key == "ref_paper" and card.get("paper_full_md_path"):
                    paper_full_manifest.append(
                        {
                            "paper_id": paper_id,
                            "title": card["title"],
                            "source_path": str(item),
                            "local_path": card["local_path"],
                            "sha256": card["sha256"],
                            "paper_full_md_path": card.get("paper_full_md_path"),
                            "parser": card.get("parser"),
                            "parser_status": card.get("parser_status"),
                            "cache_status": card.get("parser_cache_status", "disabled"),
                            "parser_artifacts": card.get("parser_artifacts"),
                            "model_version": card.get("parser_model_version"),
                            "language": card.get("parser_language"),
                            "parser_config_hash": card.get("parser_config_hash"),
                            "prompt_schema_version": card.get("parser_prompt_schema_version"),
                        }
                    )
                cards.append(card)
        if parse_errors:
            return {"status": "blocked", "missing": [], "cards": cards, "parse_errors": parse_errors}
        if not cards:
            return {"status": "blocked", "missing": ["ref files"], "cards": []}
        self._register_reference_cards(cards)
        return {"status": "ok", "missing": [], "cards": cards, "paper_full_manifest": paper_full_manifest}

    def _import_reference_file(self, source: Path, *, key: str, paper_id: str) -> tuple[Path, dict[str, Any]]:
        if key == "ref_paper" and source.suffix.lower() == ".pdf":
            return self._import_pdf_with_mineru(source, paper_id=paper_id)
        target = self.project_root / "references" / "c2c" / key / source.name
        ensure_dir(target.parent)
        shutil.copy2(source, target)
        return target, {
            "status": "ok",
            "parser": _parser_for_path(target),
            "text": extract_reference_text(target),
            "artifacts": [target.relative_to(self.project_root).as_posix()],
        }

    def _import_pdf_with_mineru(self, source: Path, *, paper_id: str) -> tuple[Path, dict[str, Any]]:
        output_dir = self.project_root / "references" / "c2c" / "ref_paper" / paper_id
        ensure_dir(output_dir)
        target = output_dir / "source.pdf"
        shutil.copy2(source, target)
        source_sha = sha256_file(target)
        metadata_path = output_dir / "mineru_result.json"
        paper_full_path = output_dir / "paper_full.md"
        pdf_cfg = self.pdf_ingest_config
        parser_config_hash = _mineru_parser_config_hash(pdf_cfg)
        cache_dir = self.project_root / ".cache" / "auto_research" / "mineru_pdf" / source_sha
        cache_md_path = cache_dir / "paper_full.md"
        cache_result_path = cache_dir / "mineru_result.json"
        shared_dir = shared_cache_root(self.project_root, self.config) / "mineru_pdf" / source_sha / parser_config_hash
        shared_md_path = shared_dir / "paper_full.md"
        shared_result_path = shared_dir / "mineru_result.json"
        cached = read_json(metadata_path, default={}) if metadata_path.exists() else {}
        if (
            paper_full_path.exists()
            and paper_full_path.stat().st_size > 0
            and isinstance(cached, dict)
            and cached.get("source_sha256") == source_sha
        ):
            _write_mineru_cache_copy(shared_md_path, shared_result_path, paper_full_path, metadata_path)
            text = paper_full_path.read_text(encoding="utf-8", errors="ignore")
            return target, _mineru_parse_payload(
                target=target,
                project_root=self.project_root,
                paper_full_path=paper_full_path,
                metadata_path=metadata_path,
                metadata=cached,
                pdf_cfg=pdf_cfg,
                cache_status="local_hit",
                text=text,
            )
        for candidate_md, candidate_result, cache_status in [
            (cache_md_path, cache_result_path, "sha_hit"),
            (shared_md_path, shared_result_path, "shared_hit"),
        ]:
            restored = _restore_mineru_cache_candidate(
                candidate_md,
                candidate_result,
                target=target,
                paper_full_path=paper_full_path,
                metadata_path=metadata_path,
                project_root=self.project_root,
                source=source,
                source_sha=source_sha,
                pdf_cfg=pdf_cfg,
                parser_config_hash=parser_config_hash,
                cache_status=cache_status,
            )
            if restored is not None:
                _write_mineru_cache_copy(cache_md_path, cache_result_path, paper_full_path, metadata_path)
                _write_mineru_cache_copy(shared_md_path, shared_result_path, paper_full_path, metadata_path)
                return target, restored
        legacy = _restore_legacy_mineru_cache(
            self.project_root,
            target=target,
            paper_full_path=paper_full_path,
            metadata_path=metadata_path,
            source=source,
            source_sha=source_sha,
            pdf_cfg=pdf_cfg,
            parser_config_hash=parser_config_hash,
        )
        if legacy is not None:
            _write_mineru_cache_copy(cache_md_path, cache_result_path, paper_full_path, metadata_path)
            _write_mineru_cache_copy(shared_md_path, shared_result_path, paper_full_path, metadata_path)
            return target, legacy

        provider = str(pdf_cfg.get("provider") or "mineru")
        if provider != "mineru":
            text = extract_reference_text(target)
            return target, {
                "status": "ok",
                "parser": _parser_for_path(target),
                "text": text,
                "artifacts": [target.relative_to(self.project_root).as_posix()],
                "fallback_used": True,
            }
        try:
            client = MinerUPdfClient(
                model_version=str(pdf_cfg.get("model_version") or "vlm"),
                language=str(pdf_cfg.get("language") or "en"),
                enable_formula=bool(pdf_cfg.get("enable_formula", True)),
                enable_table=bool(pdf_cfg.get("enable_table", True)),
                is_ocr=bool(pdf_cfg.get("is_ocr", False)),
                poll_interval_seconds=int(pdf_cfg.get("poll_interval_seconds") or 5),
                timeout_seconds=int(pdf_cfg.get("timeout_seconds") or 900),
                request_timeout_seconds=int(pdf_cfg.get("request_timeout_seconds") or 60),
            )
            result = client.parse_pdf(target, output_dir, data_id=paper_id, title=_reference_title(source, "ref_paper"))
            result["source_sha256"] = source_sha
            result["source_path"] = str(source)
            result["local_pdf_path"] = target.relative_to(self.project_root).as_posix()
            result["cache_status"] = "miss"
            result["parser_config_hash"] = parser_config_hash
            result["prompt_schema_version"] = "c2c_paper_full_markdown_v1"
            write_json(metadata_path, result)
            _write_mineru_cache_copy(cache_md_path, cache_result_path, paper_full_path, metadata_path)
            _write_mineru_cache_copy(shared_md_path, shared_result_path, paper_full_path, metadata_path)
            text = paper_full_path.read_text(encoding="utf-8", errors="ignore")
            return target, {
                "status": "ok",
                "parser": "mineru",
                "cache_status": "miss",
                "text": text,
                "paper_full_md_path": paper_full_path.relative_to(self.project_root).as_posix(),
                "mineru_result_path": metadata_path.relative_to(self.project_root).as_posix(),
                "model_version": result.get("model_version"),
                "language": result.get("language"),
                "parser_config_hash": result.get("parser_config_hash"),
                "prompt_schema_version": result.get("prompt_schema_version"),
                "artifacts": [
                    target.relative_to(self.project_root).as_posix(),
                    paper_full_path.relative_to(self.project_root).as_posix(),
                    metadata_path.relative_to(self.project_root).as_posix(),
                ],
            }
        except MinerUError as exc:
            result = {
                "provider": "mineru",
                "schema_version": "mineru_pdf_parse_result_v1",
                "created_at": now_utc(),
                "state": "failed",
                "source_sha256": source_sha,
                "source_path": str(source),
                "local_pdf_path": target.relative_to(self.project_root).as_posix(),
                "err_msg": str(exc),
                "cache_status": "miss",
                "parser_config_hash": parser_config_hash,
                "prompt_schema_version": "c2c_paper_full_markdown_v1",
            }
            write_json(metadata_path, result)
            if bool(pdf_cfg.get("fallback_to_pypdf", False)):
                text = extract_reference_text(target)
                return target, {
                    "status": "ok",
                    "parser": "pypdf_fallback",
                    "text": text,
                    "mineru_result_path": metadata_path.relative_to(self.project_root).as_posix(),
                    "artifacts": [
                        target.relative_to(self.project_root).as_posix(),
                        metadata_path.relative_to(self.project_root).as_posix(),
                    ],
                    "fallback_used": True,
                }
            return target, {
                "status": "failed",
                "parser": "mineru",
                "error": str(exc),
                "mineru_result_path": metadata_path.relative_to(self.project_root).as_posix(),
                "artifacts": [
                    target.relative_to(self.project_root).as_posix(),
                    metadata_path.relative_to(self.project_root).as_posix(),
                ],
            }

    def materialize_candidate_configs(
        self,
        candidate: dict[str, Any],
        gpu_selection: Any | None = None,
        *,
        proxy_gpu_selection: Any | None = None,
    ) -> dict[str, Any]:
        run_id = sanitize_filename(candidate.get("id") or candidate.get("title") or "candidate")
        run_root_rel = f"local/auto_research_runs/{run_id}"
        run_root = self.repo_root / run_root_rel
        ensure_dir(run_root)
        runtime_localization = self.localize_runtime_model_literals()
        selected_gpu_ids = list(getattr(gpu_selection, "selected_ids", []) or [])
        proxy_selection = proxy_gpu_selection if proxy_gpu_selection is not None else gpu_selection
        proxy_selected_gpu_ids = list(getattr(proxy_selection, "selected_ids", []) or [])
        config_overrides = c2c_candidate_config_overrides(candidate)

        train_template = self._train_template_path()
        train_config = _read_json_fallback(train_template, default={})
        if not train_config:
            train_config = self._minimal_train_config()
        self._localize_model_references(train_config)
        train_config = deep_merge(train_config, config_overrides["train"])
        train_config.setdefault("output", {})
        train_config["output"]["output_dir"] = f"{run_root_rel}/checkpoints"
        train_config.setdefault("data", {}).setdefault("kwargs", {})
        train_config["data"]["kwargs"]["num_samples"] = int(self.c2c_config.get("small_loop", {}).get("train_samples", 2048))
        train_resource_adjustment = _configure_c2c_train_resource_limits(
            train_config,
            self.c2c_config.get("small_loop", {}).get("train_resource_policy") or {},
            selected_gpu_ids=selected_gpu_ids,
        )
        _configure_disabled_wandb(train_config, run_id)
        train_config_path = run_root / "train_recipe.json"
        train_config_path.write_text(json.dumps(train_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        eval_config_paths = {}
        for dataset in self.c2c_config.get("small_loop", {}).get("eval_datasets", DEFAULT_DATASETS):
            eval_config = self._eval_template(dataset)
            self._localize_model_references(eval_config)
            eval_config = deep_merge(eval_config, config_overrides["eval"])
            eval_config.setdefault("model", {}).setdefault("rosetta_config", {})
            eval_config["model"]["rosetta_config"]["checkpoints_dir"] = f"{run_root_rel}/checkpoints/final"
            eval_config.setdefault("output", {})
            eval_config["output"]["output_dir"] = f"{run_root_rel}/results/{dataset}"
            eval_config.setdefault("eval", {})
            eval_config["eval"]["dataset"] = dataset
            eval_config["eval"]["gpu_ids"] = selected_gpu_ids or _coerce_gpu_ids(self.c2c_config.get("small_loop", {}).get("gpu_ids", [0]))
            eval_limit = self.c2c_config.get("small_loop", {}).get("eval_limit")
            if eval_limit:
                eval_config["eval"]["limit"] = int(eval_limit)
            eval_path = run_root / f"eval_{dataset}.yaml"
            eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False, allow_unicode=False), encoding="utf-8")
            eval_config_paths[dataset] = eval_path

        proxy_screen = self._materialize_proxy_screen_configs(
            run_id=run_id,
            run_root=run_root,
            run_root_rel=run_root_rel,
            config_overrides=config_overrides,
            selected_gpu_ids=proxy_selected_gpu_ids,
        )

        return {
            "run_id": run_id,
            "run_root": run_root,
            "train_config": train_config_path,
            "eval_configs": eval_config_paths,
            "proxy_screen": proxy_screen,
            "preflight_path": run_root / "preflight.json",
            "run_state_path": run_root / "run_state.json",
            "config_overrides": config_overrides,
            "has_executable_change": bool(config_overrides["train"] or config_overrides["eval"]),
            "runtime_localization": runtime_localization,
            "gpu_selection": {
                "selected_gpu_ids": selected_gpu_ids,
                "cuda_visible_devices": ",".join(str(item) for item in selected_gpu_ids),
                "policy": getattr(gpu_selection, "policy", {}) if gpu_selection else {},
                "reason": getattr(gpu_selection, "reason", "") if gpu_selection else "",
                "snapshot": getattr(gpu_selection, "snapshot", []) if gpu_selection else [],
            },
            "proxy_gpu_selection": {
                "selected_gpu_ids": proxy_selected_gpu_ids,
                "cuda_visible_devices": ",".join(str(item) for item in proxy_selected_gpu_ids),
                "policy": getattr(proxy_selection, "policy", {}) if proxy_selection else {},
                "reason": getattr(proxy_selection, "reason", "") if proxy_selection else "",
                "snapshot": getattr(proxy_selection, "snapshot", []) if proxy_selection else [],
            },
            "train_resource_adjustment": train_resource_adjustment,
            "frozen_hashes": self._frozen_hashes(train_config_path, eval_config_paths),
            "commands": self._candidate_commands(train_config_path, eval_config_paths, selected_gpu_ids),
            "candidate": candidate,
        }

    def materialize_train_oom_recovery_config(
        self,
        run_spec: dict[str, Any],
        *,
        gpu_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        recovery_cfg = deep_merge(
            copy.deepcopy(DEFAULT_C2C_FULL_TRAIN_OOM_RECOVERY),
            self.c2c_config.get("small_loop", {}).get("full_train_oom_recovery") or {},
        )
        if not recovery_cfg.get("enabled", True):
            return {"enabled": False, "status": "disabled"}
        source_path = Path(run_spec["train_config"])
        train_config = copy.deepcopy(_read_json_fallback(source_path, default={}) or {})
        if not train_config:
            return {"enabled": True, "status": "failed", "reason": f"could not read train config: {source_path}"}
        training = train_config.setdefault("training", {})
        if not isinstance(training, dict):
            train_config["training"] = training = {}
        original_training = copy.deepcopy(training)
        safe_batch = max(1, _safe_int(recovery_cfg.get("per_device_train_batch_size")) or 1)
        original_batch = max(1, _safe_int(original_training.get("per_device_train_batch_size")) or safe_batch)
        original_grad = max(1, _safe_int(original_training.get("gradient_accumulation_steps")) or 1)
        original_gpu_count = max(1, len(gpu_ids or []))
        original_effective_batch = original_batch * original_grad * original_gpu_count
        training["per_device_train_batch_size"] = safe_batch
        configured_grad = _safe_int(recovery_cfg.get("gradient_accumulation_steps"))
        if configured_grad:
            training["gradient_accumulation_steps"] = max(1, configured_grad)
        elif recovery_cfg.get("preserve_effective_batch", True):
            training["gradient_accumulation_steps"] = max(1, (original_grad * original_batch + safe_batch - 1) // safe_batch)
        recovered_effective_batch = safe_batch * max(1, _safe_int(training.get("gradient_accumulation_steps")) or 1) * original_gpu_count
        lr_adjustment = _maybe_scale_learning_rate(
            training,
            original_training,
            original_effective_batch=original_effective_batch,
            new_effective_batch=recovered_effective_batch,
            policy=recovery_cfg,
        )
        max_length = _safe_int(recovery_cfg.get("max_length"))
        if max_length:
            original_length = _safe_int(original_training.get("max_length"))
            training["max_length"] = min(original_length, max_length) if original_length else max_length
        train_samples = _safe_int(recovery_cfg.get("train_samples"))
        if train_samples:
            data_kwargs = train_config.setdefault("data", {}).setdefault("kwargs", {})
            if isinstance(data_kwargs, dict):
                original_samples = _safe_int(data_kwargs.get("num_samples"))
                data_kwargs["num_samples"] = min(original_samples, train_samples) if original_samples else train_samples
        tag = sanitize_filename(str(recovery_cfg.get("tag") or "memory_safe"))
        recovery_path = Path(run_spec["run_root"]) / f"train_recipe_{tag}.json"
        recovery_path.write_text(json.dumps(train_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        selected_gpu_ids = list(gpu_ids or [])
        commands = self._candidate_commands(recovery_path, run_spec["eval_configs"], selected_gpu_ids)
        return {
            "enabled": True,
            "status": "materialized",
            "tag": tag,
            "train_config": recovery_path,
            "command": commands["train"],
            "gpu_ids": selected_gpu_ids,
            "config_changes": {
                "original_per_device_train_batch_size": original_training.get("per_device_train_batch_size"),
                "per_device_train_batch_size": training.get("per_device_train_batch_size"),
                "original_gradient_accumulation_steps": original_training.get("gradient_accumulation_steps"),
                "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
                "original_effective_batch_size": original_effective_batch,
                "effective_batch_size": recovered_effective_batch,
                "learning_rate": training.get("learning_rate"),
                "learning_rate_adjustment": lr_adjustment,
                "original_max_length": original_training.get("max_length"),
                "max_length": training.get("max_length"),
                "train_samples": ((train_config.get("data") or {}).get("kwargs") or {}).get("num_samples"),
            },
            "sha256": sha256_file(recovery_path),
        }

    def materialize_ablation_eval_configs(
        self,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        gpu_selection: Any | None = None,
    ) -> dict[str, Any]:
        contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
        ablation_plan = candidate.get("ablation_plan") if isinstance(candidate.get("ablation_plan"), dict) else {}
        switch = contract.get("ablation_switch") or ablation_plan.get("switch")
        if not switch:
            return {"enabled": False, "status": "skipped", "reason": "candidate has no ablation_switch"}
        selected_gpu_ids = list(getattr(gpu_selection, "selected_ids", []) or [])
        run_root = Path(run_spec["run_root"])
        run_root_rel = f"local/auto_research_runs/{run_spec['run_id']}"
        ablation_root = run_root / "ablation_disabled"
        ensure_dir(ablation_root)
        config_overrides = c2c_candidate_config_overrides(candidate)
        eval_config_paths: dict[str, Path] = {}
        for dataset in self.c2c_config.get("small_loop", {}).get("eval_datasets", DEFAULT_DATASETS):
            eval_config = self._eval_template(dataset)
            self._localize_model_references(eval_config)
            eval_config = deep_merge(eval_config, config_overrides["eval"])
            eval_config.setdefault("model", {}).setdefault("rosetta_config", {})
            eval_config["model"]["rosetta_config"]["checkpoints_dir"] = f"{run_root_rel}/checkpoints/final"
            eval_config["model"]["rosetta_config"][str(switch)] = True
            eval_config.setdefault("output", {})
            eval_config["output"]["output_dir"] = f"{run_root_rel}/ablation_disabled/results/{dataset}"
            eval_config.setdefault("eval", {})
            eval_config["eval"]["dataset"] = dataset
            eval_config["eval"]["gpu_ids"] = selected_gpu_ids or _coerce_gpu_ids(self.c2c_config.get("small_loop", {}).get("gpu_ids", [0]))
            eval_limit = self.c2c_config.get("small_loop", {}).get("eval_limit")
            if eval_limit:
                eval_config["eval"]["limit"] = int(eval_limit)
            eval_path = ablation_root / f"eval_{dataset}.yaml"
            eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False, allow_unicode=False), encoding="utf-8")
            eval_config_paths[dataset] = eval_path
        return {
            "enabled": True,
            "status": "materialized",
            "switch": str(switch),
            "run_root": ablation_root,
            "eval_configs": eval_config_paths,
            "metrics_path": ablation_root / "ablation_metrics.json",
            "commands": {"eval": self._candidate_commands(Path(run_spec["train_config"]), eval_config_paths, selected_gpu_ids)["eval"]},
            "frozen_hashes": self._frozen_hashes(Path(run_spec["train_config"]), eval_config_paths),
        }

    def materialize_proxy_activation_smoke_configs(
        self,
        candidate: dict[str, Any],
        run_spec: dict[str, Any],
        gpu_selection: Any | None = None,
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.config)
        smoke_cfg = proxy_cfg.get("activation_smoke") if isinstance(proxy_cfg.get("activation_smoke"), dict) else {}
        if not smoke_cfg.get("enabled", True):
            return {"enabled": False, "status": "skipped", "reason": "activation_smoke disabled"}
        proxy_spec = run_spec.get("proxy_screen") or {}
        if not proxy_spec.get("enabled", False):
            return {"enabled": False, "status": "skipped", "reason": "proxy_screen disabled"}
        switch = _candidate_ablation_switch(candidate)
        if not switch:
            status = "failed" if smoke_cfg.get("require_ablation_switch", True) else "skipped"
            return {
                "enabled": True,
                "status": status,
                "reason": "candidate has no ablation_switch for proxy activation smoke",
                "repair_hint": "add an ablation_switch that disables only the proposed mechanism in eval rosetta_config",
                "hard_gate": bool(smoke_cfg.get("hard_gate", True)),
            }

        proxy_eval_configs = proxy_spec.get("eval_configs") if isinstance(proxy_spec.get("eval_configs"), dict) else {}
        if not proxy_eval_configs:
            return {"enabled": True, "status": "failed", "reason": "proxy eval configs missing for activation smoke"}

        selected_gpu_ids = list(getattr(gpu_selection, "selected_ids", []) or [])
        run_root = Path(run_spec["run_root"])
        run_root_rel = f"local/auto_research_runs/{run_spec['run_id']}"
        proxy_root = Path(proxy_spec.get("run_root") or run_root / "proxy")
        smoke_root = proxy_root / "activation_smoke_disabled"
        ensure_dir(smoke_root)
        smoke_root_rel = f"{run_root_rel}/proxy/activation_smoke_disabled"
        proxy_checkpoint_rel = f"{run_root_rel}/proxy/checkpoints/final"
        config_overrides = c2c_candidate_config_overrides(candidate)
        proxy_datasets = [str(dataset) for dataset in proxy_eval_configs.keys()]
        requested = [str(dataset) for dataset in (smoke_cfg.get("datasets") or []) if str(dataset) in proxy_datasets]
        if not requested:
            requested = proxy_datasets
        max_datasets = max(1, int(smoke_cfg.get("max_datasets") or 1))
        datasets = requested[:max_datasets]

        eval_config_paths: dict[str, Path] = {}
        for dataset in datasets:
            eval_config = self._eval_template(dataset)
            self._localize_model_references(eval_config)
            eval_config = deep_merge(eval_config, config_overrides["eval"])
            eval_config.setdefault("model", {}).setdefault("rosetta_config", {})
            eval_config["model"]["rosetta_config"]["checkpoints_dir"] = proxy_checkpoint_rel
            eval_config["model"]["rosetta_config"][str(switch)] = True
            eval_config.setdefault("output", {})
            eval_config["output"]["output_dir"] = f"{smoke_root_rel}/results/{dataset}"
            eval_config.setdefault("eval", {})
            eval_config["eval"]["dataset"] = dataset
            eval_config["eval"]["gpu_ids"] = selected_gpu_ids or _coerce_gpu_ids(self.c2c_config.get("small_loop", {}).get("gpu_ids", [0]))
            eval_limit = smoke_cfg.get("eval_limit")
            if eval_limit is None:
                proxy_eval_path = Path(proxy_eval_configs[dataset])
                proxy_eval_config = read_yaml(proxy_eval_path, default={}) if proxy_eval_path.exists() else {}
                eval_limit = ((proxy_eval_config.get("eval") or {}).get("limit") if isinstance(proxy_eval_config, dict) else None)
            if eval_limit:
                eval_config["eval"]["limit"] = int(eval_limit)
            eval_path = smoke_root / f"eval_{dataset}.yaml"
            eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False, allow_unicode=False), encoding="utf-8")
            eval_config_paths[dataset] = eval_path

        return {
            "enabled": True,
            "status": "materialized",
            "switch": str(switch),
            "hard_gate": bool(smoke_cfg.get("hard_gate", True)),
            "run_root": smoke_root,
            "result_root": smoke_root / "results",
            "eval_configs": eval_config_paths,
            "metrics_path": smoke_root / "activation_smoke_metrics.json",
            "datasets": datasets,
            "code_patch_validation": ((candidate.get("code_patch") or {}).get("validation") if isinstance(candidate.get("code_patch"), dict) else None),
            "config": {
                "min_abs_metric_delta": float(smoke_cfg.get("min_abs_metric_delta", 0.01) or 0.0),
                "max_prediction_files": int(smoke_cfg.get("max_prediction_files") or 8),
                "max_prediction_rows": int(smoke_cfg.get("max_prediction_rows") or 512),
                "min_prediction_diff_rate": float(smoke_cfg.get("min_prediction_diff_rate", 0.01) or 0.0),
                "min_answer_diff_rate": float(smoke_cfg.get("min_answer_diff_rate", 0.01) or 0.0),
                "min_mean_output_length_delta": float(smoke_cfg.get("min_mean_output_length_delta", 1.0) or 0.0),
                "require_observable_difference": bool(smoke_cfg.get("require_observable_difference", True)),
                "timeout_seconds": int(smoke_cfg.get("timeout_seconds") or 0) or None,
            },
            "commands": {"eval": self._candidate_commands(Path(run_spec["train_config"]), eval_config_paths, selected_gpu_ids)["eval"]},
            "frozen_hashes": self._frozen_hashes(Path(run_spec["train_config"]), eval_config_paths),
        }

    def collect_proxy_activation_smoke(
        self,
        run_spec: dict[str, Any],
        activation_spec: dict[str, Any],
    ) -> dict[str, Any]:
        if not activation_spec.get("enabled"):
            return activation_spec
        if activation_spec.get("status") != "materialized":
            return activation_spec
        datasets = [str(dataset) for dataset in activation_spec.get("datasets") or []]
        disabled_metrics = self._collect_metrics_from_result_root(Path(activation_spec["result_root"]))
        enabled_metrics = _filter_c2c_metrics_to_datasets(self.collect_proxy_metrics(run_spec), datasets)
        disabled_metrics = _filter_c2c_metrics_to_datasets(disabled_metrics, datasets)
        if disabled_metrics:
            write_json(Path(activation_spec["metrics_path"]), disabled_metrics)

        proxy_cfg = c2c_proxy_screen_config(self.config)
        output_smoke = collect_c2c_eval_smoke(
            Path(activation_spec["result_root"]),
            repo_root=self.repo_root,
            config=((proxy_cfg.get("eval_smoke") or {}) if isinstance(proxy_cfg.get("eval_smoke"), dict) else {}),
        )
        smoke_config = activation_spec.get("config") or {}
        metric_comparison = _proxy_activation_metric_comparison(enabled_metrics, disabled_metrics, min_abs_delta=smoke_config.get("min_abs_metric_delta"))
        prediction_comparison = _proxy_activation_prediction_comparison(
            Path((run_spec.get("proxy_screen") or {}).get("run_root") or Path(run_spec["run_root"]) / "proxy") / "results",
            Path(activation_spec["result_root"]),
            datasets=datasets,
            repo_root=self.repo_root,
            config=smoke_config,
        )
        comparison = {
            **metric_comparison,
            "metric_comparison": metric_comparison,
            "prediction_comparison": prediction_comparison,
            "mechanism_observed": bool(metric_comparison.get("mechanism_observed") or prediction_comparison.get("mechanism_observed")),
        }
        mechanism_trace = _proxy_activation_mechanism_trace(self.project_root, run_spec, activation_spec)
        tensor_trace = mechanism_trace.get("tensor_trace") if isinstance(mechanism_trace.get("tensor_trace"), dict) else {}
        if tensor_trace.get("status") == "changed":
            comparison["mechanism_observed"] = True
            comparison["tensor_mechanism_observed"] = True
        elif (
            tensor_trace.get("status") == "unchanged"
            and metric_comparison.get("mechanism_observed")
            and not prediction_comparison.get("mechanism_observed")
        ):
            comparison["eval_noise_suspected"] = True
            comparison["mechanism_observed"] = False
        if mechanism_trace.get("status") == "wired" and not comparison["mechanism_observed"]:
            comparison["mechanism_wired_metric_neutral"] = True
        status = "passed"
        reason = "proxy activation smoke observed enabled-vs-disabled metric or prediction change"
        repair_hint = ""
        if not enabled_metrics or not disabled_metrics:
            status = "failed"
            reason = "proxy activation smoke missing enabled or disabled metrics"
            repair_hint = "ensure proxy eval writes summary metrics for both enabled and ablation-disabled configs"
        elif _c2c_eval_smoke_hard_failure(output_smoke):
            status = "failed"
            reason = "proxy activation smoke disabled eval output health failed"
            repair_hint = "repair eval output path, prediction format, or answer parsing before full S3"
        elif smoke_config.get("require_observable_difference", True) and not comparison.get("mechanism_observed"):
            if comparison.get("eval_noise_suspected"):
                status = "failed"
                reason = "proxy score changed but mechanism tensor trace did not change"
                repair_hint = "treat this as eval noise; repair the mechanism path so enabled/disabled changes the traced tensors before full S3"
            elif mechanism_trace.get("status") == "wired":
                status = "passed"
                reason = "ablation switch is wired into eval path but produced metric-neutral proxy outputs"
                repair_hint = "mechanism is connected but effect is neutral on activation smoke; repair proxy effect or dataset regression rather than eval wiring"
            else:
                status = "failed"
                reason = "ablation switch produced no observable proxy eval metric or prediction change"
                repair_hint = "wire the mechanism and its ablation switch into the proxy eval path before full S3"

        return {
            **activation_spec,
            "status": status,
            "reason": reason,
            "repair_hint": repair_hint,
            "enabled_metrics": enabled_metrics,
            "disabled_metrics": disabled_metrics,
            "comparison": comparison,
            "output_smoke": output_smoke,
            "mechanism_trace": mechanism_trace,
        }

    def _materialize_proxy_screen_configs(
        self,
        *,
        run_id: str,
        run_root: Path,
        run_root_rel: str,
        config_overrides: dict[str, dict[str, Any]],
        selected_gpu_ids: list[int],
    ) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.config)
        proxy_root = run_root / "proxy"
        baseline_metrics_path = self.project_root / str(proxy_cfg.get("baseline_cache_path") or DEFAULT_C2C_PROXY_SCREEN["baseline_cache_path"])
        proxy_root_rel = f"{run_root_rel}/proxy"
        metrics_path = proxy_root / "proxy_metrics.json"
        enabled = bool(proxy_cfg.get("enabled", False))
        if not enabled:
            return {
                "enabled": False,
                "commands": {},
                "metrics_path": metrics_path,
                "baseline_metrics_path": baseline_metrics_path,
                "run_root": proxy_root,
            }

        ensure_dir(proxy_root)
        train_template = self._train_template_path()
        train_config = copy.deepcopy(_read_json_fallback(train_template, default={}) or self._minimal_train_config())
        self._localize_model_references(train_config)
        train_config = deep_merge(train_config, config_overrides["train"])
        train_config.setdefault("output", {})
        train_config["output"]["output_dir"] = f"{proxy_root_rel}/checkpoints"
        train_config.setdefault("data", {}).setdefault("kwargs", {})
        train_config["data"]["kwargs"]["num_samples"] = int(proxy_cfg.get("train_samples") or 128)
        _configure_proxy_train_limits(train_config, proxy_cfg, selected_gpu_ids=selected_gpu_ids)
        _configure_disabled_wandb(train_config, f"{run_id}_proxy")
        train_config_path = proxy_root / "train_recipe.json"
        train_config_path.write_text(json.dumps(train_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        small_loop = self.c2c_config.get("small_loop", {})
        proxy_datasets = proxy_cfg.get("eval_datasets") or small_loop.get("eval_datasets") or DEFAULT_DATASETS
        eval_config_paths: dict[str, Path] = {}
        for dataset in proxy_datasets:
            eval_config = self._eval_template(dataset)
            self._localize_model_references(eval_config)
            eval_config = deep_merge(eval_config, config_overrides["eval"])
            eval_config.setdefault("model", {}).setdefault("rosetta_config", {})
            eval_config["model"]["rosetta_config"]["checkpoints_dir"] = f"{proxy_root_rel}/checkpoints/final"
            eval_config.setdefault("output", {})
            eval_config["output"]["output_dir"] = f"{proxy_root_rel}/results/{dataset}"
            eval_config.setdefault("eval", {})
            eval_config["eval"]["dataset"] = dataset
            eval_config["eval"]["gpu_ids"] = selected_gpu_ids or _coerce_gpu_ids(small_loop.get("gpu_ids", [0]))
            eval_limit = proxy_cfg.get("eval_limit")
            if eval_limit:
                eval_config["eval"]["limit"] = int(eval_limit)
            eval_path = proxy_root / f"eval_{dataset}.yaml"
            eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False, allow_unicode=False), encoding="utf-8")
            eval_config_paths[str(dataset)] = eval_path

        return {
            "enabled": True,
            "mode": proxy_cfg.get("mode", "static"),
            "run_root": proxy_root,
            "train_config": train_config_path,
            "eval_configs": eval_config_paths,
            "metrics_path": metrics_path,
            "baseline_metrics_path": baseline_metrics_path,
            "commands": self._candidate_commands(train_config_path, eval_config_paths, selected_gpu_ids),
            "config": {
                "train_samples": int(proxy_cfg.get("train_samples") or 128),
                "eval_limit": int(proxy_cfg.get("eval_limit") or 0) or None,
                "eval_datasets": list(eval_config_paths.keys()),
            },
        }

    def materialize_proxy_baseline_configs(self, gpu_selection: Any | None = None) -> dict[str, Any]:
        proxy_cfg = c2c_proxy_screen_config(self.config)
        run_id = sanitize_filename(str(proxy_cfg.get("baseline_run_id") or "proxy_baseline"))
        run_root_rel = f"local/auto_research_runs/{run_id}"
        run_root = self.repo_root / run_root_rel
        ensure_dir(run_root)
        selected_gpu_ids = list(getattr(gpu_selection, "selected_ids", []) or [])
        runtime_localization = self.localize_runtime_model_literals()

        train_template = self._train_template_path()
        train_config = copy.deepcopy(_read_json_fallback(train_template, default={}) or self._minimal_train_config())
        self._localize_model_references(train_config)
        train_config.setdefault("output", {})
        train_config["output"]["output_dir"] = f"{run_root_rel}/checkpoints"
        train_config.setdefault("data", {}).setdefault("kwargs", {})
        train_config["data"]["kwargs"]["num_samples"] = int(proxy_cfg.get("train_samples") or 128)
        _configure_proxy_train_limits(train_config, proxy_cfg, selected_gpu_ids=selected_gpu_ids)
        _configure_disabled_wandb(train_config, run_id)
        train_config_path = run_root / "train_recipe.json"
        train_config_path.write_text(json.dumps(train_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        small_loop = self.c2c_config.get("small_loop", {})
        proxy_datasets = proxy_cfg.get("eval_datasets") or small_loop.get("eval_datasets") or DEFAULT_DATASETS
        eval_config_paths: dict[str, Path] = {}
        for dataset in proxy_datasets:
            eval_config = self._eval_template(dataset)
            self._localize_model_references(eval_config)
            eval_config.setdefault("model", {}).setdefault("rosetta_config", {})
            eval_config["model"]["rosetta_config"]["checkpoints_dir"] = f"{run_root_rel}/checkpoints/final"
            eval_config.setdefault("output", {})
            eval_config["output"]["output_dir"] = f"{run_root_rel}/results/{dataset}"
            eval_config.setdefault("eval", {})
            eval_config["eval"]["dataset"] = dataset
            eval_config["eval"]["gpu_ids"] = selected_gpu_ids or _coerce_gpu_ids(small_loop.get("gpu_ids", [0]))
            eval_limit = proxy_cfg.get("eval_limit")
            if eval_limit:
                eval_config["eval"]["limit"] = int(eval_limit)
            eval_path = run_root / f"eval_{dataset}.yaml"
            eval_path.write_text(yaml.safe_dump(eval_config, sort_keys=False, allow_unicode=False), encoding="utf-8")
            eval_config_paths[str(dataset)] = eval_path

        return {
            "run_id": run_id,
            "run_root": run_root,
            "train_config": train_config_path,
            "eval_configs": eval_config_paths,
            "metrics_path": self.project_root / str(proxy_cfg.get("baseline_cache_path") or DEFAULT_C2C_PROXY_SCREEN["baseline_cache_path"]),
            "runtime_localization": runtime_localization,
            "commands": self._candidate_commands(train_config_path, eval_config_paths, selected_gpu_ids),
            "config": {
                "train_samples": int(proxy_cfg.get("train_samples") or 128),
                "eval_limit": int(proxy_cfg.get("eval_limit") or 0) or None,
                "eval_datasets": list(eval_config_paths.keys()),
            },
        }

    def preflight(self, run_spec: dict[str, Any], gpu_selection: Any | None = None) -> dict[str, Any]:
        checks: list[dict[str, Any]] = []
        recovery_actions: list[dict[str, Any]] = []
        errors: list[str] = []
        warnings: list[str] = []

        env_path = Path(self.env_python).expanduser()
        env_ok = env_path.exists() and os.access(env_path, os.X_OK)
        _append_check(checks, "env_python", env_ok, path=str(env_path), executable=os.access(env_path, os.X_OK) if env_path.exists() else False)
        if not env_ok:
            errors.append(f"env_python is missing or not executable: {env_path}")

        for model_id, raw_path in self.model_map.items():
            model_path = Path(raw_path).expanduser()
            model_check = self._check_model_path(model_id, model_path)
            checks.append(model_check)
            if not model_check["ok"] and model_check.get("repairable"):
                action = self._repair_model_symlink(model_id, model_path)
                recovery_actions.append(action)
                model_check = self._check_model_path(model_id, model_path)
                model_check["after_recovery"] = True
                checks.append(model_check)
            if not model_check["ok"]:
                errors.append(model_check["reason"])

        dataset_check = self._check_dataset_cache()
        checks.append(dataset_check)
        if dataset_check.get("strict") and not dataset_check["ok"]:
            errors.append(dataset_check["reason"])
        elif not dataset_check["ok"]:
            warnings.append(dataset_check["reason"])

        output_check = self._check_output_paths(run_spec)
        checks.append(output_check)
        if not output_check["ok"]:
            errors.append(output_check["reason"])

        status = "blocked" if errors else "ok"
        payload = {
            "created_at": now_utc(),
            "status": status,
            "checks": checks,
            "errors": errors,
            "warnings": warnings,
            "recovery_actions": recovery_actions,
            "gpu_selection": {
                "selected_gpu_ids": list(getattr(gpu_selection, "selected_ids", []) or []),
                "cuda_visible_devices": getattr(gpu_selection, "cuda_visible_devices", ""),
                "policy": getattr(gpu_selection, "policy", {}) if gpu_selection else {},
                "snapshot": getattr(gpu_selection, "snapshot", []) if gpu_selection else [],
            },
            "run_id": run_spec.get("run_id"),
        }
        preflight_path = Path(run_spec.get("preflight_path") or run_spec["run_root"] / "preflight.json")
        write_json(preflight_path, payload)
        return payload

    def collect_candidate_metrics(self, run_id: str) -> dict[str, Any] | None:
        result_root = self.repo_root / "local" / "auto_research_runs" / run_id / "results"
        return self._collect_metrics_from_result_root(result_root)

    def collect_ablation_metrics(self, run_spec: dict[str, Any], ablation_spec: dict[str, Any]) -> dict[str, Any] | None:
        raw_metrics_path = ablation_spec.get("metrics_path")
        metrics_path = Path(raw_metrics_path) if raw_metrics_path else None
        if metrics_path and metrics_path.exists():
            payload = read_json(metrics_path, default={}) or {}
            if payload.get("mean") is not None or payload.get("datasets"):
                return payload
        result_root = Path(run_spec["run_root"]) / "ablation_disabled" / "results"
        return self._collect_metrics_from_result_root(result_root)

    def collect_proxy_metrics(self, run_spec: dict[str, Any]) -> dict[str, Any] | None:
        proxy_spec = run_spec.get("proxy_screen") or {}
        raw_metrics_path = proxy_spec.get("metrics_path")
        metrics_path = Path(raw_metrics_path) if raw_metrics_path else None
        if metrics_path and metrics_path.exists():
            payload = read_json(metrics_path, default={}) or {}
            if payload.get("mean") is not None or payload.get("datasets"):
                return payload
        proxy_root = Path(proxy_spec.get("run_root") or Path(run_spec["run_root"]) / "proxy")
        return self._collect_metrics_from_result_root(proxy_root / "results")

    def collect_proxy_eval_smoke(self, run_spec: dict[str, Any]) -> dict[str, Any]:
        proxy_spec = run_spec.get("proxy_screen") or {}
        proxy_root = Path(proxy_spec.get("run_root") or Path(run_spec["run_root"]) / "proxy")
        proxy_cfg = c2c_proxy_screen_config(self.config)
        return collect_c2c_eval_smoke(
            proxy_root / "results",
            repo_root=self.repo_root,
            config=(proxy_cfg.get("eval_smoke") if isinstance(proxy_cfg.get("eval_smoke"), dict) else {}),
        )

    def proxy_baseline_metrics(self, run_spec: dict[str, Any]) -> dict[str, Any] | None:
        proxy_spec = run_spec.get("proxy_screen") or {}
        raw_metrics_path = proxy_spec.get("baseline_metrics_path")
        metrics_path = Path(raw_metrics_path) if raw_metrics_path else None
        if metrics_path and metrics_path.exists():
            payload = read_json(metrics_path, default={}) or {}
            if payload.get("mean") is not None or payload.get("datasets"):
                return payload
        return self._proxy_baseline_from_config(proxy_spec)

    def collect_proxy_baseline_run_metrics(self, run_spec: dict[str, Any]) -> dict[str, Any] | None:
        return self._collect_metrics_from_result_root(Path(run_spec["run_root"]) / "results")

    def _proxy_baseline_from_config(self, proxy_spec: dict[str, Any]) -> dict[str, Any] | None:
        proxy_cfg = c2c_proxy_screen_config(self.config)
        if not proxy_cfg.get("allow_configured_baseline_fallback", True):
            return None
        baseline_scores = self.baseline.get("datasets") or {}
        proxy_datasets = (proxy_spec.get("config") or {}).get("eval_datasets") or list(baseline_scores.keys())
        datasets = {
            str(dataset): float(baseline_scores[dataset])
            for dataset in proxy_datasets
            if dataset in baseline_scores and baseline_scores.get(dataset) is not None
        }
        if not datasets:
            return None
        return {
            "mean": round(sum(datasets.values()) / len(datasets), 4),
            "datasets": datasets,
            "source": "configured_full_baseline_subset_fallback",
        }

    def _collect_metrics_from_result_root(self, result_root: Path) -> dict[str, Any] | None:
        dataset_scores = {}
        summary_files = []
        for path in sorted(result_root.rglob("*_summary.json")):
            parsed = parse_c2c_summary_json(path, self.repo_root)
            if not parsed:
                continue
            dataset = parsed["dataset"]
            dataset_scores[dataset] = parsed["accuracy_percent"]
            summary_files.append(parsed["source"])
        if not dataset_scores:
            return None
        mean = round(sum(dataset_scores.values()) / len(dataset_scores), 4)
        return {"mean": mean, "datasets": dataset_scores, "summary_files": summary_files}

    def write_mock_candidate_results(self, run_id: str, offset: float = 0.0) -> dict[str, Any]:
        baseline = self.baseline
        base_scores = baseline.get("datasets") or DEFAULT_BASELINE["datasets"]
        result_root = self.repo_root / "local" / "auto_research_runs" / run_id / "results"
        for dataset in DEFAULT_DATASETS:
            output_dir = result_root / dataset
            ensure_dir(output_dir)
            score = float(base_scores.get(dataset, baseline.get("mean", DEFAULT_BASELINE["mean"]))) + offset
            payload = {
                "model": "Rosetta",
                "dataset": dataset,
                "answer_method": "generate",
                "overall_accuracy": score / 100.0,
            }
            (output_dir / f"Rosetta_{dataset}_generate_mock_summary.json").write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        return self.collect_candidate_metrics(run_id) or {"mean": 0.0, "datasets": {}, "summary_files": []}

    def write_mock_ablation_results(self, run_spec: dict[str, Any], offset: float = -0.05) -> dict[str, Any]:
        baseline = self.baseline
        base_scores = baseline.get("datasets") or DEFAULT_BASELINE["datasets"]
        result_root = Path(run_spec["run_root"]) / "ablation_disabled" / "results"
        for dataset in DEFAULT_DATASETS:
            output_dir = result_root / dataset
            ensure_dir(output_dir)
            score = float(base_scores.get(dataset, baseline.get("mean", DEFAULT_BASELINE["mean"]))) + offset
            payload = {
                "model": "Rosetta",
                "dataset": dataset,
                "answer_method": "generate",
                "overall_accuracy": score / 100.0,
            }
            (output_dir / f"Rosetta_{dataset}_generate_ablation_disabled_summary.json").write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        return self._collect_metrics_from_result_root(result_root) or {"mean": 0.0, "datasets": {}, "summary_files": []}

    def _parse_small_loop_csvs(self, final_results: Path) -> list[dict[str, Any]]:
        records = []
        if not final_results.exists():
            return records
        for path in sorted(final_results.rglob("*small_loop_scores.csv")):
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    method = _first_present(row, ["method", "run_name", "name"]) or ""
                    record = {
                        "kind": "small_loop",
                        "method": method,
                        "receiver": row.get("receiver", ""),
                        "sharer": row.get("sharer", ""),
                        "alignment_strategy": row.get("alignment_strategy", ""),
                        "confidence_gate": row.get("confidence_gate", ""),
                        "train_samples": _safe_int(_first_present(row, ["train_samples", "num_samples", "samples"])),
                        "metrics": {
                            "mmlu-redux": _safe_float(_first_present(row, ["mmlu_redux", "mmlu-redux", "mmlu"])),
                            "ai2-arc": _safe_float(_first_present(row, ["ai2_arc_challenge", "ai2_arc", "ai2-arc", "arc_c", "arc"])),
                            "openbookqa": _safe_float(_first_present(row, ["openbookqa", "openbook_qa", "obqa"])),
                            "mean": _safe_float(_first_present(row, ["mean", "avg", "average"])),
                        },
                        "losses": {
                            "final_train_loss": _safe_float(_first_present(row, ["final_train_loss", "train_loss"])),
                            "mid_eval_loss": _safe_float(_first_present(row, ["mid_eval_loss", "mid_loss"])),
                            "final_eval_loss": _safe_float(_first_present(row, ["final_eval_loss", "eval_loss"])),
                        },
                        "source": path.relative_to(self.repo_root).as_posix(),
                    }
                    records.append(record)
        return records

    def _parse_summary_jsons(self, final_results: Path) -> list[dict[str, Any]]:
        records = []
        if not final_results.exists():
            return records
        for path in sorted(final_results.rglob("*_summary.json")):
            parsed = parse_c2c_summary_json(path, self.repo_root)
            if parsed:
                parsed["kind"] = "summary_json"
                records.append(parsed)
        return records

    def _register_reference_cards(self, cards: list[dict[str, Any]]) -> None:
        manifest_path = self.project_root / "references" / "papers" / "manifest.json"
        manifest = read_json(manifest_path, default={"updated_at": now_utc(), "papers": []})
        papers = [item for item in manifest.get("papers", []) if item.get("paper_id") not in {card["paper_id"] for card in cards}]
        for card in cards:
            papers.append({
                "paper_id": card["paper_id"],
                "title": card["title"],
                "local_pdf_path": card["local_path"] if card["local_path"].endswith(".pdf") else "",
                "local_path": card["local_path"],
                "source_path": card["source_path"],
                "sha256": card["sha256"],
                "paper_full_md_path": card.get("paper_full_md_path", ""),
                "parser": card.get("parser", ""),
                "parser_status": card.get("parser_status", ""),
                "downloaded_at": now_utc(),
            })
        manifest["updated_at"] = now_utc()
        manifest["papers"] = sorted(papers, key=lambda item: item.get("title", ""))
        write_json(manifest_path, manifest)

    def _train_template_path(self) -> Path:
        fallback = self.repo_root / "recipe/train_recipe/C2C_0.6+0.5.json"
        return fallback

    def _eval_template(self, dataset: str) -> dict[str, Any]:
        fallback = self.repo_root / "recipe/eval_recipe/unified_eval.yaml"
        payload = read_yaml(fallback, default={}) if fallback.exists() else {}
        return payload or self._minimal_eval_config(dataset)

    def _candidate_commands(self, train_config: Path, eval_configs: dict[str, Path], gpu_ids: list[int] | None = None) -> dict[str, Any]:
        python_cmd = self.env_python
        env_prefix = self._offline_env_prefix(gpu_ids=gpu_ids)
        preflight_env_prefix = self._offline_env_prefix(gpu_ids=[])
        rel_train = train_config.relative_to(self.repo_root).as_posix()
        preflight = [
            f"{preflight_env_prefix} {python_cmd} -m py_compile rosetta/model/aligner.py rosetta/model/projector.py rosetta/model/wrapper.py",
        ]
        test_path = self.repo_root / "test" / "test_aligner_span_overlap.py"
        if test_path.exists():
            preflight.append(f"{preflight_env_prefix} {python_cmd} -m pytest --no-cov test/test_aligner_span_overlap.py")
        train_launcher = self._train_launcher(num_processes=len(gpu_ids or []))
        train = f"{env_prefix} {train_launcher} script/train/SFT_train.py --config {rel_train}"
        eval_commands = [
            f"{env_prefix} {python_cmd} script/evaluation/unified_evaluator.py --config {path.relative_to(self.repo_root).as_posix()}"
            for path in eval_configs.values()
        ]
        return {"preflight": preflight, "train": train, "eval": eval_commands}

    def _offline_env_prefix(self, gpu_ids: list[int] | None = None) -> str:
        hf_home = str(Path.home() / ".cache" / "huggingface")
        dataset_cache = str(Path(hf_home) / "datasets")
        repo_pythonpath = str(self.repo_root.resolve())
        visible_devices = ",".join(str(item) for item in gpu_ids) if gpu_ids is not None else None
        if visible_devices is None:
            visible_devices = self.c2c_config.get("small_loop", {}).get("cuda_visible_devices")
        if visible_devices is None:
            configured_gpus = self.c2c_config.get("small_loop", {}).get("gpu_ids")
            configured_gpu_ids = _coerce_gpu_ids(configured_gpus)
            if configured_gpu_ids:
                visible_devices = ",".join(str(item) for item in configured_gpu_ids)
        cuda_prefix = f"CUDA_VISIBLE_DEVICES={visible_devices} " if visible_devices is not None else ""
        return (
            f"{cuda_prefix}"
            "HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
            "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
            f"HF_HOME={hf_home} HF_DATASETS_CACHE={dataset_cache} "
            f"PYTHONPATH={repo_pythonpath}:$PYTHONPATH "
            "WANDB_DISABLED=true WANDB_MODE=disabled WANDB_START_METHOD=thread WANDB_REQUIRE_SERVICE=false"
        )

    def _train_launcher(self, num_processes: int | None = None) -> str:
        small_loop = self.c2c_config.get("small_loop", {})
        python_cmd = self.env_python
        num_train_processes = int(num_processes or small_loop.get("num_train_processes") or 1)
        if num_train_processes <= 1:
            return python_cmd
        return f"{python_cmd} -m torch.distributed.run --nproc_per_node={num_train_processes}"

    @staticmethod
    def _frozen_hashes(train_config: Path, eval_configs: dict[str, Path]) -> dict[str, str]:
        paths = {"train_config": train_config, **{f"eval_config:{dataset}": path for dataset, path in eval_configs.items()}}
        return {key: sha256_file(path) for key, path in paths.items() if path.exists()}

    def _check_model_path(self, model_id: str, model_path: Path) -> dict[str, Any]:
        resolved = model_path.resolve(strict=False)
        exists = model_path.exists()
        reason = ""
        repairable = model_path.is_symlink() and not exists
        if not exists:
            reason = f"model path missing for {model_id}: {model_path}"
            return {"name": f"model:{model_id}", "ok": False, "path": str(model_path), "resolved_path": str(resolved), "is_symlink": model_path.is_symlink(), "repairable": repairable, "reason": reason}
        required_files = ["config.json"]
        tokenizer_candidates = ["tokenizer.json", "tokenizer.model", "vocab.json", "spiece.model", "tokenizer_config.json"]
        missing = [name for name in required_files if not (resolved / name).exists()]
        if not any((resolved / name).exists() for name in tokenizer_candidates):
            missing.append("tokenizer file")
        load_errors = []
        if not missing:
            load_errors = self._offline_model_load_errors(resolved)
        ok = not missing and not load_errors
        if missing:
            reason = f"model path incomplete for {model_id}: missing {', '.join(missing)}"
        elif load_errors:
            reason = f"model path failed offline load for {model_id}: {'; '.join(load_errors)}"
        return {
            "name": f"model:{model_id}",
            "ok": ok,
            "path": str(model_path),
            "resolved_path": str(resolved),
            "is_symlink": model_path.is_symlink(),
            "repairable": False,
            "missing": missing,
            "offline_load_errors": load_errors,
            "reason": reason,
        }

    @staticmethod
    def _offline_model_load_errors(model_path: Path) -> list[str]:
        try:
            from transformers import AutoConfig, AutoTokenizer
        except Exception:
            return []
        errors = []
        try:
            AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        except Exception as exc:
            errors.append(f"AutoConfig: {type(exc).__name__}: {str(exc)[:200]}")
        try:
            AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=True)
        except Exception as exc:
            errors.append(f"AutoTokenizer: {type(exc).__name__}: {str(exc)[:200]}")
        return errors

    @staticmethod
    def _repair_model_symlink(model_id: str, model_path: Path) -> dict[str, Any]:
        snapshots_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_id.replace('/', '--')}" / "snapshots"
        candidates = [
            path
            for path in sorted(snapshots_root.glob("*"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True)
            if path.is_dir() and (path / "config.json").exists()
        ]
        action = {"action": "repair_model_symlink", "model_id": model_id, "path": str(model_path), "status": "failed", "reason": ""}
        if not candidates:
            action["reason"] = f"no usable HF cache snapshot under {snapshots_root}"
            return action
        target = candidates[0]
        try:
            ensure_dir(model_path.parent)
            if model_path.is_symlink() or model_path.exists():
                model_path.unlink()
            model_path.symlink_to(target, target_is_directory=True)
        except OSError as exc:
            action["reason"] = str(exc)
            return action
        action.update({"status": "ok", "target": str(target), "reason": "rebuilt symlink from HF cache snapshot"})
        return action

    def _check_dataset_cache(self) -> dict[str, Any]:
        cache_root = Path(os.environ.get("HF_DATASETS_CACHE") or Path.home() / ".cache" / "huggingface" / "datasets")
        aliases = {
            "mmlu-redux": ["mmlu", "mmlu-redux", "edinburgh-dawg"],
            "ai2-arc": ["ai2", "arc"],
            "openbookqa": ["openbook", "openbookqa"],
        }
        missing = []
        located: dict[str, list[str]] = {}
        cache_text = ""
        if cache_root.exists():
            try:
                cache_text = "\n".join(path.name.lower() for path in cache_root.iterdir())
            except OSError:
                cache_text = ""
        for dataset in self.c2c_config.get("datasets", DEFAULT_DATASETS):
            hits = [alias for alias in aliases.get(dataset, [dataset]) if alias.lower() in cache_text]
            located[dataset] = hits
            if not hits:
                missing.append(dataset)
        strict = bool(self.c2c_config.get("small_loop", {}).get("strict_dataset_cache", not self.config.get("experiment", {}).get("simulate", False)))
        ok = cache_root.exists() and (not missing or not strict)
        return {
            "name": "dataset_cache",
            "ok": ok,
            "strict": strict,
            "path": str(cache_root),
            "located": located,
            "missing": missing,
            "reason": "" if ok else f"dataset cache missing or incomplete: {', '.join(missing) if missing else cache_root}",
        }

    def _check_output_paths(self, run_spec: dict[str, Any]) -> dict[str, Any]:
        run_root = Path(run_spec["run_root"])
        checkpoint_parent = run_root / "checkpoints"
        try:
            ensure_dir(checkpoint_parent)
            probe = checkpoint_parent / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
        except OSError as exc:
            return {"name": "checkpoint_output", "ok": False, "path": str(checkpoint_parent), "reason": f"checkpoint parent is not writable: {exc}"}
        expected = f"local/auto_research_runs/{run_spec['run_id']}/checkpoints/final"
        mismatches = []
        for dataset, path in run_spec.get("eval_configs", {}).items():
            payload = read_yaml(path, default={}) or {}
            actual = (((payload.get("model") or {}).get("rosetta_config") or {}).get("checkpoints_dir"))
            if actual != expected:
                mismatches.append({"dataset": dataset, "expected": expected, "actual": actual})
        ok = not mismatches
        return {"name": "checkpoint_output", "ok": ok, "path": str(checkpoint_parent), "expected_eval_checkpoints_dir": expected, "mismatches": mismatches, "reason": "" if ok else "eval config checkpoints_dir does not point to training final checkpoint"}

    def localize_runtime_model_literals(self) -> dict[str, Any]:
        """Patch known runtime files that hard-code HF repo ids in offline C2C runs."""
        replacements = {
            model_id: str(Path(local_path).expanduser())
            for model_id, local_path in self.model_map.items()
            if model_id and local_path
        }
        if not replacements:
            return {"status": "skipped", "reason": "model_map is empty", "files": []}

        target_files = [
            "rosetta/train/dataset_adapters.py",
        ]
        changed_files = []
        for rel_path in target_files:
            path = self.repo_root / rel_path
            if not path.exists():
                continue
            original = path.read_text(encoding="utf-8")
            updated = original
            replaced: list[dict[str, str]] = []
            for model_id, local_path in replacements.items():
                if model_id not in updated:
                    continue
                updated = updated.replace(model_id, local_path)
                replaced.append({"model_id": model_id, "local_path": local_path})
            if updated == original:
                continue
            path.write_text(updated, encoding="utf-8")
            changed_files.append(
                {
                    "path": rel_path,
                    "replacements": replaced,
                    "sha256": sha256_file(path),
                }
            )

        status = "ok" if changed_files else "noop"
        return {
            "status": status,
            "files": changed_files,
            "reason": "" if changed_files else "no known runtime model literals required localization",
        }

    def _localize_model_references(self, payload: dict[str, Any]) -> None:
        model_map = self.model_map

        def convert(value: Any) -> Any:
            if isinstance(value, str):
                return model_map.get(value, value)
            if isinstance(value, list):
                return [convert(item) for item in value]
            if isinstance(value, dict):
                return {key: convert(item) for key, item in value.items()}
            return value

        converted = convert(payload)
        payload.clear()
        payload.update(converted)

    @staticmethod
    def _minimal_train_config() -> dict[str, Any]:
        return {
            "model": {
                "base_model": "Qwen/Qwen3-0.6B",
                "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
                "is_do_alignment": False,
                "alignment_strategy": "first",
                "mapping": "last_aligned",
            },
            "training": {"num_epochs": 1, "per_device_train_batch_size": 1, "gradient_accumulation_steps": 8, "seed": 42},
            "data": {"type": "MMLUChatDataset", "kwargs": {"split": "auxiliary_train", "num_samples": 2048}},
            "output": {},
        }

    @staticmethod
    def _minimal_eval_config(dataset: str) -> dict[str, Any]:
        return {
            "model": {
                "model_name": "Rosetta",
                "rosetta_config": {
                    "base_model": "Qwen/Qwen3-0.6B",
                    "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
                    "is_do_alignment": False,
                    "alignment_strategy": "longest",
                },
                "generation_config": {"do_sample": False, "max_new_tokens": 64},
            },
            "output": {},
            "eval": {"dataset": dataset, "answer_method": "generate", "use_cot": False, "use_template": True},
        }


@dataclass
class C2CPatchGuard:
    allowed_files: list[str]
    allowed_prefixes: list[str]

    def validate_path(self, path_value: str) -> str | None:
        path = Path(path_value)
        if path.is_absolute() or ".." in path.parts:
            return "absolute paths and parent traversal are not allowed"
        normalized = path.as_posix()
        if normalized in self.allowed_files:
            return None
        if any(normalized.startswith(prefix) for prefix in self.allowed_prefixes):
            return None
        return f"path is outside C2C edit scope: {normalized}"

    def validate_edits(self, edits: list[dict[str, Any]]) -> list[str]:
        errors = []
        for idx, edit in enumerate(edits):
            path = edit.get("path", "")
            reason = self.validate_path(path)
            if reason:
                errors.append(f"edit {idx}: {reason}")
            if "old" not in edit or "new" not in edit:
                errors.append(f"edit {idx}: edit must include old and new fields")
        return errors

    def apply_edits(self, repo_root: Path, edits: list[dict[str, Any]]) -> dict[str, Any]:
        errors = self.validate_edits(edits)
        if errors:
            return {"status": "rejected", "errors": errors, "changed_files": []}
        changed = []
        for edit in edits:
            target = repo_root / edit["path"]
            if not target.exists():
                return {"status": "rejected", "errors": [f"missing target file: {edit['path']}"], "changed_files": changed}
            text = target.read_text(encoding="utf-8")
            old = edit["old"]
            if old not in text:
                return {"status": "rejected", "errors": [f"old text not found in {edit['path']}"], "changed_files": changed}
            target.write_text(text.replace(old, edit["new"], 1), encoding="utf-8")
            changed.append(edit["path"])
        return {"status": "applied", "errors": [], "changed_files": changed}


def parse_c2c_summary_json(path: Path, repo_root: Path | None = None) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if "overall_accuracy" not in payload:
        return None
    accuracy = _safe_float(payload.get("overall_accuracy"))
    if accuracy is None:
        return None
    if accuracy <= 1.0:
        accuracy *= 100.0
    source = path.as_posix()
    if repo_root:
        try:
            source = path.relative_to(repo_root).as_posix()
        except ValueError:
            pass
    return {
        "model": payload.get("model"),
        "dataset": payload.get("dataset") or path.parent.name,
        "answer_method": payload.get("answer_method"),
        "accuracy_percent": round(accuracy, 4),
        "source": source,
    }


def collect_c2c_eval_smoke(result_root: Path, *, repo_root: Path | None = None, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Collect lightweight output-health diagnostics from C2C eval artifacts."""
    config = config or {}
    if config.get("enabled") is False:
        return {
            "schema_version": "c2c_eval_smoke_v1",
            "status": "skipped",
            "reason": "eval_smoke disabled",
            "result_root": _relpath(Path(result_root), repo_root),
            "red_flags": [],
        }
    max_files = max(1, int(config.get("max_prediction_files") or 8))
    max_rows = max(1, int(config.get("max_prediction_rows") or 512))
    min_nonempty = float(config.get("min_nonempty_prediction_rate", 0.5))
    min_parse = float(config.get("min_answer_parse_rate", 0.2))
    result_root = Path(result_root)
    summary_files = []
    summary_datasets: dict[str, dict[str, Any]] = {}
    for path in sorted(result_root.rglob("*_summary.json")):
        parsed = parse_c2c_summary_json(path, repo_root)
        if not parsed:
            continue
        summary_files.append(_relpath(path, repo_root))
        summary_datasets[str(parsed.get("dataset") or path.parent.name)] = {
            "accuracy_percent": parsed.get("accuracy_percent"),
            "answer_method": parsed.get("answer_method"),
            "source": parsed.get("source"),
        }
    prediction_files = _c2c_prediction_files(result_root, max_files=max_files)
    dataset_rows: dict[str, list[dict[str, Any]]] = {}
    files_scanned = []
    for path in prediction_files:
        rows = _read_c2c_prediction_rows(path, max_rows=max_rows)
        if not rows:
            continue
        files_scanned.append(_relpath(path, repo_root))
        dataset = _infer_c2c_dataset_from_path(path)
        dataset_rows.setdefault(dataset, []).extend(rows[: max(0, max_rows - len(dataset_rows.get(dataset, [])))])
    datasets = {
        dataset: _c2c_eval_smoke_dataset(rows)
        for dataset, rows in sorted(dataset_rows.items())
    }
    total_rows = sum(item.get("sample_count", 0) for item in datasets.values())
    nonempty_count = sum(item.get("nonempty_prediction_count", 0) for item in datasets.values())
    answer_like_count = sum(item.get("answer_like_count", 0) for item in datasets.values())
    parsed_count = sum(item.get("parsed_answer_count", 0) for item in datasets.values())
    answer_distribution: dict[str, int] = {}
    for item in datasets.values():
        for answer, count in (item.get("answer_distribution") or {}).items():
            answer_distribution[answer] = answer_distribution.get(answer, 0) + int(count)
    nonempty_rate = round(nonempty_count / total_rows, 4) if total_rows else None
    answer_like_rate = round(answer_like_count / total_rows, 4) if total_rows else None
    parse_rate = round(parsed_count / total_rows, 4) if total_rows else None
    red_flags: list[str] = []
    if not summary_files:
        red_flags.append("no_summary_files")
    if not prediction_files:
        red_flags.append("no_prediction_files")
    if total_rows and nonempty_rate is not None and nonempty_rate < min_nonempty:
        red_flags.append("low_nonempty_prediction_rate")
    if total_rows and parse_rate is not None and parse_rate < min_parse:
        red_flags.append("low_answer_parse_rate")
    if total_rows and _dominant_answer_share(answer_distribution) >= 0.95:
        red_flags.append("answer_distribution_collapsed")
    if summary_datasets and all(float(item.get("accuracy_percent") or 0.0) <= 0.0 for item in summary_datasets.values()):
        red_flags.append("all_summary_scores_zero")
        if not total_rows:
            red_flags.append("all_zero_without_prediction_artifacts")
    status = "warning" if red_flags else "ok"
    return {
        "schema_version": "c2c_eval_smoke_v1",
        "status": status,
        "result_root": _relpath(result_root, repo_root),
        "summary_file_count": len(summary_files),
        "summary_files": summary_files[:40],
        "summary_datasets": summary_datasets,
        "prediction_file_count": len(prediction_files),
        "prediction_files_scanned": files_scanned,
        "sample_count": total_rows,
        "nonempty_prediction_rate": nonempty_rate,
        "answer_like_rate": answer_like_rate,
        "answer_parse_rate": parse_rate,
        "mean_output_length": _weighted_mean([item.get("mean_output_length") for item in datasets.values()], [item.get("sample_count", 0) for item in datasets.values()]),
        "answer_distribution": dict(sorted(answer_distribution.items())),
        "datasets": datasets,
        "red_flags": red_flags,
    }


def _c2c_prediction_files(result_root: Path, *, max_files: int) -> list[Path]:
    if not result_root.exists():
        return []
    candidates: list[Path] = []
    for path in sorted(result_root.rglob("*")):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name.endswith("_summary.json"):
            continue
        if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".txt"}:
            continue
        if any(marker in name for marker in ["predict", "prediction", "result", "answer", "generation", "output", "responses"]):
            candidates.append(path)
    return candidates[:max_files]


def _read_c2c_prediction_rows(path: Path, *, max_rows: int) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    try:
        if suffix == ".jsonl":
            rows = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if len(rows) >= max_rows:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
            return rows
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            if isinstance(payload, list):
                return [item for item in payload[:max_rows] if isinstance(item, dict)]
            if isinstance(payload, dict):
                for key in ["predictions", "results", "samples", "outputs", "items", "records", "examples"]:
                    value = payload.get(key)
                    if isinstance(value, list):
                        return [item for item in value[:max_rows] if isinstance(item, dict)]
                return [payload]
        if suffix == ".csv":
            rows = []
            with path.open(newline="", encoding="utf-8", errors="ignore") as handle:
                for row in csv.DictReader(handle):
                    rows.append(dict(row))
                    if len(rows) >= max_rows:
                        break
            return rows
        if suffix == ".txt":
            rows = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:max_rows]:
                if line.strip():
                    rows.append({"output": line.strip()})
            return rows
    except (OSError, json.JSONDecodeError):
        return []
    return []


def _c2c_eval_smoke_dataset(rows: list[dict[str, Any]]) -> dict[str, Any]:
    outputs = [_c2c_prediction_text(row) for row in rows]
    parsed_answers = [_parse_c2c_answer(text) or _parse_c2c_answer(_c2c_label_text(row)) for row, text in zip(rows, outputs)]
    nonempty = [text for text in outputs if text.strip()]
    answer_like = [text for text in outputs if _parse_c2c_answer(text)]
    lengths = [len(text.strip()) for text in outputs]
    distribution: dict[str, int] = {}
    for answer in parsed_answers:
        if not answer:
            continue
        distribution[answer] = distribution.get(answer, 0) + 1
    sample_count = len(rows)
    return {
        "sample_count": sample_count,
        "nonempty_prediction_count": len(nonempty),
        "nonempty_prediction_rate": round(len(nonempty) / sample_count, 4) if sample_count else None,
        "answer_like_count": len(answer_like),
        "answer_like_rate": round(len(answer_like) / sample_count, 4) if sample_count else None,
        "parsed_answer_count": sum(1 for answer in parsed_answers if answer),
        "answer_parse_rate": round(sum(1 for answer in parsed_answers if answer) / sample_count, 4) if sample_count else None,
        "mean_output_length": round(sum(lengths) / sample_count, 4) if sample_count else None,
        "max_output_length": max(lengths) if lengths else 0,
        "answer_distribution": dict(sorted(distribution.items())),
        "example_outputs": [text[:160] for text in outputs if text.strip()][:3],
    }


def _c2c_prediction_text(row: dict[str, Any]) -> str:
    keys = [
        "prediction",
        "pred",
        "output",
        "generated_text",
        "generation",
        "response",
        "model_output",
        "answer_text",
        "decoded",
        "completion",
    ]
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    nested = row.get("model") or row.get("result")
    if isinstance(nested, dict):
        return _c2c_prediction_text(nested)
    return ""


def _c2c_label_text(row: dict[str, Any]) -> str:
    for key in ["predicted_answer", "prediction_label", "predicted_label", "predicted_choice", "model_answer", "model_choice"]:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _parse_c2c_answer(text: str) -> str:
    if not text:
        return ""
    stripped = str(text).strip()
    patterns = [
        r"(?:answer|choice|option)\s*[:：]\s*([A-D])\b",
        r"\(([A-D])\)",
        r"\b([A-D])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    lowered = stripped.lower()
    for word, label in [("yes", "YES"), ("no", "NO"), ("true", "TRUE"), ("false", "FALSE")]:
        if re.fullmatch(rf"\W*{word}\W*", lowered):
            return label
    return ""


def _infer_c2c_dataset_from_path(path: Path) -> str:
    known = set(DEFAULT_DATASETS)
    for part in reversed(path.parts):
        if part in known:
            return part
    return path.parent.name


def _dominant_answer_share(distribution: dict[str, int]) -> float:
    total = sum(int(value) for value in distribution.values())
    if not total:
        return 0.0
    return max(int(value) for value in distribution.values()) / total


def _weighted_mean(values: list[Any], weights: list[Any]) -> float | None:
    total_weight = 0.0
    total = 0.0
    for value, weight in zip(values, weights):
        numeric = _safe_float(value)
        try:
            numeric_weight = float(weight)
        except (TypeError, ValueError):
            numeric_weight = 0.0
        if numeric is None or numeric_weight <= 0:
            continue
        total += numeric * numeric_weight
        total_weight += numeric_weight
    return round(total / total_weight, 4) if total_weight else None


def _candidate_ablation_switch(candidate: dict[str, Any]) -> str:
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    ablation_plan = candidate.get("ablation_plan") if isinstance(candidate.get("ablation_plan"), dict) else {}
    switch = contract.get("ablation_switch") or ablation_plan.get("switch")
    return str(switch) if switch not in (None, "") else ""


def _proxy_activation_mechanism_trace(project_root: Path, run_spec: dict[str, Any], activation_spec: dict[str, Any]) -> dict[str, Any]:
    switch = str(activation_spec.get("switch") or "")
    enabled_configs = (run_spec.get("proxy_screen") or {}).get("eval_configs") or {}
    disabled_configs = activation_spec.get("eval_configs") or {}
    enabled_rosetta = _first_eval_rosetta_config(enabled_configs.values())
    disabled_rosetta = _first_eval_rosetta_config(disabled_configs.values())
    validation_trace = _patch_validation_activation_wiring(project_root, run_spec, activation_spec)
    tensor_trace = _proxy_activation_tensor_trace(run_spec, activation_spec)
    failures: list[str] = []
    if not switch:
        failures.append("missing_ablation_switch")
    elif disabled_rosetta.get(switch) is not True:
        failures.append(f"disabled_eval_missing_{switch}")
    if switch and enabled_rosetta.get(switch) is True:
        failures.append(f"enabled_eval_sets_disable_switch_{switch}")
    wiring_check_status = validation_trace.get("status")
    if wiring_check_status not in {None, "ok", "skipped"}:
        failures.append(f"s2_5_wiring_check_{wiring_check_status}")
    if wiring_check_status is None:
        failures.append("s2_5_wiring_check_missing")
    status = "wired" if not failures else "missing"
    return {
        "status": status,
        "switch": switch,
        "failures": failures,
        "enabled_eval_rosetta_keys": sorted(enabled_rosetta.keys()),
        "disabled_eval_rosetta_keys": sorted(disabled_rosetta.keys()),
        "disabled_switch_value": disabled_rosetta.get(switch) if switch else None,
        "enabled_switch_value": enabled_rosetta.get(switch) if switch else None,
        "s2_5_wiring_check": validation_trace,
        "tensor_trace": tensor_trace,
    }


def _first_eval_rosetta_config(paths: Any) -> dict[str, Any]:
    for raw_path in paths or []:
        path = Path(raw_path)
        if not path.exists():
            continue
        payload = read_yaml(path, default={}) if path.suffix in {".yaml", ".yml"} else read_json(path, default={})
        if not isinstance(payload, dict):
            continue
        rosetta = ((payload.get("model") or {}).get("rosetta_config") or {})
        return dict(rosetta) if isinstance(rosetta, dict) else {}
    return {}


def _patch_validation_activation_wiring(project_root: Path, run_spec: dict[str, Any], activation_spec: dict[str, Any]) -> dict[str, Any]:
    validation = activation_spec.get("code_patch_validation")
    if not validation:
        validation = (((run_spec.get("candidate") or {}).get("code_patch") or {}).get("validation"))
    if not validation:
        validation = (run_spec.get("code_patch") or {}).get("validation")
    if not validation:
        return {"status": None, "reason": "candidate code_patch.validation path missing from run_spec"}
    path = Path(str(validation))
    if not path.is_absolute():
        path = project_root / path
    payload = read_json(path, default={}) if path.exists() else {}
    for check in payload.get("checks") or []:
        if isinstance(check, dict) and check.get("name") == "runtime_smoke:mechanism_activation_wiring":
            return {
                "status": check.get("status"),
                "returncode": check.get("returncode"),
                "switch": check.get("switch"),
                "failure_category": check.get("failure_category"),
                "runtime_code_refs": check.get("runtime_code_refs") or {},
                "rosetta_config": check.get("rosetta_config") or {},
                "repair_hint": check.get("repair_hint"),
            }
    return {"status": None, "reason": "runtime_smoke:mechanism_activation_wiring not found in S2.5 validation"}


def _proxy_activation_tensor_trace(run_spec: dict[str, Any], activation_spec: dict[str, Any]) -> dict[str, Any]:
    datasets = [str(dataset) for dataset in activation_spec.get("datasets") or []]
    enabled_root = Path((run_spec.get("proxy_screen") or {}).get("run_root") or Path(run_spec["run_root"]) / "proxy") / "results"
    disabled_root = Path(activation_spec.get("result_root") or "")
    enabled = _collect_activation_tensor_traces(enabled_root, datasets=datasets)
    disabled = _collect_activation_tensor_traces(disabled_root, datasets=datasets)
    if not enabled and not disabled:
        return {
            "status": "not_collected",
            "reason": "activation tensor trace artifacts were not found in enabled or disabled eval outputs",
            "expected_artifacts": [
                "activation_trace.json",
                "activation_trace.jsonl",
                "mechanism_trace.json",
                "mechanism_trace.jsonl",
            ],
        }
    comparison = _compare_activation_tensor_traces(enabled, disabled)
    return {
        "status": comparison.get("status"),
        "reason": comparison.get("reason"),
        "enabled_trace_count": len(enabled),
        "disabled_trace_count": len(disabled),
        "changed_fields": comparison.get("changed_fields") or [],
        "unchanged_fields": comparison.get("unchanged_fields") or [],
        "compared_fields": comparison.get("compared_fields") or [],
        "sample_enabled_paths": [item.get("path") for item in enabled[:3]],
        "sample_disabled_paths": [item.get("path") for item in disabled[:3]],
    }


def _collect_activation_tensor_traces(root: Path, *, datasets: list[str]) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    dataset_filter = set(datasets or [])
    patterns = [
        "activation_trace.json",
        "activation_trace.jsonl",
        "mechanism_trace.json",
        "mechanism_trace.jsonl",
        "tensor_trace.json",
        "tensor_trace.jsonl",
        "*activation*trace*.json",
        "*activation*trace*.jsonl",
        "*mechanism*trace*.json",
        "*mechanism*trace*.jsonl",
        "*tensor*trace*.json",
        "*tensor*trace*.jsonl",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(root.rglob(pattern))
    traces: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        rel_parts = path.relative_to(root).parts
        dataset = rel_parts[0] if rel_parts else ""
        if dataset_filter and dataset not in dataset_filter:
            continue
        for payload in _read_activation_trace_payloads(path):
            signature = _activation_trace_signature(payload)
            if signature:
                traces.append({"dataset": dataset, "path": path.as_posix(), "signature": signature})
    return traces


def _read_activation_trace_payloads(path: Path) -> list[Any]:
    try:
        if path.suffix == ".jsonl":
            payloads = []
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    payloads.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            return payloads
        payload = read_json(path, default=None)
        if isinstance(payload, list):
            return payload
        return [payload]
    except Exception:
        return []


def _activation_trace_signature(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    source = payload
    for key in ["tensor_trace", "activation_trace", "mechanism_trace", "trace"]:
        if isinstance(payload.get(key), dict):
            source = payload[key]
            break
    signature: dict[str, Any] = {}
    for key, value in source.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            signature[str(key)] = value
        elif isinstance(value, list) and len(value) <= 8 and all(isinstance(item, (str, int, float, bool)) or item is None for item in value):
            signature[str(key)] = value
        elif isinstance(value, dict):
            nested = {
                str(nested_key): nested_value
                for nested_key, nested_value in value.items()
                if isinstance(nested_value, (str, int, float, bool)) or nested_value is None
            }
            if nested:
                signature[str(key)] = nested
    for preferred in [
        "tensor_checksum",
        "checksum",
        "sha256",
        "mean",
        "std",
        "norm",
        "l2_norm",
        "shape",
        "numel",
        "nonzero",
        "gate_mean",
        "routing_entropy",
        "alignment_mass",
    ]:
        if preferred in payload and preferred not in signature:
            signature[preferred] = payload[preferred]
    return signature


def _compare_activation_tensor_traces(enabled: list[dict[str, Any]], disabled: list[dict[str, Any]]) -> dict[str, Any]:
    if not enabled or not disabled:
        return {"status": "missing_pair", "reason": "only one side wrote activation tensor trace artifacts"}
    enabled_map = _activation_trace_map(enabled)
    disabled_map = _activation_trace_map(disabled)
    changed: list[str] = []
    unchanged: list[str] = []
    compared: list[str] = []
    for key in sorted(set(enabled_map) & set(disabled_map)):
        compared.append(key)
        if _json_signature(enabled_map[key]) != _json_signature(disabled_map[key]):
            changed.append(key)
        else:
            unchanged.append(key)
    if changed:
        return {"status": "changed", "reason": "enabled/disabled activation tensor traces differ", "changed_fields": changed, "unchanged_fields": unchanged, "compared_fields": compared}
    if compared:
        return {"status": "unchanged", "reason": "enabled/disabled activation tensor traces are identical", "changed_fields": [], "unchanged_fields": unchanged, "compared_fields": compared}
    return {"status": "missing_pair", "reason": "activation tensor traces did not share comparable fields", "changed_fields": [], "unchanged_fields": [], "compared_fields": []}


def _activation_trace_map(traces: list[dict[str, Any]]) -> dict[str, Any]:
    mapped: dict[str, Any] = {}
    for item in traces:
        prefix = str(item.get("dataset") or "dataset")
        signature = item.get("signature") if isinstance(item.get("signature"), dict) else {}
        for key, value in signature.items():
            mapped[f"{prefix}:{key}"] = value
    return mapped


def _json_signature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _filter_c2c_metrics_to_datasets(metrics: dict[str, Any] | None, datasets: list[str]) -> dict[str, Any] | None:
    if not metrics:
        return None
    dataset_scores = metrics.get("datasets") if isinstance(metrics.get("datasets"), dict) else {}
    selected = {dataset: float(dataset_scores[dataset]) for dataset in datasets if dataset in dataset_scores and _safe_float(dataset_scores[dataset]) is not None}
    if not selected:
        return None
    return {
        "mean": round(sum(selected.values()) / len(selected), 4),
        "datasets": selected,
        "summary_files": list(metrics.get("summary_files") or []),
    }


def _proxy_activation_metric_comparison(
    enabled_metrics: dict[str, Any] | None,
    disabled_metrics: dict[str, Any] | None,
    *,
    min_abs_delta: Any,
) -> dict[str, Any]:
    threshold = float(min_abs_delta or 0.0)
    if not enabled_metrics or not disabled_metrics:
        return {
            "status": "insufficient_metrics",
            "enabled_mean": (enabled_metrics or {}).get("mean"),
            "disabled_mean": (disabled_metrics or {}).get("mean"),
            "mechanism_observed": False,
        }
    enabled_mean = _safe_float(enabled_metrics.get("mean"))
    disabled_mean = _safe_float(disabled_metrics.get("mean"))
    dataset_deltas: dict[str, float] = {}
    enabled_datasets = enabled_metrics.get("datasets") if isinstance(enabled_metrics.get("datasets"), dict) else {}
    disabled_datasets = disabled_metrics.get("datasets") if isinstance(disabled_metrics.get("datasets"), dict) else {}
    for dataset in sorted(set(enabled_datasets) & set(disabled_datasets)):
        enabled_value = _safe_float(enabled_datasets.get(dataset))
        disabled_value = _safe_float(disabled_datasets.get(dataset))
        if enabled_value is None or disabled_value is None:
            continue
        dataset_deltas[str(dataset)] = round(enabled_value - disabled_value, 4)
    mean_delta = round(enabled_mean - disabled_mean, 4) if enabled_mean is not None and disabled_mean is not None else None
    max_abs_dataset_delta = max((abs(value) for value in dataset_deltas.values()), default=0.0)
    mechanism_observed = bool(
        (mean_delta is not None and abs(mean_delta) >= threshold)
        or max_abs_dataset_delta >= threshold
    )
    return {
        "status": "ok",
        "enabled_mean": enabled_mean,
        "disabled_mean": disabled_mean,
        "enabled_minus_disabled_mean": mean_delta,
        "dataset_enabled_minus_disabled": dataset_deltas,
        "max_abs_dataset_delta": round(max_abs_dataset_delta, 4),
        "min_abs_metric_delta": threshold,
        "mechanism_observed": mechanism_observed,
    }


def _proxy_activation_prediction_comparison(
    enabled_result_root: Path,
    disabled_result_root: Path,
    *,
    datasets: list[str],
    repo_root: Path | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    max_files = max(1, int(config.get("max_prediction_files") or 8))
    max_rows = max(1, int(config.get("max_prediction_rows") or 512))
    min_prediction_diff_rate = float(config.get("min_prediction_diff_rate", 0.01) or 0.0)
    min_answer_diff_rate = float(config.get("min_answer_diff_rate", 0.01) or 0.0)
    min_length_delta = float(config.get("min_mean_output_length_delta", 1.0) or 0.0)
    enabled_by_dataset = _c2c_prediction_rows_by_dataset(enabled_result_root, datasets=datasets, max_files=max_files, max_rows=max_rows)
    disabled_by_dataset = _c2c_prediction_rows_by_dataset(disabled_result_root, datasets=datasets, max_files=max_files, max_rows=max_rows)
    dataset_comparisons: dict[str, Any] = {}
    total_compared = 0
    text_diff_count = 0
    answer_diff_count = 0
    length_deltas: list[float] = []
    for dataset in datasets:
        enabled_rows = enabled_by_dataset.get(dataset) or []
        disabled_rows = disabled_by_dataset.get(dataset) or []
        pair_count = min(len(enabled_rows), len(disabled_rows), max_rows)
        if pair_count <= 0:
            dataset_comparisons[dataset] = {
                "status": "missing_predictions",
                "enabled_row_count": len(enabled_rows),
                "disabled_row_count": len(disabled_rows),
                "compared_count": 0,
            }
            continue
        dataset_text_diff = 0
        dataset_answer_diff = 0
        enabled_lengths: list[int] = []
        disabled_lengths: list[int] = []
        enabled_distribution: dict[str, int] = {}
        disabled_distribution: dict[str, int] = {}
        for enabled_row, disabled_row in zip(enabled_rows[:pair_count], disabled_rows[:pair_count]):
            enabled_text = _c2c_prediction_text(enabled_row).strip()
            disabled_text = _c2c_prediction_text(disabled_row).strip()
            enabled_answer = _parse_c2c_answer(enabled_text) or _parse_c2c_answer(_c2c_label_text(enabled_row))
            disabled_answer = _parse_c2c_answer(disabled_text) or _parse_c2c_answer(_c2c_label_text(disabled_row))
            if enabled_text != disabled_text:
                dataset_text_diff += 1
            if enabled_answer != disabled_answer:
                dataset_answer_diff += 1
            enabled_lengths.append(len(enabled_text))
            disabled_lengths.append(len(disabled_text))
            if enabled_answer:
                enabled_distribution[enabled_answer] = enabled_distribution.get(enabled_answer, 0) + 1
            if disabled_answer:
                disabled_distribution[disabled_answer] = disabled_distribution.get(disabled_answer, 0) + 1
        mean_enabled_length = sum(enabled_lengths) / len(enabled_lengths) if enabled_lengths else 0.0
        mean_disabled_length = sum(disabled_lengths) / len(disabled_lengths) if disabled_lengths else 0.0
        mean_length_delta = round(mean_enabled_length - mean_disabled_length, 4)
        text_diff_rate = round(dataset_text_diff / pair_count, 4)
        answer_diff_rate = round(dataset_answer_diff / pair_count, 4)
        total_compared += pair_count
        text_diff_count += dataset_text_diff
        answer_diff_count += dataset_answer_diff
        length_deltas.append(abs(mean_length_delta))
        dataset_comparisons[dataset] = {
            "status": "ok",
            "enabled_row_count": len(enabled_rows),
            "disabled_row_count": len(disabled_rows),
            "compared_count": pair_count,
            "prediction_text_diff_count": dataset_text_diff,
            "prediction_text_diff_rate": text_diff_rate,
            "answer_diff_count": dataset_answer_diff,
            "answer_diff_rate": answer_diff_rate,
            "enabled_mean_output_length": round(mean_enabled_length, 4),
            "disabled_mean_output_length": round(mean_disabled_length, 4),
            "mean_output_length_delta": mean_length_delta,
            "enabled_answer_distribution": dict(sorted(enabled_distribution.items())),
            "disabled_answer_distribution": dict(sorted(disabled_distribution.items())),
            "answer_distribution_changed": enabled_distribution != disabled_distribution,
        }
    prediction_diff_rate = round(text_diff_count / total_compared, 4) if total_compared else None
    answer_diff_rate = round(answer_diff_count / total_compared, 4) if total_compared else None
    max_abs_length_delta = round(max(length_deltas), 4) if length_deltas else 0.0
    mechanism_observed = bool(
        (prediction_diff_rate is not None and prediction_diff_rate >= min_prediction_diff_rate)
        or (answer_diff_rate is not None and answer_diff_rate >= min_answer_diff_rate)
        or max_abs_length_delta >= min_length_delta
        or any((item.get("answer_distribution_changed") for item in dataset_comparisons.values() if isinstance(item, dict)))
    )
    return {
        "status": "ok" if total_compared else "missing_predictions",
        "enabled_result_root": _relpath(Path(enabled_result_root), repo_root),
        "disabled_result_root": _relpath(Path(disabled_result_root), repo_root),
        "datasets": dataset_comparisons,
        "compared_count": total_compared,
        "prediction_text_diff_count": text_diff_count,
        "prediction_diff_rate": prediction_diff_rate,
        "answer_diff_count": answer_diff_count,
        "answer_diff_rate": answer_diff_rate,
        "max_abs_mean_output_length_delta": max_abs_length_delta,
        "thresholds": {
            "min_prediction_diff_rate": min_prediction_diff_rate,
            "min_answer_diff_rate": min_answer_diff_rate,
            "min_mean_output_length_delta": min_length_delta,
        },
        "mechanism_observed": mechanism_observed,
    }


def _c2c_prediction_rows_by_dataset(
    result_root: Path,
    *,
    datasets: list[str],
    max_files: int,
    max_rows: int,
) -> dict[str, list[dict[str, Any]]]:
    wanted = set(str(dataset) for dataset in datasets)
    rows_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in wanted}
    for path in _c2c_prediction_files(Path(result_root), max_files=max_files):
        dataset = _infer_c2c_dataset_from_path(path)
        if dataset not in wanted:
            continue
        remaining = max_rows - len(rows_by_dataset.setdefault(dataset, []))
        if remaining <= 0:
            continue
        rows_by_dataset[dataset].extend(_read_c2c_prediction_rows(path, max_rows=remaining))
    return rows_by_dataset


def _c2c_eval_smoke_hard_failure(smoke: dict[str, Any] | None) -> bool:
    if not isinstance(smoke, dict):
        return False
    red_flags = set(smoke.get("red_flags") or [])
    if "no_summary_files" in red_flags:
        return True
    if "all_summary_scores_zero" in red_flags and (
        "low_nonempty_prediction_rate" in red_flags
        or "low_answer_parse_rate" in red_flags
        or "answer_distribution_collapsed" in red_flags
        or "all_zero_without_prediction_artifacts" in red_flags
    ):
        return True
    return False


def _relpath(path: Path, repo_root: Path | None) -> str:
    if repo_root:
        try:
            return path.relative_to(repo_root).as_posix()
        except ValueError:
            pass
    return path.as_posix()


def _append_check(checks: list[dict[str, Any]], name: str, ok: bool, **fields: Any) -> None:
    checks.append({"name": name, "ok": ok, "reason": "" if ok else f"{name} check failed", **fields})


def _coerce_gpu_ids(value: Any) -> list[int]:
    if value in (None, "", "auto"):
        return []
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


def c2c_candidate_config_overrides(candidate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    contract = candidate.get("experiment_contract") if isinstance(candidate.get("experiment_contract"), dict) else {}
    raw = (
        candidate.get("c2c_config_overrides")
        or candidate.get("frozen_config_overrides")
        or contract.get("config_overrides")
        or contract.get("frozen_config_overrides")
        or {}
    )
    if not isinstance(raw, dict):
        raw = {}
    train = raw.get("train") or raw.get("train_config") or {}
    eval_override = raw.get("eval") or raw.get("eval_config") or {}
    if not train and isinstance(raw.get("model"), dict):
        train = {"model": raw["model"]}
    if not eval_override and isinstance(raw.get("rosetta_config"), dict):
        eval_override = {"model": {"rosetta_config": raw["rosetta_config"]}}
    if not eval_override and isinstance(train, dict) and isinstance(train.get("model"), dict):
        projected = _project_train_model_overrides_to_eval(train["model"])
        if projected:
            eval_override = {"model": {"rosetta_config": projected}}
    return {
        "train": train if isinstance(train, dict) else {},
        "eval": eval_override if isinstance(eval_override, dict) else {},
    }


def _project_train_model_overrides_to_eval(model_overrides: dict[str, Any]) -> dict[str, Any]:
    eval_keys = {
        "alignment_strategy",
        "soft_alignment_top_k",
        "soft_alignment_score_mode",
        "soft_alignment_boundary_bonus",
        "soft_alignment_boundary_tolerance",
        "soft_alignment_min_weight",
        "soft_alignment_confidence_mode",
        "soft_alignment_confidence_alpha",
        "soft_alignment_confidence_floor",
        "soft_alignment_fallback_confidence",
        "soft_alignment_reweight_mode",
        "soft_alignment_reweight_strength",
        "soft_alignment_reweight_power",
        "soft_alignment_candidate_window",
        "cache_routing_mode",
        "cache_routing_min_utility",
        "cache_acceptance_mode",
        "cache_verifier_margin_weight",
        "cache_verifier_pathology_weight",
        "cache_controller_mode",
        "cache_controller_bucket_count",
        "bridge_memory_mode",
        "bridge_memory_slots",
        "bridge_memory_residual_weight",
        "span_graph_max_paths",
        "span_graph_consistency_weight",
    }
    return {key: value for key, value in model_overrides.items() if key in eval_keys}


def extract_reference_text(path: Path, *, max_chars: int = 200000) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader

            reader = PdfReader(str(path))
            chunks = []
            for page in reader.pages:
                chunks.append(page.extract_text() or "")
            return "\n".join(chunks)[:max_chars]
        except Exception as exc:  # pragma: no cover - depends on optional PDF parser/data
            pdftotext_result = _extract_pdf_with_pdftotext(path, max_chars=max_chars)
            if pdftotext_result:
                return pdftotext_result
            return f"PDF text extraction failed: {exc}"
    if suffix == ".json":
        try:
            return json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)[:max_chars]
        except json.JSONDecodeError:
            pass
    return path.read_text(encoding="utf-8", errors="ignore")[:max_chars]


def _iter_reference_files(source: Path, kind: str) -> list[Path]:
    if source.is_file():
        return [source]
    suffixes = {".pdf"} if kind == "ref_paper" else {".md", ".txt", ".json", ".pdf"}
    return [
        path
        for path in sorted(source.rglob("*"))
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def _reference_title(path: Path, kind: str) -> str:
    label = "paper" if kind == "ref_paper" else "review/rebuttal"
    return f"C2C reference {label}: {path.stem}"


def _mineru_parse_payload(
    *,
    target: Path,
    project_root: Path,
    paper_full_path: Path,
    metadata_path: Path,
    metadata: dict[str, Any],
    pdf_cfg: dict[str, Any],
    cache_status: str,
    text: str,
) -> dict[str, Any]:
    return {
        "status": "ok",
        "parser": "mineru",
        "cache_status": cache_status,
        "text": text,
        "paper_full_md_path": paper_full_path.relative_to(project_root).as_posix(),
        "mineru_result_path": metadata_path.relative_to(project_root).as_posix(),
        "model_version": metadata.get("model_version") or str(pdf_cfg.get("model_version") or "vlm"),
        "language": metadata.get("language") or str(pdf_cfg.get("language") or "en"),
        "parser_config_hash": metadata.get("parser_config_hash") or _mineru_parser_config_hash(pdf_cfg),
        "prompt_schema_version": metadata.get("prompt_schema_version") or "c2c_paper_full_markdown_v1",
        "artifacts": [
            target.relative_to(project_root).as_posix(),
            paper_full_path.relative_to(project_root).as_posix(),
            metadata_path.relative_to(project_root).as_posix(),
        ],
    }


def _restore_mineru_cache_candidate(
    cache_md_path: Path,
    cache_result_path: Path,
    *,
    target: Path,
    paper_full_path: Path,
    metadata_path: Path,
    project_root: Path,
    source: Path,
    source_sha: str,
    pdf_cfg: dict[str, Any],
    parser_config_hash: str,
    cache_status: str,
) -> dict[str, Any] | None:
    if not cache_md_path.exists() or cache_md_path.stat().st_size <= 0 or not cache_result_path.exists():
        return None
    cached_result = read_json(cache_result_path, default={})
    if not isinstance(cached_result, dict):
        return None
    if cached_result.get("source_sha256") and cached_result.get("source_sha256") != source_sha:
        return None
    if cached_result.get("parser_config_hash") and cached_result.get("parser_config_hash") != parser_config_hash:
        return None
    shutil.copy2(cache_md_path, paper_full_path)
    metadata = dict(cached_result)
    metadata.update(
        {
            "cache_status": cache_status,
            "restored_at": now_utc(),
            "source_sha256": source_sha,
            "source_path": str(source),
            "local_pdf_path": target.relative_to(project_root).as_posix(),
            "paper_full_md_path": paper_full_path.name,
            "parser_config_hash": parser_config_hash,
            "prompt_schema_version": metadata.get("prompt_schema_version") or "c2c_paper_full_markdown_v1",
        }
    )
    metadata.setdefault("provider", "mineru")
    metadata.setdefault("schema_version", "mineru_pdf_parse_result_v1")
    write_json(metadata_path, metadata)
    text = paper_full_path.read_text(encoding="utf-8", errors="ignore")
    return _mineru_parse_payload(
        target=target,
        project_root=project_root,
        paper_full_path=paper_full_path,
        metadata_path=metadata_path,
        metadata=metadata,
        pdf_cfg=pdf_cfg,
        cache_status=cache_status,
        text=text,
    )


def _restore_legacy_mineru_cache(
    project_root: Path,
    *,
    target: Path,
    paper_full_path: Path,
    metadata_path: Path,
    source: Path,
    source_sha: str,
    pdf_cfg: dict[str, Any],
    parser_config_hash: str,
) -> dict[str, Any] | None:
    for bundle_path in sorted(project_root.parent.glob("*/intake/c2c/static_bundle.json"), key=lambda path: path.stat().st_mtime if path.exists() else 0.0, reverse=True):
        source_project_root = bundle_path.parents[2]
        if source_project_root == project_root:
            continue
        bundle = read_json(bundle_path, default={})
        if not isinstance(bundle, dict):
            continue
        for item in bundle.get("paper_full_manifest") or []:
            if not isinstance(item, dict) or item.get("sha256") != source_sha:
                continue
            if item.get("parser_config_hash") and item.get("parser_config_hash") != parser_config_hash:
                continue
            md_rel = item.get("paper_full_md_path")
            if not isinstance(md_rel, str) or not md_rel:
                continue
            candidate_md = source_project_root / md_rel
            if not candidate_md.exists() or candidate_md.stat().st_size <= 0:
                continue
            result_rel = item.get("mineru_result_path")
            if not isinstance(result_rel, str) or not result_rel:
                result_rel = str(Path(md_rel).parent / "mineru_result.json")
            candidate_result = source_project_root / result_rel
            metadata = read_json(candidate_result, default={}) if candidate_result.exists() else {}
            if not isinstance(metadata, dict):
                metadata = {}
            shutil.copy2(candidate_md, paper_full_path)
            metadata.update(
                {
                    "provider": "mineru",
                    "schema_version": "mineru_pdf_parse_result_v1",
                    "cache_status": "legacy_project_hit",
                    "restored_at": now_utc(),
                    "legacy_cache_source_project": source_project_root.name,
                    "source_sha256": source_sha,
                    "source_path": str(source),
                    "local_pdf_path": target.relative_to(project_root).as_posix(),
                    "paper_full_md_path": paper_full_path.name,
                    "model_version": item.get("model_version") or metadata.get("model_version") or str(pdf_cfg.get("model_version") or "vlm"),
                    "language": item.get("language") or metadata.get("language") or str(pdf_cfg.get("language") or "en"),
                    "parser_config_hash": parser_config_hash,
                    "prompt_schema_version": item.get("prompt_schema_version") or metadata.get("prompt_schema_version") or "c2c_paper_full_markdown_v1",
                }
            )
            write_json(metadata_path, metadata)
            text = paper_full_path.read_text(encoding="utf-8", errors="ignore")
            return _mineru_parse_payload(
                target=target,
                project_root=project_root,
                paper_full_path=paper_full_path,
                metadata_path=metadata_path,
                metadata=metadata,
                pdf_cfg=pdf_cfg,
                cache_status="legacy_project_hit",
                text=text,
            )
    return None


def _write_mineru_cache_copy(cache_md_path: Path, cache_result_path: Path, paper_full_path: Path, metadata_path: Path) -> None:
    if not paper_full_path.exists() or not metadata_path.exists():
        return
    ensure_dir(cache_md_path.parent)
    shutil.copy2(paper_full_path, cache_md_path)
    shutil.copy2(metadata_path, cache_result_path)


def _parser_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pypdf"
    if suffix == ".json":
        return "json"
    if suffix in {".md", ".txt"}:
        return "text"
    return "file_text"


def _mineru_parser_config_hash(pdf_cfg: dict[str, Any]) -> str:
    payload = {
        "provider": str(pdf_cfg.get("provider") or "mineru"),
        "model_version": str(pdf_cfg.get("model_version") or "vlm"),
        "language": str(pdf_cfg.get("language") or "en"),
        "enable_formula": bool(pdf_cfg.get("enable_formula", True)),
        "enable_table": bool(pdf_cfg.get("enable_table", True)),
        "is_ocr": bool(pdf_cfg.get("is_ocr", False)),
        "prompt_schema_version": "c2c_paper_full_markdown_v1",
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _extract_pdf_with_pdftotext(path: Path, *, max_chars: int) -> str:
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(path), "-"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout[:max_chars]


C2C_MECHANISM_TYPES = {
    "utility_predicted_cache_routing",
    "counterfactual_training_objective",
    "semantic_span_graph_alignment",
    "verifier_guided_cache_acceptance",
    "latent_bridge_memory",
    "pathology_conditioned_controller",
}
C2C_LOCAL_TUNING_TERMS = {
    "threshold",
    "confidence floor",
    "confidence_floor",
    "top-k",
    "top_k",
    "temperature",
    "boundary bonus",
    "boundary_bonus",
    "fallback confidence",
    "fallback_confidence",
    "min weight",
    "min_weight",
    "alpha",
    "entropy penalty",
    "reweight strength",
    "reweight_strength",
}
C2C_HARD_GATE_STACK_TERMS = {
    "hard gate",
    "hard-gate",
    "binary gate",
    "binary accept",
    "accept/reject gate",
    "reject/accept",
    "stacked gate",
    "additional gate",
    "second gate",
    "only when",
    "filter out",
    "filters out",
    "reject spans",
    "reject transferred",
    "drop spans",
    "prune spans",
    "fixed threshold gate",
}
C2C_MECHANISM_TERMS = {
    "objective",
    "counterfactual",
    "routing",
    "controller",
    "verifier",
    "memory",
    "latent",
    "graph",
    "utility",
    "acceptance",
    "pathology",
    "training signal",
}
C2C_COVERAGE_STATS = [
    "baseline_transfer_coverage",
    "candidate_transfer_coverage",
    "matched_coverage_delta",
    "coverage_by_dataset",
    "coverage_by_pathology_bucket",
]
C2C_IMPLEMENTATION_SCOPES = {"bounded", "medium", "large"}
C2C_SCOPE_ORDER = {"bounded": 0, "medium": 1, "large": 2}
C2C_LARGE_SCOPE_TERMS = {
    "new training pipeline",
    "new trainer",
    "new dataset",
    "data format",
    "evaluation harness",
    "multi-stage",
    "pretraining",
    "new architecture",
    "rewrite",
    "large refactor",
}


def _c2c_text_blob(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(f"{key} {_c2c_text_blob(item)}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(_c2c_text_blob(item) for item in value)
    return str(value)


def _c2c_flatten_keys(value: Any, *, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return []
    keys: list[str] = []
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        keys.append(name)
        if isinstance(item, dict):
            keys.extend(_c2c_flatten_keys(item, prefix=name))
    return keys


def _c2c_expected_file_list(value: Any) -> list[str]:
    if value in (None, "", [], {}):
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        paths: list[str] = []
        for item in value:
            if isinstance(item, str):
                paths.append(item)
            elif isinstance(item, dict) and item.get("path"):
                paths.append(str(item["path"]))
        return paths
    if isinstance(value, dict):
        paths: list[str] = []
        for key in ("files", "expected_files", "paths"):
            paths.extend(_c2c_expected_file_list(value.get(key)))
        if value.get("path"):
            paths.append(str(value["path"]))
        return paths
    return []


def _infer_c2c_mechanism_type(idea: dict[str, Any]) -> str:
    blob = _c2c_text_blob(idea).lower()
    if "counterfactual" in blob or "dropout objective" in blob:
        return "counterfactual_training_objective"
    if "verifier" in blob or "acceptance" in blob or "margin probe" in blob:
        return "verifier_guided_cache_acceptance"
    if "latent" in blob or "bridge memory" in blob:
        return "latent_bridge_memory"
    if "graph" in blob or "path consistency" in blob:
        return "semantic_span_graph_alignment"
    if "pathology" in blob or "controller" in blob:
        return "pathology_conditioned_controller"
    if "routing" in blob or "utility" in blob:
        return "utility_predicted_cache_routing"
    return ""


def _default_expected_signature(mechanism_type: str) -> dict[str, Any]:
    signatures = {
        "utility_predicted_cache_routing": {
            "primary": "Proxy replay should reject more low-utility transferred spans while preserving useful ARC/OpenBookQA spans.",
            "stats": ["accepted_span_rate", "mean_predicted_cache_utility", "dataset_delta_vs_proxy_baseline"],
        },
        "counterfactual_training_objective": {
            "primary": "With-cache loss should improve over counterfactual cache-dropout batches on aligned spans.",
            "stats": ["paired_cache_margin", "counterfactual_dropout_loss_gap", "dataset_delta_vs_proxy_baseline"],
        },
        "semantic_span_graph_alignment": {
            "primary": "Accepted alignments should form higher-consistency span paths under fragmented tokenization.",
            "stats": ["graph_path_consistency", "ambiguous_span_rejection_rate", "dataset_delta_vs_proxy_baseline"],
        },
        "verifier_guided_cache_acceptance": {
            "primary": "Verifier-rejected examples should concentrate in prior dragging datasets and ambiguous answer formats.",
            "stats": ["verifier_accept_rate", "receiver_margin_after_transfer", "dragging_dataset_delta"],
        },
        "latent_bridge_memory": {
            "primary": "Bridge slots should reduce direct noisy span injection while retaining reusable cross-tokenizer state.",
            "stats": ["bridge_slot_norm", "direct_injection_rate", "dataset_delta_vs_proxy_baseline"],
        },
        "pathology_conditioned_controller": {
            "primary": "Controller decisions should correlate with alignment pathology stats rather than dataset labels alone.",
            "stats": ["pathology_bucket_delta", "controller_suppression_rate", "dragging_dataset_delta"],
        },
    }
    return signatures.get(mechanism_type, {"primary": "Mechanism-specific proxy statistics must move in the predicted direction.", "stats": []})


def _default_ablation_plan(idea_id: str, mechanism_type: str) -> dict[str, Any]:
    switch_key = f"ablation_disable_{sanitize_filename(idea_id)}"
    return {
        "switch": switch_key,
        "off_behavior": "fall back to the configured original C2C baseline behavior",
        "must_compare": ["proxy_baseline", "candidate_enabled", "candidate_disabled"],
        "claim": f"The {mechanism_type} component, not incidental threshold drift, explains any C2C gain.",
    }


def _default_coverage_diagnostics(mechanism_type: str) -> dict[str, Any]:
    return {
        "required": True,
        "stats": list(C2C_COVERAGE_STATS),
        "breakdowns": ["dataset", "alignment_pathology_bucket", "sample_family"],
        "purpose": (
            "Show whether score movement comes from a real mechanism or from simply reducing transferred-cache coverage."
        ),
        "mechanism_specific_stat": {
            "utility_predicted_cache_routing": "utility_weighted_coverage",
            "counterfactual_training_objective": "cache_on_cache_dropout_paired_coverage",
            "semantic_span_graph_alignment": "span_graph_path_covered_fraction",
            "verifier_guided_cache_acceptance": "verifier_weighted_coverage",
            "latent_bridge_memory": "bridge_slot_coverage",
            "pathology_conditioned_controller": "pathology_bucket_transfer_coverage",
        }.get(mechanism_type, "mechanism_coverage"),
    }


def _default_matched_coverage_ablation(idea_id: str, mechanism_type: str) -> dict[str, Any]:
    return {
        "required": True,
        "control": f"matched_coverage_{sanitize_filename(idea_id)}",
        "matching_keys": ["accepted_span_count", "cache_tokens_per_sample", "transfer_coverage_rate"],
        "comparison": "candidate_enabled_vs_baseline_matched_to_same_coverage",
        "expected_outcome": (
            f"The {mechanism_type} candidate should still improve mechanism stats and proxy/full metrics when coverage is matched."
        ),
        "guards_against": "apparent gains caused only by adding another hard reject gate or shrinking cache-transfer coverage",
    }


def _default_implementation_plan(idea_id: str, mechanism_type: str, expected_files: list[str]) -> dict[str, Any]:
    module_name = sanitize_filename(idea_id)
    new_file = f"rosetta/model/{module_name}.py"
    target_files = list(expected_files or DEFAULT_ALLOWED_FILES)
    plans = {
        "utility_predicted_cache_routing": {
            "scope": "medium",
            "required_new_files": [new_file],
            "integration_points": [
                {"path": "rosetta/model/projector.py", "symbol": "cache projection path", "change": "call the utility router before accepting transferred KV spans"},
                {"path": "rosetta/model/wrapper.py", "symbol": "runtime C2C wrapper", "change": "thread config and ablation switch into routing decisions"},
            ],
            "smoke_tests": ["test/test_aligner_span_overlap.py"],
            "decomposition_plan": [
                "Implement a small utility router module and wire it into projector/wrapper.",
                "Add config activation, ablation disable path, and focused smoke test.",
            ],
        },
        "counterfactual_training_objective": {
            "scope": "large",
            "required_new_files": [new_file],
            "integration_points": [
                {"path": "script/train/SFT_train.py", "symbol": "training loss assembly", "change": "add paired cache-on/cache-dropout objective behind a config flag"},
                {"path": "rosetta/model/projector.py", "symbol": "cache dropout path", "change": "expose deterministic cache-dropout hooks for paired batches"},
            ],
            "smoke_tests": ["test/test_aligner_span_overlap.py"],
            "decomposition_plan": [
                "Patch 1: add projector cache-dropout hook and ablation flag.",
                "Patch 2: add training loss term and minimal config activation.",
                "Patch 3: add proxy stats for paired cache margin.",
            ],
            "mvp_slice": "Implement patch 1 first; do not rewrite the full trainer in one Codex call.",
        },
        "semantic_span_graph_alignment": {
            "scope": "medium",
            "required_new_files": [new_file],
            "integration_points": [
                {"path": "rosetta/model/aligner.py", "symbol": "soft span alignment", "change": "build span graph consistency scores before KV aggregation"},
                {"path": "rosetta/model/projector.py", "symbol": "alignment metadata", "change": "consume graph consistency stats and expose ablation switch"},
            ],
            "smoke_tests": ["test/test_aligner_span_overlap.py"],
            "decomposition_plan": [
                "Add graph scoring helper.",
                "Wire graph score into existing alignment path with a disabled fallback.",
            ],
        },
        "verifier_guided_cache_acceptance": {
            "scope": "medium",
            "required_new_files": [new_file],
            "integration_points": [
                {"path": "rosetta/model/projector.py", "symbol": "cache acceptance", "change": "apply verifier decision before transferred cache injection"},
                {"path": "rosetta/model/wrapper.py", "symbol": "runtime metadata", "change": "surface verifier stats for proxy/failure attribution"},
            ],
            "smoke_tests": ["test/test_aligner_span_overlap.py"],
            "decomposition_plan": [
                "Add verifier helper with margin/pathology inputs.",
                "Wire verifier into projector path and add ablation switch.",
            ],
        },
        "latent_bridge_memory": {
            "scope": "large",
            "required_new_files": [new_file],
            "integration_points": [
                {"path": "rosetta/model/projector.py", "symbol": "projector module", "change": "insert bridge-memory slots between sharer and receiver KV states"},
                {"path": "rosetta/model/wrapper.py", "symbol": "cache lifecycle", "change": "initialize and disable bridge slots via config"},
            ],
            "smoke_tests": ["test/test_aligner_span_overlap.py"],
            "decomposition_plan": [
                "Patch 1: add bridge-memory module and disabled fallback wiring.",
                "Patch 2: train bridge slots and expose stats.",
                "Patch 3: add ablation and proxy diagnostics.",
            ],
            "mvp_slice": "Implement the bridge module and disabled fallback path before training-specific changes.",
        },
        "pathology_conditioned_controller": {
            "scope": "medium",
            "required_new_files": [new_file],
            "integration_points": [
                {"path": "rosetta/model/aligner.py", "symbol": "alignment pathology stats", "change": "compute pathology buckets from alignment metadata"},
                {"path": "rosetta/model/projector.py", "symbol": "cache controller", "change": "condition transfer strength on pathology buckets with ablation fallback"},
            ],
            "smoke_tests": ["test/test_aligner_span_overlap.py"],
            "decomposition_plan": [
                "Add pathology-stat extraction.",
                "Wire controller into projector and expose suppression stats.",
            ],
        },
    }
    plan = dict(plans.get(mechanism_type) or {})
    if not plan:
        plan = {
            "scope": "bounded",
            "required_new_files": [],
            "integration_points": [{"path": path, "symbol": "existing C2C path", "change": "implement the mechanism in-place"} for path in target_files[:3]],
            "smoke_tests": ["test/test_aligner_span_overlap.py"],
            "decomposition_plan": ["Implement a minimal mechanism slice in existing model files."],
        }
    plan.setdefault("allowed_first_patch_files", sorted(set(target_files + list(plan.get("required_new_files") or []))))
    return plan


def _infer_c2c_implementation_scope(idea: dict[str, Any], expected_files: list[str]) -> str:
    raw_scope = str(idea.get("implementation_scope") or "").strip().lower()
    if raw_scope in C2C_IMPLEMENTATION_SCOPES:
        return raw_scope
    plan = idea.get("implementation_plan") if isinstance(idea.get("implementation_plan"), dict) else {}
    raw_plan_scope = str(plan.get("scope") or "").strip().lower()
    if raw_plan_scope in C2C_IMPLEMENTATION_SCOPES:
        return raw_plan_scope
    blob = _c2c_text_blob(idea).lower()
    required_new_files = _c2c_expected_file_list(idea.get("required_new_files") or plan.get("required_new_files"))
    if len(required_new_files) >= 2 or any(term in blob for term in C2C_LARGE_SCOPE_TERMS):
        return "large"
    if required_new_files or any(path not in DEFAULT_ALLOWED_FILES for path in expected_files):
        return "medium"
    return "bounded"


def c2c_implementation_scope_report(idea: dict[str, Any]) -> dict[str, Any]:
    item = idea if isinstance(idea, dict) else {}
    contract = item.get("experiment_contract") if isinstance(item.get("experiment_contract"), dict) else {}
    expected_files = _c2c_expected_file_list(contract.get("expected_files")) or _c2c_expected_file_list(item.get("expected_files"))
    plan = item.get("implementation_plan") if isinstance(item.get("implementation_plan"), dict) else {}
    scope = _infer_c2c_implementation_scope(item, expected_files)
    required_new_files = _c2c_expected_file_list(item.get("required_new_files") or plan.get("required_new_files"))
    integration_points = item.get("integration_points") or plan.get("integration_points") or []
    smoke_tests = item.get("smoke_tests") or plan.get("smoke_tests") or []
    decomposition = item.get("decomposition_plan") or plan.get("decomposition_plan") or []
    blocked: list[str] = []
    if scope in {"medium", "large"} and not integration_points:
        blocked.append("missing integration_points")
    if scope in {"medium", "large"} and not smoke_tests:
        blocked.append("missing smoke_tests")
    if scope == "large" and not decomposition:
        blocked.append("missing decomposition_plan")
    if scope == "large" and not (plan.get("mvp_slice") or item.get("mvp_slice")):
        blocked.append("missing mvp_slice")
    return {
        "status": "pass" if not blocked else "needs_decomposition",
        "scope": scope,
        "required_new_files": required_new_files,
        "integration_points": integration_points,
        "smoke_tests": smoke_tests,
        "decomposition_plan": decomposition,
        "mvp_slice": plan.get("mvp_slice") or item.get("mvp_slice") or "",
        "expected_files": expected_files,
        "blocked_reasons": blocked,
    }


def normalize_c2c_mechanism_fields(idea: dict[str, Any], baseline: dict[str, Any] | None = None) -> dict[str, Any]:
    item = dict(idea or {})
    baseline = baseline or DEFAULT_BASELINE
    base_name = baseline.get("name", DEFAULT_BASELINE["name"])
    idea_id = str(item.get("id") or "c2c_mechanism")
    mechanism_type = str(item.get("mechanism_type") or _infer_c2c_mechanism_type(item))
    if mechanism_type:
        item["mechanism_type"] = mechanism_type
    item.setdefault("mechanism_summary", item.get("description") or item.get("hypothesis") or item.get("title") or "")
    if mechanism_type in C2C_MECHANISM_TYPES:
        if not item.get("paper_claim"):
            item["paper_claim"] = f"{mechanism_type} gives C2C a separable mechanism beyond local threshold tuning."
        if not item.get("why_baseline_fails"):
            item["why_baseline_fails"] = f"{base_name} can still inject harmful cross-tokenizer cache states when alignment evidence is ambiguous."
        if not item.get("expected_signature"):
            item["expected_signature"] = _default_expected_signature(mechanism_type)
        if not item.get("ablation_plan"):
            item["ablation_plan"] = _default_ablation_plan(idea_id, mechanism_type)
        if not item.get("coverage_diagnostics"):
            item["coverage_diagnostics"] = _default_coverage_diagnostics(mechanism_type)
        if not item.get("matched_coverage_ablation"):
            item["matched_coverage_ablation"] = _default_matched_coverage_ablation(idea_id, mechanism_type)
    if not _c2c_expected_file_list(item.get("expected_files")):
        item["expected_files"] = DEFAULT_ALLOWED_FILES
    if not item.get("verification_commands"):
        item["verification_commands"] = ["py_compile", "test_aligner_span_overlap", "small2048_train", "three_dataset_eval"]
    implementation_plan = item.get("implementation_plan") if isinstance(item.get("implementation_plan"), dict) else {}
    if mechanism_type in C2C_MECHANISM_TYPES and not implementation_plan:
        implementation_plan = _default_implementation_plan(idea_id, mechanism_type, _c2c_expected_file_list(item.get("expected_files")))
        item["implementation_plan"] = implementation_plan
    scope = _infer_c2c_implementation_scope(item, _c2c_expected_file_list(item.get("expected_files")))
    item.setdefault("implementation_scope", scope)
    if implementation_plan:
        item.setdefault("required_new_files", implementation_plan.get("required_new_files") or [])
        item.setdefault("integration_points", implementation_plan.get("integration_points") or [])
        item.setdefault("smoke_tests", implementation_plan.get("smoke_tests") or [])
        item.setdefault("decomposition_plan", implementation_plan.get("decomposition_plan") or [])
    contract = item.get("experiment_contract") if isinstance(item.get("experiment_contract"), dict) else {}
    if not isinstance(contract, dict):
        contract = {}
    contract.setdefault("primary_metric", "three_dataset_mean")
    contract.setdefault("baseline", base_name)
    contract.setdefault("expected_files", item.get("expected_files") or DEFAULT_ALLOWED_FILES)
    contract.setdefault("verification_commands", item.get("verification_commands"))
    if mechanism_type in C2C_MECHANISM_TYPES:
        if not contract.get("ablation_switch"):
            contract["ablation_switch"] = item.get("ablation_plan", {}).get("switch")
        if not contract.get("mechanism_type"):
            contract["mechanism_type"] = mechanism_type
        if not contract.get("coverage_diagnostics"):
            contract["coverage_diagnostics"] = item.get("coverage_diagnostics")
        if not contract.get("matched_coverage_ablation"):
            contract["matched_coverage_ablation"] = item.get("matched_coverage_ablation")
    contract.setdefault("implementation_scope", item.get("implementation_scope"))
    contract.setdefault("implementation_plan", item.get("implementation_plan"))
    item["experiment_contract"] = contract
    item["implementation_scope_gate"] = c2c_implementation_scope_report(item)
    return item


def c2c_idea_novelty_report(idea: dict[str, Any]) -> dict[str, Any]:
    item = idea if isinstance(idea, dict) else {}
    mechanism_type = str(item.get("mechanism_type") or _infer_c2c_mechanism_type(item))
    blob = _c2c_text_blob(item).lower()
    overrides = c2c_candidate_config_overrides(item)
    override_keys = _c2c_flatten_keys(overrides)
    expected_files = _c2c_expected_file_list(
        (item.get("experiment_contract") or {}).get("expected_files") if isinstance(item.get("experiment_contract"), dict) else item.get("expected_files")
    ) or _c2c_expected_file_list(item.get("expected_files"))
    local_hits = sorted(
        {
            term
            for term in C2C_LOCAL_TUNING_TERMS
            if term in blob or any(term.replace(" ", "_") in key.lower() or term.replace("-", "_") in key.lower() for key in override_keys)
        }
    )
    mechanism_hits = sorted({term for term in C2C_MECHANISM_TERMS if term in blob})
    hard_gate_hits = sorted(
        {
            term
            for term in C2C_HARD_GATE_STACK_TERMS
            if term in blob or any(term.replace(" ", "_").replace("-", "_") in key.lower() for key in override_keys)
        }
    )
    signals: list[str] = []
    if mechanism_type in C2C_MECHANISM_TYPES:
        signals.append("recognized_mechanism_type")
    if item.get("paper_claim"):
        signals.append("paper_claim")
    if item.get("why_baseline_fails"):
        signals.append("baseline_failure_model")
    if item.get("expected_signature"):
        signals.append("measurable_expected_signature")
    if item.get("ablation_plan") or (isinstance(item.get("experiment_contract"), dict) and item["experiment_contract"].get("ablation_switch")):
        signals.append("mechanism_ablation_plan")
    coverage_diagnostics = item.get("coverage_diagnostics")
    matched_coverage_ablation = item.get("matched_coverage_ablation")
    if coverage_diagnostics or (isinstance(item.get("experiment_contract"), dict) and item["experiment_contract"].get("coverage_diagnostics")):
        signals.append("coverage_diagnostics")
    if matched_coverage_ablation or (isinstance(item.get("experiment_contract"), dict) and item["experiment_contract"].get("matched_coverage_ablation")):
        signals.append("matched_coverage_ablation")
    if mechanism_hits:
        signals.append("mechanism_language")
    if item.get("failure_feedback_refs"):
        signals.append("uses_failure_feedback")
    if any(path.endswith(".py") for path in expected_files):
        signals.append("executable_mechanism_scope")
    config_only_local = bool(override_keys) and local_hits and not mechanism_hits and mechanism_type not in C2C_MECHANISM_TYPES
    pure_local_tuning = config_only_local or (bool(local_hits) and len(signals) < 4)
    missing_required = []
    if not (coverage_diagnostics or (isinstance(item.get("experiment_contract"), dict) and item["experiment_contract"].get("coverage_diagnostics"))):
        missing_required.append("coverage_diagnostics")
    if not (matched_coverage_ablation or (isinstance(item.get("experiment_contract"), dict) and item["experiment_contract"].get("matched_coverage_ablation"))):
        missing_required.append("matched_coverage_ablation")
    hard_gate_without_mechanism = bool(hard_gate_hits) and not (
        {"coverage_diagnostics", "matched_coverage_ablation", "mechanism_ablation_plan"} <= set(signals)
    )
    status = "pass" if len(set(signals)) >= 6 and not pure_local_tuning and not missing_required and not hard_gate_without_mechanism else "reject"
    return {
        "status": status,
        "mechanism_type": mechanism_type or "unspecified",
        "signals": sorted(set(signals)),
        "local_tuning_flags": local_hits,
        "hard_gate_stack_flags": hard_gate_hits,
        "missing_required_fields": missing_required,
        "mechanism_terms": mechanism_hits,
        "config_override_keys": override_keys[:30],
        "expected_files": expected_files[:12],
        "reason": (
            "mechanism-level idea with explicit claim, baseline failure model, signature, ablation, coverage diagnostics, and matched-coverage ablation"
            if status == "pass"
            else "idea looks like local tuning/hard-gate stacking or lacks mechanism-level coverage/ablation evidence fields"
        ),
    }


def _c2c_mechanism_idea(
    *,
    idea_id: str,
    title: str,
    description: str,
    motivation: str,
    hypothesis: str,
    mechanism_type: str,
    baseline_name: str,
    model_overrides: dict[str, Any],
    selected: bool = False,
    expected_files: list[str] | None = None,
    implementation_targets: list[dict[str, str]] | None = None,
    novelty_score: int = 9,
    feasibility_score: int = 7,
    expected_contribution: str = "",
    paper_claim: str = "",
    why_baseline_fails: str = "",
    expected_signature: dict[str, Any] | None = None,
    reviewer_risk_response: str = "",
    risk: str = "",
    novelty_against: list[str] | None = None,
    projector_params: dict[str, Any] | None = None,
    implementation_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    switch = f"ablation_disable_{sanitize_filename(idea_id)}"
    train_overrides = dict(model_overrides)
    train_overrides.setdefault(switch, False)
    files = expected_files or DEFAULT_ALLOWED_FILES
    contract = _c2c_experiment_contract(baseline_name, train_overrides, projector_params=projector_params)
    contract["expected_files"] = files
    contract["verification_commands"] = ["py_compile", "test_aligner_span_overlap", "small2048_train", "three_dataset_eval"]
    contract["implementation_targets"] = implementation_targets or [
        {"path": path, "role": "implement and expose the mechanism"}
        for path in files
    ]
    contract["ablation_switch"] = switch
    contract["mechanism_type"] = mechanism_type
    contract["coverage_diagnostics"] = _default_coverage_diagnostics(mechanism_type)
    contract["matched_coverage_ablation"] = _default_matched_coverage_ablation(idea_id, mechanism_type)
    scope_plan = implementation_plan or _default_implementation_plan(idea_id, mechanism_type, files)
    scope = _infer_c2c_implementation_scope({"implementation_plan": scope_plan, "expected_files": files}, files)
    contract["implementation_scope"] = scope
    contract["implementation_plan"] = scope_plan
    item = {
        "id": idea_id,
        "title": title,
        "description": description,
        "motivation": motivation,
        "hypothesis": hypothesis,
        "mechanism_type": mechanism_type,
        "mechanism_summary": description,
        "paper_claim": paper_claim or f"{title} creates a separable C2C mechanism beyond local threshold tuning.",
        "why_baseline_fails": why_baseline_fails or f"{baseline_name} can over-trust noisy cross-tokenizer cache states on ambiguous alignments.",
        "expected_signature": expected_signature or _default_expected_signature(mechanism_type),
        "ablation_plan": _default_ablation_plan(idea_id, mechanism_type),
        "coverage_diagnostics": _default_coverage_diagnostics(mechanism_type),
        "matched_coverage_ablation": _default_matched_coverage_ablation(idea_id, mechanism_type),
        "implementation_scope": scope,
        "implementation_plan": scope_plan,
        "required_new_files": scope_plan.get("required_new_files") or [],
        "integration_points": scope_plan.get("integration_points") or [],
        "smoke_tests": scope_plan.get("smoke_tests") or [],
        "decomposition_plan": scope_plan.get("decomposition_plan") or [],
        "novelty_score": novelty_score,
        "feasibility_score": feasibility_score,
        "expected_contribution": expected_contribution or "A mechanism-level C2C contribution with a direct ablation path.",
        "novelty_against": novelty_against or [baseline_name, "threshold/top-k/fallback tuning"],
        "reviewer_risk_response": reviewer_risk_response or "Separates mechanism evidence from mean-only tuning with proxy stats and an ablation switch.",
        "blocked_by_negative_results": False,
        "expected_files": files,
        "risk": risk or "The mechanism may be too weak in the small proxy loop to beat the imported baseline.",
        "verification_commands": ["py_compile", "test_aligner_span_overlap", "small2048_train", "three_dataset_eval"],
        "experiment_contract": contract,
        "selected": selected,
    }
    item["novelty_gate"] = c2c_idea_novelty_report(item)
    item["implementation_scope_gate"] = c2c_implementation_scope_report(item)
    return item


def default_c2c_ideas(topic: str, baseline: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    baseline = baseline or DEFAULT_BASELINE
    base_name = baseline.get("name", DEFAULT_BASELINE["name"])
    ideas = [
        _c2c_mechanism_idea(
            idea_id="utility_predicted_cache_routing",
            title="Utility-predicted cache routing",
            description="Learn a receiver-side utility predictor that routes each transferred cache span only when the predicted answer-margin gain is positive.",
            motivation=f"{base_name} already communicates cache states, but it does not estimate whether a transferred span helps the current receiver decision.",
            hypothesis="A utility-predicted router improves proxy and full-loop mean accuracy by accepting fewer harmful cross-tokenizer spans, especially on MMLU-style ambiguous options.",
            mechanism_type="utility_predicted_cache_routing",
            baseline_name=base_name,
            model_overrides={
                "cache_routing_mode": "utility_predictor",
                "cache_routing_loss_weight": 0.15,
                "cache_routing_min_utility": 0.0,
            },
            selected=True,
            expected_contribution="A decision-utility routing mechanism for cross-tokenizer KV transfer, not another fixed confidence threshold.",
            paper_claim="C2C cache sharing should be conditioned on predicted receiver utility rather than raw alignment confidence.",
            why_baseline_fails=f"{base_name} can inject aligned but answer-harmful KV states because confidence is not trained against downstream utility.",
            expected_signature={
                "primary": "Rejected spans should have lower proxy answer-margin utility than accepted spans.",
                "stats": ["accepted_span_rate", "predicted_utility_margin", "mmlu-redux_delta_vs_proxy_baseline"],
            },
            reviewer_risk_response="The utility margin and ablation switch make it possible to separate a routing contribution from threshold tuning.",
            risk="The small proxy replay may provide a noisy utility label unless paired with counterfactual cache-off evidence.",
        ),
        _c2c_mechanism_idea(
            idea_id="counterfactual_cache_dropout_objective",
            title="Counterfactual cache-dropout objective",
            description="Train the projector with paired cache-on/cache-dropout batches so the model learns when transferred KV states causally help the receiver.",
            motivation="Mean-only tuning cannot tell whether a transferred span causes the answer gain or merely correlates with easier examples.",
            hypothesis="A counterfactual objective increases the cache-on versus cache-off margin on useful spans and reduces dataset-specific regressions.",
            mechanism_type="counterfactual_training_objective",
            baseline_name=base_name,
            model_overrides={
                "counterfactual_cache_dropout": True,
                "counterfactual_margin_weight": 0.2,
                "counterfactual_dropout_prob": 0.5,
            },
            expected_contribution="A causal training signal for cross-tokenizer cache communication.",
            paper_claim="Cross-tokenizer cache transfer needs a paired counterfactual objective to distinguish helpful communication from spurious alignment.",
            why_baseline_fails=f"{base_name} optimizes the projected cache path without a paired cache-off contrast, so harmful spans can remain unpenalized.",
            expected_signature={
                "primary": "Cache-on examples should beat matched cache-dropout examples after training.",
                "stats": ["paired_cache_margin", "counterfactual_dropout_loss_gap", "dataset_delta_vs_proxy_baseline"],
            },
            reviewer_risk_response="The cache-off ablation directly answers whether the communication path caused the gain.",
            risk="Extra paired batches may be too expensive or unstable in the 2048-sample loop.",
        ),
        _c2c_mechanism_idea(
            idea_id="semantic_span_graph_alignment",
            title="Semantic span-graph alignment",
            description="Represent cross-tokenizer alignment as a local span graph and aggregate KV states over path-consistent semantic spans instead of independent token pairs.",
            motivation="Fragmented tokenization creates multi-token paths; independent soft-span weights can accept locally plausible but globally inconsistent alignments.",
            hypothesis="A span-graph consistency mechanism improves robustness on fragmented spans while preserving the existing route1 compute envelope.",
            mechanism_type="semantic_span_graph_alignment",
            baseline_name=base_name,
            model_overrides={
                "soft_alignment_score_mode": "span_graph_consistency",
                "span_graph_max_paths": 4,
                "span_graph_consistency_weight": 0.35,
            },
            expected_contribution="A graph-level alignment representation for C2C rather than token-local reweighting.",
            paper_claim="Cross-tokenizer KV transfer should preserve span-path consistency when token boundaries disagree.",
            why_baseline_fails=f"{base_name} scores candidate spans mostly independently, so fragmented paths can pass local checks while breaking semantic continuity.",
            expected_signature={
                "primary": "Accepted alignments should have higher span-path consistency in tokenizer-fragmented examples.",
                "stats": ["graph_path_consistency", "fragmented_span_accept_rate", "openbookqa_delta_vs_proxy_baseline"],
            },
            reviewer_risk_response="Path-consistency statistics are inspectable and connect directly to tokenizer mismatch concerns.",
            risk="A graph path mechanism may require careful fallback to avoid pruning useful short spans.",
        ),
    ]
    return ideas


def failure_aware_c2c_ideas(topic: str, baseline: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    baseline = baseline or DEFAULT_BASELINE
    base_name = baseline.get("name", DEFAULT_BASELINE["name"])
    return [
        _c2c_mechanism_idea(
            idea_id="verifier_guided_cache_acceptance",
            title="Verifier-guided cache acceptance",
            description="Add a lightweight verifier that accepts transferred cache spans only when receiver margin, alignment evidence, and answer-format risk agree.",
            motivation="Failure feedback often identifies one dragging dataset; the next loop should test an explicit verifier instead of another confidence-floor setting.",
            hypothesis="Verifier-guided acceptance recovers dragging datasets such as MMLU-redux while retaining useful transfer on ARC/OpenBookQA.",
            mechanism_type="verifier_guided_cache_acceptance",
            baseline_name=base_name,
            model_overrides={
                "cache_acceptance_mode": "verifier_guided",
                "cache_verifier_margin_weight": 0.4,
                "cache_verifier_pathology_weight": 0.3,
            },
            selected=True,
            expected_contribution="A verifier mechanism that turns failure evidence into a reject/accept decision for cache communication.",
            paper_claim="C2C needs a receiver-side verifier for risky transferred states, especially under option-format ambiguity.",
            why_baseline_fails=f"{base_name} lacks a verifier that can reject aligned-but-harmful cache states when a dataset-specific pathology appears.",
            expected_signature={
                "primary": "Verifier rejections should concentrate in ambiguous samples from the dragging dataset while proxy mean stays above baseline.",
                "stats": ["verifier_accept_rate", "dragging_dataset_delta", "ambiguous_sample_rejection_rate"],
            },
            reviewer_risk_response="The verifier gives a direct failure-attribution hook: which dataset and sample type was blocked, and why.",
            risk="A conservative verifier may suppress genuinely helpful cache transfer.",
        ),
        _c2c_mechanism_idea(
            idea_id="pathology_conditioned_cache_controller",
            title="Pathology-conditioned cache controller",
            description="Route cache injection through a controller conditioned on alignment pathology statistics instead of using the same mechanism for all samples.",
            motivation="OpenBookQA-only gains with MMLU regressions imply sample-pathology heterogeneity that a global threshold cannot model.",
            hypothesis="A pathology-conditioned controller improves the worst dragging dataset without relying on dataset labels.",
            mechanism_type="pathology_conditioned_controller",
            baseline_name=base_name,
            model_overrides={
                "cache_controller_mode": "alignment_pathology",
                "cache_controller_bucket_count": 4,
                "cache_controller_loss_weight": 0.1,
            },
            expected_contribution="A pathology-aware control policy for C2C transfer.",
            paper_claim="C2C transfer should be conditioned on measurable alignment pathologies, not fixed global confidence parameters.",
            why_baseline_fails=f"{base_name} applies one transfer policy across low- and high-pathology examples.",
            expected_signature={
                "primary": "Controller suppression should rise in high-pathology buckets and reduce worst-dataset regression.",
                "stats": ["pathology_bucket_delta", "controller_suppression_rate", "worst_dataset_regression"],
            },
            reviewer_risk_response="Avoids dataset-label overfitting by requiring pathology-bucket evidence in the proxy report.",
            risk="Pathology buckets may be too coarse in the cheap proxy subset.",
        ),
        _c2c_mechanism_idea(
            idea_id="latent_bridge_memory",
            title="Latent bridge memory",
            description="Insert a small learned bridge-memory slot between sharer and receiver KV states so transfer happens through reusable latent slots rather than direct noisy span injection.",
            motivation="Prior failures suggest direct token-span transfer can be brittle under tokenizer mismatch.",
            hypothesis="Latent bridge memory reduces harmful direct injection while preserving reusable cross-tokenizer context.",
            mechanism_type="latent_bridge_memory",
            baseline_name=base_name,
            model_overrides={
                "bridge_memory_mode": "latent_slots",
                "bridge_memory_slots": 4,
                "bridge_memory_residual_weight": 0.25,
            },
            expected_contribution="A memory-structure change for cross-tokenizer communication.",
            paper_claim="A latent bridge memory can decouple transferable semantics from brittle token boundary correspondences.",
            why_baseline_fails=f"{base_name} maps soft spans directly into receiver cache space, so boundary noise can propagate into attention.",
            expected_signature={
                "primary": "Bridge-enabled runs should lower direct injection rate and keep or improve proxy mean versus baseline.",
                "stats": ["bridge_slot_norm", "direct_injection_rate", "dataset_delta_vs_proxy_baseline"],
            },
            reviewer_risk_response="The disabled-bridge ablation isolates whether the new memory structure matters.",
            risk="Bridge slots may need more training than the small loop provides.",
        ),
    ]


def _c2c_experiment_contract(
    baseline_name: str,
    model_overrides: dict[str, Any],
    *,
    projector_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_model = dict(model_overrides)
    if projector_params:
        train_model["projector"] = {"params": projector_params}
    eval_rosetta = _project_train_model_overrides_to_eval(model_overrides)
    return {
        "primary_metric": "three_dataset_mean",
        "baseline": baseline_name,
        "config_overrides": {
            "train": {"model": train_model},
            "eval": {"model": {"rosetta_config": eval_rosetta}},
        },
    }


def build_c2c_ideas_with_llm(
    *,
    llm: Any,
    topic: str,
    repo_manifest: dict[str, Any],
    baseline: dict[str, Any],
    reference_cards: list[dict[str, Any]],
    rebuttal_concerns: dict[str, Any] | None = None,
    negative_memory: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    fallback = default_c2c_ideas(topic, baseline)
    if not getattr(llm, "use_real_api", False):
        return fallback
    prompt = {
        "topic": topic,
        "baseline_to_beat": baseline,
        "allowed_files": DEFAULT_ALLOWED_FILES,
        "repo_core_files": repo_manifest.get("core_files", []),
        "repo_docs": [
            {"path": item.get("path"), "snippet": (item.get("snippet") or "")[:1200]}
            for item in repo_manifest.get("docs", [])
            if item.get("exists")
        ],
        "reference_cards": [
            {"title": card.get("title"), "kind": card.get("kind"), "text": (card.get("text") or "")[:1800]}
            for card in reference_cards
        ],
        "reviewer_concern_matrix": rebuttal_concerns or {},
        "negative_result_memory": negative_memory or {},
        "required_output": (
            "a JSON list of exactly three ideas with id,title,description,motivation,hypothesis,"
            "novelty_score,feasibility_score,expected_contribution,novelty_against,reviewer_risk_response,"
            "blocked_by_negative_results,expected_files,risk,verification_commands,experiment_contract,selected,"
            "mechanism_type,paper_claim,why_baseline_fails,expected_signature,ablation_plan,"
            "coverage_diagnostics,matched_coverage_ablation,"
            "implementation_scope,implementation_plan,required_new_files,integration_points,smoke_tests,decomposition_plan"
        ),
    }
    try:
        ideas = llm.generate_json(
            instructions=(
                "You are a senior ML systems researcher designing C2C cross-tokenizer KV-cache experiments. "
                "Return only JSON. Keep every idea executable within the allowed files and the small2048 plus three-eval protocol. "
                "Do not propose pure threshold/top-k/fallback tuning. Each idea must introduce a mechanism-level change with an ablation switch. "
                "Do not propose stacking another hard accept/reject gate unless the mechanism includes coverage diagnostics and matched-coverage ablation proving gains are not just lower transfer coverage. "
                "If the idea needs new files or larger architecture changes, mark implementation_scope as medium or large and include integration_points, smoke_tests, and a decomposition_plan."
            ),
            prompt=json.dumps(prompt, ensure_ascii=False),
            default=fallback,
            agent_name="c2c-literature-agent",
        )
    except Exception:
        return fallback
    if not isinstance(ideas, list) or len(ideas) < 3:
        return fallback
    normalized = []
    for idx, idea in enumerate(ideas[:3]):
        if not isinstance(idea, dict):
            return fallback
        item = dict(idea)
        item.setdefault("id", f"c2c_idea_{idx + 1}")
        item.setdefault("title", item["id"].replace("_", " ").title())
        item.setdefault("novelty_score", 7)
        item.setdefault("feasibility_score", 7)
        item.setdefault("expected_files", DEFAULT_ALLOWED_FILES)
        item.setdefault("verification_commands", ["py_compile", "test_aligner_span_overlap", "small2048_train", "three_dataset_eval"])
        item.setdefault("reviewer_risk_response", "Tie the result to reviewer-visible concerns from rebuttal evidence.")
        item.setdefault("novelty_against", [baseline.get("name", DEFAULT_BASELINE["name"])])
        item.setdefault("blocked_by_negative_results", False)
        item.setdefault("experiment_contract", {"primary_metric": "three_dataset_mean", "baseline": baseline.get("name", DEFAULT_BASELINE["name"])})
        item = normalize_c2c_mechanism_fields(item, baseline)
        item["novelty_gate"] = c2c_idea_novelty_report(item)
        item["selected"] = idx == 0
        normalized.append(item)
    if not any((idea.get("novelty_gate") or {}).get("status") == "pass" for idea in normalized):
        return fallback
    return normalized


def _read_snippet(path: Path, *, limit: int = 2500) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")[:limit]


def _read_json_fallback(path: Path, *, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def _configure_disabled_wandb(train_config: dict[str, Any], run_name: str) -> None:
    output = train_config.setdefault("output", {})
    if not isinstance(output, dict):
        train_config["output"] = output = {}
    wandb_config = output.setdefault("wandb_config", {})
    if not isinstance(wandb_config, dict):
        output["wandb_config"] = wandb_config = {}
    wandb_config["run_name"] = run_name
    wandb_config["mode"] = "disabled"
    wandb_config.setdefault("project", "auto-research-c2c")
    wandb_config["entity"] = None


def _configure_proxy_train_limits(train_config: dict[str, Any], proxy_cfg: dict[str, Any], *, selected_gpu_ids: list[int] | None = None) -> None:
    training = train_config.setdefault("training", {})
    if not isinstance(training, dict):
        train_config["training"] = training = {}
    training["per_device_train_batch_size"] = _proxy_auto_batch_size(proxy_cfg, selected_gpu_ids=selected_gpu_ids)
    training["gradient_accumulation_steps"] = max(1, int(proxy_cfg.get("gradient_accumulation_steps") or 1))
    if proxy_cfg.get("max_length"):
        training["max_length"] = max(1, int(proxy_cfg["max_length"]))


def _configure_c2c_train_resource_limits(
    train_config: dict[str, Any],
    policy: dict[str, Any],
    *,
    selected_gpu_ids: list[int] | None = None,
) -> dict[str, Any]:
    policy = deep_merge(copy.deepcopy(DEFAULT_C2C_TRAIN_RESOURCE_POLICY), policy if isinstance(policy, dict) else {})
    if not policy.get("enabled", True):
        return {"enabled": False, "status": "disabled"}
    training = train_config.setdefault("training", {})
    if not isinstance(training, dict):
        train_config["training"] = training = {}
    original_training = copy.deepcopy(training)
    original_batch = max(1, _safe_int(training.get("per_device_train_batch_size")) or int(policy.get("reference_per_device_train_batch_size") or 4))
    original_grad = max(1, _safe_int(training.get("gradient_accumulation_steps")) or int(policy.get("reference_gradient_accumulation_steps") or 8))
    original_gpu_count = max(1, int(policy.get("reference_num_gpus") or len(selected_gpu_ids or []) or 1))
    selected_gpu_count = max(1, len(selected_gpu_ids or []))
    selected_free_mb = _selected_gpu_free_memory_mb(selected_gpu_ids)
    selected_batch = _c2c_auto_train_batch_size(policy, selected_free_mb=selected_free_mb)
    configured_batch = policy.get("per_device_train_batch_size", "auto")
    if configured_batch not in (None, "", "auto"):
        selected_batch = max(1, int(configured_batch))
    training["per_device_train_batch_size"] = selected_batch
    original_effective_batch = original_batch * original_grad * original_gpu_count
    grad_setting = policy.get("gradient_accumulation_steps", "preserve_effective_batch")
    if grad_setting not in (None, "", "preserve_effective_batch", "auto"):
        training["gradient_accumulation_steps"] = max(1, int(grad_setting))
    else:
        training["gradient_accumulation_steps"] = max(1, (original_effective_batch + (selected_batch * selected_gpu_count) - 1) // (selected_batch * selected_gpu_count))
    new_effective_batch = selected_batch * max(1, _safe_int(training.get("gradient_accumulation_steps")) or 1) * selected_gpu_count
    lr_adjustment = _maybe_scale_learning_rate(
        training,
        original_training,
        original_effective_batch=original_effective_batch,
        new_effective_batch=new_effective_batch,
        policy=policy,
    )
    return {
        "enabled": True,
        "status": "applied",
        "selected_gpu_ids": list(selected_gpu_ids or []),
        "selected_gpu_free_mb": selected_free_mb,
        "original_per_device_train_batch_size": original_training.get("per_device_train_batch_size"),
        "per_device_train_batch_size": training.get("per_device_train_batch_size"),
        "original_gradient_accumulation_steps": original_training.get("gradient_accumulation_steps"),
        "gradient_accumulation_steps": training.get("gradient_accumulation_steps"),
        "original_effective_batch_size": original_effective_batch,
        "effective_batch_size": new_effective_batch,
        "learning_rate_adjustment": lr_adjustment,
        "policy": {
            "per_device_train_batch_size": policy.get("per_device_train_batch_size"),
            "gradient_accumulation_steps": policy.get("gradient_accumulation_steps"),
            "learning_rate_scale": policy.get("learning_rate_scale"),
        },
    }


def _c2c_auto_train_batch_size(policy: dict[str, Any], *, selected_free_mb: int) -> int:
    tiers = policy.get("batch_tiers") if isinstance(policy.get("batch_tiers"), list) else []
    for tier in sorted((item for item in tiers if isinstance(item, dict)), key=lambda item: -int(item.get("min_free_mb") or 0)):
        if selected_free_mb >= int(tier.get("min_free_mb") or 0):
            return max(1, int(tier.get("per_device_train_batch_size") or 1))
    return 1


def _maybe_scale_learning_rate(
    training: dict[str, Any],
    original_training: dict[str, Any],
    *,
    original_effective_batch: int,
    new_effective_batch: int,
    policy: dict[str, Any],
) -> dict[str, Any]:
    original_lr = original_training.get("learning_rate")
    if original_lr in (None, ""):
        return {"status": "skipped", "reason": "training.learning_rate missing"}
    mode = str(policy.get("learning_rate_scale") or "none")
    try:
        original_lr_value = float(original_lr)
    except (TypeError, ValueError):
        return {"status": "skipped", "reason": "training.learning_rate is not numeric", "original_learning_rate": original_lr}
    if mode not in {"effective_batch_ratio", "linear", "auto"}:
        training["learning_rate"] = original_lr_value
        return {"status": "unchanged", "mode": mode, "learning_rate": original_lr_value}
    if original_effective_batch <= 0 or new_effective_batch <= 0:
        training["learning_rate"] = original_lr_value
        return {"status": "unchanged", "reason": "invalid effective batch", "learning_rate": original_lr_value}
    ratio = new_effective_batch / original_effective_batch
    new_lr = original_lr_value * ratio
    min_lr = policy.get("min_learning_rate")
    max_lr = policy.get("max_learning_rate")
    if min_lr not in (None, ""):
        new_lr = max(float(min_lr), new_lr)
    if max_lr not in (None, ""):
        new_lr = min(float(max_lr), new_lr)
    training["learning_rate"] = new_lr
    return {
        "status": "scaled" if abs(new_lr - original_lr_value) > 1e-15 else "unchanged",
        "mode": mode,
        "original_learning_rate": original_lr_value,
        "learning_rate": new_lr,
        "effective_batch_ratio": ratio,
    }


def _proxy_auto_batch_size(proxy_cfg: dict[str, Any], *, selected_gpu_ids: list[int] | None = None) -> int:
    configured = proxy_cfg.get("per_device_train_batch_size", "auto")
    if configured not in (None, "", "auto"):
        return max(1, int(configured))
    total_mb = _selected_gpu_total_memory_mb(selected_gpu_ids)
    if total_mb >= 70000:
        return 4
    if total_mb >= 32000:
        return 3
    if total_mb >= 20000:
        return 2
    return 1


def _selected_gpu_total_memory_mb(selected_gpu_ids: list[int] | None = None) -> int:
    try:
        from .adapters.runner import ExperimentRunner
    except Exception:
        return 0
    snapshot = ExperimentRunner._gpu_snapshot()
    if not snapshot:
        return 0
    selected = {int(item) for item in selected_gpu_ids or []}
    candidates = [
        item
        for item in snapshot
        if not selected or int(item.get("index", -1)) in selected
    ]
    totals = [int(item.get("memory_total_mb") or 0) for item in candidates]
    return min(totals) if totals else 0


def _selected_gpu_free_memory_mb(selected_gpu_ids: list[int] | None = None) -> int:
    try:
        from .adapters.runner import ExperimentRunner
    except Exception:
        return 0
    snapshot = ExperimentRunner._gpu_snapshot()
    if not snapshot:
        return 0
    selected = {int(item) for item in selected_gpu_ids or []}
    candidates = [
        item
        for item in snapshot
        if not selected or int(item.get("index", -1)) in selected
    ]
    free = [int(item.get("memory_free_mb") or 0) for item in candidates]
    return min(free) if free else 0


def _first_present(row: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _has_complete_baseline_metrics(record: dict[str, Any]) -> bool:
    metrics = record.get("metrics") or {}
    return all(metrics.get(key) is not None for key in ["mmlu-redux", "ai2-arc", "openbookqa", "mean"])


def _safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _topic_tags(text: str) -> list[str]:
    text_lc = text.lower()
    tag_map = {
        "kv_cache": ["kv", "cache"],
        "cross_tokenizer": ["tokenizer", "tokenization"],
        "multi_agent": ["multi-agent", "multi agent", "collaboration"],
        "latency": ["latency", "throughput", "speed"],
        "communication": ["communication", "sharing", "transmission"],
        "semantic_transfer": ["semantic", "latent"],
    }
    tags = []
    for tag, keywords in tag_map.items():
        if any(keyword in text_lc for keyword in keywords):
            tags.append(tag)
    return tags


def _strip_reference_section(text: str) -> str:
    match = re.search(r"(?im)^\s*(references|bibliography)\s*$", text)
    if not match:
        return text
    return text[: match.start()].strip()


def _extract_bibliography_entries(text: str) -> list[dict[str, Any]]:
    match = re.search(r"(?im)^\s*(references|bibliography)\s*$", text)
    if not match:
        return []
    refs = text[match.end() :].strip()
    raw_entries = re.split(r"\n\s*(?=(?:\[\d+\]|\d+\.|\w.+\(\d{4}\)))", refs)
    entries = []
    for idx, entry in enumerate(raw_entries):
        normalized = " ".join(entry.split())
        if len(normalized) < 20:
            continue
        entries.append(
            {
                "entry_id": f"ref_{idx + 1:03d}",
                "text": normalized[:1200],
                "year": _first_year(normalized),
            }
        )
    return entries


def _first_year(text: str) -> int | None:
    match = re.search(r"\b(19|20)\d{2}\b", text)
    return int(match.group(0)) if match else None


def _split_reference_sections(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_name = "front_matter"
    current_lines: list[str] = []
    markdown_heading_re = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$")
    heading_re = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?(abstract|introduction|background|related work|method|methods|approach|model|experiments?|evaluation|results|analysis|limitations?|discussion|conclusion)\s*$",
        flags=re.IGNORECASE,
    )
    for line in lines:
        markdown_match = markdown_heading_re.match(line)
        if markdown_match:
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = sanitize_filename(markdown_match.group(1).lower(), max_length=80)
            current_lines = [line]
            continue
        heading_match = heading_re.match(line.strip())
        if heading_match and current_lines:
            sections.append((current_name, current_lines))
            current_name = heading_match.group(1).lower().replace(" ", "_")
            current_lines = []
            continue
        if heading_match:
            current_name = heading_match.group(1).lower().replace(" ", "_")
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_name, current_lines))
    normalized = []
    for name, section_lines in sections:
        body = "\n".join(section_lines).strip()
        if body:
            normalized.append((name, body))
    return normalized or [("full_text", text)]


def _chunk_text(text: str, *, max_chars: int, overlap: int = 0) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end), text.rfind("\n", start, end))
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _structured_rebuttal_concerns(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    structured = []
    for item in matrix:
        concern_id = item.get("concern_id", "")
        for hit in item.get("evidence", []):
            for snippet_idx, snippet in enumerate(hit.get("snippets", [])):
                structured.append(
                    {
                        "concern_id": concern_id,
                        "priority": item.get("priority", "low"),
                        "paper_id": hit.get("paper_id"),
                        "source_path": hit.get("source_path"),
                        "snippet_id": f"{concern_id}:{hit.get('paper_id')}:{snippet_idx}",
                        "keyword": snippet.get("keyword"),
                        "source_snippet": snippet.get("snippet"),
                        "experiment_implication": item.get("experiment_implication") or _concern_implication(concern_id),
                        "next_round_constraint": _concern_constraint(concern_id),
                    }
                )
    return structured


def _concern_constraint(concern_id: str) -> str:
    constraints = {
        "heterogeneous_model_support": "Do not claim general cross-model support unless the experiment keeps model-pair metadata explicit.",
        "training_cost_pair_specific": "Prefer ideas that reuse the current small-loop protocol without expanding pair-specific training cost.",
        "baseline_fairness": "Every selected idea must compare against the configured original C2C baseline.",
        "failure_modes_ood": "Reject candidates that optimize mean while hiding a large per-dataset regression.",
        "multi_sharer_scaling": "Treat multi-sharer claims as future work unless one-pair stability is established.",
        "dynamic_selection": "Dynamic selection must be bounded and ablatable, not an unconstrained learned router.",
        "latency_memory": "Track changed files and command contracts so later runs can measure latency/memory.",
        "positioning_related_work": "Require novelty statements against C2C/KV communication related work.",
    }
    return constraints.get(concern_id, "Turn the concern into an explicit S2/S3 gate.")


def _keyword_snippets(text: str, keywords: list[str], *, limit: int = 3, window: int = 180) -> list[dict[str, str]]:
    text_flat = " ".join(text.split())
    text_lc = text_flat.lower()
    snippets = []
    seen = set()
    for keyword in keywords:
        keyword_lc = keyword.lower()
        start = text_lc.find(keyword_lc)
        if start < 0:
            continue
        left = max(0, start - window)
        right = min(len(text_flat), start + len(keyword) + window)
        snippet = text_flat[left:right].strip()
        key = snippet[:120]
        if key in seen:
            continue
        seen.add(key)
        snippets.append({"keyword": keyword, "snippet": snippet})
        if len(snippets) >= limit:
            break
    return snippets


def _language_for_path(path: Path) -> str:
    if path.suffix == ".py":
        return "python"
    if path.suffix in {".yaml", ".yml"}:
        return "yaml"
    if path.suffix == ".json":
        return "json"
    return "text"


def _deduplicate_strings(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(text)
    return ordered


def _chunk_keywords(text: str, *, extra_terms: list[Any] | None = None, max_keywords: int = 24) -> list[str]:
    candidates: list[str] = []
    for value in extra_terms or []:
        if not value:
            continue
        candidates.extend(re.split(r"[^A-Za-z0-9_\-]+", str(value)))
    candidates.extend(_topic_tags(text))
    text_lc = text.lower()
    controlled_terms = [
        "alignment",
        "attention",
        "baseline",
        "cache",
        "communication",
        "confidence",
        "coverage",
        "cross-tokenizer",
        "dataset",
        "evaluation",
        "experiment",
        "fallback",
        "fuser",
        "gate",
        "heterogeneous",
        "kv",
        "latency",
        "loss",
        "mmlu",
        "openbookqa",
        "projector",
        "rebuttal",
        "regression",
        "router",
        "semantic",
        "span",
        "tokenizer",
        "training",
        "utility",
        "wrapper",
    ]
    candidates.extend(term for term in controlled_terms if term in text_lc)
    candidates.extend(
        match.group(0)
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9_]{3,}\b", text)
        if len(match.group(0)) <= 40
    )
    return _deduplicate_strings(candidates)[:max_keywords]


def _is_chunkable_repo_file(path: Path, rel: str) -> bool:
    if path.suffix.lower() not in FULL_CODE_CHUNK_SUFFIXES:
        return False
    if path.stat().st_size > FULL_CODE_CHUNK_MAX_FILE_BYTES:
        return False
    parts = set(Path(rel).parts)
    if parts.intersection(FULL_CODE_CHUNK_SKIP_PARTS):
        return False
    return True


def _chunk_index_entry(chunk: dict[str, Any], *, source_type: str, ordinal: int) -> dict[str, Any]:
    text = str(chunk.get("text") or "")
    source_path = chunk.get("source_path") or chunk.get("path") or chunk.get("local_path") or ""
    entry = {
        "chunk_id": chunk.get("chunk_id") or f"{source_type}:{ordinal:05d}",
        "source_type": source_type,
        "source_path": source_path,
        "local_path": chunk.get("local_path") or "",
        "paper_id": chunk.get("paper_id") or "",
        "title": chunk.get("title") or "",
        "section": chunk.get("section") or ("file" if source_type == "code" else "full_text"),
        "section_chunk_index": chunk.get("section_chunk_index"),
        "path": chunk.get("path") or source_path,
        "language": chunk.get("language") or "",
        "symbol": chunk.get("symbol") or "",
        "symbol_kind": chunk.get("symbol_kind") or "",
        "start_line": chunk.get("start_line"),
        "end_line": chunk.get("end_line"),
        "keywords": chunk.get("keywords") or _chunk_keywords(text, extra_terms=[source_path, chunk.get("section"), chunk.get("symbol")]),
        "semantic_summary": chunk.get("semantic_summary") or ((chunk.get("semantic_enrichment") or {}).get("semantic_summary") if isinstance(chunk.get("semantic_enrichment"), dict) else ""),
        "mechanism_tags": chunk.get("mechanism_tags") or ((chunk.get("semantic_enrichment") or {}).get("mechanism_tags") if isinstance(chunk.get("semantic_enrichment"), dict) else []),
        "failure_modes": chunk.get("failure_modes") or ((chunk.get("semantic_enrichment") or {}).get("failure_modes") if isinstance(chunk.get("semantic_enrichment"), dict) else []),
        "retrieval_keywords": chunk.get("retrieval_keywords") or ((chunk.get("semantic_enrichment") or {}).get("retrieval_keywords") if isinstance(chunk.get("semantic_enrichment"), dict) else []),
        "tokens_estimate": chunk.get("tokens_estimate") or max(1, len(text) // 4),
        "char_count": len(text),
        "text_preview": " ".join(text.split())[:500],
    }
    return {key: value for key, value in entry.items() if value not in (None, "", [])}


def _collect_focus_terms_from_baseline(baseline: dict[str, Any]) -> list[str]:
    terms = []
    name = baseline.get("name")
    if name:
        terms.extend(split for split in re.split(r"[_\-\s]+", str(name)) if split)
    datasets = baseline.get("datasets") or {}
    terms.extend(str(key) for key in datasets.keys())
    return terms


def _collect_focus_terms_from_rebuttal(rebuttal_matrix: dict[str, Any]) -> list[str]:
    terms = []
    for item in rebuttal_matrix.get("matrix", []):
        concern_id = item.get("concern_id")
        if concern_id:
            terms.append(str(concern_id).replace("_", " "))
        for hit in item.get("evidence", []):
            for snippet in hit.get("snippets", []):
                keyword = snippet.get("keyword")
                if keyword:
                    terms.append(str(keyword))
    return terms


def _collect_focus_terms_from_code(code_cards: list[dict[str, Any]]) -> list[str]:
    terms = []
    for card in code_cards:
        if not card.get("exists"):
            continue
        for symbol in card.get("symbols") or []:
            name = symbol.get("name")
            if name:
                terms.append(str(name))
        for knob in card.get("config_knobs") or []:
            terms.append(str(knob))
    return terms


def _rank_chunk_targets(chunks: list[dict[str, Any]], focus_terms: list[str], *, max_items: int) -> list[dict[str, Any]]:
    scored = []
    normalized_focus = [term.lower() for term in focus_terms if term]
    for chunk in chunks:
        text = " ".join(str(chunk.get("text", "")).split())
        score = 0
        text_lc = text.lower()
        for term in normalized_focus:
            if term and term in text_lc:
                score += 1
        if chunk.get("section"):
            section = str(chunk.get("section")).lower()
            if any(term in section for term in normalized_focus):
                score += 2
        scored.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "paper_id": chunk.get("paper_id"),
                "path": chunk.get("source_path") or chunk.get("path"),
                "section": chunk.get("section"),
                "symbol": chunk.get("symbol"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "score": score,
                "snippet": text[:500],
            }
        )
    scored.sort(key=lambda item: (-item["score"], str(item.get("path") or ""), str(item.get("chunk_id") or "")))
    return scored[:max_items]


def _rank_code_symbols(code_cards: list[dict[str, Any]], focus_terms: list[str]) -> list[dict[str, Any]]:
    normalized_focus = [term.lower() for term in focus_terms if term]
    scored = []
    for card in code_cards:
        if not card.get("exists"):
            continue
        file_name = Path(str(card.get("path") or "")).stem
        file_text = " ".join(
            [
                str(card.get("path") or ""),
                str(card.get("summary_snippet") or ""),
                " ".join(str(knob) for knob in card.get("config_knobs") or []),
            ]
        ).lower()
        file_score = sum(1 for term in normalized_focus if term in file_text)
        if not card.get("symbols"):
            scored.append(
                {
                    "path": card.get("path"),
                    "symbol": file_name,
                    "kind": "file",
                    "start_line": 1,
                    "end_line": None,
                    "score": file_score,
                }
            )
            continue
        for symbol in card.get("symbols") or []:
            text = " ".join(
                [
                    str(symbol.get("name") or ""),
                    str(symbol.get("docstring") or ""),
                    " ".join(str(arg) for arg in symbol.get("args") or []),
                ]
            ).lower()
            score = sum(1 for term in normalized_focus if term in text) + file_score
            scored.append(
                {
                    "path": card.get("path"),
                    "symbol": symbol.get("name"),
                    "kind": symbol.get("kind"),
                    "start_line": symbol.get("start_line"),
                    "end_line": symbol.get("end_line"),
                    "score": score,
                }
            )
    scored.sort(key=lambda item: (-item["score"], str(item.get("path") or ""), str(item.get("symbol") or "")))
    return scored[:24]


def _merge_target_groups(
    paper_targets: list[dict[str, Any]],
    rebuttal_targets: list[dict[str, Any]],
    code_targets: list[dict[str, Any]],
    *,
    max_items: int,
) -> list[dict[str, Any]]:
    merged = []
    seen = set()
    for source, items in [("paper", paper_targets), ("rebuttal", rebuttal_targets), ("code", code_targets)]:
        for item in items:
            chunk_id = item.get("chunk_id") or f"{source}:{item.get('path')}:{item.get('symbol')}"
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            merged.append(
                {
                    "source_type": source,
                    "chunk_id": chunk_id,
                    "path": item.get("path") or item.get("source_path"),
                    "paper_id": item.get("paper_id"),
                    "section": item.get("section"),
                    "symbol": item.get("symbol"),
                    "score": item.get("score", 0),
                    "snippet": item.get("snippet", ""),
                }
            )
            if len(merged) >= max_items:
                return merged
    return merged


def _extract_code_symbols(text: str, rel_path: str) -> list[dict[str, Any]]:
    if not rel_path.endswith(".py"):
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                {
                    "name": node.name,
                    "kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "start_line": getattr(node, "lineno", None),
                    "end_line": getattr(node, "end_lineno", None),
                    "docstring": ast.get_docstring(node) or "",
                    "args": _function_args(node) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else [],
                }
            )
    symbols.sort(key=lambda item: (item.get("start_line") or 0, item.get("name") or ""))
    return symbols[:80]


def _function_args(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    args = []
    for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        args.append(arg.arg)
    if node.args.vararg:
        args.append("*" + node.args.vararg.arg)
    if node.args.kwarg:
        args.append("**" + node.args.kwarg.arg)
    return args


def _extract_imports(text: str) -> list[str]:
    imports = []
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return imports
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = "." * node.level + (node.module or "")
            imports.append(module)
    return imports[:60]


def _extract_config_knobs(text: str) -> list[str]:
    knobs = sorted(set(re.findall(r"['\"]([a-zA-Z_][a-zA-Z0-9_]*(?:alignment|confidence|soft|gate|top_k|entropy|span|token|cache)[a-zA-Z0-9_]*)['\"]", text)))
    return knobs[:120]


def _python_code_chunks(text: str, rel_path: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [
            {
                "chunk_id": f"{sanitize_filename(rel_path, max_length=60)}:0000",
                "path": rel_path,
                "language": "python",
                "start_line": 1,
                "end_line": len(text.splitlines()),
                "symbol": "",
                "section": "file",
                "source_path": rel_path,
                "keywords": _chunk_keywords(text[:2200], extra_terms=[rel_path, "python"]),
                "text": text[:2200],
            }
        ]
    lines = text.splitlines()
    chunks = []
    symbol_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(node, "lineno", None)
    ]
    symbol_nodes.sort(key=lambda node: (node.lineno, getattr(node, "name", "")))
    for idx, node in enumerate(symbol_nodes[:120]):
        start = max(1, getattr(node, "lineno", 1))
        end = max(start, getattr(node, "end_lineno", start))
        body = "\n".join(lines[start - 1 : end])
        for part_idx, chunk_text in enumerate(_chunk_text(body, max_chars=2400, overlap=160)):
            chunks.append(
                {
                    "chunk_id": f"{sanitize_filename(rel_path, max_length=60)}:{idx:04d}:{part_idx}",
                    "path": rel_path,
                    "language": "python",
                    "start_line": start,
                    "end_line": end,
                    "symbol": getattr(node, "name", ""),
                    "symbol_kind": "class" if isinstance(node, ast.ClassDef) else "function",
                    "section": getattr(node, "name", "") or "symbol",
                    "source_path": rel_path,
                    "keywords": _chunk_keywords(chunk_text, extra_terms=[rel_path, getattr(node, "name", ""), "python"]),
                    "text": chunk_text,
                }
            )
    if not chunks:
        chunks.append(
            {
                "chunk_id": f"{sanitize_filename(rel_path, max_length=60)}:0000",
                "path": rel_path,
                "language": "python",
                "start_line": 1,
                "end_line": len(lines),
                "symbol": "",
                "section": "file",
                "source_path": rel_path,
                "keywords": _chunk_keywords(text[:2400], extra_terms=[rel_path, "python"]),
                "text": text[:2400],
            }
        )
    return chunks


def _line_number_for_offset(text: str, offset: int) -> int:
    if offset < 0:
        return 1
    return text.count("\n", 0, offset) + 1


def _concern_implication(concern_id: str) -> str:
    implications = {
        "heterogeneous_model_support": "Keep model-pair metadata explicit and prefer interventions tied to tokenizer mismatch, not same-family assumptions.",
        "training_cost_pair_specific": "Track added parameters and whether a change can reuse the configured C2C training/eval protocol without pair-specific expansion.",
        "baseline_fairness": "Compare against the configured original C2C baseline and cite failed local variants instead of weak baselines.",
        "failure_modes_ood": "Require per-dataset regression checks and record failure modes before escalating an idea.",
        "multi_sharer_scaling": "Do not claim multi-sharer scaling until the small-loop result is stable on one sharer-receiver pair.",
        "dynamic_selection": "Prefer bounded dynamic selection with explicit ablation over unconstrained learned routing.",
        "latency_memory": "Record generated commands and changed files so follow-up runs can add latency and memory measurements.",
        "positioning_related_work": "Use paper cards to position the intervention against KVComm/TokenDance/C2C-style cache sharing work.",
    }
    return implications.get(concern_id, "Turn this concern into a measurable S2/S3 acceptance condition.")


def _route_family(source: str, method: str) -> str:
    label = f"{source} {method}".lower()
    for family in ["route3", "route1_alignment_v22", "route1_alignment_v21", "route1_alignment_v2", "route1_alignment"]:
        if family in label:
            return family
    if "v2." in label or "v2" in label:
        return "route1_alignment_v2x"
    return "unknown"


def _ledger_verdict(delta: float | None) -> str:
    if delta is None:
        return "missing_mean"
    if delta > 0:
        return "beats_baseline"
    if delta == 0:
        return "baseline_tie"
    return "below_baseline"


def _failure_reason(method: str, source: str) -> str:
    label = f"{method} {source}".lower()
    if "route3" in label or "learned_alignment" in label or "router" in label:
        return "Learned router variants underperformed the configured baseline, likely from anchor bias or weak direct utility supervision."
    if "delta_l2" in label:
        return "Delta L2 regularization likely suppressed useful cache-transfer corrections."
    if "layer_gate" in label or "layer_scale" in label:
        return "Coarse layer gating/scaling did not translate validation behavior into benchmark gains."
    if "adaptive_overlap" in label or "span_mlp" in label or "calibrator" in label:
        return "Span-weight calibration alone was not enough to beat the configured baseline."
    if "answer_prior" in label or "answer_margin" in label or "replay" in label:
        return "Answer-biased auxiliary objectives did not produce robust three-dataset gains."
    if "learned_affine" in label:
        return "Global learned affine confidence calibration was too coarse for cross-tokenizer transfer."
    return "Historical small-loop mean was below the configured baseline."


def _avoid_repeat_rule(method: str, source: str) -> str:
    label = f"{method} {source}".lower()
    if "route3" in label or "learned_alignment" in label or "router" in label:
        return "Avoid unconstrained learned routers unless the idea adds direct utility supervision and anti-anchor diagnostics."
    if "delta_l2" in label:
        return "Avoid plain delta regularization without a mechanism-specific diagnostic."
    if "layer_gate" in label or "layer_scale" in label:
        return "Avoid coarse layer-only scaling as the main contribution."
    if "answer_prior" in label or "answer_margin" in label or "replay" in label:
        return "Avoid answer-prior objectives as the sole source of improvement."
    return "Do not repeat below-baseline variants without a new mechanism and an explicit ablation."
