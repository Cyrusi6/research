"""Multi-agent reasoning for research idea selection."""

from __future__ import annotations

import json
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..c2c import (
    DEFAULT_ALLOWED_FILES,
    DEFAULT_BASELINE,
    c2c_candidate_config_overrides,
    c2c_idea_novelty_report,
    default_c2c_ideas,
    failure_aware_c2c_ideas,
    normalize_c2c_mechanism_fields,
)
from ..llm import ModelClient
from ..utils import compact_markdown, now_utc
from .base import AgentContext


C2C_DEBATE_ROLES = [
    "literature_scout",
    "rebuttal_analyst",
    "method_inventor",
    "skeptic_reviewer",
    "systems_feasibility",
    "experiment_designer",
    "meta_judge",
]


def _run_role_worker(
    queue: mp.Queue,
    config: dict[str, Any],
    project_root_text: str,
    role: str,
    context_payload: dict[str, Any],
    prior_round: list[dict[str, Any]],
    round_idx: int,
    fallback: dict[str, Any],
) -> None:
    try:
        project_root = Path(project_root_text)
        llm = ModelClient(config, project_root=project_root)
        service = MultiAgentReasoningService(AgentContext(project_root, config, None, llm))  # type: ignore[arg-type]
        output = service._run_role(role, context_payload, prior_round, round_idx)
        queue.put({"status": "ok", "output": output})
    except Exception as exc:  # pragma: no cover - defensive for external API workers
        queue.put({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "output": fallback})


def _run_meta_worker(
    queue: mp.Queue,
    config: dict[str, Any],
    project_root_text: str,
    context_payload: dict[str, Any],
    round_summaries: list[dict[str, Any]],
    fallback_ideas: list[dict[str, Any]],
    fallback: dict[str, Any],
) -> None:
    try:
        project_root = Path(project_root_text)
        llm = ModelClient(config, project_root=project_root)
        service = MultiAgentReasoningService(AgentContext(project_root, config, None, llm))  # type: ignore[arg-type]
        output = service._run_meta_judge_sync(context_payload, round_summaries, fallback_ideas, fallback)
        queue.put({"status": "ok", "output": output})
    except Exception as exc:  # pragma: no cover - defensive for external API workers
        queue.put({"status": "failed", "error": str(exc), "traceback": traceback.format_exc(), "output": fallback})


def _multiprocessing_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX fallback
        return mp.get_context()


@dataclass
class MultiAgentReasoningService:
    context: AgentContext

    def run_c2c_debate(
        self,
        *,
        topic: str,
        repo_card: dict[str, Any],
        paper_cards: list[dict[str, Any]],
        paper_chunks: list[dict[str, Any]] | None = None,
        rebuttal_matrix: dict[str, Any],
        rebuttal_chunks: list[dict[str, Any]] | None = None,
        code_cards: list[dict[str, Any]] | None = None,
        code_chunks: list[dict[str, Any]] | None = None,
        retrieval_plan: dict[str, Any] | None = None,
        followup_bundle: dict[str, Any] | None = None,
        negative_memory: dict[str, Any],
        baseline: dict[str, Any],
        feedback: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        rounds = int(self.context.config.get("ideation", {}).get("debate", {}).get("rounds", 2) or 2)
        feedback_constraints = _feedback_constraints(feedback or [])
        failure_feedback_index = _feedback_index(feedback or [], feedback_constraints)
        feedback_constraints["failure_feedback_refs"] = failure_feedback_index
        fallback_ideas = _fallback_ideas_for_feedback(topic, baseline, feedback_constraints)
        transcript = []
        context_payload = {
            "topic": topic,
            "baseline": baseline,
            "allowed_files": DEFAULT_ALLOWED_FILES,
            "repo_card": _compact(repo_card, 4000),
            "paper_cards": _compact(paper_cards, 3000),
            "paper_evidence_index": _compact(_evidence_index(paper_chunks or [], max_items=10), 7000),
            "rebuttal_matrix": _compact(rebuttal_matrix, 5000),
            "structured_rebuttal_concerns": _compact(rebuttal_matrix.get("structured_concerns", []), 7000),
            "rebuttal_evidence_index": _compact(_evidence_index(rebuttal_chunks or [], max_items=10), 7000),
            "code_cards": _compact(_code_card_index(code_cards or []), 8000),
            "code_evidence_index": _compact(_code_chunk_index(code_chunks or [], max_items=12), 9000),
            "retrieval_plan": _compact(retrieval_plan or {}, 11000),
            "followup_bundle": _compact(followup_bundle or {}, 9000),
            "negative_memory": _compact(negative_memory, 3000),
            "prior_failure_feedback": _compact(feedback or [], 7000),
            "failure_feedback_index": failure_feedback_index,
            "failure_feedback_summary": _feedback_summary(feedback_constraints, failure_feedback_index),
            "failure_constraints": feedback_constraints,
            "fallback_ideas": fallback_ideas,
            "fallback_ideas_brief": _compact(fallback_ideas, 4000),
        }
        prior_round = []
        round_summaries: list[dict[str, Any]] = []
        quality_flags: list[dict[str, Any]] = []
        for round_idx in range(1, rounds + 1):
            round_outputs = self._run_round_roles(context_payload, prior_round, round_idx)
            transcript.append({"round": round_idx, "outputs": round_outputs})
            round_summaries.append(_round_summary(round_idx, round_outputs))
            prior_round = round_outputs
        quality_flags = _debate_quality_flags(transcript)

        meta = self._run_meta_judge(context_payload, round_summaries, fallback_ideas)
        quality_flags.extend(_meta_quality_flags(meta))
        meta["decision_chain"] = _normalize_decision_chain(meta.get("decision_chain"), fallback=_decision_chain_from_transcript(transcript, meta))
        decision_chain = meta["decision_chain"]
        ideas = self._normalize_ideas(
            meta.get("selected_ideas"),
            fallback_ideas,
            baseline,
            feedback_constraints,
            transcript=transcript,
            meta=meta,
        )
        return {
            "roles": C2C_DEBATE_ROLES,
            "rounds": transcript,
            "meta_judge": meta,
            "decision_chain": decision_chain,
            "selected_ideas": ideas,
            "negative_constraints": self._negative_constraints(rebuttal_matrix, negative_memory, feedback or [], meta, feedback_constraints),
            "run_log": self._load_run_log(),
            "quality_flags": quality_flags,
        }

    def _run_round_roles(self, context_payload: dict[str, Any], prior_round: list[dict[str, Any]], round_idx: int) -> list[dict[str, Any]]:
        config = self.context.config.get("ideation", {}).get("debate", {})
        if not self.context.llm.use_real_api or not config.get("parallel", True):
            return [self._run_role(role, context_payload, prior_round, round_idx) for role in C2C_DEBATE_ROLES[:-1]]
        default_timeout_seconds = int(config.get("agent_timeout_seconds") or self.context.config.get("llm", {}).get("timeout_seconds") or 180)
        role_timeout_seconds = config.get("role_timeout_seconds") or {}
        mp_context = _multiprocessing_context()
        workers = []
        started_at = time.monotonic()
        for role in C2C_DEBATE_ROLES[:-1]:
            timeout_seconds = _role_timeout_seconds(role, default_timeout_seconds, role_timeout_seconds)
            fallback = _fallback_role_output(role, context_payload)
            self._write_progress(role=role, round_idx=round_idx, status="running", started_at=now_utc(), timeout_seconds=timeout_seconds)
            queue: mp.Queue = mp_context.Queue(maxsize=1)
            process = mp_context.Process(
                target=_run_role_worker,
                args=(queue, self.context.config, str(self.context.project_root), role, context_payload, prior_round, round_idx, fallback),
            )
            try:
                process.start()
            except Exception as exc:
                output = dict(fallback)
                output["status"] = "error_fallback"
                output["fallback_reason"] = f"{role} worker failed to start: {exc}"
                self._write_progress(role=role, round_idx=round_idx, status="error_fallback", finished_at=now_utc(), fallback_reason=output["fallback_reason"])
                workers.append({"role": role, "fallback": fallback, "queue": queue, "process": None, "startup_output": output})
                continue
            workers.append(
                {
                    "role": role,
                    "fallback": fallback,
                    "queue": queue,
                    "process": process,
                    "startup_output": None,
                    "timeout_seconds": timeout_seconds,
                    "deadline": started_at + timeout_seconds,
                }
            )

        for worker in workers:
            if worker.get("process") is None:
                continue
            remaining = max(0.0, worker["deadline"] - time.monotonic())
            worker["process"].join(remaining)

        outputs = []
        for worker in workers:
            role = worker["role"]
            fallback = worker["fallback"]
            process = worker["process"]
            queue = worker["queue"]
            timeout_seconds = int(worker.get("timeout_seconds") or default_timeout_seconds)
            if process is None:
                outputs.append(worker["startup_output"])
                continue
            if process.is_alive():
                process.terminate()
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join()
                output = dict(fallback)
                output["status"] = "timeout_fallback"
                output["fallback_reason"] = f"{role} timed out after {timeout_seconds}s"
                self._write_progress(role=role, round_idx=round_idx, status="timeout_fallback", finished_at=now_utc(), fallback_reason=output["fallback_reason"])
                outputs.append(output)
                continue
            try:
                payload = queue.get_nowait()
            except Exception:
                payload = {"status": "failed", "error": f"{role} exited without returning payload"}
            if isinstance(payload, dict) and payload.get("status") == "ok" and isinstance(payload.get("output"), dict):
                output = payload["output"]
                output.setdefault("role", role)
                output.setdefault("status", "ok")
                self._write_progress(role=role, round_idx=round_idx, status="ok", finished_at=now_utc())
                outputs.append(output)
                continue
            output = dict(fallback)
            output["status"] = "error_fallback"
            output["fallback_reason"] = str((payload or {}).get("error") or f"{role} failed")
            self._write_progress(role=role, round_idx=round_idx, status="error_fallback", finished_at=now_utc(), fallback_reason=output["fallback_reason"])
            outputs.append(output)
        return outputs

    def _write_progress(self, **record: Any) -> None:
        progress_path = self.context.project_root / "literature" / "c2c" / "idea_debate_progress.jsonl"
        try:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            return

    def _load_run_log(self) -> dict[str, Any]:
        progress_path = self.context.project_root / "literature" / "c2c" / "idea_debate_progress.jsonl"
        if not progress_path.exists():
            return {"progress_path": "literature/c2c/idea_debate_progress.jsonl", "events": []}
        events = []
        for line in progress_path.read_text(encoding="utf-8").splitlines()[-80:]:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return {
            "progress_path": "literature/c2c/idea_debate_progress.jsonl",
            "events": events,
            "fallback_events": [event for event in events if "fallback" in str(event.get("status", ""))],
        }

    def _run_role(self, role: str, context_payload: dict[str, Any], prior_round: list[dict[str, Any]], round_idx: int) -> dict[str, Any]:
        fallback = _fallback_role_output(role, context_payload)
        if not self.context.llm.use_real_api:
            return fallback
        prompt = {
            "role": role,
            "round": round_idx,
            "context": context_payload,
            "prior_round_outputs": prior_round,
            "required_fields": [
                "role",
                "claims",
                "evidence",
                "counterevidence",
                "conclusion",
                "reviewer_concerns",
                "risks",
                "missing_evidence",
                "proposed_ideas",
                "kill_criteria",
                "score",
                "evidence_refs",
                "counterevidence_refs",
                "code_refs",
                "failure_feedback_refs",
            ],
            "evidence_rules": [
                "Use paper_evidence_index, rebuttal_evidence_index, and code_evidence_index ids whenever possible.",
                "Every proposed idea should cite at least one paper/rebuttal evidence item and one code path or symbol.",
                "Do not use bibliography entries as method evidence; references are only for related-work expansion.",
                "Do not propose pure threshold/top-k/confidence-floor/fallback tuning. Proposed ideas must introduce a new C2C mechanism.",
                "Every proposed idea must include mechanism_type, paper_claim, why_baseline_fails, expected_signature, and ablation_plan.",
                "Do not propose another hard accept/reject gate layered on top of the baseline unless the idea includes coverage_diagnostics and matched_coverage_ablation proving the gain is not just lower transfer coverage.",
                "Every proposed idea must include coverage_diagnostics and matched_coverage_ablation.",
                "Every proposed idea must include implementation_scope. If scope is medium or large, include integration_points and smoke_tests; if large, include decomposition_plan and mvp_slice.",
                "Prefer followup_bundle.cross_source_targets when narrowing the evidence set.",
                "If failure_feedback_index is non-empty, every proposed idea must attach failure_feedback_refs and explain how it differs from failed ideas.",
                "Use failure_attribution evidence directly: name dragging datasets, sample_type_failures, mixed_gain_patterns, and patch_risk_labels when proposing a fix.",
                "Do not repeat failure_constraints.failed_idea_ids, failure_constraints.failed_titles, blocked_idea_patterns, or avoid_repeat_rules.",
            ],
        }
        schema = {"type": "object", "required": ["role", "claims", "evidence", "counterevidence", "conclusion", "risks", "proposed_ideas", "score"]}
        try:
            payload = self.context.llm.generate_json_with_schema(
                instructions=(
                    "You are one specialist in a multi-agent ML research debate. "
                    "Return a research decision chain with evidence, counterevidence, and a conclusion. "
                    "Ground each claim in retrieval_plan targets, evidence ids, snippets, code symbols, and rebuttal concerns. "
                    "Treat failure_feedback_index as prior experimental evidence, not as optional context. "
                    "Do not invent file paths."
                ),
                prompt=json.dumps(prompt, ensure_ascii=False),
                default=fallback,
                schema=schema,
                agent_name=role,
            )
        except Exception:
            return fallback
        if not isinstance(payload, dict):
            return fallback
        payload.setdefault("role", role)
        return payload

    def _run_meta_judge(self, context_payload: dict[str, Any], round_summaries: list[dict[str, Any]], fallback_ideas: list[dict[str, Any]]) -> dict[str, Any]:
        fallback = {
            "role": "meta_judge",
            "decision_rationale": "Fallback selection from deterministic C2C ideas.",
            "decision_chain": {
                "evidence": ["Deterministic C2C evidence inventory and historical results were available."],
                "counterevidence": ["Existing failure feedback indicates some low-confidence ideas should not be repeated."],
                "conclusion": "Choose the strongest bounded candidate with explicit recovery from prior regressions.",
            },
            "selected_ideas": fallback_ideas,
            "constraints": [],
        }
        config = self.context.config.get("ideation", {}).get("debate", {})
        if self.context.llm.use_real_api and config.get("parallel", True):
            timeout_seconds = int(config.get("meta_timeout_seconds") or config.get("agent_timeout_seconds") or self.context.config.get("llm", {}).get("timeout_seconds") or 180)
            return self._run_meta_judge_process(context_payload, round_summaries, fallback_ideas, fallback, timeout_seconds=timeout_seconds)
        return self._run_meta_judge_sync(context_payload, round_summaries, fallback_ideas, fallback)

    def _run_meta_judge_process(
        self,
        context_payload: dict[str, Any],
        round_summaries: list[dict[str, Any]],
        fallback_ideas: list[dict[str, Any]],
        fallback: dict[str, Any],
        *,
        timeout_seconds: int,
    ) -> dict[str, Any]:
        self._write_progress(role="meta_judge", round_idx="final", status="running", started_at=now_utc())
        mp_context = _multiprocessing_context()
        queue: mp.Queue = mp_context.Queue(maxsize=1)
        process = mp_context.Process(
            target=_run_meta_worker,
            args=(queue, self.context.config, str(self.context.project_root), context_payload, round_summaries, fallback_ideas, fallback),
        )
        try:
            process.start()
        except Exception as exc:
            output = dict(fallback)
            output["status"] = "error_fallback"
            output["fallback_reason"] = f"meta_judge worker failed to start: {exc}"
            self._write_progress(role="meta_judge", round_idx="final", status="error_fallback", finished_at=now_utc(), fallback_reason=output["fallback_reason"])
            return output
        process.join(timeout_seconds)
        if process.is_alive():
            process.terminate()
            process.join(5)
            if process.is_alive():
                process.kill()
                process.join()
            output = dict(fallback)
            output["status"] = "timeout_fallback"
            output["fallback_reason"] = f"meta_judge timed out after {timeout_seconds}s"
            self._write_progress(role="meta_judge", round_idx="final", status="timeout_fallback", finished_at=now_utc(), fallback_reason=output["fallback_reason"])
            return output
        try:
            payload = queue.get_nowait()
        except Exception:
            payload = {"status": "failed", "error": "meta_judge exited without returning payload"}
        if isinstance(payload, dict) and payload.get("status") == "ok" and isinstance(payload.get("output"), dict):
            output = payload["output"]
            output.setdefault("role", "meta_judge")
            output.setdefault("status", "ok")
            self._write_progress(role="meta_judge", round_idx="final", status="ok", finished_at=now_utc())
            return output
        output = dict(fallback)
        output["status"] = "error_fallback"
        output["fallback_reason"] = str((payload or {}).get("error") or "meta_judge failed")
        self._write_progress(role="meta_judge", round_idx="final", status="error_fallback", finished_at=now_utc(), fallback_reason=output["fallback_reason"])
        return output

    def _run_meta_judge_sync(
        self,
        context_payload: dict[str, Any],
        round_summaries: list[dict[str, Any]],
        fallback_ideas: list[dict[str, Any]],
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.context.llm.use_real_api:
            return fallback
        schema = {"type": "object", "required": ["selected_ideas", "decision_rationale", "decision_chain"]}
        try:
            payload = self.context.llm.generate_json_with_schema(
                instructions=(
                    "You are the meta judge. Select 3 to 5 executable C2C ideas. "
                    "Every idea must include experiment_contract with frozen config_overrides, expected_files, verification_commands, and selected. "
                    "Reject pure local tuning ideas; selected ideas must introduce a mechanism-level change with mechanism_type, paper_claim, why_baseline_fails, expected_signature, and ablation_plan. "
                    "Reject hard-gate stacking unless coverage_diagnostics and matched_coverage_ablation are explicit and testable. "
                    "Every selected idea must include coverage_diagnostics and matched_coverage_ablation. "
                    "If a selected idea needs new files or broad changes, require implementation_scope plus integration_points, smoke_tests, decomposition_plan, and mvp_slice. "
                    "Summarize the final decision as evidence, counterevidence, and conclusion. "
                    "Every selected idea should cite retrieval_plan targets, code paths/symbols, and paper or rebuttal evidence ids when available. "
                    "When failure_feedback_index is non-empty, each selected idea must include failure_feedback_refs and explain how it avoids those failures. "
                    "Use failure_attribution evidence explicitly, including dragging datasets, sample type failures, mixed gain patterns, and patch risk labels. "
                    "Do not select ideas listed in failure_constraints.failed_idea_ids or failure_constraints.failed_titles. "
                    "Do not violate failure_constraints.avoid_repeat_rules or blocked_idea_patterns."
                ),
                prompt=json.dumps({"context": context_payload, "round_summaries": round_summaries}, ensure_ascii=False),
                default=fallback,
                schema=schema,
                agent_name="meta_judge",
            )
        except Exception:
            return fallback
        return payload if isinstance(payload, dict) else fallback

    @staticmethod
    def _normalize_ideas(
        raw: Any,
        fallback: list[dict[str, Any]],
        baseline: dict[str, Any],
        feedback_constraints: dict[str, Any],
        *,
        transcript: list[dict[str, Any]],
        meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if not isinstance(raw, list):
            raw = []
        normalized = []
        profiles = {
            item["id"]: item
            for item in [
                *default_c2c_ideas("C2C", baseline),
                *failure_aware_c2c_ideas("C2C", baseline),
                *fallback,
            ]
        }
        for idx, idea in enumerate(raw[:8]):
            if not isinstance(idea, dict):
                continue
            item = dict(idea)
            item.setdefault("id", f"debated_c2c_idea_{idx + 1}")
            item.setdefault("title", str(item["id"]).replace("_", " ").title())
            if _is_forbidden_idea(item, feedback_constraints):
                continue
            profile = profiles.get(item["id"])
            profile = normalize_c2c_mechanism_fields(dict(profile), baseline) if profile else {}
            item.setdefault("description", item.get("title", "C2C idea"))
            item.setdefault("motivation", "Selected by multi-agent evidence debate.")
            item.setdefault("hypothesis", "This idea improves the C2C three-dataset mean.")
            item.setdefault("novelty_score", 7)
            item.setdefault("feasibility_score", 7)
            item.setdefault("expected_contribution", "Improved cross-tokenizer KV-cache communication.")
            item.setdefault("novelty_against", [baseline.get("name", DEFAULT_BASELINE["name"])])
            item.setdefault("reviewer_risk_response", "Addresses reviewer concerns from rebuttal evidence.")
            item.setdefault("blocked_by_negative_results", False)
            item.setdefault("expected_files", DEFAULT_ALLOWED_FILES)
            item.setdefault("risk", f"May not beat the configured baseline {baseline.get('name', DEFAULT_BASELINE['name'])}.")
            item.setdefault("verification_commands", ["py_compile", "test_aligner_span_overlap", "small2048_train", "three_dataset_eval"])
            item.setdefault("experiment_contract", profile.get("experiment_contract") or {"primary_metric": "three_dataset_mean", "baseline": baseline.get("name", DEFAULT_BASELINE["name"])})
            for mechanism_key in [
                "mechanism_type",
                "mechanism_summary",
                "paper_claim",
                "why_baseline_fails",
                "expected_signature",
                "ablation_plan",
                "coverage_diagnostics",
                "matched_coverage_ablation",
                "implementation_scope",
                "implementation_plan",
                "required_new_files",
                "integration_points",
                "smoke_tests",
                "decomposition_plan",
            ]:
                if profile.get(mechanism_key) not in (None, "", [], {}):
                    item.setdefault(mechanism_key, profile.get(mechanism_key))
            item["evidence_refs"] = _merge_ref_lists(item.get("evidence_refs"), _extract_evidence_refs(transcript, item))
            item["counterevidence_refs"] = _merge_ref_lists(item.get("counterevidence_refs"), _extract_counterevidence_refs(transcript, item))
            item["code_refs"] = _merge_ref_lists(item.get("code_refs"), _extract_code_refs(transcript, item))
            item["chunk_refs"] = _merge_ref_lists(item.get("chunk_refs"), _extract_chunk_refs(transcript, item))
            item["failure_feedback_refs"] = _merge_ref_lists(
                item.get("failure_feedback_refs"),
                _extract_failure_feedback_refs(transcript, item),
                _fallback_failure_feedback_refs(feedback_constraints, item),
            )
            item["decision_chain"] = _decision_chain_from_idea(item, transcript=transcript, meta=meta)
            item["evidence_refs"] = _merge_ref_lists(item.get("evidence_refs"), item.get("decision_chain", {}).get("evidence_items"), _fallback_evidence_refs())
            item["counterevidence_refs"] = _merge_ref_lists(
                item.get("counterevidence_refs"),
                item.get("decision_chain", {}).get("counterevidence_items"),
                _fallback_counterevidence_refs(feedback_constraints),
            )
            item["code_refs"] = _merge_ref_lists(item.get("code_refs"), _extract_code_refs_from_refs(item.get("evidence_refs")), _fallback_code_refs())
            item["chunk_refs"] = _merge_ref_lists(item.get("chunk_refs"), item.get("evidence_refs"), item.get("counterevidence_refs"), _fallback_chunk_refs())
            item.setdefault("evidence", item.get("decision_chain", {}).get("evidence", []))
            item.setdefault("counterevidence", item.get("decision_chain", {}).get("counterevidence", []))
            item.setdefault("conclusion", item.get("decision_chain", {}).get("conclusion", item.get("hypothesis", "")))
            item.setdefault("decision_rationale", item.get("decision_chain", {}).get("conclusion", item.get("hypothesis", "")))
            overrides = c2c_candidate_config_overrides(item)
            if not overrides["train"] and not overrides["eval"]:
                item["experiment_contract"] = profile.get("experiment_contract") or item["experiment_contract"]
            item = normalize_c2c_mechanism_fields(item, baseline)
            item["novelty_gate"] = c2c_idea_novelty_report(item)
            if item["novelty_gate"]["status"] != "pass":
                continue
            normalized.append(item)
        seen = {item["id"] for item in normalized}
        for idea in fallback:
            if len(normalized) >= 5:
                break
            if idea["id"] in seen or _is_forbidden_idea(idea, feedback_constraints):
                continue
            item = dict(idea)
            item["failure_feedback_refs"] = _merge_ref_lists(
                item.get("failure_feedback_refs"),
                _fallback_failure_feedback_refs(feedback_constraints, item),
            )
            item = normalize_c2c_mechanism_fields(item, baseline)
            item["novelty_gate"] = c2c_idea_novelty_report(item)
            if item["novelty_gate"]["status"] != "pass":
                continue
            normalized.append(item)
            seen.add(idea["id"])
        if len(normalized) < 3:
            for item in _recovery_fallback_ideas(baseline, feedback_constraints):
                if len(normalized) >= 5:
                    break
                if item["id"] in seen or _is_forbidden_idea(item, feedback_constraints):
                    continue
                candidate = dict(item)
                candidate["failure_feedback_refs"] = _merge_ref_lists(
                    candidate.get("failure_feedback_refs"),
                    _fallback_failure_feedback_refs(feedback_constraints, candidate),
                )
                candidate["evidence_refs"] = _merge_ref_lists(candidate.get("evidence_refs"), _fallback_evidence_refs())
                candidate["counterevidence_refs"] = _merge_ref_lists(candidate.get("counterevidence_refs"), _fallback_counterevidence_refs(feedback_constraints))
                candidate["code_refs"] = _merge_ref_lists(candidate.get("code_refs"), _fallback_code_refs())
                candidate["chunk_refs"] = _merge_ref_lists(candidate.get("chunk_refs"), _fallback_chunk_refs())
                candidate.setdefault("decision_chain", _fallback_candidate_decision_chain(candidate, meta, feedback_constraints))
                candidate = normalize_c2c_mechanism_fields(candidate, baseline)
                candidate["novelty_gate"] = c2c_idea_novelty_report(candidate)
                if candidate["novelty_gate"]["status"] != "pass":
                    continue
                normalized.append(candidate)
                seen.add(candidate["id"])
        for idx, item in enumerate(normalized):
            item["selected"] = idx == 0
        return normalized[:5]

    @staticmethod
    def _negative_constraints(
        rebuttal_matrix: dict[str, Any],
        negative_memory: dict[str, Any],
        feedback: list[dict[str, Any]],
        meta: dict[str, Any],
        feedback_constraints: dict[str, Any],
    ) -> dict[str, Any]:
        avoid_rules = feedback_constraints.get("avoid_repeat_rules", [])
        blocked_patterns = sorted(set((negative_memory.get("blocked_idea_patterns", []) or []) + (feedback_constraints.get("blocked_idea_patterns", []) or []) + avoid_rules))
        return {
            "reviewer_concerns": rebuttal_matrix.get("top_concerns", []),
            "blocked_idea_patterns": blocked_patterns,
            "failure_feedback_rules": avoid_rules,
            "forbidden_idea_ids": feedback_constraints.get("failed_idea_ids", []),
            "forbidden_titles": feedback_constraints.get("failed_titles", []),
        "failure_modes": feedback_constraints.get("failure_modes", []),
        "dataset_regressions": feedback_constraints.get("dataset_regressions", {}),
        "dragging_datasets": feedback_constraints.get("dragging_datasets", []),
        "sample_type_failures": feedback_constraints.get("sample_type_failures", []),
        "patch_risk_labels": feedback_constraints.get("patch_risk_labels", []),
        "patch_risk_files": feedback_constraints.get("patch_risk_files", []),
        "mixed_gain_patterns": feedback_constraints.get("mixed_gain_patterns", []),
        "next_round_suggestions": feedback_constraints.get("next_round_suggestions", []),
        "latest_reason": feedback_constraints.get("latest_reason"),
        "failure_feedback_refs": feedback_constraints.get("failure_feedback_refs", [])[:8],
            "meta_constraints": meta.get("constraints", []),
        }


def c2c_debate_markdown(payload: dict[str, Any]) -> str:
    lines = ["# C2C Idea Debate", ""]
    meta = payload.get("meta_judge") or {}
    quality_flags = payload.get("quality_flags") or []
    if quality_flags:
        lines.append("## Quality Flags")
        for flag in quality_flags[:8]:
            if not isinstance(flag, dict):
                continue
            lines.append(f"- {flag.get('type', 'flag')}: {flag.get('message', '')}")
        lines.append("")
    lines.append("## Decision Chain")
    decision_chain = _normalize_decision_chain(meta.get("decision_chain"), fallback=_normalize_decision_chain(payload.get("decision_chain")))
    lines.append(f"- Evidence: {', '.join(decision_chain.get('evidence', [])[:3]) or 'n/a'}")
    lines.append(f"- Counterevidence: {', '.join(decision_chain.get('counterevidence', [])[:3]) or 'n/a'}")
    lines.append(f"- Conclusion: {decision_chain.get('conclusion', meta.get('decision_rationale', 'n/a'))}")
    lines.append("")
    lines.append("## Selected Ideas")
    for idea in payload.get("selected_ideas", []):
        selected = " selected" if idea.get("selected") else ""
        chain = idea.get("decision_chain") or {}
        lines.append(f"- `{idea.get('id')}`{selected}: {idea.get('title')} (novelty={idea.get('novelty_score')}, feasibility={idea.get('feasibility_score')})")
        if idea.get("mechanism_type"):
            gate = idea.get("novelty_gate") or {}
            lines.append(f"  - Mechanism: {idea.get('mechanism_type')} ({gate.get('status', 'unchecked')})")
        if idea.get("implementation_scope"):
            scope_gate = idea.get("implementation_scope_gate") or {}
            lines.append(f"  - Implementation scope: {idea.get('implementation_scope')} ({scope_gate.get('status', 'unchecked')})")
        if chain:
            lines.append(f"  - Evidence: {', '.join(chain.get('evidence', [])[:2]) or 'n/a'}")
            lines.append(f"  - Counterevidence: {', '.join(chain.get('counterevidence', [])[:2]) or 'n/a'}")
            lines.append(f"  - Conclusion: {chain.get('conclusion', idea.get('decision_rationale', 'n/a'))}")
        failure_refs = idea.get("failure_feedback_refs") or []
        if failure_refs:
            labels = [str(ref.get("source_label") or ref.get("chunk_id") or ref.get("source_path")) for ref in failure_refs[:2] if isinstance(ref, dict)]
            lines.append(f"  - Failure feedback: {', '.join(labels) or 'n/a'}")
    lines.append("")
    lines.append("## Reviewer Constraints")
    constraints = payload.get("negative_constraints") or {}
    for concern in constraints.get("reviewer_concerns", [])[:8]:
        lines.append(f"- {concern}")
    return compact_markdown("\n".join(lines))


def _role_timeout_seconds(role: str, default_timeout_seconds: int, role_timeout_seconds: Any) -> int:
    if isinstance(role_timeout_seconds, dict):
        value = role_timeout_seconds.get(role)
        if value is not None:
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                return default_timeout_seconds
    return default_timeout_seconds


def _debate_quality_flags(transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_role: dict[str, list[dict[str, Any]]] = {}
    for round_item in transcript:
        round_idx = round_item.get("round")
        for output in round_item.get("outputs") or []:
            if not isinstance(output, dict):
                continue
            role = str(output.get("role") or "")
            if not role:
                continue
            per_role.setdefault(role, []).append(
                {
                    "round": round_idx,
                    "status": output.get("status"),
                    "fallback_reason": output.get("fallback_reason"),
                }
            )
    flags = []
    for role, events in sorted(per_role.items()):
        fallback_events = [event for event in events if "fallback" in str(event.get("status", ""))]
        ok_events = [event for event in events if event.get("status") == "ok"]
        if fallback_events and ok_events:
            flags.append(
                {
                    "type": "gpt_recovered_after_timeout" if any("timeout" in str(event.get("status", "")) for event in fallback_events) else "gpt_recovered_after_fallback",
                    "role": role,
                    "fallback_rounds": [event.get("round") for event in fallback_events],
                    "ok_rounds": [event.get("round") for event in ok_events],
                    "message": f"{role} used fallback in round(s) {[event.get('round') for event in fallback_events]} but returned GPT output in round(s) {[event.get('round') for event in ok_events]}.",
                    "reasons": [event.get("fallback_reason") for event in fallback_events if event.get("fallback_reason")],
                }
            )
        elif fallback_events:
            flags.append(
                {
                    "type": "fallback_only_role",
                    "role": role,
                    "fallback_rounds": [event.get("round") for event in fallback_events],
                    "message": f"{role} only produced fallback output in round(s) {[event.get('round') for event in fallback_events]}.",
                    "reasons": [event.get("fallback_reason") for event in fallback_events if event.get("fallback_reason")],
                }
            )
    return flags


def _meta_quality_flags(meta: dict[str, Any]) -> list[dict[str, Any]]:
    status = str(meta.get("status") or "")
    if "fallback" not in status:
        return []
    reason = meta.get("fallback_reason")
    flag_type = "meta_timeout_fallback" if "timeout" in status else "meta_error_fallback"
    return [
        {
            "type": flag_type,
            "role": "meta_judge",
            "message": f"meta_judge did not complete with GPT output; final selection used fallback. Reason: {reason or status}.",
            "reasons": [reason] if reason else [],
        }
    ]


def _fallback_role_output(role: str, context_payload: dict[str, Any]) -> dict[str, Any]:
    ideas = context_payload.get("fallback_ideas") or context_payload.get("fallback_ideas_brief") or default_c2c_ideas(context_payload.get("topic", "C2C"), context_payload.get("baseline"))
    failure_feedback_refs = (context_payload.get("failure_feedback_index") or [])[:4]
    counterevidence = _fallback_counterevidence_refs(context_payload.get("failure_constraints") or {})
    if failure_feedback_refs:
        counterevidence.extend(failure_feedback_refs[:2])
    return {
        "role": role,
        "claims": [f"{role} used deterministic local C2C evidence."],
        "evidence": _fallback_evidence_refs(),
        "counterevidence": counterevidence,
        "conclusion": f"{role} recommends ideas that preserve the baseline while adding one bounded intervention.",
        "reviewer_concerns": (context_payload.get("rebuttal_matrix") or {}).get("top_concerns", []),
        "risks": [f"The candidate may not beat the configured baseline {(context_payload.get('baseline') or {}).get('name', DEFAULT_BASELINE['name'])}."],
        "missing_evidence": ["Ablation is required after a small-loop win."],
        "proposed_ideas": ideas,
        "kill_criteria": ["mean <= baseline mean", "dataset regression > threshold"],
        "score": 6,
        "evidence_refs": _fallback_evidence_refs(),
        "counterevidence_refs": _fallback_counterevidence_refs(context_payload.get("failure_constraints") or {}),
        "failure_feedback_refs": failure_feedback_refs,
        "code_refs": _fallback_code_refs(),
        "chunk_refs": _fallback_chunk_refs(),
    }


def _decision_chain_from_idea(idea: dict[str, Any], *, transcript: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    chain = idea.get("decision_chain")
    if isinstance(chain, dict) and chain.get("evidence") and chain.get("counterevidence") and chain.get("conclusion"):
        return chain
    evidence = []
    counterevidence = []
    evidence_items = []
    counterevidence_items = []
    for round_payload in transcript:
        round_idx = round_payload.get("round")
        for output in round_payload.get("outputs", []):
            if not isinstance(output, dict):
                continue
            if _idea_matches_proposed(output.get("proposed_ideas", []), idea):
                evidence.extend(_chain_items_from_role_output(output.get("evidence"), "evidence"))
                counterevidence.extend(_chain_items_from_role_output(output.get("counterevidence"), "counterevidence"))
                evidence_items.extend(_structured_chain_items_from_role_output(output.get("evidence"), "evidence", role=output.get("role"), round_idx=round_idx))
                counterevidence_items.extend(_structured_chain_items_from_role_output(output.get("counterevidence"), "counterevidence", role=output.get("role"), round_idx=round_idx))
    if not evidence:
        evidence = [f"Selected by meta judge: {meta.get('decision_rationale', 'n/a')}"]
    if not evidence_items:
        evidence_items = [
            {
                "source_type": "summary",
                "source_label": "meta_judge selection",
                "snippet": meta.get("decision_rationale", "n/a"),
                "why_relevant": "No explicit evidence item was attached by the agent.",
            }
        ]
    if not counterevidence:
        counterevidence = ["No explicit counterevidence was attached by the agent; rely on reviewer constraints and regression checks."]
    if not counterevidence_items:
        counterevidence_items = [
            {
                "source_type": "summary",
                "source_label": "reviewer constraints",
                "snippet": "No explicit counterevidence item was attached by the agent.",
                "why_relevant": "Rely on reviewer constraints and regression checks.",
            }
        ]
    conclusion = idea.get("hypothesis") or idea.get("decision_rationale") or meta.get("decision_rationale", "n/a")
    return {
        "evidence": evidence[:4],
        "counterevidence": counterevidence[:4],
        "evidence_items": evidence_items[:4],
        "counterevidence_items": counterevidence_items[:4],
        "conclusion": conclusion,
    }


def _decision_chain_from_transcript(transcript: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    evidence: list[str] = []
    counterevidence: list[str] = []
    evidence_items: list[dict[str, Any]] = []
    counterevidence_items: list[dict[str, Any]] = []
    for round_payload in transcript:
        round_idx = round_payload.get("round")
        for output in round_payload.get("outputs", []):
            if not isinstance(output, dict):
                continue
            evidence.extend(_chain_items_from_role_output(output.get("evidence"), "evidence"))
            counterevidence.extend(_chain_items_from_role_output(output.get("counterevidence"), "counterevidence"))
            evidence_items.extend(_structured_chain_items_from_role_output(output.get("evidence"), "evidence", role=output.get("role"), round_idx=round_idx))
            counterevidence_items.extend(_structured_chain_items_from_role_output(output.get("counterevidence"), "counterevidence", role=output.get("role"), round_idx=round_idx))
    if not evidence:
        evidence.append("Deterministic C2C repo card, paper cards, rebuttal matrix, and negative memory were available.")
    if not evidence_items:
        evidence_items.append(
            {
                "source_type": "summary",
                "source_label": "deterministic local evidence",
                "snippet": "Deterministic C2C repo card, paper cards, rebuttal matrix, and negative memory were available.",
                "why_relevant": "The candidate search space is bounded by local evidence and reviewer concerns.",
            }
        )
    if not counterevidence:
        counterevidence.append("Reviewer concerns and failure memory constrain the candidate search space.")
    if not counterevidence_items:
        counterevidence_items.append(
            {
                "source_type": "summary",
                "source_label": "failure memory",
                "snippet": "Reviewer concerns and failure memory constrain the candidate search space.",
                "why_relevant": "Past failures should prevent repeating the same low-confidence ideas.",
            }
        )
    conclusion = meta.get("decision_rationale") or "Choose the strongest bounded candidate with explicit regression controls."
    return {
        "evidence": evidence[:5],
        "counterevidence": counterevidence[:5],
        "evidence_items": evidence_items[:5],
        "counterevidence_items": counterevidence_items[:5],
        "conclusion": conclusion,
    }


def _normalize_decision_chain(value: Any, *, fallback: dict[str, Any] | None = None) -> dict[str, Any]:
    fallback = fallback or {
        "evidence": [],
        "counterevidence": [],
        "evidence_items": [],
        "counterevidence_items": [],
        "conclusion": "",
    }
    if isinstance(value, dict):
        return {
            "evidence": _chain_list(value.get("evidence") or value.get("supporting_evidence") or []),
            "counterevidence": _chain_list(value.get("counterevidence") or value.get("risks") or value.get("limitations") or []),
            "evidence_items": value.get("evidence_items") if isinstance(value.get("evidence_items"), list) else [],
            "counterevidence_items": value.get("counterevidence_items") if isinstance(value.get("counterevidence_items"), list) else [],
            "conclusion": str(value.get("conclusion") or value.get("decision") or value.get("rationale") or fallback.get("conclusion") or ""),
        }
    if isinstance(value, list):
        evidence = []
        counterevidence = []
        conclusion = ""
        for item in value:
            if isinstance(item, dict):
                label = str(item.get("type") or item.get("kind") or item.get("role") or "").lower()
                text = item.get("text") or item.get("claim") or item.get("summary") or item.get("rationale") or item.get("conclusion")
                if not text:
                    text = json.dumps(item, ensure_ascii=False)
                if "counter" in label or "risk" in label or "limit" in label:
                    counterevidence.append(str(text))
                elif "conclusion" in label or "decision" in label:
                    conclusion = str(text)
                else:
                    evidence.append(str(text))
            elif item:
                evidence.append(str(item))
        return {
            "evidence": evidence[:5] or _chain_list(fallback.get("evidence") or []),
            "counterevidence": counterevidence[:5] or _chain_list(fallback.get("counterevidence") or []),
            "evidence_items": fallback.get("evidence_items") or [],
            "counterevidence_items": fallback.get("counterevidence_items") or [],
            "conclusion": conclusion or str(fallback.get("conclusion") or ""),
        }
    return fallback


def _chain_list(value: Any) -> list[str]:
    if isinstance(value, list):
        result = []
        for item in value:
            if isinstance(item, dict):
                result.append(str(item.get("text") or item.get("claim") or item.get("summary") or item.get("snippet") or json.dumps(item, ensure_ascii=False)))
            elif item:
                result.append(str(item))
        return result
    if value:
        return [str(value)]
    return []


def _round_summary(round_idx: int, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    role_summaries = []
    for output in outputs:
        if not isinstance(output, dict):
            continue
        decision_chain = output.get("decision_chain") or {}
        evidence_refs = output.get("evidence_refs") or []
        counterevidence_refs = output.get("counterevidence_refs") or []
        code_refs = output.get("code_refs") or []
        failure_feedback_refs = output.get("failure_feedback_refs") or []
        proposed_ideas = []
        for idea in output.get("proposed_ideas") or []:
            if isinstance(idea, dict):
                proposed_ideas.append(
                    {
                        "id": idea.get("id"),
                        "title": idea.get("title"),
                        "selected": bool(idea.get("selected")),
                        "novelty_score": idea.get("novelty_score"),
                        "feasibility_score": idea.get("feasibility_score"),
                    }
                )
        role_summaries.append(
            {
                "role": output.get("role"),
                "status": output.get("status"),
                "score": output.get("score"),
                "claims": (output.get("claims") or [])[:3],
                "top_evidence": evidence_refs[:4],
                "top_counterevidence": counterevidence_refs[:4],
                "top_code_refs": code_refs[:4],
                "failure_feedback_refs": failure_feedback_refs[:4],
                "proposed_ideas": proposed_ideas[:4],
                "one_line_conclusion": decision_chain.get("conclusion") or output.get("conclusion") or output.get("decision_rationale"),
                "risks": (output.get("risks") or [])[:3],
            }
        )
    return {
        "round": round_idx,
        "role_count": len(role_summaries),
        "role_summaries": role_summaries,
        "selected_roles": [item["role"] for item in role_summaries if item.get("status") == "ok"][:6],
        "timeout_roles": [item["role"] for item in role_summaries if "timeout" in str(item.get("status", ""))],
        "error_roles": [item["role"] for item in role_summaries if item.get("status") not in {"ok", "timeout_fallback"} and "timeout" not in str(item.get("status", ""))],
        "selected_idea_ids": [
            idea.get("id")
            for item in role_summaries
            for idea in item.get("proposed_ideas", [])
            if isinstance(idea, dict) and idea.get("selected") and idea.get("id")
        ][:8],
    }


def _fallback_evidence_refs() -> list[dict[str, Any]]:
    return [
        {
            "source_type": "repo_artifact",
            "source_path": "literature/c2c/repo_card.json",
            "chunk_id": "literature/c2c/repo_card.json::baseline_surface",
            "source_label": "repo_card baseline surface",
            "snippet": "Deterministic repo card and baseline evidence were available.",
            "why_relevant": "The intervention must stay within the allowed C2C edit surface.",
        },
        {
            "source_type": "repo_artifact",
            "source_path": "literature/c2c/rebuttal_concern_matrix.json",
            "chunk_id": "literature/c2c/rebuttal_concern_matrix.json::top_concerns",
            "source_label": "rebuttal concern matrix",
            "snippet": "Reviewer concern matrix highlights tokenizer mismatch and regression risks.",
            "why_relevant": "Candidate ideas must address per-dataset regressions, not only mean gain.",
        },
    ]


def _fallback_counterevidence_refs(feedback_constraints: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    feedback_constraints = feedback_constraints or {}
    refs = [
        {
            "source_type": "repo_artifact",
            "source_path": "literature/c2c/negative_result_memory.json",
            "chunk_id": "literature/c2c/negative_result_memory.json::failed_variants",
            "source_label": "negative result memory",
            "snippet": "Prior below-baseline variants already failed under similar boundaries.",
            "why_relevant": "Do not repeat the same low-confidence gate without a new mechanism.",
            "implication": "Do not repeat the same low-confidence gate without a new mechanism.",
        }
    ]
    if feedback_constraints.get("avoid_repeat_rules"):
        refs.append(
            {
                "source_type": "failure_feedback",
                "source_path": "meta/negative_memory.jsonl",
                "chunk_id": "meta/negative_memory.jsonl::avoid_repeat_rules",
                "source_label": "failure feedback avoid-repeat rules",
                "snippet": " | ".join(feedback_constraints.get("avoid_repeat_rules", [])[:3]),
                "why_relevant": "Historical failures constrain regeneration and candidate selection.",
            }
        )
    return refs


def _fallback_code_refs() -> list[dict[str, Any]]:
    return [
        {
            "source_type": "code",
            "source_path": "rosetta/model/aligner.py",
            "chunk_id": "rosetta/model/aligner.py::aligner",
            "symbol": "aligner",
            "symbol_kind": "file",
            "start_line": 1,
            "end_line": None,
            "source_label": "rosetta/model/aligner.py",
        },
        {
            "source_type": "code",
            "source_path": "rosetta/model/projector.py",
            "chunk_id": "rosetta/model/projector.py::projector",
            "symbol": "projector",
            "symbol_kind": "file",
            "start_line": 1,
            "end_line": None,
            "source_label": "rosetta/model/projector.py",
        },
        {
            "source_type": "code",
            "source_path": "rosetta/model/wrapper.py",
            "chunk_id": "rosetta/model/wrapper.py::wrapper",
            "symbol": "wrapper",
            "symbol_kind": "file",
            "start_line": 1,
            "end_line": None,
            "source_label": "rosetta/model/wrapper.py",
        },
    ]


def _fallback_chunk_refs() -> list[dict[str, Any]]:
    return [
        {
            "source_type": "repo_artifact",
            "source_path": "literature/c2c/retrieval_plan.json",
            "chunk_id": "literature/c2c/retrieval_plan.json::fallback evidence",
            "source_label": "retrieval plan fallback evidence",
            "snippet": "fallback evidence",
        }
    ]


def _fallback_candidate_decision_chain(idea: dict[str, Any], meta: dict[str, Any], feedback_constraints: dict[str, Any]) -> dict[str, Any]:
    evidence = _fallback_evidence_refs()
    counterevidence = _fallback_counterevidence_refs(feedback_constraints)
    return {
        "evidence": [_chain_items_from_role_output(evidence, "evidence")[0], _chain_items_from_role_output(evidence, "evidence")[1]],
        "counterevidence": _chain_items_from_role_output(counterevidence, "counterevidence")[:3],
        "evidence_items": evidence,
        "counterevidence_items": counterevidence,
        "conclusion": idea.get("hypothesis") or meta.get("decision_rationale") or "Choose a bounded recovery candidate with explicit regression controls.",
    }


def _idea_matches_proposed(proposed_ideas: Any, idea: dict[str, Any]) -> bool:
    if not isinstance(proposed_ideas, list):
        return False
    target_id = str(idea.get("id") or "")
    target_title = str(idea.get("title") or "").strip().lower()
    for item in proposed_ideas:
        if isinstance(item, dict):
            candidate_id = str(item.get("idea_id") or item.get("id") or "")
            candidate_title = str(item.get("title") or "").strip().lower()
            if candidate_id == target_id or (target_title and candidate_title == target_title):
                return True
        elif isinstance(item, str):
            text = item.strip().lower()
            if target_id and target_id in text:
                return True
            if target_title and target_title in text:
                return True
    return False


def _chain_items_from_role_output(items: Any, kind: str) -> list[str]:
    results: list[str] = []
    if not isinstance(items, list):
        return results
    for item in items:
        if isinstance(item, dict):
            source = item.get("source_label") or item.get("source_path") or item.get("source") or item.get("paper_id") or item.get("path") or "unknown"
            snippet = item.get("snippet") or item.get("claim") or item.get("text") or ""
            implication = item.get("implication") or item.get("why_relevant") or item.get("why") or ""
            text = " | ".join(part for part in [str(source), str(snippet), str(implication)] if part)
            if text:
                results.append(text)
        elif isinstance(item, str):
            results.append(item)
    return results


def _extract_evidence_refs(transcript: list[dict[str, Any]], idea: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for round_idx, output in _matching_idea_outputs(transcript, idea):
        refs.extend(_collect_refs_from_output(output, ["evidence_refs", "evidence"], role=output.get("role"), round_idx=round_idx))
    return _dedupe_ref_items(refs)[:8]


def _extract_counterevidence_refs(transcript: list[dict[str, Any]], idea: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for round_idx, output in _matching_idea_outputs(transcript, idea):
        refs.extend(_collect_refs_from_output(output, ["counterevidence_refs", "counterevidence"], role=output.get("role"), round_idx=round_idx))
    return _dedupe_ref_items(refs)[:8]


def _extract_code_refs(transcript: list[dict[str, Any]], idea: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for round_idx, output in _matching_idea_outputs(transcript, idea):
        refs.extend(
            ref
            for ref in _collect_refs_from_output(output, ["code_refs", "evidence"], role=output.get("role"), round_idx=round_idx)
            if ref.get("source_type") == "code"
        )
    return _dedupe_ref_items(refs)[:8]


def _extract_chunk_refs(transcript: list[dict[str, Any]], idea: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for round_idx, output in _matching_idea_outputs(transcript, idea):
        refs.extend(
            ref
            for ref in _collect_refs_from_output(output, ["chunk_refs", "evidence"], role=output.get("role"), round_idx=round_idx)
            if ref.get("chunk_id") or ref.get("section") or ref.get("start_line") is not None or ref.get("end_line") is not None
        )
    return _dedupe_ref_items(refs)[:8]


def _extract_code_refs_from_refs(refs: Any) -> list[dict[str, Any]]:
    if not isinstance(refs, list):
        return []
    normalized = [_normalize_ref_item(ref) for ref in refs]
    return _dedupe_ref_items([ref for ref in normalized if ref and ref.get("source_type") == "code"])[:8]


def _extract_failure_feedback_refs(transcript: list[dict[str, Any]], idea: dict[str, Any]) -> list[dict[str, Any]]:
    refs = []
    for round_idx, output in _matching_idea_outputs(transcript, idea):
        refs.extend(
            ref
            for ref in _collect_refs_from_output(
                output,
                ["failure_feedback_refs", "counterevidence_refs", "counterevidence", "evidence_refs", "evidence"],
                role=output.get("role"),
                round_idx=round_idx,
            )
            if _is_failure_feedback_ref(ref)
        )
    return _dedupe_ref_items(refs)[:8]


def _is_failure_feedback_ref(ref: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(ref.get(key) or "").lower()
        for key in ["source_type", "source_path", "chunk_id", "source_label", "snippet", "why_relevant"]
    )
    return any(marker in haystack for marker in ["failure_feedback", "failed_ideas_round", "meta/negative_memory.jsonl", "dragging_datasets", "sample_type_failures", "patch_risk"])


def _matching_idea_outputs(transcript: list[dict[str, Any]], idea: dict[str, Any]):
    for round_payload in transcript:
        round_idx = round_payload.get("round")
        for output in round_payload.get("outputs", []):
            if not isinstance(output, dict) or not _idea_matches_proposed(output.get("proposed_ideas", []), idea):
                continue
            yield round_idx, output


def _structured_refs_from_items(items: Any, *, role: str | None, round_idx: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    if not isinstance(items, list):
        return refs
    for item in items:
        ref = _normalize_ref_item(item, role=role, round_idx=round_idx)
        if ref:
            refs.append(ref)
    return refs


def _collect_refs_from_output(output: dict[str, Any], field_names: list[str], *, role: str | None, round_idx: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for field_name in field_names:
        refs.extend(_structured_refs_from_items(output.get(field_name), role=role, round_idx=round_idx))
    return refs


def _structured_chain_items_from_role_output(items: Any, kind: str, *, role: str | None, round_idx: Any) -> list[dict[str, Any]]:
    refs = _structured_refs_from_items(items, role=role, round_idx=round_idx)
    for ref in refs:
        ref.setdefault("kind", kind)
    return refs


def _normalize_ref_item(item: Any, *, role: str | None = None, round_idx: Any = None) -> dict[str, Any] | None:
    if isinstance(item, str):
        item = _parse_string_ref_item(item)
    if not isinstance(item, dict):
        return None
    raw = dict(item)
    source_path = str(raw.get("source_path") or raw.get("path") or raw.get("source") or "").strip()
    source_type = _infer_source_type(raw, source_path)
    chunk_id = str(raw.get("chunk_id") or raw.get("source_id") or "").strip()
    paper_id = str(raw.get("paper_id") or "").strip()
    symbol = str(raw.get("symbol") or "").strip()
    symbol_kind = str(raw.get("symbol_kind") or raw.get("kind") or "").strip()
    start_line = raw.get("start_line")
    end_line = raw.get("end_line")
    snippet = str(raw.get("snippet") or raw.get("claim") or raw.get("text") or raw.get("summary") or "").strip()
    why_relevant = str(raw.get("why_relevant") or raw.get("why") or raw.get("implication") or "").strip()
    source_label = str(raw.get("source_label") or "").strip()
    if not source_label:
        label_bits = [bit for bit in [source_path or paper_id or chunk_id or source_type, symbol if symbol else "", f"{start_line}-{end_line}" if start_line or end_line else ""] if bit]
        source_label = "::".join(label_bits[:2]) if len(label_bits) >= 2 else (label_bits[0] if label_bits else source_type)
    normalized = {
        "source_type": source_type,
        "source_path": source_path,
        "paper_id": paper_id,
        "chunk_id": chunk_id,
        "section": str(raw.get("section") or "").strip(),
        "symbol": symbol,
        "symbol_kind": symbol_kind,
        "start_line": start_line,
        "end_line": end_line,
        "snippet": snippet,
        "why_relevant": why_relevant,
        "source_label": source_label,
        "role": role or str(raw.get("role") or ""),
        "round": round_idx if round_idx is not None else raw.get("round"),
        "score": raw.get("score"),
        "idea_id": raw.get("idea_id"),
        "decision": raw.get("decision"),
        "failure_mode": raw.get("failure_mode"),
        "dataset_regressions": raw.get("dataset_regressions") or {},
        "dragging_datasets": raw.get("dragging_datasets") or [],
        "sample_type_failures": raw.get("sample_type_failures") or [],
        "patch_risk_labels": raw.get("patch_risk_labels") or [],
        "patch_risk_files": raw.get("patch_risk_files") or [],
        "mixed_gain_patterns": raw.get("mixed_gain_patterns") or [],
        "avoid_repeat_rule": raw.get("avoid_repeat_rule"),
    }
    if not normalized["chunk_id"] and normalized["source_path"] and (normalized["section"] or normalized["symbol"] or normalized["start_line"] or normalized["end_line"]):
        normalized["chunk_id"] = _derive_chunk_id(normalized)
    if not normalized["source_label"]:
        normalized["source_label"] = normalized["source_path"] or normalized["chunk_id"] or normalized["paper_id"] or normalized["source_type"]
    return normalized


def _parse_string_ref_item(value: str) -> dict[str, Any]:
    text = value.strip()
    if not text:
        return {}
    if " | " in text:
        parts = [part.strip() for part in text.split(" | ") if part.strip()]
        data: dict[str, Any] = {"source_label": parts[0] if parts else text, "snippet": parts[1] if len(parts) > 1 else text}
        if len(parts) > 2:
            data["why_relevant"] = parts[2]
        return data
    if "::" in text:
        source, snippet = text.split("::", 1)
        return {"source_path": source.strip(), "source_label": source.strip(), "snippet": snippet.strip()}
    return {"source_label": text, "snippet": text, "source_type": "summary"}


def _infer_source_type(item: dict[str, Any], source_path: str) -> str:
    explicit = str(item.get("source_type") or item.get("kind") or "").strip().lower()
    if explicit in {"paper", "rebuttal", "code", "repo_artifact", "summary", "memory", "failure_feedback"}:
        return explicit
    lowered = f"{source_path} {item.get('source', '')} {item.get('source_label', '')} {item.get('section', '')}".lower()
    if any(marker in lowered for marker in ["failure_feedback", "failed_ideas_round", "meta/negative_memory.jsonl"]):
        return "failure_feedback"
    if any(marker in lowered for marker in ["rosetta/", "script/", "recipe/", "test/"]) or source_path.endswith((".py", ".sh", ".json", ".yaml", ".yml")):
        return "code" if any(marker in lowered for marker in ["rosetta/", "script/", "recipe/", "test/"]) or source_path.endswith(".py") else "repo_artifact"
    if any(marker in lowered for marker in ["rebuttal", "review", "openreview", "reviewer"]):
        return "rebuttal"
    if any(marker in lowered for marker in ["paper", "ref_paper", "bibliography"]) or source_path.endswith(".pdf"):
        return "paper"
    if any(marker in lowered for marker in ["repo_card", "rebuttal_concern_matrix", "negative_result_memory", "retrieval_plan", "idea_debate", "negative_constraints"]):
        return "repo_artifact"
    return "summary" if str(item.get("snippet") or item.get("claim") or item.get("text") or "").strip() else "unknown"


def _derive_chunk_id(ref: dict[str, Any]) -> str:
    source_path = ref.get("source_path") or ref.get("paper_id") or ref.get("source_label") or "unknown"
    pieces = [str(source_path)]
    if ref.get("section"):
        pieces.append(str(ref["section"]))
    if ref.get("symbol"):
        pieces.append(str(ref["symbol"]))
    line_bits = []
    if ref.get("start_line") is not None:
        line_bits.append(str(ref["start_line"]))
    if ref.get("end_line") is not None:
        line_bits.append(str(ref["end_line"]))
    if line_bits:
        pieces.append("-".join(line_bits))
    return "::".join(piece for piece in pieces if piece)


def _dedupe_ref_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for item in items:
        if not isinstance(item, dict):
            continue
        key = (
            item.get("source_type"),
            item.get("source_path"),
            item.get("paper_id"),
            item.get("chunk_id"),
            item.get("symbol"),
            item.get("section"),
            item.get("start_line"),
            item.get("end_line"),
            item.get("snippet", "")[:180],
            item.get("why_relevant", "")[:120],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _merge_ref_lists(*values: Any) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for value in values:
        if value in (None, ""):
            continue
        if isinstance(value, list):
            for item in value:
                ref = _normalize_ref_item(item)
                if ref:
                    refs.append(ref)
        else:
            ref = _normalize_ref_item(value)
            if ref:
                refs.append(ref)
    return _dedupe_ref_items(refs)


def _evidence_index(chunks: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    index = []
    for chunk in chunks[:max_items]:
        text = " ".join(str(chunk.get("text", "")).split())
        index.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "paper_id": chunk.get("paper_id"),
                "kind": chunk.get("kind"),
                "section": chunk.get("section"),
                "source_path": chunk.get("source_path"),
                "snippet": text[:900],
            }
        )
    return index


def _code_card_index(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = []
    for card in cards:
        if not card.get("exists"):
            index.append({"path": card.get("path"), "exists": False})
            continue
        index.append(
            {
                "path": card.get("path"),
                "language": card.get("language"),
                "symbols": [
                    {
                        "name": item.get("name"),
                        "kind": item.get("kind"),
                        "start_line": item.get("start_line"),
                        "end_line": item.get("end_line"),
                        "args": item.get("args", []),
                    }
                    for item in (card.get("symbols") or [])[:30]
                ],
                "config_knobs": (card.get("config_knobs") or [])[:40],
                "imports": (card.get("imports") or [])[:20],
                "summary_snippet": " ".join(str(card.get("summary_snippet", "")).split())[:700],
            }
        )
    return index


def _code_chunk_index(chunks: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    index = []
    for chunk in chunks[:max_items]:
        text = " ".join(str(chunk.get("text", "")).split())
        index.append(
            {
                "chunk_id": chunk.get("chunk_id"),
                "path": chunk.get("path"),
                "symbol": chunk.get("symbol"),
                "symbol_kind": chunk.get("symbol_kind"),
                "start_line": chunk.get("start_line"),
                "end_line": chunk.get("end_line"),
                "snippet": text[:900],
            }
        )
    return index


def _compact(value: Any, max_chars: int) -> Any:
    text = json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return value
    return {"truncated_json": text[:max_chars]}


def _feedback_index(feedback: list[dict[str, Any]], feedback_constraints: dict[str, Any], *, max_items: int = 12) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for idx, item in enumerate(feedback):
        if not isinstance(item, dict):
            continue
        ref = _feedback_ref_from_item(item, idx)
        if ref:
            refs.append(ref)
        for key in ["feedback_items", "entries"]:
            nested_items = item.get(key)
            if not isinstance(nested_items, list):
                continue
            for nested_idx, nested in enumerate(nested_items[:4]):
                if isinstance(nested, dict):
                    ref = _feedback_ref_from_item(nested, idx * 10 + nested_idx + 1)
                    if ref:
                        refs.append(ref)
    if not refs and (
        feedback_constraints.get("avoid_repeat_rules")
        or feedback_constraints.get("failed_idea_ids")
        or feedback_constraints.get("failure_modes")
        or feedback_constraints.get("dragging_datasets")
        or feedback_constraints.get("sample_type_failures")
        or feedback_constraints.get("mixed_gain_patterns")
        or feedback_constraints.get("patch_risk_labels")
    ):
        snippet_parts = []
        if feedback_constraints.get("failed_idea_ids"):
            snippet_parts.append(f"failed_idea_ids={','.join(feedback_constraints['failed_idea_ids'][:5])}")
        if feedback_constraints.get("avoid_repeat_rules"):
            snippet_parts.append(f"avoid_repeat={feedback_constraints['avoid_repeat_rules'][0]}")
        if feedback_constraints.get("failure_modes"):
            snippet_parts.append(f"failure_modes={','.join(feedback_constraints['failure_modes'][:4])}")
        if feedback_constraints.get("dragging_datasets"):
            snippet_parts.append(f"dragging_datasets={json.dumps(feedback_constraints['dragging_datasets'][:3], ensure_ascii=False, sort_keys=True)}")
        if feedback_constraints.get("sample_type_failures"):
            snippet_parts.append(f"sample_type_failures={','.join(feedback_constraints['sample_type_failures'][:4])}")
        if feedback_constraints.get("mixed_gain_patterns"):
            snippet_parts.append(f"mixed_patterns={','.join(feedback_constraints['mixed_gain_patterns'][:3])}")
        if feedback_constraints.get("patch_risk_labels"):
            snippet_parts.append(f"patch_risk_labels={','.join(feedback_constraints['patch_risk_labels'][:4])}")
        refs.append(
            _normalize_ref_item(
                {
                    "source_type": "failure_feedback",
                    "source_path": "meta/negative_memory.jsonl",
                    "chunk_id": "meta/negative_memory.jsonl::summary",
                    "source_label": "failure feedback summary",
                    "snippet": " | ".join(snippet_parts),
                    "why_relevant": "Prior failed ideas and avoid-repeat rules constrain the next S1/S2 search.",
                }
            )
            or {}
        )
    return [ref for ref in _dedupe_ref_items(refs)[:max_items] if ref]


def _feedback_ref_from_item(item: dict[str, Any], idx: int) -> dict[str, Any] | None:
    sources = _feedback_nested_sources(item)
    kind = _first_feedback_value(sources, "kind") or "c2c_failure_feedback"
    iteration = _first_feedback_value(sources, "iteration", "latest_iteration")
    idea_id = _first_feedback_value(sources, "idea_id", "id", "candidate_id", "latest_idea_id")
    title = _first_feedback_value(sources, "title", "latest_title")
    decision = _first_feedback_value(sources, "decision", "latest_decision")
    failure_mode = _first_feedback_value(sources, "failure_mode", "latest_failure_mode")
    reason = _first_feedback_value(sources, "reason", "latest_reason")
    summary_text = _first_feedback_value(sources, "summary_text")
    dataset_regressions = _first_feedback_value(sources, "dataset_regressions") or {}
    dragging_datasets = _first_feedback_value(sources, "dragging_datasets") or []
    sample_type_failures = _first_feedback_value(sources, "sample_type_failures") or []
    patch_risk_labels = _first_feedback_value(sources, "patch_risk_labels") or []
    patch_risk_files = _first_feedback_value(sources, "patch_risk_files") or []
    mixed_gain_patterns = _first_feedback_value(sources, "mixed_gain_patterns") or []
    avoid_rule = _first_feedback_value(sources, "avoid_repeat_rule")
    if not avoid_rule:
        rules = _feedback_list_values(sources, "avoid_repeat_rules")
        avoid_rule = rules[0] if rules else None
    failed_ids = _feedback_list_values(sources, "failed_idea_ids", "failed_candidate_ids")
    failed_titles = _feedback_list_values(sources, "failed_titles", "failed_candidate_titles")
    if not any([idea_id, title, reason, summary_text, avoid_rule, failed_ids, failed_titles, failure_mode, dataset_regressions, dragging_datasets, sample_type_failures, patch_risk_labels, patch_risk_files, mixed_gain_patterns]):
        return None

    source_path = _feedback_source_path(item, kind)
    chunk_bits = [source_path or "failure_feedback", str(iteration or idx)]
    if idea_id:
        chunk_bits.append(str(idea_id))
    elif kind:
        chunk_bits.append(str(kind))
    snippet_parts = []
    if title:
        snippet_parts.append(f"title={title}")
    if reason:
        snippet_parts.append(f"reason={reason}")
    if summary_text:
        snippet_parts.append(str(summary_text))
    if avoid_rule:
        snippet_parts.append(f"avoid_repeat={avoid_rule}")
    if failed_ids:
        snippet_parts.append(f"failed_idea_ids={','.join(str(item) for item in failed_ids[:5])}")
    if failed_titles:
        snippet_parts.append(f"failed_titles={','.join(str(item) for item in failed_titles[:3])}")
    if failure_mode:
        snippet_parts.append(f"failure_mode={failure_mode}")
    if isinstance(dataset_regressions, dict) and dataset_regressions:
        snippet_parts.append(f"dataset_regressions={json.dumps(dataset_regressions, ensure_ascii=False, sort_keys=True)}")
    if isinstance(dragging_datasets, list) and dragging_datasets:
        snippet_parts.append(f"dragging_datasets={json.dumps(dragging_datasets[:3], ensure_ascii=False, sort_keys=True)}")
    if isinstance(sample_type_failures, list) and sample_type_failures:
        snippet_parts.append(f"sample_type_failures={','.join(str(item) for item in sample_type_failures[:4])}")
    if isinstance(patch_risk_labels, list) and patch_risk_labels:
        snippet_parts.append(f"patch_risk_labels={','.join(str(item) for item in patch_risk_labels[:4])}")
    if isinstance(patch_risk_files, list) and patch_risk_files:
        snippet_parts.append(f"patch_risk_files={','.join(str(item) for item in patch_risk_files[:3])}")
    if isinstance(mixed_gain_patterns, list) and mixed_gain_patterns:
        snippet_parts.append(f"mixed_patterns={','.join(str(item) for item in mixed_gain_patterns[:3])}")
    return _normalize_ref_item(
        {
            "source_type": "failure_feedback",
            "source_path": source_path,
            "chunk_id": "::".join(chunk_bits),
            "source_label": str(title or kind or "failure feedback"),
            "snippet": " | ".join(snippet_parts)[:1200],
            "why_relevant": "Prior experiment feedback must be used to avoid repeating failed mechanisms and dataset regressions.",
            "idea_id": idea_id,
            "decision": decision,
            "failure_mode": failure_mode,
            "dataset_regressions": dataset_regressions if isinstance(dataset_regressions, dict) else {},
            "dragging_datasets": dragging_datasets if isinstance(dragging_datasets, list) else [],
            "sample_type_failures": sample_type_failures if isinstance(sample_type_failures, list) else [],
            "patch_risk_labels": patch_risk_labels if isinstance(patch_risk_labels, list) else [],
            "patch_risk_files": patch_risk_files if isinstance(patch_risk_files, list) else [],
            "mixed_gain_patterns": mixed_gain_patterns if isinstance(mixed_gain_patterns, list) else [],
            "avoid_repeat_rule": avoid_rule,
        }
    )


def _feedback_nested_sources(item: dict[str, Any]) -> list[dict[str, Any]]:
    sources = [item]
    for key in [
        "summary",
        "summary_entry",
        "entry",
        "entry_snapshot",
        "candidate",
        "candidate_snapshot",
        "latest_candidate_snapshot",
        "latest_entry_snapshot",
        "posthoc_review",
    ]:
        value = item.get(key)
        if isinstance(value, dict):
            sources.append(value)
    return sources


def _first_feedback_value(sources: list[dict[str, Any]], *keys: str) -> Any:
    for key in keys:
        for source in sources:
            value = source.get(key)
            if value not in (None, "", [], {}):
                return value
    return None


def _feedback_list_values(sources: list[dict[str, Any]], *keys: str) -> list[Any]:
    values: list[Any] = []
    for key in keys:
        for source in sources:
            value = source.get(key)
            if isinstance(value, list):
                values.extend(item for item in value if item not in (None, ""))
            elif value not in (None, "", [], {}):
                values.append(value)
    return values


def _feedback_source_path(item: dict[str, Any], kind: Any) -> str:
    for key in ["feedback_round_path", "source_path", "path"]:
        value = item.get(key)
        if value:
            return str(value)
    for key in ["artifacts", "sources"]:
        value = item.get(key)
        if isinstance(value, list) and value:
            return str(value[0])
    if str(kind or "") == "c2c_feedback_summary":
        return "meta/negative_memory.jsonl"
    return "experiment/results/failure_feedback.json"


def _feedback_summary(feedback_constraints: dict[str, Any], feedback_index: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "has_feedback": bool(feedback_index),
        "failed_idea_ids": feedback_constraints.get("failed_idea_ids", []),
        "failed_titles": feedback_constraints.get("failed_titles", []),
        "avoid_repeat_rules": feedback_constraints.get("avoid_repeat_rules", []),
        "blocked_idea_patterns": feedback_constraints.get("blocked_idea_patterns", []),
        "failure_modes": feedback_constraints.get("failure_modes", []),
        "dataset_regressions": feedback_constraints.get("dataset_regressions", {}),
        "dragging_datasets": feedback_constraints.get("dragging_datasets", []),
        "sample_type_failures": feedback_constraints.get("sample_type_failures", []),
        "patch_risk_labels": feedback_constraints.get("patch_risk_labels", []),
        "patch_risk_files": feedback_constraints.get("patch_risk_files", []),
        "mixed_gain_patterns": feedback_constraints.get("mixed_gain_patterns", []),
        "next_round_suggestions": feedback_constraints.get("next_round_suggestions", []),
        "latest_reason": feedback_constraints.get("latest_reason"),
        "latest_decision": feedback_constraints.get("latest_decision"),
        "feedback_ref_ids": [ref.get("chunk_id") for ref in feedback_index[:8] if ref.get("chunk_id")],
    }


def _fallback_failure_feedback_refs(feedback_constraints: dict[str, Any], idea: dict[str, Any]) -> list[dict[str, Any]]:
    refs = list(feedback_constraints.get("failure_feedback_refs") or [])
    if not refs:
        return []
    idea_id = str(idea.get("id") or "")
    title = str(idea.get("title") or "").strip().lower()
    relevant = []
    for ref in refs:
        snippet = str(ref.get("snippet") or "").lower()
        ref_idea = str(ref.get("idea_id") or "")
        if ref_idea == idea_id or (title and title in snippet):
            relevant.append(ref)
    return _dedupe_ref_items([*relevant, *refs])[:4]


def _recovery_fallback_ideas(baseline: dict[str, Any], feedback_constraints: dict[str, Any]) -> list[dict[str, Any]]:
    base_name = baseline.get("name", DEFAULT_BASELINE["name"])
    ideas = []
    for idx, profile in enumerate([*failure_aware_c2c_ideas("C2C", baseline), *default_c2c_ideas("C2C", baseline)]):
        if len(ideas) >= 3:
            break
        item = normalize_c2c_mechanism_fields(dict(profile), baseline)
        item["id"] = f"recovery_{item['id']}"
        item["title"] = f"Recovery {item['title']}"
        item["motivation"] = "Generated by deterministic recovery fallback because failure feedback exhausted the first mechanism pool."
        item["expected_contribution"] = "A mechanism-level recovery candidate that explicitly responds to prior C2C failure feedback."
        item["novelty_against"] = [base_name, "failed feedback variants", "local threshold tuning"]
        item["selected"] = idx == 0
        item["novelty_gate"] = c2c_idea_novelty_report(item)
        ideas.append(item)
    return ideas


def _fallback_ideas_for_feedback(topic: str, baseline: dict[str, Any], feedback_constraints: dict[str, Any]) -> list[dict[str, Any]]:
    pool = failure_aware_c2c_ideas(topic, baseline) if feedback_constraints.get("failed_idea_ids") or feedback_constraints.get("failed_titles") else []
    pool.extend(default_c2c_ideas(topic, baseline))
    ideas = []
    seen = set()
    for idea in pool:
        if _is_forbidden_idea(idea, feedback_constraints) or idea["id"] in seen:
            continue
        item = dict(idea)
        item["selected"] = not ideas
        ideas.append(item)
        seen.add(item["id"])
        if len(ideas) >= 5:
            break
    return ideas or default_c2c_ideas(topic, baseline)


C2C_FAILED_DECISIONS = {None, "not_viable", "failed_no_metrics", "partial", "blocked", "patch_rejected", "proxy_rejected", "proxy_repairable"}


def _feedback_constraints(feedback: list[dict[str, Any]]) -> dict[str, Any]:
    failed_ids: set[str] = set()
    failed_titles: set[str] = set()
    avoid_rules: list[str] = []
    blocked_patterns: list[str] = []
    failure_modes: list[str] = []
    next_round_suggestions: list[str] = []
    dataset_regressions: dict[str, float] = {}
    dragging_datasets: list[dict[str, Any]] = []
    sample_type_failures: set[str] = set()
    patch_risk_labels: set[str] = set()
    patch_risk_files: set[str] = set()
    mixed_gain_patterns: set[str] = set()
    latest_reason = None
    latest_decision = None
    latest_failure_mode = None

    def add_candidate(candidate: dict[str, Any] | None) -> None:
        if not isinstance(candidate, dict):
            return
        candidate_id = candidate.get("idea_id") or candidate.get("id") or candidate.get("candidate_id")
        title = candidate.get("title")
        decision = candidate.get("decision")
        if candidate_id and decision in C2C_FAILED_DECISIONS:
            failed_ids.add(str(candidate_id))
        if title and decision in C2C_FAILED_DECISIONS:
            failed_titles.add(str(title).strip().lower())

    def add_dataset_regressions(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for dataset, delta in value.items():
            try:
                number = float(delta)
            except (TypeError, ValueError):
                continue
            dataset_regressions[str(dataset)] = max(dataset_regressions.get(str(dataset), float("-inf")), number)

    def add_list_values(target: list[str], value: Any) -> None:
        if isinstance(value, list):
            for entry in value:
                if entry:
                    target.append(str(entry))
        elif value:
            target.append(str(value))

    def add_failure_attribution(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for item in value.get("dragging_datasets") or []:
            if isinstance(item, dict) and item.get("dataset"):
                dragging_datasets.append(item)
        for item in value.get("sample_type_failures") or []:
            if isinstance(item, dict) and item.get("sample_family"):
                sample_type_failures.add(str(item["sample_family"]))
            elif item:
                sample_type_failures.add(str(item))
        for pattern in value.get("mixed_gain_patterns") or []:
            if pattern:
                mixed_gain_patterns.add(str(pattern))
        patch_risk = value.get("patch_risk") or {}
        for label in patch_risk.get("risk_labels") or []:
            if label:
                patch_risk_labels.add(str(label))
        for risk_file in patch_risk.get("risk_files") or []:
            if isinstance(risk_file, dict) and risk_file.get("path"):
                patch_risk_files.add(str(risk_file["path"]))

    queue = [item for item in feedback if isinstance(item, dict)]
    seen_objects: set[int] = set()
    for item in queue:
        if not isinstance(item, dict):
            continue
        object_id = id(item)
        if object_id in seen_objects:
            continue
        seen_objects.add(object_id)
        add_candidate(item)
        add_candidate(item.get("entry"))
        add_candidate(item.get("entry_snapshot"))
        add_candidate(item.get("candidate"))
        add_candidate(item.get("candidate_snapshot"))
        add_candidate(item.get("latest_candidate_snapshot"))
        add_candidate(item.get("latest_entry_snapshot"))
        for nested_key in [
            "summary",
            "summary_entry",
            "entry",
            "entry_snapshot",
            "candidate",
            "candidate_snapshot",
            "latest_candidate_snapshot",
            "latest_entry_snapshot",
            "posthoc_review",
        ]:
            nested = item.get(nested_key)
            if isinstance(nested, dict):
                queue.append(nested)
        for list_key in ["entries", "feedback_items"]:
            nested_items = item.get(list_key)
            if isinstance(nested_items, list):
                queue.extend(nested for nested in nested_items if isinstance(nested, dict))
        for candidate in item.get("candidate_results") or []:
            add_candidate(candidate)
            if isinstance(candidate, dict):
                add_dataset_regressions(candidate.get("dataset_regressions"))
                add_list_values(avoid_rules, candidate.get("avoid_repeat_rules"))
                add_failure_attribution(candidate.get("failure_attribution"))
        for candidate_id in item.get("failed_idea_ids") or item.get("failed_candidate_ids") or []:
            failed_ids.add(str(candidate_id))
        for title in item.get("failed_titles") or item.get("failed_candidate_titles") or []:
            failed_titles.add(str(title).strip().lower())
        rule = item.get("avoid_repeat_rule")
        if rule:
            avoid_rules.append(str(rule))
        for rule in item.get("avoid_repeat_rules") or []:
            if rule:
                avoid_rules.append(str(rule))
        for pattern in item.get("blocked_idea_patterns") or []:
            if pattern:
                blocked_patterns.append(str(pattern))
        if item.get("failure_mode"):
            failure_modes.append(str(item["failure_mode"]))
        for mode in item.get("failure_modes") or []:
            if mode:
                failure_modes.append(str(mode))
        add_list_values(next_round_suggestions, item.get("next_round_suggestions"))
        add_dataset_regressions(item.get("dataset_regressions"))
        add_failure_attribution(item.get("failure_attribution"))
        add_failure_attribution((item.get("candidate_snapshot") or {}).get("failure_attribution"))
        add_failure_attribution((item.get("entry_snapshot") or {}).get("failure_attribution"))
        latest_reason = item.get("latest_reason") or item.get("reason") or latest_reason
        latest_decision = item.get("latest_decision") or item.get("decision") or latest_decision
        latest_failure_mode = item.get("latest_failure_mode") or item.get("failure_mode") or latest_failure_mode
        entry = item.get("entry") if isinstance(item.get("entry"), dict) else {}
        if entry.get("avoid_repeat_rule"):
            avoid_rules.append(str(entry["avoid_repeat_rule"]))
    return {
        "failed_idea_ids": sorted(failed_ids),
        "failed_titles": sorted(failed_titles),
        "avoid_repeat_rules": sorted(set(avoid_rules)),
        "blocked_idea_patterns": sorted(set(blocked_patterns + avoid_rules)),
        "failure_modes": sorted(set(failure_modes)),
        "dataset_regressions": {key: round(value, 4) for key, value in dataset_regressions.items()},
        "dragging_datasets": _dedupe_dragging_datasets(dragging_datasets),
        "sample_type_failures": sorted(sample_type_failures),
        "patch_risk_labels": sorted(patch_risk_labels),
        "patch_risk_files": sorted(patch_risk_files),
        "mixed_gain_patterns": sorted(mixed_gain_patterns),
        "next_round_suggestions": list(dict.fromkeys(next_round_suggestions))[:12],
        "latest_reason": latest_reason,
        "latest_decision": latest_decision,
        "latest_failure_mode": latest_failure_mode,
    }


def _is_forbidden_idea(idea: dict[str, Any], feedback_constraints: dict[str, Any]) -> bool:
    idea_id = str(idea.get("id") or "")
    title = str(idea.get("title") or "").strip().lower()
    return idea_id in set(feedback_constraints.get("failed_idea_ids") or []) or title in set(feedback_constraints.get("failed_titles") or [])


def _dedupe_dragging_datasets(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict) or not item.get("dataset"):
            continue
        dataset = str(item["dataset"])
        try:
            regression = float(item.get("regression") or 0.0)
            current = float((by_dataset.get(dataset) or {}).get("regression") or float("-inf"))
        except (TypeError, ValueError):
            regression = 0.0
            current = float("-inf")
        if regression >= current:
            by_dataset[dataset] = dict(item)
    return sorted(by_dataset.values(), key=lambda item: item.get("regression", 0.0), reverse=True)
