"""S1 literature and idea gate."""

from __future__ import annotations

import json
from pathlib import Path

from auto_research.evidence_refs import resolve_s1_evidence_refs

from .base import StageGateValidator, load_schema, validate_min_schema


class S1GateValidator(StageGateValidator):
    stage_key = "S1_literature"
    validator_name = "s1_literature_gate_v1"

    def validate(self):
        direction_path = self.require_file("literature/direction.json", check_name="direction_json_exists")
        evidence_bundle_path = self.require_file("literature/evidence_bundle.json", check_name="evidence_bundle_json_exists")
        scorecard_path = self.require_file("literature/direction_scorecard.json", check_name="direction_scorecard_json_exists")
        novelty_path = self.require_file("literature/novelty_audit.json", check_name="novelty_audit_json_exists")
        manifest_path = self.require_file("references/papers/manifest.json", check_name="reference_manifest_exists")
        if not direction_path or not evidence_bundle_path or not scorecard_path or not novelty_path or not manifest_path:
            return self.finalize()

        direction = self.read_json_artifact("literature/direction.json")
        evidence_bundle = self.read_json_artifact("literature/evidence_bundle.json")
        scorecard = self.read_json_artifact("literature/direction_scorecard.json")
        novelty_audit = self.read_json_artifact("literature/novelty_audit.json")
        if direction is None or evidence_bundle is None or scorecard is None or novelty_audit is None:
            return self.finalize()

        direction_errors = validate_min_schema(direction, load_schema("direction.schema.json"))
        direction_errors.extend(_direction_semantic_errors(direction))
        if direction_errors:
            self.retry_check("direction_schema", "direction.json does not satisfy direction contract", artifact="literature/direction.json", details={"errors": direction_errors[:10]})
        else:
            self.pass_check("direction_schema", artifact="literature/direction.json", details={"direction_id": direction.get("direction_id")})

        bundle_errors = validate_min_schema(evidence_bundle, load_schema("evidence_bundle.schema.json"))
        if bundle_errors:
            self.retry_check("evidence_bundle_schema", "evidence_bundle.json does not satisfy evidence bundle contract", artifact="literature/evidence_bundle.json", details={"errors": bundle_errors[:10]})
        else:
            self.pass_check("evidence_bundle_schema", artifact="literature/evidence_bundle.json", details={"item_count": len(evidence_bundle.get("items") or []) if isinstance(evidence_bundle, dict) else 0})

        scorecard_errors = validate_min_schema(scorecard, load_schema("direction_scorecard.schema.json"))
        if scorecard_errors:
            self.retry_check("direction_scorecard_schema", "direction_scorecard.json does not satisfy direction scorecard contract", artifact="literature/direction_scorecard.json", details={"errors": scorecard_errors[:10]})
        else:
            self.pass_check("direction_scorecard_schema", artifact="literature/direction_scorecard.json")

        novelty_errors = validate_min_schema(novelty_audit, load_schema("novelty_audit.schema.json"))
        if novelty_errors:
            self.retry_check("novelty_audit_schema", "novelty_audit.json does not satisfy novelty audit contract", artifact="literature/novelty_audit.json", details={"errors": novelty_errors[:10]})
        else:
            self.pass_check("novelty_audit_schema", artifact="literature/novelty_audit.json")

        ideas = self._optional_legacy_ideas_check()

        manifest = self.read_json_artifact("references/papers/manifest.json")
        paper_metadata_exists = (self.project_root / "literature" / "papers" / "metadata.json").exists()
        if isinstance(manifest, dict) and (manifest.get("papers") or paper_metadata_exists):
            self.pass_check("reference_material_registered", artifact="references/papers/manifest.json")
        else:
            self.retry_check("reference_material_registered", "no reference papers registered", artifact="references/papers/manifest.json")

        c2c_manifest = self.project_root / "intake" / "c2c" / "static_bundle.json"
        if c2c_manifest.exists():
            self._validate_c2c_s1_contract(ideas if isinstance(ideas, list) else [])
        elif (self.project_root / "literature" / "evidence_session.json").exists():
            self._validate_generic_codex_evidence_agent_contract(ideas if isinstance(ideas, list) else [])
        self._validate_s1_novelty_audit()
        return self.finalize()

    def _optional_legacy_ideas_check(self):
        path = self.project_root / "literature" / "ideas.json"
        if not path.exists():
            self.pass_check("ideas_json_compatibility", message="legacy ideas.json mirror not present; direction.json is authoritative")
            return []
        try:
            ideas = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.pass_check("ideas_json_compatibility", message="legacy ideas.json mirror is invalid but no longer gates S1")
            self.artifacts_checked.append("literature/ideas.json")
            return []
        self.artifacts_checked.append("literature/ideas.json")
        schema_errors = validate_min_schema(ideas, load_schema("idea.schema.json"))
        if schema_errors:
            self.pass_check(
                "ideas_json_compatibility",
                artifact="literature/ideas.json",
                message="legacy ideas.json mirror does not satisfy old idea contract",
                details={"errors": schema_errors[:10]},
            )
        else:
            self.pass_check("ideas_json_compatibility", artifact="literature/ideas.json", details={"idea_count": len(ideas) if isinstance(ideas, list) else 0})
        return ideas

    def _validate_s1_novelty_audit(self) -> None:
        path = self.project_root / "literature" / "novelty_audit.json"
        if not path.exists():
            path = self.project_root / "literature" / "c2c" / "novelty_audit.json"
        if not path.exists():
            self.pass_check("s1_novelty_audit", details={"status": "not_configured"})
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            self.retry_check("s1_novelty_audit", "novelty audit artifact is not valid JSON", artifact=str(path.relative_to(self.project_root)))
            return
        if isinstance(payload, dict) and payload.get("schema_version") == "auto_research_novelty_audit_v1":
            latest = payload.get("latest") if isinstance(payload.get("latest"), dict) else {}
            if payload.get("passed") is True or payload.get("status") == "skipped" or payload.get("enabled") is False:
                self.pass_check("s1_novelty_audit", artifact=str(path.relative_to(self.project_root)), details={"status": payload.get("status"), "threshold": payload.get("threshold")})
            else:
                self.retry_check("s1_novelty_audit", "S1 novelty audit did not pass", artifact=str(path.relative_to(self.project_root)), details={"latest": latest or payload})
            return
        audits = payload
        if not isinstance(audits, list) or not audits:
            self.retry_check("s1_novelty_audit", "novelty audit artifact must be a non-empty list", artifact=str(path.relative_to(self.project_root)))
            return
        latest = next((item for item in reversed(audits) if isinstance(item, dict)), {})
        if latest.get("status") == "skipped" or latest.get("enabled") is False:
            self.pass_check("s1_novelty_audit", artifact=str(path.relative_to(self.project_root)), details={"status": latest.get("status"), "reason": latest.get("reason")})
        elif latest.get("passed") is True:
            audit = latest.get("audit") if isinstance(latest.get("audit"), dict) else {}
            self.pass_check("s1_novelty_audit", artifact=str(path.relative_to(self.project_root)), details={"novelty_score": audit.get("novelty_score"), "threshold": latest.get("threshold")})
        else:
            self.retry_check("s1_novelty_audit", "S1 novelty audit did not pass", artifact=str(path.relative_to(self.project_root)), details={"latest": latest})

    def _validate_generic_codex_evidence_agent_contract(self, ideas: list[dict]) -> None:
        required = [
            "literature/evidence_requests.json",
            "literature/evidence_bundle.json",
            "literature/direction.json",
            "literature/evidence_session.json",
            "literature/evidence_ref_report.json",
        ]
        missing = [rel for rel in required if not (self.project_root / rel).exists()]
        if missing:
            self.retry_check(
                "s1_codex_evidence_agent_artifacts",
                "S1 Codex evidence agent artifacts missing",
                artifact="literature/evidence_session.json",
                details={"missing": missing},
            )
            return
        direction = self._safe_json("literature/direction.json")
        bundle = self._safe_json("literature/evidence_bundle.json")
        session = self._safe_json("literature/evidence_session.json")
        ref_report = self._safe_json("literature/evidence_ref_report.json")
        errors = []
        if not isinstance(direction, dict) or not direction.get("direction_id") or not direction.get("hypothesis"):
            errors.append("direction.json must include direction_id and hypothesis")
        if not isinstance(bundle, dict) or not bundle.get("items"):
            errors.append("evidence_bundle.items must be non-empty")
        if not isinstance(session, dict) or session.get("status") != "ok":
            errors.append("evidence_session.status must be ok")
        if not isinstance(ref_report, dict):
            errors.append("evidence_ref_report must exist")
        elif ref_report.get("status") != "pass":
            errors.append("evidence_ref_report.status must be pass")
        if ideas and len(ideas) != 1:
            errors.append("legacy ideas.json mirror should contain exactly one selected high-level direction")
        if not errors and ideas:
            payload = {"evidence_bundle": bundle, "selected_ideas": ideas}
            live_report = resolve_s1_evidence_refs(self.project_root, payload, mode="generic")
            if live_report.get("status") != "pass":
                errors.append("live evidence ref resolution failed")
        if errors:
            self.retry_check(
                "s1_codex_evidence_agent_contract",
                "S1 Codex evidence agent output is incomplete",
                artifact="literature/evidence_session.json",
                details={"errors": errors},
            )
        else:
            self.pass_check("s1_codex_evidence_agent_contract", artifact="literature/evidence_session.json")

    def _validate_c2c_s1_contract(self, ideas: list[dict]) -> None:
        required = [
            "intake/c2c/static_bundle.json",
            "intake/c2c/evidence_brief.json",
            "intake/c2c/repo_card.json",
            "intake/c2c/result_ledger.csv",
            "intake/c2c/paper_cards.json",
            "intake/c2c/rebuttal_concern_matrix.json",
            "intake/c2c/negative_result_memory.json",
            "intake/c2c/baseline_evidence.json",
            "literature/idea_debate.json",
            "literature/negative_constraints.json",
            "literature/c2c/baseline_evidence.json",
            "literature/c2c/rebuttal_concern_matrix.json",
        ]
        missing = [rel for rel in required if not (self.project_root / rel).exists()]
        if missing:
            self.retry_check("c2c_s1_evidence_bundle", f"C2C S1 evidence missing: {', '.join(Path(item).name for item in missing)}", details={"missing": missing})
        else:
            self.pass_check("c2c_s1_evidence_bundle", details={"required_count": len(required)})

        invalid_contracts = []
        for idea in ideas[:3]:
            idea_id = idea.get("id") or idea.get("title") or "unknown"
            missing_fields = []
            for field in ["hypothesis", "expected_files", "verification_commands", "reviewer_risk_response", "evidence_refs", "counterevidence_refs", "code_refs"]:
                if not idea.get(field):
                    missing_fields.append(field)
            if missing_fields:
                invalid_contracts.append({"idea": idea_id, "missing": missing_fields})
        if invalid_contracts:
            self.retry_check(
                "c2c_idea_experiment_contract",
                "C2C idea missing structured evidence, counterevidence, code refs, or experiment contract fields",
                artifact="literature/ideas.json",
                details={"invalid": invalid_contracts},
            )
        else:
            self.pass_check("c2c_idea_experiment_contract", artifact="literature/ideas.json")

        debate = self._safe_json("literature/idea_debate.json")
        if debate:
            fallback_roles = []
            for key, value in debate.items():
                if isinstance(value, dict) and str(value.get("status", "")).endswith("fallback"):
                    fallback_roles.append(key)
            if fallback_roles:
                self.retry_check("c2c_debate_gpt_completion", "C2C debate contains fallback agent outputs", artifact="literature/idea_debate.json", details={"fallback_roles": fallback_roles})
            else:
                self.pass_check("c2c_debate_gpt_completion", artifact="literature/idea_debate.json")
            if debate.get("strategy") == "codex_resume_evidence_agent":
                self._validate_c2c_codex_evidence_agent_contract(debate)

    def _validate_c2c_codex_evidence_agent_contract(self, debate: dict) -> None:
        required = [
            "literature/direction.json",
            "literature/evidence_bundle.json",
            "literature/c2c/evidence_requests.json",
            "literature/c2c/evidence_bundle.json",
            "literature/c2c/direction_decision.json",
            "literature/c2c/evidence_session.json",
            "literature/c2c/evidence_ref_report.json",
        ]
        missing = [rel for rel in required if not (self.project_root / rel).exists()]
        if missing:
            self.retry_check(
                "c2c_s1_codex_evidence_agent_artifacts",
                "C2C S1 Codex evidence agent artifacts missing",
                artifact="literature/idea_debate.json",
                details={"missing": missing},
            )
            return
        direction = self._safe_json("literature/direction.json")
        legacy_direction = self._safe_json("literature/c2c/direction_decision.json")
        bundle = self._safe_json("literature/c2c/evidence_bundle.json")
        session = self._safe_json("literature/c2c/evidence_session.json")
        ref_report = self._safe_json("literature/c2c/evidence_ref_report.json")
        errors = []
        if not isinstance(direction, dict) or not direction.get("direction_id") or not direction.get("hypothesis"):
            errors.append("direction.json must include direction_id and hypothesis")
        if not isinstance(legacy_direction, dict) or not legacy_direction.get("direction_id") or not legacy_direction.get("core_hypothesis"):
            errors.append("c2c direction_decision compatibility mirror must include direction_id and core_hypothesis")
        if not isinstance(bundle, dict) or not bundle.get("items"):
            errors.append("evidence_bundle.items must be non-empty")
        if not isinstance(session, dict) or session.get("status") != "ok":
            errors.append("evidence_session.status must be ok")
        if not isinstance(ref_report, dict):
            errors.append("evidence_ref_report must exist")
        elif ref_report.get("status") != "pass":
            errors.append("evidence_ref_report.status must be pass")
        if not debate.get("selected_ideas") or len(debate.get("selected_ideas") or []) != 1:
            errors.append("Codex S1 must pass exactly one high-level direction card to S2")
        if not errors:
            payload = {"evidence_bundle": bundle, "selected_ideas": debate.get("selected_ideas") or []}
            live_report = resolve_s1_evidence_refs(self.project_root, payload, mode="c2c")
            if live_report.get("status") != "pass":
                errors.append("live evidence ref resolution failed")
        if errors:
            self.retry_check(
                "c2c_s1_codex_evidence_agent_contract",
                "C2C S1 Codex evidence agent output is incomplete",
                artifact="literature/idea_debate.json",
                details={"errors": errors},
            )
        else:
            self.pass_check("c2c_s1_codex_evidence_agent_contract", artifact="literature/idea_debate.json")

    def _safe_json(self, rel_path: str):
        path = self.project_root / rel_path
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None


def _direction_semantic_errors(direction: object) -> list[str]:
    if not isinstance(direction, dict):
        return ["direction must be an object"]
    errors: list[str] = []
    for field in ["direction_id", "mechanism_axis", "integration_point", "control_signal", "hypothesis", "why_baseline_fails"]:
        if not str(direction.get(field) or "").strip():
            errors.append(f"{field} must be non-empty")
    if not isinstance(direction.get("expected_metric_signature"), dict) or not direction.get("expected_metric_signature"):
        errors.append("expected_metric_signature must be a non-empty object")
    for field in ["required_evidence_refs", "counterevidence_refs", "implementation_surface_refs", "go_to_s2_conditions", "return_to_s1_conditions"]:
        if not isinstance(direction.get(field), list) or not direction.get(field):
            errors.append(f"{field} must be a non-empty list")
    return errors
