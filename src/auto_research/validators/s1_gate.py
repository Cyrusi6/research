"""S1 literature and idea gate."""

from __future__ import annotations

import json
from pathlib import Path

from .base import StageGateValidator, load_schema, validate_min_schema


class S1GateValidator(StageGateValidator):
    stage_key = "S1_literature"
    validator_name = "s1_literature_gate_v1"

    def validate(self):
        ideas_path = self.require_file("literature/ideas.json", check_name="ideas_json_exists")
        manifest_path = self.require_file("references/papers/manifest.json", check_name="reference_manifest_exists")
        if not ideas_path or not manifest_path:
            return self.finalize()

        ideas = self.read_json_artifact("literature/ideas.json")
        if ideas is None:
            return self.finalize()
        schema_errors = validate_min_schema(ideas, load_schema("idea.schema.json"))
        if schema_errors:
            self.retry_check("ideas_schema", "ideas.json does not satisfy idea contract", artifact="literature/ideas.json", details={"errors": schema_errors[:10]})
        else:
            self.pass_check("ideas_schema", artifact="literature/ideas.json", details={"idea_count": len(ideas)})

        if isinstance(ideas, list):
            low_scores = [
                idea.get("id") or idea.get("title") or f"idea_{idx}"
                for idx, idea in enumerate(ideas)
                if float(idea.get("novelty_score") or 0) < 4 or float(idea.get("feasibility_score") or 0) < 4
            ]
            if low_scores:
                self.fail_check("idea_score_thresholds", "idea score below threshold", artifact="literature/ideas.json", details={"ideas": low_scores[:10]})
            else:
                self.pass_check("idea_score_thresholds", artifact="literature/ideas.json")

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
        return self.finalize()

    def _validate_generic_codex_evidence_agent_contract(self, ideas: list[dict]) -> None:
        required = [
            "literature/evidence_requests.json",
            "literature/evidence_bundle.json",
            "literature/direction_decision.json",
            "literature/evidence_session.json",
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
        direction = self._safe_json("literature/direction_decision.json")
        bundle = self._safe_json("literature/evidence_bundle.json")
        session = self._safe_json("literature/evidence_session.json")
        errors = []
        if not isinstance(direction, dict) or not direction.get("direction_id") or not direction.get("core_hypothesis"):
            errors.append("direction_decision must include direction_id and core_hypothesis")
        if not isinstance(bundle, dict) or not bundle.get("items"):
            errors.append("evidence_bundle.items must be non-empty")
        if not isinstance(session, dict) or session.get("status") != "ok":
            errors.append("evidence_session.status must be ok")
        if len(ideas) != 1:
            errors.append("S1 Codex evidence agent must pass exactly one high-level idea/direction to S2")
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
            "literature/c2c/evidence_requests.json",
            "literature/c2c/evidence_bundle.json",
            "literature/c2c/direction_decision.json",
            "literature/c2c/evidence_session.json",
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
        direction = self._safe_json("literature/c2c/direction_decision.json")
        bundle = self._safe_json("literature/c2c/evidence_bundle.json")
        session = self._safe_json("literature/c2c/evidence_session.json")
        errors = []
        if not isinstance(direction, dict) or not direction.get("direction_id") or not direction.get("core_hypothesis"):
            errors.append("direction_decision must include direction_id and core_hypothesis")
        if not isinstance(bundle, dict) or not bundle.get("items"):
            errors.append("evidence_bundle.items must be non-empty")
        if not isinstance(session, dict) or session.get("status") != "ok":
            errors.append("evidence_session.status must be ok")
        if not debate.get("selected_ideas") or len(debate.get("selected_ideas") or []) != 1:
            errors.append("Codex S1 must pass exactly one high-level direction card to S2")
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
