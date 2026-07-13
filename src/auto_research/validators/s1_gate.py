"""Strict S1 DirectionSpec v2 gate."""

from __future__ import annotations

from auto_research.config import bootstrap_profile_enabled
from auto_research.domain_contracts import contract_errors, validate_direction_identity

from .base import StageGateValidator, load_schema, validate_min_schema


class S1GateValidator(StageGateValidator):
    stage_key = "S1_literature"
    validator_name = "s1_literature_gate_v2"

    def validate(self):
        required = [
            "literature/direction.json",
            "literature/evidence_bundle.json",
            "literature/direction_scorecard.json",
            "literature/novelty_audit.json",
            "references/papers/manifest.json",
        ]
        if not all(self.require_file(path, retry=True) for path in required):
            return self.finalize()
        direction = self.read_json_artifact("literature/direction.json")
        evidence = self.read_json_artifact("literature/evidence_bundle.json")
        scorecard = self.read_json_artifact("literature/direction_scorecard.json")
        novelty = self.read_json_artifact("literature/novelty_audit.json")
        manifest = self.read_json_artifact("references/papers/manifest.json")
        if not all(isinstance(item, dict) for item in [direction, evidence, scorecard, novelty, manifest]):
            self.retry_check("s1_authoritative_json", "S1 authoritative artifacts must be JSON objects")
            return self.finalize()

        errors = contract_errors(direction, "direction_v2.schema.json")
        try:
            validate_direction_identity(direction)
        except ValueError as exc:
            errors.append(str(exc))
        if errors:
            self.retry_check("direction_v2", "DirectionSpec v2 validation failed", artifact="literature/direction.json", details={"errors": errors[:20]})
        else:
            self.pass_check("direction_v2", artifact="literature/direction.json", details={"direction_id": direction["direction_id"], "direction_hash": direction["direction_hash"]})

        self._schema_check("evidence_bundle", evidence, "evidence_bundle.schema.json", "literature/evidence_bundle.json")
        self._schema_check("direction_scorecard", scorecard, "direction_scorecard.schema.json", "literature/direction_scorecard.json")
        self._schema_check("novelty_audit", novelty, "novelty_audit.schema.json", "literature/novelty_audit.json")
        items = [item for item in evidence.get("items") or [] if isinstance(item, dict)]
        evidence_ids = {
            str(item.get("claim_id") or item.get("chunk_id") or item.get("source_path") or item.get("id") or "")
            for item in items
        }
        missing_claims = [claim_id for claim_id in [*direction.get("support_claim_ids", []), *direction.get("counter_claim_ids", [])] if claim_id not in evidence_ids]
        if missing_claims:
            self.retry_check("direction_claim_evidence", "DirectionSpec claim ids must resolve in evidence_bundle", details={"missing_claim_ids": missing_claims})
        else:
            self.pass_check("direction_claim_evidence", details={"resolved_claim_count": len(evidence_ids)})

        if not (manifest.get("papers") or (self.project_root / "literature" / "papers" / "metadata.json").exists()):
            self.retry_check("reference_material_registered", "no reference papers registered", artifact="references/papers/manifest.json")
        else:
            self.pass_check("reference_material_registered", artifact="references/papers/manifest.json")

        simulate = bool((self.config.get("experiment") or {}).get("simulate"))
        if novelty.get("enabled") is False and not bootstrap_profile_enabled(self.config) and not simulate:
            self.retry_check("novelty_gate", "novelty audit may only be disabled in bootstrap profile", artifact="literature/novelty_audit.json")
        elif novelty.get("passed") is not True:
            self.retry_check("novelty_gate", "novelty audit did not pass", artifact="literature/novelty_audit.json")
        else:
            self.pass_check("novelty_gate", artifact="literature/novelty_audit.json")

        if (self.project_root / "intake" / "c2c" / "static_bundle.json").exists():
            self._validate_c2c_quality(direction)
        return self.finalize()

    def _schema_check(self, name, payload, schema_name, artifact):
        errors = validate_min_schema(payload, load_schema(schema_name))
        if errors:
            self.retry_check(name, f"{artifact} schema validation failed", artifact=artifact, details={"errors": errors[:20]})
        else:
            self.pass_check(name, artifact=artifact)

    def _validate_c2c_quality(self, direction):
        artifacts = {
            "evidence_quality": ("literature/c2c/evidence_quality_score.json", "s1_evidence_quality.schema.json"),
            "retrieval_trace": ("literature/c2c/evidence_retrieval_trace.json", "s1_evidence_retrieval_trace.schema.json"),
            "direction_fingerprint": ("literature/c2c/direction_fingerprint.json", "s1_direction_fingerprint.schema.json"),
        }
        for name, (path, schema) in artifacts.items():
            if not self.require_file(path, check_name=f"{name}_exists"):
                continue
            payload = self.read_json_artifact(path)
            errors = validate_min_schema(payload, load_schema(schema)) if isinstance(payload, dict) else ["expected object"]
            if isinstance(payload, dict) and payload.get("direction_id") not in {None, direction["direction_id"]}:
                errors.append("direction_id mismatch")
            if name == "evidence_quality" and isinstance(payload, dict) and payload.get("gate") != "pass":
                errors.append("evidence quality gate did not pass")
            if errors:
                self.retry_check(name, f"{path} failed strict validation", artifact=path, details={"errors": errors[:20]})
            else:
                self.pass_check(name, artifact=path)
