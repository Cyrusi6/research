"""S3 experiment gate."""

from __future__ import annotations

import math

from .base import StageGateValidator, load_schema, validate_min_schema
from ..config import bootstrap_proxy_only_enabled
from ..utils import sha256_file


class S3GateValidator(StageGateValidator):
    stage_key = "S3_experiment"
    validator_name = "s3_experiment_gate_v1"

    def validate(self):
        required = [
            "experiment/results/main_results.json",
            "experiment/results/ablation_results.json",
            "experiment/results/hypothesis_verification.md",
        ]
        missing = [rel for rel in required if not (self.project_root / rel).exists()]
        if missing:
            self.retry_check("experiment_outputs_exist", f"missing experiment outputs: {', '.join(path.split('/')[-1] for path in missing)}", details={"missing": missing})
            return self.finalize()
        for rel in required:
            self.pass_check(f"{rel}_exists", artifact=rel)

        main_results = self.read_json_artifact("experiment/results/main_results.json")
        if not isinstance(main_results, dict):
            return self.finalize()
        self._validate_s3_candidate_selection(main_results)
        if bootstrap_proxy_only_enabled(self.config):
            self._validate_bootstrap_proxy_reached(main_results)
            return self.finalize()
        if self._requires_c2c_proxy_contracts(main_results):
            self._validate_c2c_proxy_contracts(main_results)
        if "candidate_results" in main_results and "baseline" in main_results:
            self._validate_c2c_acceptance(main_results)
        return self.finalize()

    def _validate_bootstrap_proxy_reached(self, main_results: dict) -> None:
        candidates = [item for item in main_results.get("candidate_results") or [] if isinstance(item, dict)]
        reached = []
        for candidate in candidates:
            proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
            metrics = proxy.get("metrics") if isinstance(proxy.get("metrics"), dict) else {}
            try:
                mean = float(metrics.get("mean"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(mean) or str(proxy.get("status") or "").strip().lower() in {"failed", "blocked", "resource_retry", "baseline_blocked"}:
                continue
            reached.append(
                {
                    "candidate_id": candidate.get("id"),
                    "mean": mean,
                    "proxy_status": proxy.get("status"),
                }
            )
        if not reached:
            self.retry_check(
                "bootstrap_proxy_reached",
                "Bootstrap profile requires at least one cheap proxy mean metric",
                artifact="experiment/results/main_results.json",
            )
            return
        self.pass_check(
            "bootstrap_proxy_reached",
            artifact="experiment/results/main_results.json",
            details={"candidates": reached},
        )

    def _validate_s3_candidate_selection(self, main_results: dict) -> None:
        selection_path = self.project_root / "experiment/results/s3_candidate_selection.json"
        if not selection_path.exists():
            return
        selection = self.read_json_artifact("experiment/results/s3_candidate_selection.json")
        if not isinstance(selection, dict):
            return
        selected_id = selection.get("selected_candidate_id")
        candidate_ids = [
            item.get("id")
            for item in main_results.get("candidate_results") or []
            if isinstance(item, dict) and item.get("id")
        ]
        if selected_id and candidate_ids and any(str(candidate_id) != str(selected_id) for candidate_id in candidate_ids):
            self.fail_check(
                "s3_candidate_selection_locked",
                "S3 candidate results are not locked to the selected patch_manifest candidate",
                artifact="experiment/results/s3_candidate_selection.json",
                details={
                    "selected_candidate_id": selected_id,
                    "candidate_ids": candidate_ids,
                },
            )
        else:
            self.pass_check(
                "s3_candidate_selection_locked",
                artifact="experiment/results/s3_candidate_selection.json",
                details={
                    "selected_candidate_id": selected_id,
                    "candidate_ids": candidate_ids,
                },
            )
        self._validate_s3_artifact_locks(selection)

    def _validate_s3_artifact_locks(self, selection: dict) -> None:
        locks = []
        lock_keys = [
            "patch_manifest",
            "selected_patch",
            "selected_patched_repo_snapshot_lock",
            "selected_implementation_contract",
            "selected_patch_gate_report",
            "selected_planner_gate_report",
            "selected_variant_scorecard",
        ]
        for key in lock_keys:
            lock = selection.get(key)
            if isinstance(lock, dict) and lock.get("rel_path") and lock.get("exists") is not False:
                locks.append((key, lock))
        mismatches = []
        for key, lock in locks:
            rel_path = str(lock.get("rel_path") or "")
            path = self.project_root / rel_path
            if not path.exists() or not path.is_file():
                mismatches.append({"name": key, "rel_path": rel_path, "reason": "missing"})
                continue
            actual = sha256_file(path)
            expected = lock.get("sha256")
            if expected and actual != expected:
                mismatches.append({"name": key, "rel_path": rel_path, "expected_sha256": expected, "actual_sha256": actual})
        if selection.get("selected_candidate_id"):
            for key in ["selected_patch_gate_report", "selected_planner_gate_report", "selected_variant_scorecard"]:
                lock = selection.get(key)
                if not isinstance(lock, dict) or not lock.get("rel_path"):
                    mismatches.append({"name": key, "reason": "missing_lock"})
                elif lock.get("exists") is False:
                    mismatches.append({"name": key, "rel_path": lock.get("rel_path"), "reason": "missing"})
        if mismatches:
            self.fail_check(
                "s3_s2_5_artifact_lock_sha256",
                "S3 S2.5 artifact lock changed after candidate selection",
                artifact="experiment/results/s3_candidate_selection.json",
                details={"mismatches": mismatches},
            )
        else:
            self.pass_check(
                "s3_s2_5_artifact_lock_sha256",
                artifact="experiment/results/s3_candidate_selection.json",
                details={"locked_artifacts": [key for key, _ in locks]},
            )

    def _requires_c2c_proxy_contracts(self, main_results: dict) -> bool:
        candidates = [item for item in main_results.get("candidate_results") or [] if isinstance(item, dict)]
        candidate_proxies = [
            candidate.get("proxy_screen")
            for candidate in candidates
            if isinstance(candidate.get("proxy_screen"), dict)
        ]
        if candidate_proxies:
            return any(proxy.get("enabled") is not False and proxy.get("status") != "skipped" for proxy in candidate_proxies)
        proxy_cfg = (((self.config.get("c2c") or {}).get("small_loop") or {}).get("proxy_screen") or {}) if isinstance(self.config.get("c2c"), dict) else {}
        if proxy_cfg.get("enabled") is True:
            return True
        for candidate in candidates:
            proxy = candidate.get("proxy_screen") if isinstance(candidate.get("proxy_screen"), dict) else {}
            if proxy and proxy.get("enabled") is not False and proxy.get("status") != "skipped":
                return True
        for rel in [
            "experiment/results/c2c_proxy_baseline_fingerprint.json",
            "experiment/results/c2c_proxy_decision_report.json",
            "experiment/results/c2c_effective_proxy_policy.json",
        ]:
            if (self.project_root / rel).exists():
                return True
        return False

    def _validate_c2c_proxy_contracts(self, main_results: dict) -> None:
        artifacts = {
            "baseline_fingerprint": (
                "experiment/results/c2c_proxy_baseline_fingerprint.json",
                "c2c_proxy_baseline_fingerprint.schema.json",
            ),
            "cache_report": (
                "experiment/results/c2c_proxy_cache_report.json",
                "c2c_proxy_cache_report.schema.json",
            ),
            "effective_policy": (
                "experiment/results/c2c_effective_proxy_policy.json",
                "c2c_effective_proxy_policy.schema.json",
            ),
            "decision_report": (
                "experiment/results/c2c_proxy_decision_report.json",
                "c2c_proxy_decision_report.schema.json",
            ),
            "calibration_policy": (
                "experiment/results/c2c_proxy_calibration_policy.json",
                "c2c_proxy_calibration_policy.schema.json",
            ),
        }
        payloads = {}
        missing = []
        schema_errors = {}
        for key, (rel, schema_name) in artifacts.items():
            path = self.require_file(rel, check_name=f"{key}_exists", retry=True)
            if path is None:
                missing.append(rel)
                continue
            payload = self.read_json_artifact(rel)
            payloads[key] = payload
            errors = validate_min_schema(payload, load_schema(schema_name)) if isinstance(payload, dict) else ["expected object"]
            if errors:
                schema_errors[key] = errors
        if missing:
            self.retry_check(
                "c2c_proxy_contract_artifacts_exist",
                "missing C2C proxy contract artifacts",
                details={"missing": missing},
            )
            return
        if schema_errors:
            self.retry_check(
                "c2c_proxy_contract_schema_valid",
                "C2C proxy contract schema validation failed",
                details={"errors": schema_errors},
            )
            return
        self.pass_check("c2c_proxy_contract_schema_valid", details={"artifacts": [rel for rel, _ in artifacts.values()]})

        fingerprint = payloads.get("baseline_fingerprint") or {}
        if not str(fingerprint.get("fingerprint_hash") or "").strip():
            self.retry_check(
                "proxy_baseline_fingerprint_valid",
                "C2C proxy baseline fingerprint_hash is missing",
                artifact="experiment/results/c2c_proxy_baseline_fingerprint.json",
            )
        else:
            self.pass_check("proxy_baseline_fingerprint_valid", artifact="experiment/results/c2c_proxy_baseline_fingerprint.json")

        cache_report = payloads.get("cache_report") or {}
        if cache_report.get("cache_status") not in {"hit", "miss", "invalidated", "missing"} or cache_report.get("action") not in {"reuse", "rerun_baseline", "block"}:
            self.retry_check(
                "proxy_cache_hit_or_rerun",
                "C2C proxy cache report has an invalid status/action",
                artifact="experiment/results/c2c_proxy_cache_report.json",
                details={"cache_report": cache_report},
            )
        elif cache_report.get("action") == "block":
            self.retry_check(
                "proxy_cache_hit_or_rerun",
                "C2C proxy baseline cache is blocked",
                artifact="experiment/results/c2c_proxy_cache_report.json",
                details={"cache_report": cache_report},
            )
        else:
            self.pass_check("proxy_cache_hit_or_rerun", artifact="experiment/results/c2c_proxy_cache_report.json", details={"cache_report": cache_report})

        policy = payloads.get("effective_policy") or {}
        effective = policy.get("effective_policy") if isinstance(policy.get("effective_policy"), dict) else {}
        if effective.get("min_proxy_mean_delta") is None or effective.get("max_proxy_dataset_regression") is None:
            self.retry_check(
                "proxy_policy_schema_valid",
                "effective proxy policy is missing required thresholds",
                artifact="experiment/results/c2c_effective_proxy_policy.json",
                details={"effective_policy": effective},
            )
        else:
            self.pass_check("proxy_policy_schema_valid", artifact="experiment/results/c2c_effective_proxy_policy.json", details={"policy_hash": policy.get("policy_hash")})

        decision = payloads.get("decision_report") or {}
        if decision.get("effective_policy_hash") != policy.get("policy_hash"):
            self.fail_check(
                "proxy_decision_uses_effective_policy_hash",
                "C2C proxy decision does not reference the effective proxy policy hash",
                artifact="experiment/results/c2c_proxy_decision_report.json",
                details={"decision_hash": decision.get("effective_policy_hash"), "policy_hash": policy.get("policy_hash")},
            )
        else:
            self.pass_check("proxy_decision_uses_effective_policy_hash", artifact="experiment/results/c2c_proxy_decision_report.json")

        self._validate_proxy_decision_metrics(decision)
        self._validate_neutral_proxy_worthiness(decision, effective)

        selected_decision = decision.get("decision")
        if selected_decision in {"proxy_rejected", "proxy_repairable", "blocked"}:
            self.retry_check(
                selected_decision,
                f"C2C proxy decision routed before full S3: {selected_decision}",
                artifact="experiment/results/c2c_proxy_decision_report.json",
                details={"route_hint": decision.get("route_hint"), "failure_class": decision.get("failure_class"), "reason_codes": decision.get("reason_codes") or []},
            )
        elif selected_decision in {"proxy_pass", "neutral_proxy_full_s3"}:
            self.pass_check(
                "proxy_decision_allows_full_s3",
                artifact="experiment/results/c2c_proxy_decision_report.json",
                details={"decision": selected_decision, "route_hint": decision.get("route_hint")},
            )
        else:
            self.retry_check(
                "proxy_decision_enum_valid",
                "C2C proxy decision is not a recognized routing decision",
                artifact="experiment/results/c2c_proxy_decision_report.json",
                details={"decision": selected_decision},
            )

    def _validate_proxy_decision_metrics(self, decision: dict) -> None:
        proxy_metrics = decision.get("proxy_metrics") if isinstance(decision.get("proxy_metrics"), dict) else {}
        baseline_metrics = decision.get("paired_baseline_metrics") if isinstance(decision.get("paired_baseline_metrics"), dict) else {}
        deltas = decision.get("deltas") if isinstance(decision.get("deltas"), dict) else {}
        if proxy_metrics.get("mean") is not None and baseline_metrics.get("mean") is not None:
            expected = round(float(proxy_metrics["mean"]) - float(baseline_metrics["mean"]), 4)
            reported = deltas.get("mean_delta")
            if reported is None or abs(float(reported) - expected) > 1e-4:
                self.fail_check(
                    "proxy_decision_matches_metrics",
                    "C2C proxy decision mean_delta does not match proxy and paired baseline metrics",
                    artifact="experiment/results/c2c_proxy_decision_report.json",
                    details={"expected_mean_delta": expected, "reported_mean_delta": reported},
                )
                return
        dataset_deltas = deltas.get("dataset_deltas") if isinstance(deltas.get("dataset_deltas"), dict) else {}
        if dataset_deltas:
            worst = max(max(0.0, -float(value)) for value in dataset_deltas.values())
            reported_worst = float(deltas.get("worst_dataset_regression") or 0.0)
            if abs(worst - reported_worst) > 1e-4:
                self.fail_check(
                    "proxy_decision_matches_metrics",
                    "C2C proxy decision worst_dataset_regression does not match dataset deltas",
                    artifact="experiment/results/c2c_proxy_decision_report.json",
                    details={"expected_worst_dataset_regression": worst, "reported_worst_dataset_regression": reported_worst},
                )
                return
        self.pass_check("proxy_decision_matches_metrics", artifact="experiment/results/c2c_proxy_decision_report.json")

    def _validate_neutral_proxy_worthiness(self, decision: dict, effective_policy: dict) -> None:
        if decision.get("decision") != "neutral_proxy_full_s3":
            return
        worthiness_rel = "experiment/results/c2c_full_s3_worthiness.json"
        path = self.require_file(worthiness_rel, check_name="neutral_proxy_requires_worthiness_score", retry=True)
        if path is None:
            return
        worthiness = self.read_json_artifact(worthiness_rel)
        errors = validate_min_schema(worthiness, load_schema("c2c_full_s3_worthiness.schema.json")) if isinstance(worthiness, dict) else ["expected object"]
        if errors:
            self.retry_check(
                "neutral_proxy_requires_worthiness_score",
                "neutral proxy full S3 worthiness schema validation failed",
                artifact=worthiness_rel,
                details={"errors": errors},
            )
            return
        score = float((worthiness or {}).get("score") or 0.0)
        threshold = float(effective_policy.get("full_s3_worthiness_min_score", (worthiness or {}).get("threshold", 0.60)) or 0.0)
        if score < threshold or (worthiness or {}).get("decision") != "run_full_s3":
            self.retry_check(
                "full_s3_worthiness_threshold_passed_if_neutral",
                "neutral proxy is not worthy of full S3 budget",
                artifact=worthiness_rel,
                details={"score": score, "threshold": threshold, "worthiness_decision": (worthiness or {}).get("decision")},
            )
        else:
            self.pass_check(
                "full_s3_worthiness_threshold_passed_if_neutral",
                artifact=worthiness_rel,
                details={"score": score, "threshold": threshold},
            )

    def _validate_c2c_acceptance(self, main_results: dict) -> None:
        acceptance = main_results.get("acceptance") or {}
        if acceptance and not acceptance.get("passed"):
            self.fail_check(
                "c2c_acceptance_passed",
                f"C2C candidate did not clear acceptance: {acceptance.get('reason', 'unknown')}",
                artifact="experiment/results/main_results.json",
                details={"acceptance": acceptance},
            )
            return
        self.pass_check("c2c_acceptance_passed", artifact="experiment/results/main_results.json", details={"acceptance": acceptance})

        baseline = main_results.get("baseline") or {}
        best = main_results.get("best_candidate") or {}
        metrics = best.get("metrics") or {}
        if metrics.get("mean") is None:
            self.fail_check("c2c_best_candidate_mean_metric", "C2C best candidate has no mean metric", artifact="experiment/results/main_results.json")
            return
        self.pass_check("c2c_best_candidate_mean_metric", artifact="experiment/results/main_results.json", details={"mean": metrics.get("mean")})

        baseline_mean = float(baseline.get("mean") or acceptance.get("baseline_mean") or 0.0)
        min_delta = float(acceptance.get("min_delta_to_pass", best.get("acceptance_rule", {}).get("min_delta_to_pass", 0.0)))
        if float(metrics["mean"]) < baseline_mean + min_delta:
            self.fail_check(
                "c2c_baseline_delta",
                "C2C best candidate did not exceed baseline threshold",
                artifact="experiment/results/main_results.json",
                details={"mean": metrics["mean"], "baseline_mean": baseline_mean, "min_delta_to_pass": min_delta},
            )
        else:
            self.pass_check("c2c_baseline_delta", artifact="experiment/results/main_results.json")

        max_regression = float(acceptance.get("max_dataset_regression", best.get("acceptance_rule", {}).get("max_dataset_regression", 999.0)))
        worst_regression = float(best.get("worst_dataset_regression") or 0.0)
        if worst_regression > max_regression:
            self.fail_check(
                "c2c_dataset_regression",
                "C2C best candidate has excessive per-dataset regression",
                artifact="experiment/results/main_results.json",
                details={"worst_dataset_regression": worst_regression, "max_dataset_regression": max_regression},
            )
        else:
            self.pass_check("c2c_dataset_regression", artifact="experiment/results/main_results.json")
