"""S3 experiment gate."""

from __future__ import annotations

from .base import StageGateValidator


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
        if "candidate_results" in main_results and "baseline" in main_results:
            self._validate_c2c_acceptance(main_results)
        return self.finalize()

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
