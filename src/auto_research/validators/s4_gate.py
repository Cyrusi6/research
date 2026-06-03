"""S4 writing gate."""

from __future__ import annotations

from .base import StageGateValidator


class S4GateValidator(StageGateValidator):
    stage_key = "S4_writing"
    validator_name = "s4_writing_gate_v1"

    def validate(self):
        self.require_file("paper/main.tex", check_name="paper_main_tex_exists")
        audit_path = self.require_file("paper/claim_audit.json", check_name="claim_audit_exists")
        if not audit_path:
            return self.finalize()

        audit = self.read_json_artifact("paper/claim_audit.json")
        threshold = self.config.get("writing", {}).get("claim_verification", {}).get("min_pass_rate", 0.8)
        if isinstance(audit, dict) and float(audit.get("pass_rate") or 0) >= float(threshold):
            self.pass_check("claim_audit_pass_rate", artifact="paper/claim_audit.json", details={"pass_rate": audit.get("pass_rate"), "threshold": threshold})
        else:
            self.fail_check("claim_audit_pass_rate", "claim audit below threshold", artifact="paper/claim_audit.json", details={"threshold": threshold})

        compile_report_path = self.project_root / "paper" / "compile_report.json"
        if compile_report_path.exists():
            compile_report = self.read_json_artifact("paper/compile_report.json") or {}
            require_compile = self.config.get("writing", {}).get("require_compile", False)
            if require_compile and compile_report.get("status") != "ok":
                self.fail_check("latex_compile", "latex compile failed", artifact="paper/compile_report.json", details={"status": compile_report.get("status")})
            else:
                self.pass_check("latex_compile", artifact="paper/compile_report.json", details={"status": compile_report.get("status")})
        return self.finalize()
