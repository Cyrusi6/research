"""Ledger-backed exactly-once lifecycle for side-effecting phase commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contract_store import ContractStore, canonical_contract_bytes, contract_digest
from .phase_execution import AuthoritativePhaseContext, PhaseAuthorization, PhaseAuthority

PHASE_COMMAND_SCHEMA_VERSION = "auto_research_phase_command_v1"
PHASE_RUN_RECEIPT_SCHEMA_VERSION = "auto_research_phase_run_receipt_v2"


class CommandJournalError(RuntimeError):
    pass


class LedgerCommandAuthority(PhaseAuthority, Protocol):
    def start_phase_command(self, command: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def complete_phase_command(self, command_id: str, receipt_ref: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def mark_phase_command_unknown(self, command_id: str, reason: str) -> Mapping[str, Any]: ...
    def phase_command(self, command_id: str) -> Mapping[str, Any] | None: ...


@dataclass(frozen=True)
class CommandExecutionResult:
    exit_code: int
    stdout_hash: str
    stderr_hash: str
    outputs: tuple[Mapping[str, Any], ...]
    external_job_id: str | None = None


@dataclass(frozen=True)
class LedgerCommandJournal:
    """Uses ResearchEventLedger events; it owns no mutable command database."""

    project_root: Path
    ledger: LedgerCommandAuthority

    def run_once(
        self,
        context: AuthoritativePhaseContext,
        *,
        command_id: str,
        argv: Sequence[str],
        cwd: str,
        source_snapshot_hash: str,
        expected_outputs: Sequence[str],
        runner: Callable[[], CommandExecutionResult],
        recovered_receipt_ref: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        authorization = self._authorize(context)
        command = self._command_payload(
            context,
            authorization,
            command_id=command_id,
            argv=argv,
            cwd=cwd,
            source_snapshot_hash=source_snapshot_hash,
            expected_outputs=expected_outputs,
        )
        existing = self.ledger.phase_command(command_id)
        if existing is not None:
            self._assert_command_identity(existing, command)
            if existing.get("status") == "completed":
                return existing
            if recovered_receipt_ref is not None:
                return self.ledger.complete_phase_command(command_id, recovered_receipt_ref)
            if existing.get("status") in {"started", "unknown"}:
                if existing.get("status") == "started":
                    return self.ledger.mark_phase_command_unknown(
                        command_id,
                        "started command has no trustworthy receipt and cannot be silently rerun",
                    )
                return existing
            raise CommandJournalError("unknown authoritative command status")

        started = self.ledger.start_phase_command(command)
        if started.get("status") != "started":
            raise CommandJournalError("Ledger did not commit PhaseCommandStarted")
        self._authorize(context)
        try:
            result = runner()
        except BaseException:
            self.ledger.mark_phase_command_unknown(command_id, "runner exited before a trustworthy receipt was committed")
            raise
        if not isinstance(result, CommandExecutionResult):
            self.ledger.mark_phase_command_unknown(command_id, "runner returned no typed command result")
            raise CommandJournalError("runner must return CommandExecutionResult")
        receipt = self._receipt(command, started, result)
        receipt_ref = ContractStore(self.project_root).put_json(
            receipt,
            schema_file="phase_run_receipt_v2.schema.json",
        )
        return self.ledger.complete_phase_command(command_id, receipt_ref)

    def _authorize(self, context: AuthoritativePhaseContext) -> PhaseAuthorization:
        authorization = self.ledger.authorize_phase(context)
        if not isinstance(authorization, PhaseAuthorization):
            raise CommandJournalError("Ledger phase authorization is missing or malformed")
        if authorization.authorization_hash != context.authorization_hash:
            raise CommandJournalError("Ledger phase authorization changed before command execution")
        return authorization

    @staticmethod
    def _command_payload(
        context: AuthoritativePhaseContext,
        authorization: PhaseAuthorization,
        *,
        command_id: str,
        argv: Sequence[str],
        cwd: str,
        source_snapshot_hash: str,
        expected_outputs: Sequence[str],
    ) -> dict[str, Any]:
        if not command_id or not argv or any(not isinstance(item, str) or not item for item in argv):
            raise CommandJournalError("command identity and argv must be non-empty")
        command_plan = {
            "argv": list(argv),
            "cwd": cwd,
            "source_snapshot_hash": source_snapshot_hash,
            "expected_outputs": sorted(expected_outputs),
        }
        command_hash = contract_digest(canonical_contract_bytes(command_plan))
        return {
            "schema_version": PHASE_COMMAND_SCHEMA_VERSION,
            "command_id": command_id,
            "command_hash": command_hash,
            "command_plan": command_plan,
            "attempt_id": context.attempt_id,
            "lifecycle_generation": context.lifecycle_generation,
            "phase": context.phase,
            "phase_execution_id": context.phase_execution_id,
            "phase_start_event_id": context.phase_start_event_id,
            "producer_run_id": context.producer_run_id,
            "implementation_hash": context.implementation_hash,
            "attempt_input_hash": context.attempt_input_hash,
            "authorization_hash": authorization.authorization_hash,
            "provenance_mode": context.provenance_mode,
            "idempotency_key": contract_digest(canonical_contract_bytes({
                "command_id": command_id,
                "command_hash": command_hash,
                "attempt_id": context.attempt_id,
                "generation": context.lifecycle_generation,
                "phase_execution_id": context.phase_execution_id,
            })),
        }

    @staticmethod
    def _receipt(command: Mapping[str, Any], started: Mapping[str, Any], result: CommandExecutionResult) -> dict[str, Any]:
        started_event_id = started.get("event_id")
        started_event_hash = started.get("event_hash")
        started_at = started.get("created_at")
        if not all(isinstance(value, str) and value for value in (started_event_id, started_event_hash, started_at)):
            raise CommandJournalError("PhaseCommandStarted event identity is incomplete")
        return {
            "schema_version": PHASE_RUN_RECEIPT_SCHEMA_VERSION,
            "command_id": command["command_id"],
            "command_hash": command["command_hash"],
            "started_event_id": started_event_id,
            "started_event_hash": started_event_hash,
            "attempt_id": command["attempt_id"],
            "lifecycle_generation": command["lifecycle_generation"],
            "phase": command["phase"],
            "phase_execution_id": command["phase_execution_id"],
            "phase_start_event_id": command["phase_start_event_id"],
            "producer_run_id": command["producer_run_id"],
            "implementation_hash": command["implementation_hash"],
            "attempt_input_hash": command["attempt_input_hash"],
            "provenance_mode": command["provenance_mode"],
            "started_at": started_at,
            "completed_at": _now_utc(),
            "exit_code": result.exit_code,
            "stdout_hash": result.stdout_hash,
            "stderr_hash": result.stderr_hash,
            "external_job_id": result.external_job_id,
            "outputs": [dict(item) for item in result.outputs],
        }

    @staticmethod
    def _assert_command_identity(existing: Mapping[str, Any], command: Mapping[str, Any]) -> None:
        authoritative = existing.get("command") if isinstance(existing.get("command"), Mapping) else existing
        for field in (
            "command_id", "command_hash", "attempt_id", "lifecycle_generation", "phase",
            "phase_execution_id", "phase_start_event_id", "producer_run_id", "implementation_hash",
            "attempt_input_hash", "authorization_hash", "idempotency_key",
        ):
            if authoritative.get(field) != command.get(field):
                raise CommandJournalError(f"command replay conflicts on {field}")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "CommandExecutionResult", "CommandJournalError", "LedgerCommandAuthority", "LedgerCommandJournal",
    "PHASE_COMMAND_SCHEMA_VERSION", "PHASE_RUN_RECEIPT_SCHEMA_VERSION",
]
