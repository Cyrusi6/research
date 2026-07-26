"""Ledger-backed exactly-once lifecycle for side-effecting phase commands."""

from __future__ import annotations

import json
import os
import re
import stat
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

from .contract_store import ContractStore, canonical_contract_bytes, contract_digest
from .domain_contracts import PHASE_COMMAND_SCHEMA_VERSION, PHASE_RUN_RECEIPT_SCHEMA_VERSION
from .phase_command_plan import validate_phase_command_plan
from .phase_receipts import validate_phase_run_receipt
from .phase_execution import (
    AuthoritativePhaseContext,
    PhaseArtifactInventory,
    PhaseAuthorization,
    PhaseAuthority,
)

PHASE_COMMAND_SCHEMA_FILE = "phase_command_v5.schema.json"
PHASE_RUN_RECEIPT_SCHEMA_FILE = "phase_run_receipt_v5.schema.json"
COMMAND_RECEIPT_LOCATOR_SCHEMA_VERSION = "auto_research_command_receipt_locator_v1"
_COMMAND_LOCATOR_ROOT = Path("meta") / "command_receipts"
_SAFE_COMMAND_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


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
    derivation_ref: Mapping[str, Any] | None = None
    derivation_hash: str | None = None
    raw_outputs: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    external_job_id: str | None = None
    stdout: str | None = None
    stderr: str | None = None

    @property
    def status(self) -> str:
        return "ok" if self.exit_code == 0 else "failed"

    def as_runner_result(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": self.status,
            "returncode": self.exit_code,
            "stdout": self.stdout or "",
            "stderr": self.stderr or "",
        }
        if self.external_job_id is not None:
            result["external_job_id"] = self.external_job_id
        return result


class CommandJournalResult(dict[str, Any]):
    """Ledger operation result plus receipt-recovered runtime projections."""

    execution_result: CommandExecutionResult
    artifact_inventory: PhaseArtifactInventory
    receipt: Mapping[str, Any]
    receipt_ref: Mapping[str, Any]

    def __init__(
        self,
        operation: Mapping[str, Any],
        *,
        execution_result: CommandExecutionResult,
        artifact_inventory: PhaseArtifactInventory,
        receipt: Mapping[str, Any],
        receipt_ref: Mapping[str, Any],
    ) -> None:
        super().__init__(operation)
        self.execution_result = execution_result
        self.artifact_inventory = artifact_inventory
        self.receipt = dict(receipt)
        self.receipt_ref = dict(receipt_ref)


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
        command_spec_id: str | None = None,
        argv: Sequence[str],
        cwd: str,
        source_snapshot_hash: str,
        expected_outputs: Sequence[str],
        environment: Mapping[str, str] | None = None,
        inherited_environment: Sequence[str] | None = None,
        runner: Callable[[], CommandExecutionResult],
        retry_policy: Mapping[str, Any] | None = None,
        resource_policy: Mapping[str, Any] | None = None,
        resume_policy: Mapping[str, Any] | None = None,
        recovered_receipt_ref: Mapping[str, Any] | None = None,
    ) -> CommandJournalResult | Mapping[str, Any]:
        authorization = self._authorize(context)
        command = self._command_payload(
            context,
            authorization,
            command_id=command_id,
            command_spec_id=command_spec_id,
            argv=argv,
            cwd=cwd,
            source_snapshot_hash=source_snapshot_hash,
            expected_outputs=expected_outputs,
            environment=environment,
            inherited_environment=inherited_environment,
            retry_policy=retry_policy,
            resource_policy=resource_policy,
            resume_policy=resume_policy,
        )
        existing = self.ledger.phase_command(command_id)
        if existing is not None:
            self._assert_command_identity(existing, command)
            status = existing.get("status")
            if status == "completed":
                return self._recover_completed(context, command, existing)
            receipt_ref = recovered_receipt_ref or self._read_receipt_locator(command)
            if receipt_ref is not None:
                if status != "started":
                    raise CommandJournalError("durable receipt cannot reconcile a non-started command")
                self._validate_receipt_ref(command, existing, receipt_ref)
                completed = self.ledger.complete_phase_command(command_id, receipt_ref)
                self._authorize(context)
                return self._recover_completed(context, command, completed)
            if status in {"started", "unknown"}:
                if status == "started":
                    return self.ledger.mark_phase_command_unknown(
                        command_id,
                        "started command has no trustworthy receipt and cannot be silently rerun",
                    )
                return existing
            raise CommandJournalError("unknown authoritative command status")

        started = self.ledger.start_phase_command(command)
        if started.get("status") != "started":
            raise CommandJournalError("Ledger did not commit PhaseCommandStarted")
        try:
            result = runner()
        except BaseException:
            self.ledger.mark_phase_command_unknown(command_id, "runner exited before a trustworthy receipt was committed")
            raise
        self._authorize(context)
        if not isinstance(result, CommandExecutionResult):
            self.ledger.mark_phase_command_unknown(command_id, "runner returned no typed command result")
            raise CommandJournalError("runner must return CommandExecutionResult")
        store = ContractStore(self.project_root)
        for output in result.outputs:
            reference = output.get("contract_ref") if isinstance(output.get("contract_ref"), Mapping) else output
            try:
                store.verify(reference)
            except (OSError, TypeError, ValueError) as error:
                self.ledger.mark_phase_command_unknown(command_id, "runner output is not an immutable contract")
                raise CommandJournalError(f"runner output reference is invalid: {error}") from error
        self._validate_result_derivation(store, result)
        receipt = self._receipt(command, started, result, store=store)
        receipt_ref = store.put_json(
            receipt,
            schema_file=PHASE_RUN_RECEIPT_SCHEMA_FILE,
        )
        self._write_receipt_locator(command, receipt_ref)
        completed = self.ledger.complete_phase_command(command_id, receipt_ref)
        return self._recover_completed(context, command, completed)

    def recover_completed(
        self,
        context: AuthoritativePhaseContext,
        command_id: str,
    ) -> CommandJournalResult:
        command_record = self.ledger.phase_command(command_id)
        if not isinstance(command_record, Mapping) or command_record.get("status") != "completed":
            raise CommandJournalError("command is not authoritatively completed")
        command = command_record.get("command")
        if not isinstance(command, Mapping):
            raise CommandJournalError("completed command identity is missing")
        self._authorize(context)
        return self._recover_completed(context, command, command_record)

    def reconcile_started(
        self,
        context: AuthoritativePhaseContext,
        command_id: str,
    ) -> CommandJournalResult | Mapping[str, Any]:
        """Reconcile one previously-started command without invoking its runner."""

        record = self.ledger.phase_command(command_id)
        if not isinstance(record, Mapping):
            raise CommandJournalError("started command projection is unavailable")
        if record.get("status") == "completed":
            return self.recover_completed(context, command_id)
        if record.get("status") == "unknown":
            return record
        if record.get("status") != "started":
            raise CommandJournalError("only a started command can be reconciled")
        command = record.get("command")
        if not isinstance(command, Mapping):
            raise CommandJournalError("started command identity is missing")
        self._assert_command_matches_context(command, context)
        self._authorize(context)
        receipt_ref = self._read_receipt_locator(command)
        if receipt_ref is None:
            return self.ledger.mark_phase_command_unknown(
                command_id,
                "started command has no trustworthy receipt and cannot be silently rerun",
            )
        self._validate_receipt_ref(command, record, receipt_ref)
        completed = self.ledger.complete_phase_command(command_id, receipt_ref)
        self._authorize(context)
        return self._recover_completed(context, command, completed)

    def _recover_completed(
        self,
        context: AuthoritativePhaseContext,
        command: Mapping[str, Any],
        operation: Mapping[str, Any],
    ) -> CommandJournalResult:
        self._assert_command_matches_context(command, context)
        receipt_ref = operation.get("receipt_ref") or operation.get("receipt")
        if not isinstance(receipt_ref, Mapping):
            raise CommandJournalError("completed command is missing its durable receipt reference")
        receipt = self._validate_receipt_ref(command, operation, receipt_ref)
        execution_result = self._execution_result_from_receipt(receipt)
        inventory = self._inventory_from_receipt(context, command, receipt, receipt_ref)
        return CommandJournalResult(
            operation,
            execution_result=execution_result,
            artifact_inventory=inventory,
            receipt=receipt,
            receipt_ref=receipt_ref,
        )

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
        command_spec_id: str | None = None,
        argv: Sequence[str],
        cwd: str,
        source_snapshot_hash: str,
        expected_outputs: Sequence[str],
        environment: Mapping[str, str] | None = None,
        inherited_environment: Sequence[str] | None = None,
        retry_policy: Mapping[str, Any] | None = None,
        resource_policy: Mapping[str, Any] | None = None,
        resume_policy: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not _SAFE_COMMAND_ID.fullmatch(command_id):
            raise CommandJournalError("command_id must be a safe identifier")
        if not argv or any(not isinstance(item, str) or not item for item in argv):
            raise CommandJournalError("command argv must be non-empty")
        store = ContractStore(context.project_root)
        try:
            frozen_plan = store.read_json(
                context.command_plan_hash,
                schema_file="phase_command_plan_v4.schema.json",
            )
            validate_phase_command_plan(
                frozen_plan,
                expected_evidence_kinds=context.expected_evidence_kinds,
            )
            command_plan_blob = store.verify(context.command_plan_hash)
        except (OSError, TypeError, ValueError) as error:
            raise CommandJournalError(f"frozen PhaseCommandPlan is unavailable or invalid: {error}") from error
        if frozen_plan["phase"] != context.phase:
            raise CommandJournalError("frozen PhaseCommandPlan phase differs from authorization")
        requested_output_kinds = sorted(expected_outputs)
        candidates = [
            item
            for item in frozen_plan["commands"]
            if (command_spec_id is None or item["command_spec_id"] == command_spec_id)
            and item["argv"] == list(argv)
            and item["cwd"] == cwd
            and item["environment"] == dict(environment or {})
            and item["inherited_environment"] == sorted(inherited_environment or ["HOME", "PATH", "TMPDIR"])
            and (
                not requested_output_kinds
                or sorted(output["kind"] for output in item["expected_outputs"]) == requested_output_kinds
            )
        ]
        if len(candidates) != 1:
            raise CommandJournalError("command does not exactly match one frozen PhaseCommandPlan entry")
        command_spec = dict(candidates[0])
        frozen_source_hash = command_spec["source_snapshot_hash"]
        if source_snapshot_hash != frozen_source_hash:
            raise CommandJournalError("command source snapshot differs from the frozen plan")
        LedgerCommandJournal._validate_authorized_cwd(context.project_root, cwd)
        for supplied, field_name in (
            (retry_policy, "retry_policy"),
            (resource_policy, "resource_policy"),
            (resume_policy, "resume_policy"),
        ):
            if supplied is not None and dict(supplied) != command_spec[field_name]:
                raise CommandJournalError(f"command {field_name} differs from the frozen plan")
        command_hash = contract_digest(canonical_contract_bytes(command_spec))
        command_plan_ref = dict(command_plan_blob)
        return {
            "schema_version": PHASE_COMMAND_SCHEMA_VERSION,
            "command_id": command_id,
            "command_hash": command_hash,
            "command_spec_id": command_spec["command_spec_id"],
            "command_spec": command_spec,
            "command_plan_hash": context.command_plan_hash,
            "command_plan_ref": command_plan_ref,
            "attempt_id": context.attempt_id,
            "lifecycle_generation": context.lifecycle_generation,
            "phase": context.phase,
            "phase_execution_id": context.phase_execution_id,
            "phase_start_event_id": context.phase_start_event_id,
            "producer_run_id": context.producer_run_id,
            "implementation_hash": context.implementation_hash,
            "attempt_input_hash": context.attempt_input_hash,
            "authorization_hash": authorization.authorization_hash,
            "provenance_mode": context.provenance_mode.replace("_", "-"),
            "idempotency_key": contract_digest(canonical_contract_bytes({
                "command_id": command_id,
                "command_hash": command_hash,
                "command_plan_hash": context.command_plan_hash,
                "attempt_id": context.attempt_id,
                "generation": context.lifecycle_generation,
                "phase_execution_id": context.phase_execution_id,
            })),
        }

    @staticmethod
    def _validate_authorized_cwd(project_root: Path, cwd: str) -> None:
        candidate = Path(cwd)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        try:
            candidate = candidate.absolute()
        except OSError as error:
            raise CommandJournalError(f"command cwd is unavailable: {error}") from error
        try:
            candidate.relative_to(project_root.absolute())
        except ValueError as error:
            raise CommandJournalError("command cwd must remain inside the authoritative project root") from error
        current = Path(candidate.anchor)
        for component in candidate.parts[1:]:
            current /= component
            try:
                mode = current.lstat().st_mode
            except FileNotFoundError:
                raise CommandJournalError("command cwd must exist before PhaseCommandStarted")
            if stat.S_ISLNK(mode):
                raise CommandJournalError("command cwd contains a symlink component")
        if not candidate.is_dir():
            raise CommandJournalError("command cwd must be an existing directory")

    @staticmethod
    def _receipt(
        command: Mapping[str, Any],
        started: Mapping[str, Any],
        result: CommandExecutionResult,
        *,
        store: ContractStore,
    ) -> dict[str, Any]:
        started_event_id = started.get("event_id") or started.get("started_event_id")
        started_event_hash = started.get("event_hash") or started.get("started_event_hash")
        started_at = started.get("created_at")
        if not all(isinstance(value, str) and value for value in (started_event_id, started_event_hash, started_at)):
            raise CommandJournalError("PhaseCommandStarted event identity is incomplete")
        stdout = (result.stdout or "").encode("utf-8")
        stderr = (result.stderr or "").encode("utf-8")
        stdout_ref = store.put_bytes(stdout)
        stderr_ref = store.put_bytes(stderr)
        if stdout_ref["digest"] != result.stdout_hash or stderr_ref["digest"] != result.stderr_hash:
            raise CommandJournalError("command result log hashes do not match durable log bytes")
        return {
            "schema_version": PHASE_RUN_RECEIPT_SCHEMA_VERSION,
            "command_id": command["command_id"],
            "command_hash": command["command_hash"],
            "command_spec_id": command["command_spec_id"],
            "command_plan_hash": command["command_plan_hash"],
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
            "receipt_locator": str(
                Path("meta") / "contracts" / "sha256" / command["command_hash"][:2] / f"{command['command_hash']}.json"
            ),
            "started_at": started_at,
            "completed_at": _now_utc(),
            "exit_code": result.exit_code,
            "stdout_hash": result.stdout_hash,
            "stderr_hash": result.stderr_hash,
            "stdout_ref": stdout_ref,
            "stderr_ref": stderr_ref,
            "external_job_id": result.external_job_id,
            "outputs": LedgerCommandJournal._receipt_outputs(command, result),
            "raw_outputs": LedgerCommandJournal._receipt_raw_outputs(command, result),
            "derivation_ref": dict(result.derivation_ref) if result.derivation_ref is not None else None,
            "derivation_hash": result.derivation_hash,
        }

    @staticmethod
    def _validate_result_derivation(store: ContractStore, result: CommandExecutionResult) -> None:
        reference = result.derivation_ref
        digest = result.derivation_hash
        if (reference is None) != (digest is None):
            raise CommandJournalError("command derivation reference and hash must be supplied together")
        if reference is None:
            return
        if not isinstance(reference, Mapping) or not isinstance(digest, str):
            raise CommandJournalError("command derivation identity is malformed")
        try:
            verified = store.verify(reference)
        except (OSError, TypeError, ValueError) as error:
            raise CommandJournalError(f"command derivation reference is invalid: {error}") from error
        if verified["digest"] != digest:
            raise CommandJournalError("command derivation hash differs from its immutable reference")

    def _validate_receipt_ref(
        self,
        command: Mapping[str, Any],
        record: Mapping[str, Any],
        receipt_ref: Mapping[str, Any],
    ) -> dict[str, Any]:
        del command
        try:
            return validate_phase_run_receipt(self.project_root, record, receipt_ref)
        except (OSError, TypeError, ValueError) as error:
            raise CommandJournalError(f"durable command receipt is invalid: {error}") from error

    @staticmethod
    def _receipt_outputs(
        command: Mapping[str, Any],
        result: CommandExecutionResult,
    ) -> list[dict[str, Any]]:
        expected_outputs = tuple(command["command_spec"]["expected_outputs"])
        if result.exit_code != 0 and not result.outputs:
            return []
        if len(result.outputs) != len(expected_outputs):
            raise CommandJournalError("command result output count differs from frozen command spec")
        normalized: list[dict[str, Any]] = []
        for expected, supplied in zip(expected_outputs, result.outputs, strict=True):
            if "contract_ref" in supplied:
                reference = supplied["contract_ref"]
                kind = supplied.get("kind")
                schema_version = supplied.get("schema_version")
            else:
                reference = supplied
                kind = expected["kind"]
                schema_version = expected["schema_version"]
            if supplied.get("output_id", expected["output_id"]) != expected["output_id"]:
                raise CommandJournalError("command result output_id differs from frozen command spec")
            if kind != expected["kind"] or schema_version != expected["schema_version"]:
                raise CommandJournalError("command result output kind/schema differs from frozen command spec")
            normalized.append({
                "output_id": expected["output_id"],
                "kind": kind,
                "schema_version": schema_version,
                "content_hash": reference["digest"],
                "contract_ref": dict(reference),
                "producer_run_id": command["producer_run_id"],
                "phase": command["phase"],
                "lifecycle_generation": command["lifecycle_generation"],
            })
        return normalized

    @staticmethod
    def _receipt_raw_outputs(
        command: Mapping[str, Any],
        result: CommandExecutionResult,
    ) -> list[dict[str, Any]]:
        if result.exit_code != 0:
            if result.raw_outputs:
                raise CommandJournalError("failed command cannot publish scientific raw outputs")
            return []
        normalized: list[dict[str, Any]] = []
        seen: set[str] = set()
        for supplied in result.raw_outputs:
            output_id = str(supplied.get("output_id") or "")
            if not output_id or output_id in seen:
                raise CommandJournalError("command raw output_id must be unique and non-empty")
            seen.add(output_id)
            reference = supplied.get("contract_ref")
            if not isinstance(reference, Mapping):
                raise CommandJournalError("command raw output requires an immutable ContractRef")
            normalized.append({
                "output_id": output_id,
                "kind": str(supplied.get("kind") or ""),
                "schema_version": str(supplied.get("schema_version") or ""),
                "content_hash": str(reference.get("digest") or ""),
                "contract_ref": dict(reference),
                "producer_run_id": command["producer_run_id"],
                "phase": command["phase"],
                "lifecycle_generation": command["lifecycle_generation"],
                "command_spec_id": command["command_spec_id"],
                "locator": str(supplied.get("locator") or ""),
                "locator_type": str(supplied.get("locator_type") or ""),
                "dataset_id": supplied.get("dataset_id"),
                "role": supplied.get("role"),
            })
        return normalized

    def _execution_result_from_receipt(self, receipt: Mapping[str, Any]) -> CommandExecutionResult:
        store = ContractStore(self.project_root)
        stdout_raw = store.read_bytes(receipt["stdout_ref"])
        stderr_raw = store.read_bytes(receipt["stderr_ref"])
        if contract_digest(stdout_raw) != receipt["stdout_hash"] or contract_digest(stderr_raw) != receipt["stderr_hash"]:
            raise CommandJournalError("durable command log hash mismatch")
        return CommandExecutionResult(
            exit_code=int(receipt["exit_code"]),
            stdout_hash=str(receipt["stdout_hash"]),
            stderr_hash=str(receipt["stderr_hash"]),
            outputs=tuple(dict(item) for item in receipt["outputs"]),
            derivation_ref=(
                dict(receipt["derivation_ref"])
                if isinstance(receipt.get("derivation_ref"), Mapping)
                else None
            ),
            derivation_hash=(
                str(receipt["derivation_hash"])
                if isinstance(receipt.get("derivation_hash"), str)
                else None
            ),
            raw_outputs=tuple(dict(item) for item in receipt.get("raw_outputs") or []),
            external_job_id=str(receipt["external_job_id"]) if receipt.get("external_job_id") else None,
            stdout=stdout_raw.decode("utf-8"),
            stderr=stderr_raw.decode("utf-8"),
        )

    @staticmethod
    def _inventory_from_receipt(
        context: AuthoritativePhaseContext,
        command: Mapping[str, Any],
        receipt: Mapping[str, Any],
        receipt_ref: Mapping[str, Any],
    ) -> PhaseArtifactInventory:
        artifacts = []
        for output in receipt["outputs"]:
            if output["kind"] == "evidence_derivation_manifest":
                continue
            reference = output["contract_ref"]
            artifacts.append({
                "kind": output["kind"],
                "source_path": str(reference["relative_path"]),
                "content_hash": str(reference["digest"]),
                "receipt_hash": str(receipt_ref["digest"]),
                "producer_run_id": context.producer_run_id,
            })
        complete = {item["kind"] for item in artifacts} == set(context.expected_evidence_kinds)
        return PhaseArtifactInventory(context=context, artifacts=tuple(artifacts), complete=complete)

    def _locator_path(self, command: Mapping[str, Any]) -> Path:
        digest = contract_digest(str(command["command_id"]).encode("utf-8"))
        return self.project_root / _COMMAND_LOCATOR_ROOT / f"{digest}.json"

    def _write_receipt_locator(self, command: Mapping[str, Any], receipt_ref: Mapping[str, Any]) -> None:
        payload = {
            "schema_version": COMMAND_RECEIPT_LOCATOR_SCHEMA_VERSION,
            "command_id": command["command_id"],
            "command_hash": command["command_hash"],
            "receipt_ref": dict(receipt_ref),
        }
        path = self._locator_path(command)
        directory_fd = self._open_locator_directory(create=True)
        temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
        raw = canonical_contract_bytes(payload)
        try:
            file_fd = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory_fd,
            )
            try:
                with os.fdopen(file_fd, "wb", closefd=False) as stream:
                    stream.write(raw)
                    stream.flush()
                os.fsync(file_fd)
            finally:
                os.close(file_fd)
            os.replace(temporary_name, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
            os.close(directory_fd)

    def _read_receipt_locator(self, command: Mapping[str, Any]) -> Mapping[str, Any] | None:
        path = self._locator_path(command)
        try:
            directory_fd = self._open_locator_directory(create=False)
        except FileNotFoundError:
            return None
        try:
            file_fd = os.open(path.name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=directory_fd)
        except FileNotFoundError:
            os.close(directory_fd)
            return None
        except OSError as error:
            os.close(directory_fd)
            raise CommandJournalError(f"durable receipt locator is unavailable: {error}") from error
        try:
            metadata = os.fstat(file_fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise CommandJournalError("durable receipt locator is not a regular file")
            chunks = []
            while True:
                chunk = os.read(file_fd, 65536)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(file_fd)
            os.close(directory_fd)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CommandJournalError("durable receipt locator is malformed") from error
        if not isinstance(payload, dict) or canonical_contract_bytes(payload) != raw:
            raise CommandJournalError("durable receipt locator is not canonical")
        if set(payload) != {"schema_version", "command_id", "command_hash", "receipt_ref"}:
            raise CommandJournalError("durable receipt locator fields are invalid")
        if payload["schema_version"] != COMMAND_RECEIPT_LOCATOR_SCHEMA_VERSION:
            raise CommandJournalError("durable receipt locator schema is unsupported")
        if payload["command_id"] != command["command_id"] or payload["command_hash"] != command["command_hash"]:
            raise CommandJournalError("durable receipt locator command identity mismatch")
        if not isinstance(payload["receipt_ref"], Mapping):
            raise CommandJournalError("durable receipt locator reference is invalid")
        return payload["receipt_ref"]

    def _open_locator_directory(self, *, create: bool) -> int:
        root_fd = os.open(self.project_root, os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0))
        current_fd = root_fd
        try:
            for component in _COMMAND_LOCATOR_ROOT.parts:
                if create:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=current_fd)
                    except FileExistsError:
                        pass
                next_fd = os.open(
                    component,
                    os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=current_fd,
                )
                if current_fd != root_fd:
                    os.close(current_fd)
                current_fd = next_fd
            os.close(root_fd)
            return current_fd
        except BaseException:
            if current_fd != root_fd:
                os.close(current_fd)
            os.close(root_fd)
            raise

    @staticmethod
    def _assert_command_matches_context(command: Mapping[str, Any], context: AuthoritativePhaseContext) -> None:
        for field_name in (
            "attempt_id", "lifecycle_generation", "phase", "phase_execution_id", "phase_start_event_id",
            "producer_run_id", "implementation_hash", "attempt_input_hash", "provenance_mode",
        ):
            expected = getattr(context, field_name)
            if field_name == "provenance_mode":
                expected = expected.replace("_", "-")
            if command.get(field_name) != expected:
                raise CommandJournalError(f"completed command {field_name} differs from current phase context")
        if command.get("authorization_hash") != context.authorization_hash:
            raise CommandJournalError("completed command authorization differs from current phase context")

    @staticmethod
    def _assert_command_identity(existing: Mapping[str, Any], command: Mapping[str, Any]) -> None:
        authoritative = existing.get("command") if isinstance(existing.get("command"), Mapping) else existing
        for field_name in (
            "command_id", "command_hash", "attempt_id", "lifecycle_generation", "phase", "phase_execution_id",
            "phase_start_event_id", "producer_run_id", "implementation_hash", "attempt_input_hash",
            "authorization_hash", "provenance_mode", "idempotency_key",
            "command_spec_id", "command_plan_hash",
        ):
            if authoritative.get(field_name) != command.get(field_name):
                raise CommandJournalError(f"command replay conflicts on {field_name}")
        if authoritative.get("command_spec") != command.get("command_spec"):
            raise CommandJournalError("command replay conflicts on frozen command spec")
        if authoritative.get("command_plan_ref") != command.get("command_plan_ref"):
            raise CommandJournalError("command replay conflicts on frozen command plan reference")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "COMMAND_RECEIPT_LOCATOR_SCHEMA_VERSION", "CommandExecutionResult", "CommandJournalError",
    "CommandJournalResult", "LedgerCommandAuthority", "LedgerCommandJournal", "PHASE_COMMAND_SCHEMA_FILE",
    "PHASE_COMMAND_SCHEMA_VERSION", "PHASE_RUN_RECEIPT_SCHEMA_FILE", "PHASE_RUN_RECEIPT_SCHEMA_VERSION",
]
