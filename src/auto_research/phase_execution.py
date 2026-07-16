"""Fail-closed execution shells for authoritative experiment phases."""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from .contract_store import ContractStore
from .domain_contracts import PHASE_EXECUTION_MANIFEST_SCHEMA_VERSION, canonical_hash

_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")
_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,255}$")
_PHASES = {"proxy", "full"}
_PROVENANCE_MODES = {"synthetic", "local_external", "production"}
_PHASE_STATES = {"proxy": "PROXY_RUNNING", "full": "FULL_RUNNING"}
_IDENTITY_FIELDS = (
    "attempt_id",
    "direction_semantic_hash",
    "direction_spec_hash",
    "variant_semantic_hash",
    "variant_spec_hash",
    "trial_spec_hash",
    "lifecycle_generation",
    "implementation_hash",
    "attempt_input_hash",
    "phase",
    "phase_execution_id",
    "phase_start_event_id",
    "producer_run_id",
)


@dataclass(frozen=True)
class PhaseAuthorization:
    """Exact authorization read from the authoritative SQLite ledger."""

    attempt_id: str
    lifecycle_generation: int
    phase: str
    phase_execution_id: str
    phase_start_event_id: str
    phase_start_event_hash: str
    phase_start_sequence: int
    producer_run_id: str
    implementation_hash: str
    attempt_input_hash: str
    trial_spec_hash: str
    command_plan_hash: str
    phase_contract_hash: str
    expected_evidence_kinds: tuple[str, ...]
    adapter_identity: str
    provenance_mode: str
    state: str
    proxy_authorization_required: bool = True
    proxy_commit_event_id: str | None = None
    proxy_commit_event_hash: str | None = None
    proxy_outcome_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValueError("authorization attempt_id must be non-empty")
        if self.phase not in _PHASES:
            raise ValueError("authorization phase must be proxy or full")
        if self.state != _PHASE_STATES[self.phase]:
            raise ValueError("authorization state does not match phase")
        if isinstance(self.lifecycle_generation, bool) or self.lifecycle_generation < 0:
            raise ValueError("authorization lifecycle_generation is invalid")
        if isinstance(self.phase_start_sequence, bool) or self.phase_start_sequence < 1:
            raise ValueError("authorization phase_start_sequence is invalid")
        for value, label in (
            (self.phase_execution_id, "phase_execution_id"),
            (self.producer_run_id, "producer_run_id"),
            (self.adapter_identity, "adapter_identity"),
        ):
            if not _SAFE_ID_RE.fullmatch(value):
                raise ValueError(f"authorization {label} is invalid")
        if not _EVENT_ID_RE.fullmatch(self.phase_start_event_id):
            raise ValueError("authorization phase_start_event_id is invalid")
        for field_name in (
            "phase_start_event_hash",
            "implementation_hash",
            "attempt_input_hash",
            "trial_spec_hash",
            "command_plan_hash",
            "phase_contract_hash",
        ):
            if not _SHA256_RE.fullmatch(getattr(self, field_name)):
                raise ValueError(f"authorization {field_name} must be SHA-256")
        if self.provenance_mode not in _PROVENANCE_MODES:
            raise ValueError("authorization provenance_mode is invalid")
        kinds = tuple(self.expected_evidence_kinds)
        if not kinds or any(not isinstance(kind, str) or not kind for kind in kinds) or len(kinds) != len(set(kinds)):
            raise ValueError("authorization expected_evidence_kinds must be unique and non-empty")
        object.__setattr__(self, "expected_evidence_kinds", tuple(sorted(kinds)))
        proxy_fields = (self.proxy_commit_event_id, self.proxy_commit_event_hash, self.proxy_outcome_hash)
        if not isinstance(self.proxy_authorization_required, bool):
            raise ValueError("proxy_authorization_required must be boolean")
        if self.phase == "full" and self.proxy_authorization_required:
            if any(value is None for value in proxy_fields):
                raise ValueError("full authorization requires committed proxy identity")
            if not _EVENT_ID_RE.fullmatch(str(self.proxy_commit_event_id)):
                raise ValueError("proxy commit event id is invalid")
            if not _SHA256_RE.fullmatch(str(self.proxy_commit_event_hash)) or not _SHA256_RE.fullmatch(str(self.proxy_outcome_hash)):
                raise ValueError("proxy authorization hashes are invalid")
        elif any(value is not None for value in proxy_fields):
            raise ValueError("proxy phase cannot carry full authorization")

    @property
    def authorization_hash(self) -> str:
        return canonical_hash(self.as_dict())

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "lifecycle_generation": self.lifecycle_generation,
            "phase": self.phase,
            "phase_execution_id": self.phase_execution_id,
            "phase_start_event_id": self.phase_start_event_id,
            "phase_start_event_hash": self.phase_start_event_hash,
            "phase_start_sequence": self.phase_start_sequence,
            "producer_run_id": self.producer_run_id,
            "implementation_hash": self.implementation_hash,
            "attempt_input_hash": self.attempt_input_hash,
            "trial_spec_hash": self.trial_spec_hash,
            "command_plan_hash": self.command_plan_hash,
            "phase_contract_hash": self.phase_contract_hash,
            "expected_evidence_kinds": list(self.expected_evidence_kinds),
            "adapter_identity": self.adapter_identity,
            "provenance_mode": self.provenance_mode,
            "state": self.state,
            "proxy_authorization_required": self.proxy_authorization_required,
            "proxy_commit_event_id": self.proxy_commit_event_id,
            "proxy_commit_event_hash": self.proxy_commit_event_hash,
            "proxy_outcome_hash": self.proxy_outcome_hash,
        }


@dataclass(frozen=True)
class AuthoritativePhaseContext:
    """Immutable phase context constructed from a Ledger authorization."""

    project_root: Path
    attempt_id: str
    direction_semantic_hash: str
    direction_spec_hash: str
    variant_semantic_hash: str
    variant_spec_hash: str
    trial_spec_hash: str
    lifecycle_generation: int
    implementation_hash: str
    attempt_input_hash: str
    phase: str
    phase_execution_id: str
    phase_start_event_id: str
    producer_run_id: str
    command_plan_hash: str
    phase_contract_hash: str
    expected_evidence_kinds: tuple[str, ...]
    adapter_identity: str
    provenance_mode: str
    authorization_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        if not self.attempt_id:
            raise ValueError("attempt_id must be non-empty")
        for field_name in (
            "direction_semantic_hash", "direction_spec_hash", "variant_semantic_hash", "variant_spec_hash",
            "trial_spec_hash", "implementation_hash", "attempt_input_hash", "command_plan_hash",
            "phase_contract_hash", "authorization_hash",
        ):
            if not _SHA256_RE.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if isinstance(self.lifecycle_generation, bool) or not isinstance(self.lifecycle_generation, int) or self.lifecycle_generation < 0:
            raise ValueError("lifecycle_generation must be a non-negative integer")
        if self.phase not in _PHASES:
            raise ValueError("phase must be proxy or full")
        for value, label in ((self.phase_execution_id, "phase_execution_id"), (self.producer_run_id, "producer_run_id"), (self.adapter_identity, "adapter_identity")):
            if not _SAFE_ID_RE.fullmatch(value):
                raise ValueError(f"{label} is invalid")
        if not _EVENT_ID_RE.fullmatch(self.phase_start_event_id):
            raise ValueError("phase_start_event_id is invalid")
        if self.provenance_mode not in _PROVENANCE_MODES:
            raise ValueError("provenance_mode is invalid")
        kinds = tuple(self.expected_evidence_kinds)
        if not kinds or len(kinds) != len(set(kinds)):
            raise ValueError("expected_evidence_kinds must be unique and non-empty")
        object.__setattr__(self, "expected_evidence_kinds", tuple(sorted(kinds)))

    @classmethod
    def from_authorization(
        cls,
        project_root: Path,
        attempt: Mapping[str, Any],
        authorization: PhaseAuthorization,
    ) -> AuthoritativePhaseContext:
        for field_name in ("attempt_id", "lifecycle_generation", "implementation_hash", "attempt_input_hash", "trial_spec_hash"):
            if attempt.get(field_name) != getattr(authorization, field_name):
                raise ValueError(f"attempt {field_name} differs from Ledger authorization")
        return cls(
            project_root=project_root,
            attempt_id=authorization.attempt_id,
            direction_semantic_hash=attempt["direction_semantic_hash"],
            direction_spec_hash=attempt["direction_spec_hash"],
            variant_semantic_hash=attempt["variant_semantic_hash"],
            variant_spec_hash=attempt["variant_spec_hash"],
            trial_spec_hash=authorization.trial_spec_hash,
            lifecycle_generation=authorization.lifecycle_generation,
            implementation_hash=authorization.implementation_hash,
            attempt_input_hash=authorization.attempt_input_hash,
            phase=authorization.phase,
            phase_execution_id=authorization.phase_execution_id,
            phase_start_event_id=authorization.phase_start_event_id,
            producer_run_id=authorization.producer_run_id,
            command_plan_hash=authorization.command_plan_hash,
            phase_contract_hash=authorization.phase_contract_hash,
            expected_evidence_kinds=authorization.expected_evidence_kinds,
            adapter_identity=authorization.adapter_identity,
            provenance_mode=authorization.provenance_mode,
            authorization_hash=authorization.authorization_hash,
        )

    def identity(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in _IDENTITY_FIELDS}


@dataclass(frozen=True)
class PhaseArtifactInventory:
    context: AuthoritativePhaseContext
    artifacts: Sequence[Mapping[str, str]]
    complete: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.context, AuthoritativePhaseContext):
            raise TypeError("inventory context must be AuthoritativePhaseContext")
        normalized: list[Mapping[str, str]] = []
        seen: set[str] = set()
        for artifact in self.artifacts:
            required = {"kind", "source_path", "content_hash", "receipt_hash", "producer_run_id"}
            if not isinstance(artifact, Mapping) or set(artifact) != required:
                raise ValueError("artifact entries require kind, source_path, content_hash, receipt_hash, producer_run_id")
            kind = artifact["kind"]
            if kind in seen:
                raise ValueError("artifact kinds must be unique")
            if kind not in self.context.expected_evidence_kinds:
                raise ValueError("artifact kind was not authorized")
            if not _safe_relative_path(artifact["source_path"]):
                raise ValueError("artifact source_path must be safe and relative")
            if not _SHA256_RE.fullmatch(artifact["content_hash"]) or not _SHA256_RE.fullmatch(artifact["receipt_hash"]):
                raise ValueError("artifact hashes must be SHA-256")
            if artifact["producer_run_id"] != self.context.producer_run_id:
                raise ValueError("artifact producer_run_id differs from phase context")
            seen.add(kind)
            normalized.append(MappingProxyType(dict(artifact)))
        if not isinstance(self.complete, bool):
            raise ValueError("artifact inventory complete flag must be boolean")
        if self.complete and seen != set(self.context.expected_evidence_kinds):
            raise ValueError("artifact inventory must exactly match authorized evidence kinds")
        object.__setattr__(self, "artifacts", tuple(normalized))

    @classmethod
    def combine(
        cls,
        context: AuthoritativePhaseContext,
        inventories: Sequence[PhaseArtifactInventory],
    ) -> PhaseArtifactInventory:
        artifacts: list[Mapping[str, str]] = []
        for inventory in inventories:
            if inventory.context != context:
                raise ValueError("cannot combine inventories from different phase contexts")
            artifacts.extend(inventory.artifacts)
        return cls(context=context, artifacts=tuple(artifacts), complete=True)


class TypedPhaseFailure(RuntimeError):
    def __init__(self, failure_class: str, message: str, *, phase: str, executor: str, retryable: bool = False, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.failure_class = failure_class
        self.phase = phase
        self.executor = executor
        self.retryable = retryable
        self.details = dict(details or {})


@dataclass(frozen=True)
class ExecutorCapability:
    """Opaque, callback-scoped proof that a strict executor owns execution."""

    executor_name: str
    adapter_identity: str
    attempt_id: str
    lifecycle_generation: int
    phase: str
    phase_execution_id: str
    authorization_hash: str
    _nonce: object = field(repr=False, compare=False)


_ACTIVE_EXECUTOR_CAPABILITY: ContextVar[ExecutorCapability | None] = ContextVar(
    "auto_research_active_executor_capability",
    default=None,
)


def _required_executor_names(context: AuthoritativePhaseContext) -> frozenset[str]:
    if context.provenance_mode == "synthetic":
        return frozenset({"synthetic"})
    try:
        plan = ContractStore(context.project_root).read_json(
            context.command_plan_hash,
            schema_file="phase_command_plan_v2.schema.json",
        )
    except (OSError, TypeError, ValueError):
        return frozenset()
    adapter_id = str((plan.get("adapter_identity") or {}).get("adapter_id") or "").lower()
    plan_provenance = str((plan.get("adapter_identity") or {}).get("provenance_mode") or "")
    if plan_provenance == "synthetic":
        return frozenset({"synthetic"})
    if "c2c" in adapter_id:
        return frozenset({"c2c_proxy" if context.phase == "proxy" else "c2c_full"})
    return frozenset({"generic_external"})


def require_executor_capability(context: AuthoritativePhaseContext) -> ExecutorCapability:
    """Fail closed unless called inside the currently authorized executor callback."""

    capability = _ACTIVE_EXECUTOR_CAPABILITY.get()
    if capability is None:
        raise TypedPhaseFailure(
            "executor_capability_missing",
            "side-effect command requires current PhaseExecutor capability",
            phase=context.phase,
            executor="unowned",
        )
    expected = (
        context.attempt_id,
        context.lifecycle_generation,
        context.phase,
        context.phase_execution_id,
        context.authorization_hash,
    )
    actual = (
        capability.attempt_id,
        capability.lifecycle_generation,
        capability.phase,
        capability.phase_execution_id,
        capability.authorization_hash,
    )
    required_executors = _required_executor_names(context)
    if (
        actual != expected
        or capability.adapter_identity != context.adapter_identity
        or capability.executor_name not in required_executors
    ):
        raise TypedPhaseFailure(
            "executor_capability_mismatch",
            "PhaseExecutor capability does not authorize this phase context",
            phase=context.phase,
            executor=capability.executor_name,
        )
    return capability


@runtime_checkable
class PhaseAuthority(Protocol):
    def authorize_phase(self, context: AuthoritativePhaseContext) -> PhaseAuthorization: ...


@dataclass(frozen=True)
class ResearchLedgerPhaseAuthority:
    """Reads the latest attempt and phase-start event from SQLite on every check."""

    ledger: Any

    def context_for_attempt(self, project_root: Path, attempt_id: str, phase: str) -> AuthoritativePhaseContext:
        authorization, attempt = self._read_authorization(attempt_id, phase)
        return AuthoritativePhaseContext.from_authorization(project_root, attempt, authorization)

    def authorize_phase(self, context: AuthoritativePhaseContext) -> PhaseAuthorization:
        authorization, _ = self._read_authorization(context.attempt_id, context.phase)
        if authorization.authorization_hash != context.authorization_hash:
            raise ValueError("phase context does not match latest SQLite authorization")
        return authorization

    def _read_authorization(self, attempt_id: str, phase: str) -> tuple[PhaseAuthorization, Mapping[str, Any]]:
        state = self.ledger.state()
        attempt = (state.get("attempts") or {}).get(attempt_id)
        if not isinstance(attempt, Mapping):
            raise ValueError("authoritative attempt is missing")
        execution = ((attempt.get("phase_executions") or {}).get(phase))
        if not isinstance(execution, Mapping):
            raise ValueError("authoritative phase execution is missing")
        event = next(
            (
                item for item in reversed(self.ledger.events())
                if item.get("event_id") == execution.get("phase_start_event_id")
            ),
            None,
        )
        if not isinstance(event, Mapping) or event.get("event_type") != ("ProxyPhaseStarted" if phase == "proxy" else "FullPhaseStarted"):
            raise ValueError("authoritative phase-start event is missing")
        manifest = (event.get("payload") or {}).get("phase_execution_manifest")
        if not isinstance(manifest, Mapping) or manifest.get("schema_version") != PHASE_EXECUTION_MANIFEST_SCHEMA_VERSION:
            raise ValueError(f"authoritative {PHASE_EXECUTION_MANIFEST_SCHEMA_VERSION} is required")
        proxy = attempt.get("committed_proxy_outcome") if phase == "full" else None
        runtime = attempt["frozen_trial_spec"]["execution_contract"]["runtime_config"]
        collector = str(runtime.get("collector") or "generic")
        adapter_identity = "adapter-" + re.sub(r"[^A-Za-z0-9_-]", "-", collector).strip("-")
        if len(adapter_identity) < 8:
            adapter_identity = "adapter-generic"
        authorization = PhaseAuthorization(
            attempt_id=attempt["attempt_id"], lifecycle_generation=attempt["lifecycle_generation"], phase=phase,
            phase_execution_id=manifest["phase_execution_id"], phase_start_event_id=event["event_id"],
            phase_start_event_hash=event["event_hash"], phase_start_sequence=event["sequence"],
            producer_run_id=manifest["producer_run_id"], implementation_hash=attempt["implementation_hash"],
            attempt_input_hash=attempt["attempt_input_hash"], trial_spec_hash=attempt["trial_spec_hash"],
            command_plan_hash=manifest["command_plan_hash"], phase_contract_hash=manifest["phase_contract_hash"],
            expected_evidence_kinds=tuple(manifest["expected_evidence_kinds"]), adapter_identity=adapter_identity[:127],
            provenance_mode=manifest["provenance_mode"].replace("-", "_"), state=attempt["state"],
            proxy_authorization_required=phase != "full" or attempt.get("attempt_kind") == "proxy_full",
            proxy_commit_event_id=proxy.get("event_id") if isinstance(proxy, Mapping) else None,
            proxy_commit_event_hash=proxy.get("event_hash") if isinstance(proxy, Mapping) else None,
            proxy_outcome_hash=proxy.get("outcome_hash") if isinstance(proxy, Mapping) else None,
        )
        return authorization, attempt


@runtime_checkable
class PhaseExecutor(Protocol):
    def execute(self, context: AuthoritativePhaseContext) -> PhaseArtifactInventory: ...


PhaseRunner = Callable[[AuthoritativePhaseContext], PhaseArtifactInventory]


@dataclass(frozen=True)
class _StrictPhaseExecutor:
    authority: PhaseAuthority
    runner: PhaseRunner
    executor_name: str = field(init=False, default="phase")
    required_phase: str | None = field(init=False, default=None)

    def execute(self, context: AuthoritativePhaseContext) -> PhaseArtifactInventory:
        if not isinstance(context, AuthoritativePhaseContext):
            raise TypeError("context must be AuthoritativePhaseContext")
        if self.required_phase is not None and context.phase != self.required_phase:
            raise TypedPhaseFailure("phase_mismatch", f"{self.executor_name} requires {self.required_phase}", phase=context.phase, executor=self.executor_name)
        authorization = self._authorize(context)
        if authorization.authorization_hash != context.authorization_hash:
            raise TypedPhaseFailure("authority_identity_mismatch", "Ledger authorization differs from phase context", phase=context.phase, executor=self.executor_name)
        required_executors = _required_executor_names(context)
        if self.executor_name not in required_executors or authorization.adapter_identity != context.adapter_identity:
            raise TypedPhaseFailure(
                "executor_adapter_mismatch",
                f"{self.executor_name} is not authorized for adapter {context.adapter_identity}",
                phase=context.phase,
                executor=self.executor_name,
            )
        capability = ExecutorCapability(
            executor_name=self.executor_name,
            adapter_identity=context.adapter_identity,
            attempt_id=context.attempt_id,
            lifecycle_generation=context.lifecycle_generation,
            phase=context.phase,
            phase_execution_id=context.phase_execution_id,
            authorization_hash=context.authorization_hash,
            _nonce=object(),
        )
        token: Token[ExecutorCapability | None] = _ACTIVE_EXECUTOR_CAPABILITY.set(capability)
        try:
            inventory = self.runner(context)
            self._assert_authorization(context, authorization)
        finally:
            _ACTIVE_EXECUTOR_CAPABILITY.reset(token)
        if not isinstance(inventory, PhaseArtifactInventory) or inventory.context != context or not inventory.complete:
            raise TypedPhaseFailure("artifact_identity_mismatch", "runner returned inventory for a different phase", phase=context.phase, executor=self.executor_name)
        return inventory

    def __call__(self, context: AuthoritativePhaseContext) -> PhaseArtifactInventory:
        return self.execute(context)

    def _authorize(self, context: AuthoritativePhaseContext) -> PhaseAuthorization:
        try:
            authorization = self.authority.authorize_phase(context)
        except TypedPhaseFailure:
            raise
        except Exception as error:
            raise TypedPhaseFailure("authority_check_failed", f"SQLite phase authority check failed: {error}", phase=context.phase, executor=self.executor_name) from error
        if not isinstance(authorization, PhaseAuthorization):
            raise TypedPhaseFailure("invalid_authority_verdict", "authority must return PhaseAuthorization", phase=context.phase, executor=self.executor_name)
        return authorization

    def _assert_authorization(
        self,
        context: AuthoritativePhaseContext,
        expected: PhaseAuthorization,
    ) -> None:
        current = self._authorize(context)
        if current != expected or current.authorization_hash != context.authorization_hash:
            raise TypedPhaseFailure(
                "authority_changed",
                "SQLite phase authorization changed during executor callback",
                phase=context.phase,
                executor=self.executor_name,
            )


@dataclass(frozen=True)
class C2CProxyPhaseExecutor(_StrictPhaseExecutor):
    executor_name: str = field(init=False, default="c2c_proxy")
    required_phase: str = field(init=False, default="proxy")


@dataclass(frozen=True)
class C2CFullPhaseExecutor(_StrictPhaseExecutor):
    executor_name: str = field(init=False, default="c2c_full")
    required_phase: str = field(init=False, default="full")


@dataclass(frozen=True)
class GenericExternalPhaseExecutor(_StrictPhaseExecutor):
    executor_name: str = field(init=False, default="generic_external")


@dataclass(frozen=True)
class SyntheticPhaseExecutor(_StrictPhaseExecutor):
    executor_name: str = field(init=False, default="synthetic")


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


__all__ = [
    "AuthoritativePhaseContext", "C2CFullPhaseExecutor", "C2CProxyPhaseExecutor", "GenericExternalPhaseExecutor",
    "ExecutorCapability", "PHASE_EXECUTION_MANIFEST_SCHEMA_VERSION", "PhaseArtifactInventory", "PhaseAuthorization",
    "PhaseAuthority", "PhaseExecutor", "ResearchLedgerPhaseAuthority", "SyntheticPhaseExecutor", "TypedPhaseFailure",
    "require_executor_capability",
]
