"""Strict execution shells for authoritative experiment phases."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_EXECUTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{7,127}$")
_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9:._/-]{0,255}$")
_PHASES = {"proxy", "full"}
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
class AuthoritativePhaseContext:
    """Immutable phase identity expected to match the SQLite authority exactly."""

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
    proxy_authorization: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", Path(self.project_root).resolve())
        if not isinstance(self.attempt_id, str) or not self.attempt_id:
            raise ValueError("attempt_id must be non-empty")
        for field_name in (
            "direction_semantic_hash",
            "direction_spec_hash",
            "variant_semantic_hash",
            "variant_spec_hash",
            "trial_spec_hash",
            "implementation_hash",
            "attempt_input_hash",
        ):
            if not _SHA256_RE.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase SHA-256 digest")
        if isinstance(self.lifecycle_generation, bool) or not isinstance(self.lifecycle_generation, int) or self.lifecycle_generation < 0:
            raise ValueError("lifecycle_generation must be a non-negative integer")
        if self.phase not in _PHASES:
            raise ValueError("phase must be proxy or full")
        if not _EXECUTION_ID_RE.fullmatch(self.phase_execution_id):
            raise ValueError("phase_execution_id is invalid")
        if not _EVENT_ID_RE.fullmatch(self.phase_start_event_id):
            raise ValueError("phase_start_event_id is invalid")
        if not _EXECUTION_ID_RE.fullmatch(self.producer_run_id):
            raise ValueError("producer_run_id is invalid")
        authorization = _normalize_proxy_authorization(self.proxy_authorization)
        object.__setattr__(self, "proxy_authorization", authorization)
        if self.phase == "full" and authorization is None:
            raise ValueError("full phase context requires proxy authorization")

    @classmethod
    def from_attempt(
        cls,
        project_root: Path,
        attempt: Mapping[str, Any],
        phase: str,
    ) -> AuthoritativePhaseContext:
        phase_executions = attempt.get("phase_executions")
        execution = phase_executions.get(phase) if isinstance(phase_executions, Mapping) else None
        if not isinstance(execution, Mapping):
            raise ValueError(f"attempt has no authoritative {phase} phase execution")
        values = {field_name: attempt[field_name] for field_name in _IDENTITY_FIELDS[:9]}
        return cls(
            project_root=project_root,
            **values,
            phase=phase,
            phase_execution_id=execution["phase_execution_id"],
            phase_start_event_id=execution["phase_start_event_id"],
            producer_run_id=execution["producer_run_id"],
            proxy_authorization=attempt.get("committed_proxy_outcome") if phase == "full" else None,
        )

    def identity(self) -> dict[str, Any]:
        return {field_name: getattr(self, field_name) for field_name in _IDENTITY_FIELDS}


@dataclass(frozen=True)
class PhaseArtifactInventory:
    """Artifacts produced by exactly one authoritative phase execution."""

    context: AuthoritativePhaseContext
    artifacts: Sequence[Mapping[str, str]]

    def __post_init__(self) -> None:
        if not isinstance(self.context, AuthoritativePhaseContext):
            raise TypeError("inventory context must be AuthoritativePhaseContext")
        normalized: list[Mapping[str, str]] = []
        seen: set[str] = set()
        for artifact in self.artifacts:
            if not isinstance(artifact, Mapping) or set(artifact) != {"kind", "source_path", "producer_run_id"}:
                raise ValueError("artifact entries require exactly kind, source_path, and producer_run_id")
            kind = artifact.get("kind")
            source_path = artifact.get("source_path")
            producer_run_id = artifact.get("producer_run_id")
            if not isinstance(kind, str) or not kind:
                raise ValueError("artifact kind must be non-empty")
            if kind in seen:
                raise ValueError(f"duplicate artifact kind: {kind}")
            if not isinstance(source_path, str) or not _safe_relative_path(source_path):
                raise ValueError(f"artifact source_path is not a safe project-relative path: {source_path!r}")
            if producer_run_id != self.context.producer_run_id:
                raise ValueError("artifact producer_run_id does not match phase context")
            seen.add(kind)
            normalized.append(
                MappingProxyType(
                    {"kind": kind, "source_path": source_path, "producer_run_id": producer_run_id}
                )
            )
        if not normalized:
            raise ValueError("phase artifact inventory must not be empty")
        object.__setattr__(self, "artifacts", tuple(normalized))

    @classmethod
    def from_artifacts(
        cls,
        context: AuthoritativePhaseContext,
        artifacts: Sequence[Mapping[str, str]],
    ) -> PhaseArtifactInventory:
        return cls(context=context, artifacts=artifacts)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "auto_research_phase_execution_manifest_v1",
            **self.context.identity(),
            "artifacts": [dict(artifact) for artifact in self.artifacts],
        }


class TypedPhaseFailure(RuntimeError):
    """Typed executor failure suitable for deterministic failure routing."""

    def __init__(
        self,
        failure_class: str,
        message: str,
        *,
        phase: str | None = None,
        executor: str | None = None,
        retryable: bool = False,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(failure_class, str) or not failure_class:
            raise ValueError("failure_class must be non-empty")
        if not isinstance(message, str) or not message:
            raise ValueError("message must be non-empty")
        if phase is not None and phase not in _PHASES:
            raise ValueError("failure phase must be proxy or full")
        super().__init__(message)
        self.failure_class = failure_class
        self.phase = phase
        self.executor = executor
        self.retryable = bool(retryable)
        self.details = MappingProxyType(dict(details or {}))

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_class": self.failure_class,
            "message": str(self),
            "phase": self.phase,
            "executor": self.executor,
            "retryable": self.retryable,
            "details": dict(self.details),
        }


@runtime_checkable
class PhaseAuthorityChecker(Protocol):
    """Injected adapter that verifies a phase identity against SQLite authority."""

    def __call__(self, context: AuthoritativePhaseContext) -> AuthoritativePhaseContext | bool | None:
        ...


@runtime_checkable
class PhaseExecutor(Protocol):
    """Protocol implemented by all strict phase execution shells."""

    def execute(self, context: AuthoritativePhaseContext) -> PhaseArtifactInventory:
        ...


PhaseRunner = Callable[[AuthoritativePhaseContext], PhaseArtifactInventory]


@dataclass(frozen=True)
class _StrictPhaseExecutor:
    authority_checker: PhaseAuthorityChecker
    runner: PhaseRunner
    executor_name: str = field(init=False, default="phase")
    required_phase: str | None = field(init=False, default=None)

    def execute(self, context: AuthoritativePhaseContext) -> PhaseArtifactInventory:
        if not isinstance(context, AuthoritativePhaseContext):
            raise TypedPhaseFailure(
                "invalid_phase_context",
                "executor requires AuthoritativePhaseContext",
                executor=self.executor_name,
            )
        if self.required_phase is not None and context.phase != self.required_phase:
            raise TypedPhaseFailure(
                "phase_mismatch",
                f"{self.executor_name} requires {self.required_phase} context",
                phase=context.phase,
                executor=self.executor_name,
            )
        self._check_authority(context)
        try:
            inventory = self.runner(context)
        except TypedPhaseFailure:
            raise
        except Exception as error:
            raise TypedPhaseFailure(
                "phase_execution_failed",
                f"{self.executor_name} runner failed: {error}",
                phase=context.phase,
                executor=self.executor_name,
                retryable=True,
                details={"exception_type": type(error).__name__},
            ) from error
        if not isinstance(inventory, PhaseArtifactInventory):
            raise TypedPhaseFailure(
                "invalid_artifact_inventory",
                "runner must return PhaseArtifactInventory",
                phase=context.phase,
                executor=self.executor_name,
            )
        if inventory.context != context:
            raise TypedPhaseFailure(
                "artifact_identity_mismatch",
                "artifact inventory is bound to a different phase identity",
                phase=context.phase,
                executor=self.executor_name,
            )
        return inventory

    def __call__(self, context: AuthoritativePhaseContext) -> PhaseArtifactInventory:
        return self.execute(context)

    def _check_authority(self, context: AuthoritativePhaseContext) -> None:
        try:
            verdict = self.authority_checker(context)
        except TypedPhaseFailure:
            raise
        except Exception as error:
            raise TypedPhaseFailure(
                "authority_check_failed",
                f"SQLite phase authority check failed: {error}",
                phase=context.phase,
                executor=self.executor_name,
                details={"exception_type": type(error).__name__},
            ) from error
        if verdict is False:
            raise TypedPhaseFailure(
                "authority_rejected",
                "SQLite authority rejected the phase identity",
                phase=context.phase,
                executor=self.executor_name,
            )
        if isinstance(verdict, AuthoritativePhaseContext) and verdict != context:
            raise TypedPhaseFailure(
                "authority_identity_mismatch",
                "SQLite authority returned a different phase identity",
                phase=context.phase,
                executor=self.executor_name,
            )
        if verdict is not None and verdict is not True and not isinstance(verdict, AuthoritativePhaseContext):
            raise TypedPhaseFailure(
                "invalid_authority_verdict",
                "authority checker returned an unsupported verdict",
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


def _normalize_proxy_authorization(value: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("proxy_authorization must be a mapping")
    required = {"event_id", "outcome_hash", "decision"}
    if not required.issubset(value):
        raise ValueError("proxy_authorization requires event_id, outcome_hash, and decision")
    if set(value) - (required | {"event_hash"}):
        raise ValueError("proxy_authorization contains unsupported fields")
    if value.get("decision") != "RUN_FULL":
        raise ValueError("full phase requires RUN_FULL proxy authorization")
    event_id = value.get("event_id")
    outcome_hash = value.get("outcome_hash")
    if not isinstance(event_id, str) or not _EVENT_ID_RE.fullmatch(event_id):
        raise ValueError("proxy_authorization event_id is invalid")
    if not isinstance(outcome_hash, str) or not _SHA256_RE.fullmatch(outcome_hash):
        raise ValueError("proxy_authorization outcome_hash is invalid")
    event_hash = value.get("event_hash")
    if event_hash is not None and (not isinstance(event_hash, str) or not _SHA256_RE.fullmatch(event_hash)):
        raise ValueError("proxy_authorization event_hash is invalid")
    normalized = {"event_id": event_id, "outcome_hash": outcome_hash, "decision": "RUN_FULL"}
    if event_hash is not None:
        normalized["event_hash"] = event_hash
    return MappingProxyType(normalized)


def _safe_relative_path(value: str) -> bool:
    if not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


__all__ = [
    "AuthoritativePhaseContext",
    "C2CFullPhaseExecutor",
    "C2CProxyPhaseExecutor",
    "GenericExternalPhaseExecutor",
    "PhaseArtifactInventory",
    "PhaseAuthorityChecker",
    "PhaseExecutor",
    "SyntheticPhaseExecutor",
    "TypedPhaseFailure",
]
