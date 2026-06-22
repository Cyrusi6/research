from __future__ import annotations

import argparse
import ast
import hashlib
import copy
import inspect
import json
import os
from pathlib import Path
from typing import Any
from types import SimpleNamespace
import sys

import yaml


RUNTIME_FILES = [
    Path("rosetta/model/wrapper.py"),
    Path("rosetta/model/projector.py"),
    Path("rosetta/model/aligner.py"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Lightweight C2C forward activation probe.")
    parser.add_argument("--enabled-config", required=True)
    parser.add_argument("--disabled-config", required=True)
    parser.add_argument("--switch", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    enabled_cfg = _read_config(Path(args.enabled_config))
    disabled_cfg = _read_config(Path(args.disabled_config))
    enabled_rosetta = _rosetta_config(enabled_cfg)
    disabled_rosetta = _rosetta_config(disabled_cfg)

    switch = str(args.switch)
    runtime_trace = _runtime_switch_trace(Path.cwd(), switch)
    static_evidence = _static_activation_evidence(enabled_rosetta, disabled_rosetta, switch, runtime_trace)
    forward_evidence = _repo_small_batch_forward_probe(enabled_rosetta, disabled_rosetta, switch)

    payload = _merge_probe_evidence(static_evidence, forward_evidence)
    Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if payload.get("mechanism_observed") else 1


def _static_activation_evidence(
    enabled_rosetta: dict[str, Any],
    disabled_rosetta: dict[str, Any],
    switch: str,
    runtime_trace: dict[str, Any],
) -> dict[str, Any]:
    changed_fields: list[str] = []
    unchanged_fields: list[str] = []

    if enabled_rosetta.get(switch) != disabled_rosetta.get(switch):
        changed_fields.append(f"rosetta_config.{switch}")
    else:
        unchanged_fields.append(f"rosetta_config.{switch}")

    if runtime_trace["switch_refs"]:
        changed_fields.append("runtime_code.switch_refs")
    else:
        unchanged_fields.append("runtime_code.switch_refs")

    if runtime_trace["config_read_refs"]:
        changed_fields.append("runtime_code.config_read_refs")
    else:
        unchanged_fields.append("runtime_code.config_read_refs")

    config_hash_changed = _stable_hash(enabled_rosetta) != _stable_hash(disabled_rosetta)
    if config_hash_changed:
        changed_fields.append("rosetta_config.hash")
    else:
        unchanged_fields.append("rosetta_config.hash")

    mechanism_observed = bool(
        disabled_rosetta.get(switch) is True
        and runtime_trace["switch_refs"]
        and runtime_trace["config_read_refs"]
    )
    failures = []
    if disabled_rosetta.get(switch) is not True:
        failures.append("disabled_config_missing_true_switch")
    if not runtime_trace["switch_refs"]:
        failures.append("runtime_forward_missing_switch_ref")
    if not runtime_trace["config_read_refs"]:
        failures.append("runtime_forward_missing_config_read")

    return {
        "probe_type": "builtin_static_forward_trace",
        "mechanism_observed": mechanism_observed,
        "compared_fields": [
            f"rosetta_config.{switch}",
            "rosetta_config.hash",
            "runtime_code.switch_refs",
            "runtime_code.config_read_refs",
        ],
        "changed_fields": list(dict.fromkeys(changed_fields)),
        "unchanged_fields": list(dict.fromkeys(unchanged_fields)),
        "enabled": {
            "switch_value": enabled_rosetta.get(switch),
            "rosetta_hash": _stable_hash(enabled_rosetta),
        },
        "disabled": {
            "switch_value": disabled_rosetta.get(switch),
            "rosetta_hash": _stable_hash(disabled_rosetta),
        },
        "trace": runtime_trace,
        "failures": failures,
    }


def _repo_small_batch_forward_probe(
    enabled_rosetta: dict[str, Any],
    disabled_rosetta: dict[str, Any],
    switch: str,
) -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return _forward_probe_failed("torch_import_failed", exc)

    try:
        sys.path.insert(0, str(Path.cwd()))
        from rosetta.model.projector import create_projector
    except Exception as exc:
        return _forward_probe_failed("projector_import_failed", exc)

    projector_spec = _projector_spec(enabled_rosetta)
    torch.manual_seed(1234)
    try:
        enabled_projector = _build_probe_projector(create_projector, projector_spec, enabled_rosetta, switch)
        disabled_projector = _build_probe_projector(create_projector, projector_spec, disabled_rosetta, switch)
        disabled_projector.load_state_dict(copy.deepcopy(enabled_projector.state_dict()), strict=False)
        enabled_projector.eval()
        disabled_projector.eval()
        source_kv, target_kv = _synthetic_kv_batch(torch)
        with torch.no_grad():
            enabled_output = enabled_projector(source_kv, target_kv)
            disabled_output = disabled_projector(source_kv, target_kv)
        tensor_checks = _tensor_diff_checks(torch, enabled_projector, disabled_projector, enabled_output, disabled_output)
        wrapper_probe = _repo_wrapper_cache_projection_probe(
            torch,
            enabled_projector,
            disabled_projector,
            switch,
        )
    except Exception as exc:
        return _forward_probe_failed("small_batch_forward_failed", exc, projector_spec=projector_spec)

    wrapper_tensor_checks = wrapper_probe.get("tensor_checks") if isinstance(wrapper_probe, dict) else []
    all_tensor_checks = [*tensor_checks, *(wrapper_tensor_checks if isinstance(wrapper_tensor_checks, list) else [])]
    changed_fields = [item["name"] for item in all_tensor_checks if item.get("changed")]
    unchanged_fields = [item["name"] for item in all_tensor_checks if not item.get("changed")]
    wrapper_changed_fields = [
        item["name"]
        for item in (wrapper_tensor_checks if isinstance(wrapper_tensor_checks, list) else [])
        if item.get("changed")
    ]
    failures = []
    if wrapper_probe.get("status") == "ok":
        if not wrapper_probe.get("projector_called"):
            failures.append("wrapper_projector_not_called")
        if not wrapper_changed_fields:
            failures.append("enabled_disabled_wrapper_cache_identical")
    elif not changed_fields:
        failures.extend(wrapper_probe.get("failures") or ["wrapper_cache_projection_probe_failed"])
    if not changed_fields:
        failures.append("enabled_disabled_forward_tensors_identical")
    mechanism_observed = bool(wrapper_changed_fields) if wrapper_probe.get("status") == "ok" else bool(changed_fields)
    return {
        "probe_type": "repo_small_batch_forward",
        "mechanism_observed": mechanism_observed and not failures,
        "compared_fields": [item["name"] for item in all_tensor_checks],
        "changed_fields": changed_fields,
        "unchanged_fields": unchanged_fields,
        "tensor_checks": all_tensor_checks,
        "projector_tensor_checks": tensor_checks,
        "wrapper_probe": wrapper_probe,
        "cache_key_diff": wrapper_probe.get("cache_key_diff"),
        "cache_value_diff": wrapper_probe.get("cache_value_diff"),
        "projector_called": wrapper_probe.get("projector_called"),
        "switch_seen_by_forward": wrapper_probe.get("switch_seen_by_forward"),
        "projector_spec": _compact_projector_spec(projector_spec),
        "enabled": {
            "switch_value": enabled_rosetta.get(switch),
            "rosetta_hash": _stable_hash(enabled_rosetta),
        },
        "disabled": {
            "switch_value": disabled_rosetta.get(switch),
            "rosetta_hash": _stable_hash(disabled_rosetta),
        },
        "failures": failures,
    }


def _repo_wrapper_cache_projection_probe(
    torch_mod: Any,
    enabled_projector: Any,
    disabled_projector: Any,
    switch: str,
) -> dict[str, Any]:
    try:
        from rosetta.model.wrapper import RosettaModel
    except Exception as exc:
        return _wrapper_probe_failed("wrapper_import_failed", exc)

    try:
        enabled_cache, enabled_calls = _run_fake_rosetta_forward_projection(torch_mod, RosettaModel, enabled_projector)
        disabled_cache, disabled_calls = _run_fake_rosetta_forward_projection(torch_mod, RosettaModel, disabled_projector)
        tensor_checks = _cache_tensor_diff_checks(torch_mod, enabled_cache, disabled_cache)
    except Exception as exc:
        return _wrapper_probe_failed("wrapper_forward_projection_failed", exc)

    key_diffs = [item["max_abs_diff"] for item in tensor_checks if item["name"].endswith(".key")]
    value_diffs = [item["max_abs_diff"] for item in tensor_checks if item["name"].endswith(".value")]
    changed_fields = [item["name"] for item in tensor_checks if item.get("changed")]
    trace = _runtime_switch_trace(Path.cwd(), switch)
    projector_called = bool(enabled_calls.get("projector_forward_calls") and disabled_calls.get("projector_forward_calls"))
    failures = []
    if not projector_called:
        failures.append("wrapper_projector_not_called")
    if not changed_fields:
        failures.append("enabled_disabled_wrapper_cache_identical")
    return {
        "status": "ok",
        "probe_type": "wrapper_cache_projection",
        "projector_called": projector_called,
        "projector_forward_calls": {
            "enabled": int(enabled_calls.get("projector_forward_calls") or 0),
            "disabled": int(disabled_calls.get("projector_forward_calls") or 0),
        },
        "switch_seen_by_forward": bool(trace.get("forward_refs") or _projector_has_switch_attr(enabled_projector, switch) or _projector_has_switch_attr(disabled_projector, switch)),
        "cache_key_diff": max(key_diffs) if key_diffs else None,
        "cache_value_diff": max(value_diffs) if value_diffs else None,
        "changed_fields": changed_fields,
        "unchanged_fields": [item["name"] for item in tensor_checks if not item.get("changed")],
        "tensor_checks": tensor_checks,
        "trace": trace,
        "failures": failures,
    }


def _run_fake_rosetta_forward_projection(torch_mod: Any, RosettaModel: Any, projector: Any) -> tuple[Any, dict[str, int]]:
    base_model = _FakeCausalLM(torch_mod, offset=0.05)
    source_model = _FakeCausalLM(torch_mod, offset=1.25)
    model = RosettaModel(
        [base_model, source_model],
        base_model_idx=0,
        projector_list=[projector],
        multi_source_fusion_mode="parallel",
    )
    model.set_projector_config(
        source_model_idx=1,
        source_model_layer_idx=0,
        target_model_idx=0,
        target_model_layer_idx=0,
        projector_idx=0,
    )
    tracked_projector = model.projector_list[0]
    call_state = {"projector_forward_calls": 0}
    original_forward = tracked_projector.forward

    def counted_forward(*args: Any, **kwargs: Any) -> Any:
        call_state["projector_forward_calls"] += 1
        return original_forward(*args, **kwargs)

    tracked_projector.forward = counted_forward
    try:
        input_ids = torch_mod.tensor([[5, 7, 11]], dtype=torch_mod.long)
        attention_mask = torch_mod.ones_like(input_ids)
        kv_cache_index = [
            torch_mod.tensor([[[1, 0], [1, 0]]], dtype=torch_mod.long),
            torch_mod.tensor([[[-1, 0]]], dtype=torch_mod.long),
        ]
        with torch_mod.no_grad():
            output = model.forward(
                kv_cache_index=kv_cache_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
    finally:
        tracked_projector.forward = original_forward
    return output.past_key_values, call_state


class _FakeCausalLM:
    def __new__(cls, torch_mod: Any, offset: float) -> Any:
        class FakeCausalLM(torch_mod.nn.Module):
            def __init__(self, offset_value: float) -> None:
                super().__init__()
                self._probe_param = torch_mod.nn.Parameter(torch_mod.zeros((), dtype=torch_mod.float32), requires_grad=False)
                self.offset = float(offset_value)
                self.config = SimpleNamespace(num_hidden_layers=1)
                self.is_gradient_checkpointing = False

            @property
            def device(self) -> Any:
                return self._probe_param.device

            @property
            def dtype(self) -> Any:
                return self._probe_param.dtype

            def gradient_checkpointing_disable(self) -> None:
                self.is_gradient_checkpointing = False

            def gradient_checkpointing_enable(self) -> None:
                self.is_gradient_checkpointing = True

            def forward(self, input_ids: Any = None, past_key_values: Any = None, use_cache: bool = True, **kwargs: Any) -> Any:
                del use_cache, kwargs
                from transformers.cache_utils import DynamicCache

                if input_ids is None:
                    input_ids = torch_mod.zeros((1, 1), dtype=torch_mod.long, device=self.device)
                input_ids = input_ids.to(self.device)
                batch_size, seq_len = input_ids.shape
                heads = 2
                head_dim = 4
                base = torch_mod.arange(
                    batch_size * heads * seq_len * head_dim,
                    dtype=self.dtype,
                    device=self.device,
                ).view(batch_size, heads, seq_len, head_dim)
                token_signal = input_ids[:, None, :, None].to(dtype=self.dtype) / 100.0
                key = base / 50.0 + token_signal + self.offset
                value = torch_mod.sin(base / 25.0 + token_signal + self.offset)
                cache = past_key_values if past_key_values is not None else DynamicCache()
                cache.update(key, value, 0)
                logits = torch_mod.zeros(batch_size, seq_len, 4, dtype=self.dtype, device=self.device)
                return SimpleNamespace(logits=logits, past_key_values=cache)

        return FakeCausalLM(offset)


def _cache_tensor_diff_checks(torch_mod: Any, enabled_cache: Any, disabled_cache: Any) -> list[dict[str, Any]]:
    checks = []
    enabled_keys = list(getattr(enabled_cache, "key_cache", []) or [])
    disabled_keys = list(getattr(disabled_cache, "key_cache", []) or [])
    enabled_values = list(getattr(enabled_cache, "value_cache", []) or [])
    disabled_values = list(getattr(disabled_cache, "value_cache", []) or [])
    layer_count = min(len(enabled_keys), len(disabled_keys), len(enabled_values), len(disabled_values))
    for layer_idx in range(layer_count):
        checks.append(_tensor_check(torch_mod, f"wrapper_cache.layer{layer_idx}.key", enabled_keys[layer_idx], disabled_keys[layer_idx]))
        checks.append(_tensor_check(torch_mod, f"wrapper_cache.layer{layer_idx}.value", enabled_values[layer_idx], disabled_values[layer_idx]))
    return checks


def _projector_has_switch_attr(projector: Any, switch: str) -> bool:
    return bool(switch and hasattr(projector, switch))


def _wrapper_probe_failed(reason: str, exc: Exception) -> dict[str, Any]:
    return {
        "status": "failed",
        "probe_type": "wrapper_cache_projection",
        "projector_called": False,
        "switch_seen_by_forward": False,
        "cache_key_diff": None,
        "cache_value_diff": None,
        "tensor_checks": [],
        "error": {"type": type(exc).__name__, "message": str(exc)[-1000:]},
        "failures": [reason],
    }


def _merge_probe_evidence(static_evidence: dict[str, Any], forward_evidence: dict[str, Any]) -> dict[str, Any]:
    if forward_evidence.get("probe_type") == "repo_small_batch_forward":
        payload = dict(forward_evidence)
        payload["static_trace"] = {
            "mechanism_observed": static_evidence.get("mechanism_observed"),
            "changed_fields": static_evidence.get("changed_fields") or [],
            "unchanged_fields": static_evidence.get("unchanged_fields") or [],
            "failures": static_evidence.get("failures") or [],
            "trace": static_evidence.get("trace") or {},
        }
        return payload
    payload = dict(static_evidence)
    payload["probe_type"] = "repo_small_batch_forward_failed_static_trace"
    payload["fallback_reason"] = forward_evidence.get("fallback_reason") or "repo_small_batch_forward_unavailable"
    payload["forward_probe_error"] = forward_evidence.get("error") or {}
    payload["forward_probe_projector_spec"] = forward_evidence.get("projector_spec") or {}
    failures = list(payload.get("failures") or [])
    failures.append(str(payload["fallback_reason"]))
    payload["failures"] = list(dict.fromkeys(failures))
    payload["mechanism_observed"] = False
    return payload


def _projector_spec(rosetta_config: dict[str, Any]) -> dict[str, Any]:
    projector = rosetta_config.get("projector") if isinstance(rosetta_config.get("projector"), dict) else {}
    params = projector.get("params") if isinstance(projector.get("params"), dict) else {}
    return {
        "type": str(projector.get("type") or "C2CProjector"),
        "params": dict(params),
    }


def _build_probe_projector(create_projector: Any, spec: dict[str, Any], rosetta_config: dict[str, Any], switch: str) -> Any:
    params = dict(spec.get("params") or {})
    params.update(_probe_mechanism_params(rosetta_config, switch))
    params.setdefault("hidden_dim", 16)
    params.setdefault("intermediate_dim", 16)
    params.setdefault("num_layers", 3)
    params.setdefault("dropout", 0.0)
    params.setdefault("source_num_heads", 2)
    params.setdefault("target_num_heads", 2)
    params.setdefault("dtype", __import__("torch").float32)
    params = _constructor_safe_projector_params(str(spec.get("type") or "C2CProjector"), params)
    return create_projector(
        str(spec.get("type") or "C2CProjector"),
        source_dim=4,
        target_dim=4,
        **params,
    )


def _probe_mechanism_params(rosetta_config: dict[str, Any], switch: str) -> dict[str, Any]:
    result = {}
    for key, value in rosetta_config.items():
        if key in {"base_model", "teacher_model", "projector", "checkpoints_dir"}:
            continue
        if key == switch:
            result[key] = value
            continue
        if any(token in str(key).lower() for token in ["alignment", "router", "gate", "cache", "prior", "span", "confidence", "weight", "disable"]):
            result[key] = value
    return result


def _constructor_safe_projector_params(projector_type: str, params: dict[str, Any]) -> dict[str, Any]:
    projector_cls = _projector_class(projector_type)
    if projector_cls is None:
        return params
    try:
        signature = inspect.signature(projector_cls.__init__)
    except (TypeError, ValueError):
        return params
    ast_allowed = _projector_init_param_names_from_source(projector_type)
    if ast_allowed:
        return {key: value for key, value in params.items() if key in ast_allowed}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return params
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if name not in {"self", "source_dim", "target_dim"}
        and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {key: value for key, value in params.items() if key in allowed}


def _projector_class(projector_type: str) -> Any | None:
    module = sys.modules.get("rosetta.model.projector")
    if module is None:
        return None
    getter = getattr(module, "get_projector_class", None)
    if callable(getter):
        try:
            return getter(projector_type)
        except Exception:
            pass
    candidate = getattr(module, projector_type, None)
    return candidate if isinstance(candidate, type) else None


def _projector_init_param_names_from_source(projector_type: str) -> set[str]:
    path = Path.cwd() / "rosetta/model/projector.py"
    if not path.exists():
        return set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
    except SyntaxError:
        return set()
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != projector_type:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name == "__init__":
                return {
                    arg.arg
                    for arg in [*child.args.args, *child.args.kwonlyargs]
                    if arg.arg not in {"self", "source_dim", "target_dim"}
                }
    return set()


def _synthetic_kv_batch(torch_mod: Any) -> tuple[tuple[Any, Any], tuple[Any, Any]]:
    shape = (1, 2, 4, 4)
    source_key = torch_mod.linspace(-0.7, 0.8, steps=shape[0] * shape[1] * shape[2] * shape[3], dtype=torch_mod.float32).view(*shape)
    source_value = torch_mod.cos(source_key * 1.7)
    target_key = torch_mod.linspace(0.3, -0.5, steps=shape[0] * shape[1] * shape[2] * shape[3], dtype=torch_mod.float32).view(*shape)
    target_value = torch_mod.sin(target_key * 2.3)
    return (source_key, source_value), (target_key, target_value)


def _tensor_diff_checks(torch_mod: Any, enabled_projector: Any, disabled_projector: Any, enabled_output: Any, disabled_output: Any) -> list[dict[str, Any]]:
    checks = []
    for idx, name in enumerate(["projector_output.key", "projector_output.value"]):
        checks.append(_tensor_check(torch_mod, name, enabled_output[idx], disabled_output[idx]))
    for attr in ["gate_logit", "key_weight", "value_weight"]:
        if hasattr(enabled_projector, attr) and hasattr(disabled_projector, attr):
            checks.append(_tensor_check(torch_mod, f"projector_param.{attr}", getattr(enabled_projector, attr), getattr(disabled_projector, attr)))
    for attr in ["last_alignment_scores", "last_router_logits", "last_cache_weights", "last_projector_output"]:
        if hasattr(enabled_projector, attr) and hasattr(disabled_projector, attr):
            enabled_value = getattr(enabled_projector, attr)
            disabled_value = getattr(disabled_projector, attr)
            if enabled_value is not None and disabled_value is not None:
                checks.append(_tensor_check(torch_mod, f"instrumentation.{attr}", enabled_value, disabled_value))
    return checks


def _tensor_check(torch_mod: Any, name: str, enabled_value: Any, disabled_value: Any) -> dict[str, Any]:
    enabled_tensor = enabled_value.detach().float().cpu() if hasattr(enabled_value, "detach") else torch_mod.as_tensor(enabled_value, dtype=torch_mod.float32)
    disabled_tensor = disabled_value.detach().float().cpu() if hasattr(disabled_value, "detach") else torch_mod.as_tensor(disabled_value, dtype=torch_mod.float32)
    max_abs_diff = float((enabled_tensor - disabled_tensor).abs().max().item())
    mean_abs_diff = float((enabled_tensor - disabled_tensor).abs().mean().item())
    return {
        "name": name,
        "changed": bool(max_abs_diff > 1e-7),
        "max_abs_diff": round(max_abs_diff, 10),
        "mean_abs_diff": round(mean_abs_diff, 10),
        "enabled_sha256": _tensor_hash(enabled_tensor),
        "disabled_sha256": _tensor_hash(disabled_tensor),
        "shape": list(enabled_tensor.shape),
    }


def _tensor_hash(tensor: Any) -> str:
    array = tensor.contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def _forward_probe_failed(reason: str, exc: Exception, *, projector_spec: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "probe_type": "repo_small_batch_forward_failed",
        "mechanism_observed": False,
        "fallback_reason": reason,
        "projector_spec": _compact_projector_spec(projector_spec or {}),
        "error": {"type": type(exc).__name__, "message": str(exc)[-1000:]},
        "failures": [reason],
    }


def _compact_projector_spec(spec: dict[str, Any]) -> dict[str, Any]:
    params = spec.get("params") if isinstance(spec.get("params"), dict) else {}
    return {
        "type": spec.get("type"),
        "param_keys": sorted(str(key) for key in params.keys())[:80],
    }


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".json":
        return json.loads(text or "{}")
    loaded = yaml.safe_load(text) or {}
    return loaded if isinstance(loaded, dict) else {}


def _rosetta_config(config: dict[str, Any]) -> dict[str, Any]:
    model = config.get("model") if isinstance(config.get("model"), dict) else {}
    rosetta = model.get("rosetta_config") if isinstance(model.get("rosetta_config"), dict) else {}
    return dict(rosetta or model)


def _runtime_switch_trace(repo_root: Path, switch: str) -> dict[str, Any]:
    switch_refs = []
    config_read_refs = []
    forward_refs = []
    syntax_errors = []
    for rel_path in RUNTIME_FILES:
        path = repo_root / rel_path
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if switch in text:
            switch_refs.append(rel_path.as_posix())
        if "rosetta_config" in text or "config.get" in text or ".get(" in text:
            config_read_refs.append(rel_path.as_posix())
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            syntax_errors.append({"path": rel_path.as_posix(), "error": str(exc)})
            continue
        if _forward_mentions_switch_or_config(tree, switch):
            forward_refs.append(rel_path.as_posix())
    return {
        "switch_refs": list(dict.fromkeys(switch_refs)),
        "config_read_refs": list(dict.fromkeys(config_read_refs)),
        "forward_refs": list(dict.fromkeys(forward_refs)),
        "syntax_errors": syntax_errors,
    }


def _forward_mentions_switch_or_config(tree: ast.AST, switch: str) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "forward":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                if child.value == switch or child.value == "rosetta_config":
                    return True
            if isinstance(child, ast.Name) and child.id == "rosetta_config":
                return True
    return False


def _stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    raise SystemExit(main())
