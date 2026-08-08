"""Resolve portable OCR profiles against machine-local installations.

This module is the narrow bridge between shareable execution profiles and
machine-specific executable/cache records. It remains standard-library-only at
import time and probes backend runtimes in isolated, secret-scrubbed
subprocesses.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import selectors
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .ocr_process_lifecycle import isolated_subprocess_signal_guard

SCHEMA_VERSION = 1
_MAX_PROBE_OUTPUT_BYTES = 64 * 1024
_PROBE_READ_BYTES = 8 * 1024
_SUBPROCESS_RUNNER = subprocess.run
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CUDA_RUNTIME_MODULE_ROOTS = ("torch", "vllm")
_SAFE_ENV_NAMES = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LANGUAGE",
        "LC_ALL",
        "LC_CTYPE",
        "LD_LIBRARY_PATH",
        "NVIDIA_VISIBLE_DEVICES",
        "PATH",
        "TZ",
    }
)
_RUNTIME_PROBE_CODE = r"""
import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, distributions, version

engine, attention_backend, dtype, *package_names = sys.argv[1:]
try:
    import torch

    packages = {}
    for name in package_names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    engine_version = packages.get(engine)
    if not isinstance(engine_version, str) or not engine_version:
        raise RuntimeError("OCR engine distribution is unavailable")
    missing = sorted(
        name for name, value in packages.items() if value is None
    )
    if missing:
        raise RuntimeError("OCR runtime distributions are unavailable")
    installed = sorted(
        f"{item.metadata.get('Name', item.name)}=={item.version}"
        for item in distributions()
    )
    installed_sha256 = hashlib.sha256(
        json.dumps(
            installed,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    result = {
        "ok": True,
        "runtime": {
            "python": platform.python_version(),
            "engine": engine,
            "engine_version": engine_version,
            "torch": str(torch.__version__),
            "torch_cuda": str(torch.version.cuda),
            "packages": packages,
            "attention_backend_requested": attention_backend,
            "dtype": dtype,
            "installed_distributions_count": len(installed),
            "installed_distributions_sha256": installed_sha256,
        },
    }
except BaseException as exc:
    result = {"ok": False, "error_type": type(exc).__name__}

print(json.dumps(result, sort_keys=True, separators=(",", ":")))
raise SystemExit(0 if result["ok"] else 1)
""".strip()


class OcrExecutionError(ValueError):
    """Raised when a structured OCR execution cannot be resolved safely."""


def default_profile_directory(
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the XDG directory containing portable OCR profiles."""

    environment = os.environ if environ is None else environ
    configured = environment.get("XDG_CONFIG_HOME")
    if configured:
        root = Path(configured).expanduser()
    else:
        home = environment.get("HOME")
        root = (Path(home).expanduser() if home else Path.home()) / ".config"
    if not root.is_absolute():
        raise OcrExecutionError(
            "OCR profile configuration root is not absolute"
        )
    return root / "sneeze" / "tpn" / "ocr-profiles"


def bundled_profile_directory() -> Path:
    """Return the read-only profile directory shipped with the plugin."""

    return Path(__file__).resolve().parent / "ocr_profiles_data"


def validate_bundled_profile_registry() -> tuple[dict[str, Any], ...]:
    """Require bundled profiles and the reviewed allowlist to agree."""

    from .ocr_profiles import (
        TRUSTED_QUALIFIED_PROFILE_SHA256S,
        list_execution_profiles,
    )

    directory = bundled_profile_directory()
    profiles = (
        list_execution_profiles(paths=(directory,))
        if directory.is_dir()
        else ()
    )
    if any(
        profile["qualification"]["state"] != "qualified"
        for profile in profiles
    ):
        raise OcrExecutionError(
            "bundled OCR profiles must all be qualified"
        )
    bundled_hashes = {profile["profile_sha256"] for profile in profiles}
    trusted_hashes = set(TRUSTED_QUALIFIED_PROFILE_SHA256S)
    if bundled_hashes != trusted_hashes:
        raise OcrExecutionError(
            "bundled OCR profiles and the reviewed trust registry differ"
        )
    return tuple(copy.deepcopy(profile) for profile in profiles)


def profile_search_paths(
    value: str | os.PathLike[str] | None = None,
) -> tuple[Path, ...]:
    """Resolve a comma-separated profile search path or the XDG default."""

    raw_values = (
        [item.strip() for item in str(value).split(",") if item.strip()]
        if value is not None
        else [
            str(bundled_profile_directory()),
            str(default_profile_directory()),
        ]
    )
    validate_bundled_profile_registry()
    if not raw_values:
        raise OcrExecutionError("OCR profile search path cannot be empty")
    paths = []
    for raw in raw_values:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            raise OcrExecutionError(
                "OCR profile search paths must be absolute"
            )
        paths.append(path)
    return tuple(dict.fromkeys(paths))


def _safe_probe_environment(
    installation: Mapping[str, Any],
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment_source = os.environ if source is None else source
    environment = {
        name: value
        for name, value in environment_source.items()
        if name in _SAFE_ENV_NAMES
        and isinstance(value, str)
        and "\x00" not in value
    }
    cuda = installation["cuda"]
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": cuda["visible_devices"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONNOUSERSITE": "1",
        }
    )
    return environment


def _run_bounded_runtime_probe(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> subprocess.CompletedProcess[str]:
    """Run one runtime probe with bounded retained stdout and stderr."""

    # Tests and callers may inject an already-completed runner. The default
    # real runner takes the streaming path so a producer cannot allocate
    # unbounded parent-process memory before the size check runs.
    if runner is not _SUBPROCESS_RUNNER:
        try:
            completed = runner(
                argv,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=dict(environment),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OcrExecutionError(
                "runtime probe command failed: "
                f"{type(exc).__name__}"
            ) from exc
        for name, value in (
            ("stdout", completed.stdout),
            ("stderr", completed.stderr),
        ):
            if (
                len(value.encode("utf-8", errors="replace"))
                > _MAX_PROBE_OUTPUT_BYTES
            ):
                raise OcrExecutionError(
                    f"runtime probe {name} exceeded 64 KiB"
                )
        return completed

    with isolated_subprocess_signal_guard() as lifecycle:
        return _run_bounded_runtime_probe_guarded(
            argv,
            environment=environment,
            timeout_seconds=timeout_seconds,
            lifecycle=lifecycle,
        )


def _run_bounded_runtime_probe_guarded(
    argv: list[str],
    *,
    environment: Mapping[str, str],
    timeout_seconds: float,
    lifecycle: Any,
) -> subprocess.CompletedProcess[str]:
    process: subprocess.Popen[bytes] | None = None
    streams: dict[str, Any] = {}
    chunks: dict[str, list[bytes]] = {"stdout": [], "stderr": []}
    sizes = {"stdout": 0, "stderr": 0}
    selector = selectors.DefaultSelector()

    def stop_process() -> None:
        if process is None:
            return
        # Descendants can keep inherited pipes open after the direct child
        # exits, so always target the private process group first.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError:
            if process.poll() is None:
                try:
                    process.kill()
                except OSError:
                    pass
        try:
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass

    try:
        try:
            with lifecycle.spawn_window():
                process = subprocess.Popen(
                    argv,
                    env=dict(environment),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
        except OSError as exc:
            raise OcrExecutionError(
                f"runtime probe command failed: {type(exc).__name__}"
            ) from exc

        streams = {
            "stdout": process.stdout,
            "stderr": process.stderr,
        }
        if any(  # pragma: no cover - Popen PIPE invariant
            stream is None for stream in streams.values()
        ):
            raise OcrExecutionError("runtime probe pipes were not created")
        for name, stream in streams.items():
            assert stream is not None
            selector.register(stream, selectors.EVENT_READ, name)
        deadline = time.monotonic() + timeout_seconds
        while selector.get_map():
            lifecycle.checkpoint()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(argv, timeout_seconds)
            for key, _mask in selector.select(timeout=min(remaining, 0.1)):
                name = key.data
                allowance = _MAX_PROBE_OUTPUT_BYTES - sizes[name]
                chunk = os.read(
                    key.fd,
                    min(_PROBE_READ_BYTES, allowance + 1),
                )
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                if len(chunk) > allowance:
                    raise OcrExecutionError(
                        f"runtime probe {name} exceeded 64 KiB"
                    )
                chunks[name].append(chunk)
                sizes[name] += len(chunk)

        remaining = deadline - time.monotonic()
        if process.poll() is None and remaining <= 0:
            raise subprocess.TimeoutExpired(argv, timeout_seconds)
        returncode = lifecycle.wait(
            process,
            timeout_seconds=max(0, remaining),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcrExecutionError(
            f"runtime probe command failed: {type(exc).__name__}"
        ) from exc
    finally:
        # The handler remains passive until the whole private group is killed
        # and reaped, including normal and partially initialized paths.
        with lifecycle.cleanup_window():
            stop_process()
            selector.close()
            for stream in streams.values():
                if stream is not None and not stream.closed:
                    stream.close()

    return subprocess.CompletedProcess(
        argv,
        returncode,
        stdout=b"".join(chunks["stdout"]).decode(
            "utf-8", errors="replace"
        ),
        stderr=b"".join(chunks["stderr"]).decode(
            "utf-8", errors="replace"
        ),
    )


def probe_backend_runtime(
    installation: Mapping[str, Any],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Probe the exact runtime tuple without loading the model."""

    from .ocr_backends import get_backend

    backend = get_backend(str(installation.get("backend_id", "")))
    executable = installation.get("python_executable", {}).get("path")
    if not isinstance(executable, str) or not Path(executable).is_absolute():
        raise OcrExecutionError(
            f"installation for {backend.backend_id} has no absolute Python"
        )
    cuda = installation.get("cuda")
    if not isinstance(cuda, Mapping):
        raise OcrExecutionError(
            f"installation for {backend.backend_id} has no CUDA binding"
        )
    attention = cuda.get("attention_backend")
    if not isinstance(attention, str) or not attention:
        attention = (
            "auto"
            if backend.backend_id == "deepseek-vllm"
            else backend.default_attention_backend
        )
    engine = (
        "vllm" if backend.backend_id == "deepseek-vllm" else "transformers"
    )
    package_names = [name for name, _version in backend.runtime_packages]
    argv = [
        executable,
        "-I",
        "-B",
        "-c",
        _RUNTIME_PROBE_CODE,
        engine,
        attention,
        "bfloat16",
        *package_names,
    ]
    completed = _run_bounded_runtime_probe(
        argv,
        environment=_safe_probe_environment(installation, environment),
        timeout_seconds=120,
        runner=runner,
    )
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    try:
        result = json.loads(lines[-1]) if lines else None
    except ValueError as exc:
        raise OcrExecutionError(
            f"runtime probe returned invalid JSON for {backend.backend_id}"
        ) from exc
    if (
        completed.returncode != 0
        or not isinstance(result, Mapping)
        or result.get("ok") is not True
        or not isinstance(result.get("runtime"), Mapping)
    ):
        error_type = (
            result.get("error_type")
            if isinstance(result, Mapping)
            else "InvalidProbeResult"
        )
        raise OcrExecutionError(
            f"runtime probe failed for {backend.backend_id}: {error_type}"
        )
    return copy.deepcopy(dict(result["runtime"]))


def build_execution_observation(
    *,
    pipeline_id: str,
    hardware: Mapping[str, Any],
    installations: Mapping[str, Mapping[str, Any]],
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    """Build the hardware/model/runtime observation used for matching."""

    from .ocr_profiles import get_pipeline_definition

    pipeline = get_pipeline_definition(pipeline_id)
    runtime_by_backend: dict[str, dict[str, Any]] = {}
    stages = {}
    for stage in pipeline["stages"]:
        backend_id = stage["backend_id"]
        installation = installations.get(backend_id)
        if not isinstance(installation, Mapping):
            raise OcrExecutionError(
                f"no machine installation is registered for {backend_id}"
            )
        snapshot = installation.get("snapshot")
        snapshot_files_sha256 = (
            snapshot.get("files_sha256")
            if isinstance(snapshot, Mapping)
            else None
        )
        if not isinstance(
            snapshot_files_sha256, str
        ) or not _SHA256_RE.fullmatch(snapshot_files_sha256):
            raise OcrExecutionError(
                "installation for "
                f"{backend_id} has no exact model snapshot identity"
            )
        implementation = installation.get("implementation")
        implementation_sha256 = (
            implementation.get("sha256")
            if isinstance(implementation, Mapping)
            else None
        )
        if not isinstance(
            implementation_sha256, str
        ) or not _SHA256_RE.fullmatch(implementation_sha256):
            raise OcrExecutionError(
                f"installation for {backend_id} has no exact "
                "implementation identity"
            )
        if backend_id not in runtime_by_backend:
            runtime_by_backend[backend_id] = probe_backend_runtime(
                installation,
                runner=runner,
            )
            runtime_by_backend[backend_id]["implementation_sha256"] = (
                implementation_sha256
            )
        stages[stage["stage_id"]] = {
            "backend_id": backend_id,
            "capability": stage["capability"],
            "model": {
                **copy.deepcopy(stage["model"]),
                "snapshot_files_sha256": snapshot_files_sha256,
            },
            "runtime": copy.deepcopy(runtime_by_backend[backend_id]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "hardware": copy.deepcopy(dict(hardware)),
        "stages": stages,
    }


def resolve_execution_profile(
    selection: str,
    *,
    pipeline_id: str | None,
    memory_tier: str,
    observation: Mapping[str, Any],
    search_paths: Sequence[str | os.PathLike[str]],
    allow_provisional: bool = False,
) -> dict[str, Any]:
    """Resolve auto, explicit-path, or profile-ID selection and explain it."""

    from .ocr_profiles import (
        TRUSTED_QUALIFIED_PROFILE_SHA256S,
        OcrProfileError,
        auto_select_execution_profile,
        get_pipeline_definition,
        list_execution_profiles,
        load_execution_profile,
        profile_match_report,
        show_execution_profile,
    )

    normalized = selection.strip()
    if not normalized:
        raise OcrExecutionError("execution profile selection cannot be empty")
    validate_bundled_profile_registry()
    paths = tuple(Path(item).expanduser() for item in search_paths)
    try:
        source = None
        if normalized == "auto":
            if not pipeline_id:
                raise OcrExecutionError(
                    "--execution-profile auto requires --pipeline"
                )
            profiles = list_execution_profiles(
                paths=tuple(path for path in paths if path.exists())
            )
            profile = auto_select_execution_profile(
                profiles,
                observation,
                pipeline_id=pipeline_id,
                memory_tier=memory_tier,
                require_qualified=not allow_provisional,
                trusted_qualified_profile_sha256s=(
                    TRUSTED_QUALIFIED_PROFILE_SHA256S
                ),
            )
            source = "auto"
        else:
            candidate = Path(normalized).expanduser()
            path_like = (
                candidate.is_absolute()
                or normalized.startswith((".", "~"))
                or "/" in normalized
                or normalized.endswith(".json")
            )
            if path_like:
                profile = load_execution_profile(candidate)
                source = str(candidate.resolve(strict=True))
            else:
                profile = show_execution_profile(
                    normalized,
                    paths=tuple(path for path in paths if path.exists()),
                )
                source = "id:" + normalized
        canonical_pipeline = profile["pipeline_id"]
        if pipeline_id:
            requested = get_pipeline_definition(pipeline_id)["pipeline_id"]
            if canonical_pipeline != requested:
                raise OcrExecutionError(
                    "execution profile pipeline differs from --pipeline"
                )
        report = profile_match_report(
            profile,
            observation,
            memory_tier=memory_tier,
            require_qualified=not allow_provisional,
            trusted_qualified_profile_sha256s=(
                TRUSTED_QUALIFIED_PROFILE_SHA256S
            ),
        )
        if not report["matched"]:
            raise OcrExecutionError(
                "execution profile does not match this execution: "
                + ", ".join(report["reasons"])
            )
    except OcrExecutionError:
        raise
    except (OSError, OcrProfileError) as exc:
        raise OcrExecutionError(str(exc)) from exc
    return {
        "profile": profile,
        "profile_source": source,
        "match": report,
    }


def assert_in_process_deepseek_installation(
    installation: Mapping[str, Any],
    *,
    environ: Mapping[str, str] | None = None,
) -> None:
    """Ensure the process has the registered interpreter/GPU binding."""

    expected = installation.get("python_executable", {}).get("path")
    if not isinstance(expected, str):
        raise OcrExecutionError("DeepSeek installation has no Python binding")
    if Path(sys.executable).resolve() != Path(expected).resolve():
        raise OcrExecutionError(
            "structured OCR must run from the registered DeepSeek Python "
            f"interpreter: {expected}"
        )
    cuda = installation.get("cuda")
    if not isinstance(cuda, Mapping):
        raise OcrExecutionError("DeepSeek installation has no CUDA binding")
    expected_visible = cuda.get("visible_devices")
    if not isinstance(expected_visible, str) or not expected_visible:
        raise OcrExecutionError(
            "DeepSeek installation has no CUDA_VISIBLE_DEVICES binding"
        )
    environment = os.environ if environ is None else environ
    actual_order = environment.get("CUDA_DEVICE_ORDER")
    if actual_order != "PCI_BUS_ID":
        raise OcrExecutionError(
            "structured OCR requires CUDA_DEVICE_ORDER=PCI_BUS_ID before "
            "starting Sneeze"
        )
    actual_visible = environment.get("CUDA_VISIBLE_DEVICES")
    if actual_visible != expected_visible:
        raise OcrExecutionError(
            "structured OCR requires CUDA_VISIBLE_DEVICES to match the "
            f"registered DeepSeek binding: {expected_visible}"
        )
    imported_runtime_roots = sorted(
        root
        for root in _CUDA_RUNTIME_MODULE_ROOTS
        if any(
            name == root or name.startswith(root + ".")
            for name in sys.modules
        )
    )
    if imported_runtime_roots:
        raise OcrExecutionError(
            "structured OCR must verify the registered CUDA binding before "
            "importing CUDA runtime modules: "
            + ", ".join(imported_runtime_roots)
        )


def stage_runtime_mismatches(
    profile: Mapping[str, Any],
    *,
    stage_id: str,
    runtime: Mapping[str, Any],
) -> list[str]:
    """Return exact selector mismatches for one observed stage runtime."""

    from .ocr_profiles import validate_execution_profile

    value = validate_execution_profile(profile)
    try:
        expected = value["stages"][stage_id]["runtime_selector"]
    except KeyError as exc:
        raise OcrExecutionError(f"unknown profile stage: {stage_id}") from exc

    def compare(
        wanted: Mapping[str, Any],
        observed: Mapping[str, Any],
        prefix: str,
    ) -> list[str]:
        reasons = []
        for key, expected_value in wanted.items():
            label = f"{prefix}.{key}"
            if key not in observed:
                reasons.append(f"{label} is missing")
                continue
            actual = observed[key]
            if isinstance(expected_value, Mapping):
                if not isinstance(actual, Mapping):
                    reasons.append(f"{label} differs")
                else:
                    reasons.extend(compare(expected_value, actual, label))
            elif actual != expected_value:
                reasons.append(f"{label} differs")
        return reasons

    return compare(expected, runtime, f"stages.{stage_id}.runtime")


def stage_model_mismatches(
    profile: Mapping[str, Any],
    *,
    stage_id: str,
    model: Mapping[str, Any],
) -> list[str]:
    """Compare a provider's loaded model bytes with the selected profile."""

    from .ocr_profiles import validate_execution_profile

    value = validate_execution_profile(profile)
    try:
        expected = value["stages"][stage_id]["model"]
    except KeyError as exc:
        raise OcrExecutionError(f"unknown profile stage: {stage_id}") from exc
    reasons = []
    for key in ("requested", "revision", "snapshot_files_sha256"):
        if model.get(key) != expected.get(key):
            reasons.append(f"stages.{stage_id}.model.{key} differs")
    resolved_revision = model.get("resolved_revision")
    if resolved_revision != expected.get("revision"):
        reasons.append(
            f"stages.{stage_id}.model.resolved_revision differs"
        )
    return reasons


def deepseek_engine_identity_mismatches(
    profile: Mapping[str, Any],
    *,
    installation: Mapping[str, Any],
    engine_metadata: Mapping[str, Any],
) -> list[str]:
    """Compare the loaded DeepSeek engine with its sealed identities.

    The preflight runtime probe proves the complete package tuple before model
    loading. This check binds the in-process engine that was actually created
    to the same profile, registered snapshot, and selected attention backend.
    """

    from .ocr_profiles import validate_execution_profile

    value = validate_execution_profile(profile)
    stage = value["stages"].get("grounded_ocr")
    if not isinstance(stage, Mapping) or stage.get("backend_id") != (
        "deepseek-vllm"
    ):
        raise OcrExecutionError(
            "execution profile has no DeepSeek grounded-OCR stage"
        )

    reasons: list[str] = []
    expected_model = stage["model"]
    expected_runtime = stage["runtime_selector"]
    installed_model = installation.get("model")
    installed_snapshot = installation.get("snapshot")
    installed_cuda = installation.get("cuda")
    loaded_model = engine_metadata.get("model")
    loaded_runtime = engine_metadata.get("runtime")

    if installation.get("backend_id") != "deepseek-vllm":
        reasons.append("installation.backend_id differs")
    if not isinstance(installed_model, Mapping):
        reasons.append("installation.model is missing")
        installed_model = {}
    if not isinstance(installed_snapshot, Mapping):
        reasons.append("installation.snapshot is missing")
        installed_snapshot = {}
    if not isinstance(installed_cuda, Mapping):
        reasons.append("installation.cuda is missing")
        installed_cuda = {}
    if not isinstance(loaded_model, Mapping):
        reasons.append("engine.model is missing")
        loaded_model = {}
    if not isinstance(loaded_runtime, Mapping):
        reasons.append("engine.runtime is missing")
        loaded_runtime = {}

    for key in ("id", "revision"):
        expected_key = "requested" if key == "id" else key
        if installed_model.get(key) != expected_model.get(expected_key):
            reasons.append(f"installation.model.{key} differs from profile")
    if installed_snapshot.get("files_sha256") != expected_model.get(
        "snapshot_files_sha256"
    ):
        reasons.append(
            "installation.snapshot.files_sha256 differs from profile"
        )

    snapshot_path = installed_snapshot.get("path")
    resolved_path = loaded_model.get("resolved_path")
    if not isinstance(snapshot_path, str) or not snapshot_path:
        reasons.append("installation.snapshot.path is missing")
    if not isinstance(resolved_path, str) or not resolved_path:
        reasons.append("engine.model.resolved_path is missing")
    elif isinstance(snapshot_path, str) and snapshot_path:
        if os.path.realpath(resolved_path) != os.path.realpath(snapshot_path):
            reasons.append(
                "engine.model.resolved_path differs from installation"
            )

    requested_model = loaded_model.get("requested")
    requested_matches_profile = requested_model == expected_model.get(
        "requested"
    )
    requested_matches_snapshot = (
        isinstance(requested_model, str)
        and requested_model
        and isinstance(snapshot_path, str)
        and snapshot_path
        and os.path.realpath(requested_model)
        == os.path.realpath(snapshot_path)
    )
    if not requested_matches_profile and not requested_matches_snapshot:
        reasons.append("engine.model.requested differs from installation")
    expected_revision = expected_model.get("revision")
    loaded_revision = loaded_model.get("revision")
    if loaded_revision not in {None, expected_revision}:
        reasons.append("engine.model.revision differs from profile")
    if loaded_model.get("snapshot") != expected_revision:
        reasons.append("engine.model.snapshot differs from profile")

    runtime_fields = {
        "python": "python",
        "engine_version": "vllm",
        "torch": "torch",
        "torch_cuda": "torch_cuda",
        "installed_distributions_count": "installed_distributions_count",
        "installed_distributions_sha256": "installed_distributions_sha256",
    }
    if expected_runtime.get("engine") != "vllm":
        reasons.append("profile runtime engine is not vllm")
    for expected_key, loaded_key in runtime_fields.items():
        actual = loaded_runtime.get(loaded_key)
        if actual is None:
            reasons.append(f"engine.runtime.{loaded_key} is missing")
        elif actual != expected_runtime.get(expected_key):
            reasons.append(
                f"engine.runtime.{loaded_key} differs from profile"
            )

    expected_packages = expected_runtime.get("packages", {})
    for package_name, loaded_key in (
        ("transformers", "transformers"),
        ("pillow", "pillow"),
    ):
        expected_version = expected_packages.get(package_name)
        actual_version = loaded_runtime.get(loaded_key)
        if (
            expected_version is not None
            and actual_version != expected_version
        ):
            reasons.append(
                f"engine.runtime.{loaded_key} differs from profile package"
            )

    expected_dtype = str(expected_runtime.get("dtype", "")).removeprefix(
        "torch."
    )
    actual_dtype = str(loaded_runtime.get("dtype", "")).removeprefix(
        "torch."
    )
    if not actual_dtype:
        reasons.append("engine.runtime.dtype is missing")
    elif actual_dtype != expected_dtype:
        reasons.append("engine.runtime.dtype differs from profile")

    selected_attention = loaded_runtime.get("attention_backend_selected")
    if not isinstance(selected_attention, str) or not selected_attention:
        reasons.append(
            "engine.runtime.attention_backend_selected is missing"
        )
    elif selected_attention != expected_runtime.get("attention_backend"):
        reasons.append(
            "engine.runtime.attention_backend_selected differs from profile"
        )
    configured_attention = installed_cuda.get("attention_backend")
    if (
        isinstance(configured_attention, str)
        and configured_attention
        and configured_attention != "auto"
        and selected_attention != configured_attention
    ):
        reasons.append(
            "engine.runtime.attention_backend_selected differs from "
            "installation"
        )
    return list(dict.fromkeys(reasons))


def build_profile_provider_spec(
    *,
    profile: Mapping[str, Any],
    profile_path: str | os.PathLike[str],
    memory_tier: str,
    installation: Mapping[str, Any],
    allow_provisional: bool = False,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Build a local-only MinerU provider spec bound to one sealed profile."""

    from .ocr_backends import get_backend
    from .ocr_profiles import validate_execution_profile

    value = validate_execution_profile(profile)
    if value["pipeline_id"] != "local-mineru25":
        raise OcrExecutionError(
            "profile-driven production currently supports local-mineru25 only"
        )
    if memory_tier not in {"25", "50", "max"}:
        raise OcrExecutionError("memory tier must be 25, 50, or max")
    tier = value["memory_tiers"][memory_tier]
    if not tier["supported"]:
        raise OcrExecutionError(f"memory tier {memory_tier} is unsupported")
    backend = get_backend("mineru25-pro-2605")
    if installation.get("backend_id") != backend.backend_id:
        raise OcrExecutionError("MinerU installation binding is invalid")
    executable = installation.get("python_executable", {}).get("path")
    cache_dir = installation.get("hf_cache", {}).get("path")
    adapter = installation.get("adapter", {}).get("path")
    cuda = installation.get("cuda")
    if not all(
        isinstance(item, str) and item
        for item in (executable, cache_dir, adapter)
    ):
        raise OcrExecutionError("MinerU installation paths are incomplete")
    if not isinstance(cuda, Mapping):
        raise OcrExecutionError("MinerU installation CUDA binding is missing")
    timeout = (
        backend.default_timeout_seconds
        if timeout_seconds is None
        else timeout_seconds
    )
    if (
        not isinstance(timeout, (int, float))
        or isinstance(timeout, bool)
        or timeout <= 0
    ):
        raise OcrExecutionError(
            "provider timeout must be a finite positive number"
        )
    try:
        timeout_is_finite = math.isfinite(timeout)
    except OverflowError:
        timeout_is_finite = False
    if not timeout_is_finite:
        raise OcrExecutionError(
            "provider timeout must be a finite positive number"
        )
    profile_file = Path(profile_path).expanduser().resolve(strict=True)
    settings = copy.deepcopy(tier["stages"]["markdown"]["settings"])
    argv = [
        executable,
        "-I",
        "-B",
        adapter,
        "--model-id",
        backend.model_id,
        "--revision",
        backend.model_revision,
        "--device",
        cuda["device"],
        "--attention-backend",
        cuda.get("attention_backend") or backend.default_attention_backend,
        "--execution-profile",
        str(profile_file),
        "--profile-sha256",
        value["profile_sha256"],
        "--memory-tier",
        memory_tier,
    ]
    if allow_provisional:
        argv.append("--allow-provisional-profile")
    return {
        "schema_version": 1,
        "provider_id": backend.provider_id,
        "argv": argv,
        "egress": "none",
        "model": {
            "requested": backend.model_id,
            "revision": backend.model_revision,
        },
        "lanes": {
            backend.default_lane: {
                "prompt": backend.provider_prompt,
                "settings": settings,
                "include_raw_text": False,
            }
        },
        "timeout_seconds": timeout,
        "secret_env_names": [],
        "env": {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": cuda["visible_devices"],
            "HF_HUB_CACHE": cache_dir,
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        },
    }


__all__ = [
    "OcrExecutionError",
    "assert_in_process_deepseek_installation",
    "build_execution_observation",
    "build_profile_provider_spec",
    "deepseek_engine_identity_mismatches",
    "default_profile_directory",
    "probe_backend_runtime",
    "profile_search_paths",
    "validate_bundled_profile_registry",
    "resolve_execution_profile",
    "stage_runtime_mismatches",
]
