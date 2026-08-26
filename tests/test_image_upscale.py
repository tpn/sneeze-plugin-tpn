# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from io import StringIO

import pytest


def _make_trusted_test_runtime(tmp_path):
    runtime = tmp_path / "runtime"
    python_base = (
        tmp_path / "uv-managed/python" / "cpython-3.12.13-linux-x86_64-gnu"
    )
    python_home = python_base / "bin"
    python_home.mkdir(parents=True)
    for directory in (
        python_base.parent.parent,
        python_base.parent,
        python_base,
        python_home,
    ):
        directory.chmod(0o700)
    build_identity = python_base / "BUILD"
    build_identity.write_text("test\n", encoding="ascii")
    build_identity.chmod(0o600)
    managed_python = python_home / "python3.12"
    managed_python.write_bytes(b"")
    managed_python.chmod(0o700)
    python = runtime / "bin/python"
    python.parent.mkdir(parents=True)
    runtime.chmod(0o700)
    python.parent.chmod(0o700)
    python.symlink_to(managed_python)
    (python.parent / "python3").symlink_to("python")
    config = runtime / "pyvenv.cfg"
    config.write_text(
        f"home = {python_home}\n",
        encoding="utf-8",
    )
    config.chmod(0o600)
    return runtime, python


def test_setup_dry_run_is_reproducible_and_does_not_write(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    uv = tmp_path / "uv-test"
    uv.write_bytes(b"")
    uv.chmod(0o700)
    monkeypatch.setattr(
        image_upscale.shutil,
        "which",
        lambda _executable: str(uv),
    )

    runtime = tmp_path / "runtime"
    models = tmp_path / "models"
    receipt = image_upscale.setup_upscaler(
        runtime_dir=runtime,
        model_cache=models,
        uv_executable="uv-test",
        dry_run=True,
    )

    assert receipt["dry_run"] is True
    assert receipt["model"]["sha256"] == image_upscale.MODEL_SHA256
    assert (
        receipt["model"]["download_mirror"]["url"] == image_upscale.MODEL_URL
    )
    assert receipt["model"]["upstream"]["repository"] == "XPixelGroup/HAT"
    assert "official Google Drive" in receipt["model"]["sha256_provenance"]
    assert (
        receipt["runtime"]["pytorch_index_url"]
        == image_upscale.PYTORCH_INDEX_URL
    )
    assert (
        receipt["runtime"]["pypi_index_url"] == image_upscale.PYPI_INDEX_URL
    )
    assert receipt["commands"][0] == [
        str(uv),
        "--no-config",
        "venv",
        "--managed-python",
        "--python",
        "3.12.13",
        str(runtime),
    ]
    assert "torch==2.11.0" in receipt["commands"][1]
    assert (
        receipt["commands"][1][
            receipt["commands"][1].index("--default-index") + 1
        ]
        == image_upscale.PYTORCH_INDEX_URL
    )
    assert "spandrel==0.4.2" in receipt["commands"][2]
    assert (
        receipt["commands"][2][
            receipt["commands"][2].index("--default-index") + 1
        ]
        == image_upscale.PYPI_INDEX_URL
    )
    assert not runtime.exists()
    assert not models.exists()


def test_setup_sanitizes_package_resolution_environment(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    runtime = tmp_path / "runtime"
    uv = tmp_path / "uv"
    uv.write_bytes(b"")
    uv.chmod(0o700)
    python = runtime / "bin/python"
    calls = []
    hostile_names = (
        "ALL_PROXY",
        "CUDA_INJECTION64_PATH",
        "CURL_CA_BUNDLE",
        "DYLD_INSERT_LIBRARIES",
        "FTP_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NVIDIA_DRIVER_CAPABILITIES",
        "NO_PROXY",
        "PIP_INDEX_URL",
        "PYTHONBREAKPOINT",
        "PYTHONEXECUTABLE",
        "PYTHONINSPECT",
        "PYTHONSTARTUP",
        "PYTHONWARNINGS",
        "PYTORCH_KERNEL_CACHE_PATH",
        "REQUESTS_CA_BUNDLE",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "TORCH_HOME",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "UV_CONFIG_FILE",
        "UV_DEFAULT_INDEX",
        "UV_PYTHON",
        "VIRTUAL_ENV",
    )
    for name in hostile_names:
        monkeypatch.setenv(name, "/hostile-package-resolution")
    monkeypatch.setattr(
        image_upscale.shutil,
        "which",
        lambda _executable: str(uv),
    )
    monkeypatch.setattr(image_upscale, "_ensure_model", lambda _path: None)
    monkeypatch.setattr(
        image_upscale,
        "_validate_upscale_trust",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        image_upscale,
        "model_identity",
        lambda *_args, **_kwargs: {},
    )

    probe = {
        "cuda_available": True,
        "gpu": {"bfloat16_supported": True},
        "python": "3.12.13",
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "torchvision": "0.26.0+cu130",
        "spandrel": "0.4.2",
        "pillow": "12.3.0",
        "numpy": "2.5.2",
    }

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        if "venv" in argv:
            created_runtime, created_python = _make_trusted_test_runtime(
                tmp_path
            )
            assert created_runtime == runtime
            assert created_python == python
        stdout = json.dumps(probe) if "--probe" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    image_upscale.setup_upscaler(
        runtime_dir=runtime,
        model_cache=tmp_path / "models",
        command_runner=runner,
    )

    assert len(calls) == 4
    for argv, kwargs in calls:
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
        for name in hostile_names:
            assert name not in kwargs["env"]
        if argv[0] == str(uv):
            assert kwargs["umask"] == 0o077
        else:
            assert "umask" not in kwargs
    probe_argv = next(argv for argv, _kwargs in calls if "--probe" in argv)
    assert probe_argv[:3] == [
        str(python),
        "-P",
        str(image_upscale.runtime_script()),
    ]


def test_setup_rejects_untrusted_existing_runtime_before_commands(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    uv = tmp_path / "uv"
    uv.write_bytes(b"")
    uv.chmod(0o700)
    runtime, python = _make_trusted_test_runtime(tmp_path)
    python.chmod(0o720)
    called = False

    monkeypatch.setattr(
        image_upscale.shutil,
        "which",
        lambda _executable: str(uv),
    )

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(image_upscale.ImageUpscaleError, match="not trusted"):
        image_upscale.setup_upscaler(
            runtime_dir=runtime,
            model_cache=tmp_path / "models",
            command_runner=runner,
        )

    assert called is False


def test_setup_rejects_untrusted_fresh_runtime_parent_before_commands(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    uv = tmp_path / "uv"
    uv.write_bytes(b"")
    uv.chmod(0o700)
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir()
    unsafe_parent.chmod(0o770)
    called = False

    monkeypatch.setattr(
        image_upscale.shutil,
        "which",
        lambda _executable: str(uv),
    )

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(image_upscale.ImageUpscaleError, match="not trusted"):
        image_upscale.setup_upscaler(
            runtime_dir=unsafe_parent / "runtime",
            model_cache=tmp_path / "models",
            command_runner=runner,
        )

    assert called is False


def test_setup_rejects_untrusted_fresh_runtime_leaf_before_commands(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    uv = tmp_path / "uv"
    uv.write_bytes(b"")
    uv.chmod(0o700)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    runtime.chmod(0o770)
    called = False

    monkeypatch.setattr(
        image_upscale.shutil,
        "which",
        lambda _executable: str(uv),
    )

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(image_upscale.ImageUpscaleError, match="not trusted"):
        image_upscale.setup_upscaler(
            runtime_dir=runtime,
            model_cache=tmp_path / "models",
            command_runner=runner,
        )

    assert called is False


def test_setup_prepares_model_cache_privately_under_permissive_umask(
    tmp_path, monkeypatch
):
    import os
    import stat

    from sneeze.tpn.upscale import image_upscale

    uv = tmp_path / "uv"
    uv.write_bytes(b"")
    uv.chmod(0o700)
    runtime, _python = _make_trusted_test_runtime(tmp_path)
    model_cache = tmp_path / "models"
    monkeypatch.setattr(
        image_upscale.shutil,
        "which",
        lambda _executable: str(uv),
    )
    monkeypatch.setattr(image_upscale, "_ensure_model", lambda _path: None)
    monkeypatch.setattr(
        image_upscale,
        "_validate_upscale_trust",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        image_upscale,
        "model_identity",
        lambda *_args, **_kwargs: {},
    )

    probe = {
        "cuda_available": True,
        "gpu": {"bfloat16_supported": True},
        "python": "3.12.13",
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "torchvision": "0.26.0+cu130",
        "spandrel": "0.4.2",
        "pillow": "12.3.0",
        "numpy": "2.5.2",
    }

    def runner(argv, **_kwargs):
        stdout = json.dumps(probe) if "--probe" in argv else ""
        return subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr="")

    previous_umask = os.umask(0o002)
    try:
        image_upscale.setup_upscaler(
            runtime_dir=runtime,
            model_cache=model_cache,
            command_runner=runner,
        )
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(model_cache.stat().st_mode) == 0o700


def test_trusted_runtime_fixture_ignores_permissive_umask(tmp_path):
    import os

    from sneeze.tpn.upscale import image_upscale

    previous_umask = os.umask(0o002)
    try:
        runtime, python = _make_trusted_test_runtime(tmp_path)
    finally:
        os.umask(previous_umask)

    image_upscale._validate_runtime_trust(runtime, python)


def test_setup_secures_uv_runtime_lock_without_following_symlinks(tmp_path):
    import stat

    from sneeze.tpn.upscale import image_upscale

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    lock = runtime / ".lock"
    lock.write_bytes(b"")
    lock.chmod(0o666)

    image_upscale._secure_runtime_lock(runtime)

    assert stat.S_IMODE(lock.stat().st_mode) == 0o600

    lock.unlink()
    victim = tmp_path / "victim"
    victim.write_bytes(b"owned")
    victim.chmod(0o644)
    lock.symlink_to(victim)

    with pytest.raises(image_upscale.ImageUpscaleError, match="secure"):
        image_upscale._secure_runtime_lock(runtime)

    assert stat.S_IMODE(victim.stat().st_mode) == 0o644


def test_runtime_trust_rejects_external_interpreter_symlink_chain(tmp_path):
    from sneeze.tpn.upscale import image_upscale

    runtime, python = _make_trusted_test_runtime(tmp_path)
    python.unlink()
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o770)
    intermediate = shared / "interpreter"
    intermediate.symlink_to("/usr/bin/python3")
    python.symlink_to(intermediate)

    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match="must be canonical",
    ):
        image_upscale._validate_runtime_trust(runtime, python)


def test_runtime_trust_rejects_non_managed_python_before_tree_walk(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    runtime, python = _make_trusted_test_runtime(tmp_path)
    python.unlink()
    system_python = tmp_path / "system/bin/python3.12"
    system_python.parent.mkdir(parents=True)
    system_python.write_bytes(b"")
    system_python.chmod(0o700)
    python.symlink_to(system_python)

    def unexpected_tree_walk(*_args, **_kwargs):
        raise AssertionError(
            "non-managed Python must fail before a tree walk"
        )

    monkeypatch.setattr(
        image_upscale,
        "validate_trusted_tree",
        unexpected_tree_walk,
    )

    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match="exact uv-managed Python",
    ):
        image_upscale._validate_runtime_trust(runtime, python)


def test_runtime_trust_rejects_external_directory_before_tree_walk(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    runtime, python = _make_trusted_test_runtime(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    (runtime / "external").symlink_to(external, target_is_directory=True)

    def unexpected_tree_walk(*_args, **_kwargs):
        raise AssertionError("trusted-tree walk must follow scope validation")

    monkeypatch.setattr(
        image_upscale,
        "validate_trusted_tree",
        unexpected_tree_walk,
    )

    with pytest.raises(image_upscale.ImageUpscaleError, match="leaves"):
        image_upscale._validate_runtime_trust(runtime, python)


def test_runtime_trust_rejects_external_config_before_read(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from sneeze.tpn.upscale import image_upscale

    runtime, python = _make_trusted_test_runtime(tmp_path)
    config = runtime / "pyvenv.cfg"
    config.unlink()
    external = tmp_path / "external-config"
    external.write_text("home = ignored\n", encoding="utf-8")
    config.symlink_to(external)
    original_read_text = Path.read_text

    def guarded_read_text(path, *args, **kwargs):
        if path == config:
            raise AssertionError("config read must follow trust validation")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)

    with pytest.raises(image_upscale.ImageUpscaleError, match="not trusted"):
        image_upscale._validate_runtime_trust(runtime, python)


def test_symlink_scope_rejects_walk_errors(tmp_path, monkeypatch):
    from sneeze.tpn.upscale import image_upscale

    def failed_walk(_root, *, followlinks, onerror):
        assert followlinks is False
        onerror(PermissionError("unreadable"))
        return ()

    monkeypatch.setattr(image_upscale.os, "walk", failed_walk)

    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match="Could not scan",
    ):
        image_upscale._validate_symlink_scope(tmp_path)


def test_model_download_is_hash_verified_and_atomic(tmp_path, monkeypatch):
    import hashlib
    import stat
    from io import BytesIO

    from sneeze.tpn.upscale import image_upscale

    payload = b"verified-checkpoint"
    expected = hashlib.sha256(payload).hexdigest()
    model = tmp_path / "models" / image_upscale.MODEL_FILENAME
    monkeypatch.setattr(image_upscale, "MODEL_SHA256", expected)
    monkeypatch.setattr(image_upscale, "MODEL_SIZE_BYTES", len(payload))
    monkeypatch.setattr(
        image_upscale.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(payload),
    )
    process_status = tmp_path / "proc-status"
    process_status.write_text("Name:\ttest\nUmask:\t0027\n", encoding="ascii")
    monkeypatch.setattr(image_upscale, "_PROC_STATUS", process_status)

    image_upscale._ensure_model(model)

    assert model.read_bytes() == payload
    assert stat.S_IMODE(model.stat().st_mode) == 0o640
    assert not list(model.parent.glob(".*.download"))


def test_model_download_mismatch_removes_partial_file(tmp_path, monkeypatch):
    from io import BytesIO

    from sneeze.tpn.upscale import image_upscale

    model = tmp_path / "models" / image_upscale.MODEL_FILENAME
    monkeypatch.setattr(image_upscale, "MODEL_SHA256", "0" * 64)
    monkeypatch.setattr(image_upscale, "MODEL_SIZE_BYTES", len(b"tampered"))
    monkeypatch.setattr(
        image_upscale.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(b"tampered"),
    )

    with pytest.raises(image_upscale.ImageUpscaleError, match="mismatch"):
        image_upscale._ensure_model(model)

    assert not model.exists()
    assert not list(model.parent.glob(".*.download"))


def test_model_download_rejects_oversized_response(tmp_path, monkeypatch):
    from io import BytesIO

    from sneeze.tpn.upscale import image_upscale

    model = tmp_path / "models" / image_upscale.MODEL_FILENAME
    monkeypatch.setattr(image_upscale, "MODEL_SIZE_BYTES", 3)
    monkeypatch.setattr(
        image_upscale.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: BytesIO(b"oversized"),
    )

    with pytest.raises(image_upscale.ImageUpscaleError, match="exceeded"):
        image_upscale._ensure_model(model)

    assert not model.exists()
    assert not list(model.parent.glob(".*.download"))


def test_model_download_wraps_incomplete_response(tmp_path, monkeypatch):
    import http.client

    from sneeze.tpn.upscale import image_upscale

    class IncompleteResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            raise http.client.IncompleteRead(b"partial", 1)

    model = tmp_path / "models" / image_upscale.MODEL_FILENAME
    monkeypatch.setattr(
        image_upscale.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: IncompleteResponse(),
    )

    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match="Could not download",
    ):
        image_upscale._ensure_model(model)

    assert not model.exists()
    assert not list(model.parent.glob(".*.download"))


def test_model_cache_rejects_corrupt_file_and_directory(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    model = tmp_path / image_upscale.MODEL_FILENAME
    model.write_bytes(b"corrupt")
    monkeypatch.setattr(image_upscale, "MODEL_SHA256", "0" * 64)

    with pytest.raises(image_upscale.ImageUpscaleError, match="mismatch"):
        image_upscale._ensure_model(model)

    model.unlink()
    model.mkdir()
    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match="not a regular file",
    ):
        image_upscale._ensure_model(model)


def test_runtime_probe_requires_pinned_cuda_bf16_tuple():
    from sneeze.tpn.upscale.image_upscale import (
        ImageUpscaleError,
        _validate_runtime_probe,
    )

    qualified = {
        "cuda_available": True,
        "gpu": {"bfloat16_supported": True},
        "python": "3.12.13",
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "torchvision": "0.26.0+cu130",
        "spandrel": "0.4.2",
        "pillow": "12.3.0",
        "numpy": "2.5.2",
    }

    _validate_runtime_probe(qualified)

    unqualified = dict(qualified, torch="2.11.0+cu128")
    with pytest.raises(ImageUpscaleError, match="expected torch"):
        _validate_runtime_probe(unqualified)


def test_runtime_tiled_matches_untiled_nearest():
    import importlib.util
    import sys

    if importlib.util.find_spec("torch") is None:
        pytest.skip("Torch is installed only in the isolated upscale runtime")

    script = """
import torch

from sneeze.tpn.upscale.image_upscale_runtime import _run_tiled


class NearestX2:
    scale = 2
    output_channels = 3

    def __call__(self, tensor):
        return torch.nn.functional.interpolate(
            tensor,
            scale_factor=self.scale,
            mode="nearest",
        )


for height, width, tile_size, overlap in (
    (7, 11, 4, 1),
    (3, 2, 8, 1),
    (5, 9, 4, 3),
):
    tensor = torch.arange(
        3 * height * width,
        dtype=torch.float32,
    ).reshape(1, 3, height, width)
    model = NearestX2()
    expected = model(tensor)
    actual = _run_tiled(
        model,
        tensor,
        torch=torch,
        tile_size=tile_size,
        overlap=overlap,
    )
    if not torch.equal(actual, expected):
        raise AssertionError(
            f"tiled mismatch for {(height, width, tile_size, overlap)}"
        )
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert completed.returncode == 0, completed.stdout


def test_default_output_is_non_destructive_sibling(tmp_path):
    from sneeze.tpn.upscale.image_upscale import default_output_path

    source = tmp_path / "slide.png"
    assert default_output_path(source) == tmp_path / "slide-4k.png"
    assert default_output_path(source, 2560) == tmp_path / "slide-2560w.png"

    target_dir = tmp_path / "shared"
    target_dir.mkdir()
    target = target_dir / "target.png"
    target.write_bytes(b"target")
    source.symlink_to(target)
    assert default_output_path(source) == tmp_path / "slide-4k.png"


def test_run_upscale_refuses_existing_output_before_runtime(tmp_path):
    from sneeze.tpn.upscale.image_upscale import (
        ImageUpscaleError,
        run_upscale,
    )

    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    source.write_bytes(b"source")
    output.write_bytes(b"owned")

    with pytest.raises(ImageUpscaleError, match="Output already exists"):
        run_upscale(source=source, output=output)

    assert output.read_bytes() == b"owned"


def test_run_upscale_rejects_output_directory_before_runtime(tmp_path):
    from sneeze.tpn.upscale.image_upscale import (
        ImageUpscaleError,
        run_upscale,
    )

    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    source.write_bytes(b"source")
    output.mkdir()

    with pytest.raises(ImageUpscaleError, match="must not be a directory"):
        run_upscale(source=source, output=output, force=True)


@pytest.mark.parametrize("target_exists", (False, True))
def test_run_upscale_rejects_symlink_output_before_runtime(
    tmp_path, target_exists
):
    from sneeze.tpn.upscale.image_upscale import (
        ImageUpscaleError,
        run_upscale,
    )

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    target = tmp_path / "unrelated.png"
    if target_exists:
        target.write_bytes(b"unrelated")
    output = tmp_path / "output.png"
    output.symlink_to(target)

    with pytest.raises(ImageUpscaleError, match="symbolic link"):
        run_upscale(source=source, output=output, force=True)

    assert output.is_symlink()
    if target_exists:
        assert target.read_bytes() == b"unrelated"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"output": "output.jpg"}, r"\.png suffix"),
        ({"target_width": 0}, "greater than zero"),
        (
            {"tile_size": 64, "tile_overlap": 64},
            "smaller than tile size",
        ),
        ({"dtype": "float16"}, "Dtype must be"),
    ),
)
def test_run_upscale_rejects_invalid_options_before_runtime(
    tmp_path, kwargs, message
):
    from sneeze.tpn.upscale.image_upscale import (
        ImageUpscaleError,
        run_upscale,
    )

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    invocation = dict(kwargs)
    if "output" in invocation:
        invocation["output"] = tmp_path / invocation["output"]

    with pytest.raises(ImageUpscaleError, match=message):
        run_upscale(source=source, **invocation)


def test_run_upscale_rejects_source_as_output_before_runtime(tmp_path):
    from sneeze.tpn.upscale.image_upscale import (
        ImageUpscaleError,
        run_upscale,
    )

    source = tmp_path / "source.png"
    source.write_bytes(b"source")

    with pytest.raises(ImageUpscaleError, match="must be different"):
        run_upscale(source=source, output=source, force=True)


def test_run_upscale_validates_model_hash_before_child(tmp_path):
    from sneeze.tpn.upscale.image_upscale import (
        ImageUpscaleError,
        run_upscale,
    )

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    runtime, python = _make_trusted_test_runtime(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / "HAT-L_SRx2_ImageNet-pretrain.pth").write_bytes(b"wrong")

    with pytest.raises(ImageUpscaleError, match="Model SHA-256 mismatch"):
        run_upscale(
            source=source,
            runtime_dir=runtime,
            model_cache=models,
        )


def test_run_upscale_rejects_untrusted_runtime_before_child(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    runtime, python = _make_trusted_test_runtime(tmp_path)
    python.chmod(0o720)
    models = tmp_path / "models"
    models.mkdir()
    (models / image_upscale.MODEL_FILENAME).write_bytes(b"model")
    called = False

    monkeypatch.setattr(
        image_upscale,
        "model_identity",
        lambda *_args, **_kwargs: {},
    )

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match="not trusted",
    ):
        image_upscale.run_upscale(
            source=source,
            output=tmp_path / "output.png",
            runtime_dir=runtime,
            model_cache=models,
            command_runner=runner,
        )

    assert called is False


def test_run_upscale_launches_isolated_runtime_and_writes_receipt(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.upscale import image_upscale

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    output = tmp_path / "source-4k.png"
    runtime, python = _make_trusted_test_runtime(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    model = models / image_upscale.MODEL_FILENAME
    model.write_bytes(b"model")
    receipt_path = tmp_path / "receipt.json"
    calls = []
    reported_dimensions = [3840, 2160]
    reported_implementation_hash = [
        image_upscale.file_sha256(image_upscale.runtime_script())
    ]
    hostile_names = (
        "CUDA_INJECTION64_PATH",
        "DYLD_INSERT_LIBRARIES",
        "LD_AUDIT",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "NVIDIA_DRIVER_CAPABILITIES",
        "PYTHONBREAKPOINT",
        "PYTHONEXECUTABLE",
        "PYTHONHOME",
        "PYTHONINSPECT",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONWARNINGS",
        "PYTORCH_KERNEL_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
        "TRITON_CACHE_DIR",
        "__PYVENV_LAUNCHER__",
    )
    for name in hostile_names:
        monkeypatch.setenv(name, "/hostile-parent-python-path")

    monkeypatch.setattr(
        image_upscale,
        "model_identity",
        lambda *_args, **_kwargs: {},
    )

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        output.write_bytes(b"upscaled")
        payload = {
            "implementation": {
                "path": str(image_upscale.runtime_script()),
                "sha256": reported_implementation_hash[0],
            },
            "source": {
                "path": str(source),
                "sha256": image_upscale.file_sha256(source),
            },
            "output": {
                "path": str(output),
                "dimensions": list(reported_dimensions),
                "sha256": image_upscale.file_sha256(output),
            },
            "parameters": {"target_height": 2160},
        }
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    receipt = image_upscale.run_upscale(
        source=source,
        output=output,
        runtime_dir=runtime,
        model_cache=models,
        cuda_visible_devices="0",
        receipt_path=receipt_path,
        command_runner=runner,
    )

    argv, kwargs = calls[0]
    assert argv[:3] == [
        str(python),
        "-P",
        str(image_upscale.runtime_script()),
    ]
    assert argv[argv.index("--target-width") + 1] == "3840"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    for name in hostile_names:
        assert name not in kwargs["env"]
    assert receipt["receipt_path"] == str(receipt_path)
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted == receipt

    output.unlink()
    reported_dimensions[1] = 1080
    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match=f"dimensions.*unverified output.*{output}",
    ):
        image_upscale.run_upscale(
            source=source,
            output=output,
            runtime_dir=runtime,
            model_cache=models,
            command_runner=runner,
        )

    output.unlink()
    reported_dimensions[1] = 2160
    reported_implementation_hash[0] = "0" * 64
    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match="implementation identity",
    ):
        image_upscale.run_upscale(
            source=source,
            output=output,
            runtime_dir=runtime,
            model_cache=models,
            command_runner=runner,
        )


def test_run_upscale_preserves_receipt_that_appears_during_inference(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.upscale import image_upscale

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    output = tmp_path / "source-4k.png"
    receipt_path = tmp_path / "receipt.json"
    runtime, python = _make_trusted_test_runtime(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / image_upscale.MODEL_FILENAME).write_bytes(b"model")

    monkeypatch.setattr(
        image_upscale,
        "model_identity",
        lambda *_args, **_kwargs: {},
    )

    def runner(argv, **_kwargs):
        output.write_bytes(b"upscaled")
        receipt_path.write_text("owned", encoding="utf-8")
        payload = {
            "implementation": {
                "path": str(image_upscale.runtime_script()),
                "sha256": image_upscale.file_sha256(
                    image_upscale.runtime_script()
                ),
            },
            "source": {
                "path": str(source),
                "sha256": image_upscale.file_sha256(source),
            },
            "output": {
                "path": str(output),
                "dimensions": [3840, 2160],
                "sha256": image_upscale.file_sha256(output),
            },
            "parameters": {"target_height": 2160},
        }
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=json.dumps(payload),
            stderr="",
        )

    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match="Receipt appeared while upscaling",
    ) as caught:
        image_upscale.run_upscale(
            source=source,
            output=output,
            runtime_dir=runtime,
            model_cache=models,
            receipt_path=receipt_path,
            command_runner=runner,
        )

    assert receipt_path.read_text(encoding="utf-8") == "owned"
    assert output.read_bytes() == b"upscaled"
    assert str(output) in str(caught.value)


def test_runtime_png_publish_honors_umask_and_is_atomic(
    tmp_path, monkeypatch
):
    import stat

    from sneeze.tpn.upscale import image_upscale_runtime

    temporary = tmp_path / "temporary.png"
    output = tmp_path / "output.png"
    temporary.write_bytes(b"complete")
    output.write_bytes(b"owned")
    process_status = tmp_path / "proc-status"
    process_status.write_text("Name:\ttest\nUmask:\t0027\n", encoding="ascii")
    monkeypatch.setattr(
        image_upscale_runtime,
        "_PROC_STATUS",
        process_status,
    )

    with pytest.raises(ValueError, match="Output appeared during upscale"):
        image_upscale_runtime._publish_png(
            str(temporary), output, force=False
        )

    assert output.read_bytes() == b"owned"
    assert temporary.read_bytes() == b"complete"

    output.unlink()
    image_upscale_runtime._publish_png(str(temporary), output, force=False)

    assert output.read_bytes() == b"complete"
    assert stat.S_IMODE(output.stat().st_mode) == 0o640
    assert not temporary.exists()

    victim = tmp_path / "victim.png"
    victim.write_bytes(b"victim")
    symlink_output = tmp_path / "symlink-output.png"
    symlink_output.symlink_to(victim)
    replacement = tmp_path / "replacement.png"
    replacement.write_bytes(b"replacement")

    with pytest.raises(ValueError, match="symbolic link"):
        image_upscale_runtime._publish_png(
            str(replacement), symlink_output, force=True
        )

    assert symlink_output.is_symlink()
    assert victim.read_bytes() == b"victim"
    assert replacement.read_bytes() == b"replacement"


def test_receipt_publish_honors_umask(tmp_path, monkeypatch):
    import stat

    from sneeze.tpn.upscale import image_upscale

    process_status = tmp_path / "proc-status"
    process_status.write_text("Name:\ttest\nUmask:\t0027\n", encoding="ascii")
    monkeypatch.setattr(image_upscale, "_PROC_STATUS", process_status)
    receipt = tmp_path / "receipt.json"

    image_upscale._write_json_atomic(
        receipt,
        {"kind": "test"},
        force=False,
    )

    assert json.loads(receipt.read_text(encoding="utf-8")) == {"kind": "test"}
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o640


def test_file_modes_preserve_owner_access_under_restrictive_umask(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale, image_upscale_runtime

    process_status = tmp_path / "proc-status"
    process_status.write_text("Name:\ttest\nUmask:\t0477\n", encoding="ascii")
    monkeypatch.setattr(image_upscale, "_PROC_STATUS", process_status)
    monkeypatch.setattr(
        image_upscale_runtime,
        "_PROC_STATUS",
        process_status,
    )

    assert image_upscale._file_mode_from_umask() == 0o600
    assert image_upscale_runtime._file_mode_from_umask() == 0o600


def test_run_upscale_refuses_receipt_aliases_before_child(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    source_dir = tmp_path / "source-dir"
    source_dir.mkdir()
    source = source_dir / "source.png"
    source.write_bytes(b"source")
    runtime, python = _make_trusted_test_runtime(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    model = models / image_upscale.MODEL_FILENAME
    model.write_bytes(b"model")
    called = False

    monkeypatch.setattr(
        image_upscale,
        "model_identity",
        lambda *_args, **_kwargs: {},
    )

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True

    source_dir_alias = tmp_path / "source-dir-alias"
    source_dir_alias.symlink_to(source_dir, target_is_directory=True)
    hardlink_alias = tmp_path / "source-hardlink.png"
    hardlink_alias.hardlink_to(source)
    receipt_aliases = (
        source,
        source_dir_alias / source.name,
        hardlink_alias,
    )

    for receipt_alias in receipt_aliases:
        with pytest.raises(
            image_upscale.ImageUpscaleError,
            match="Receipt path must be separate",
        ):
            image_upscale.run_upscale(
                source=source,
                output=tmp_path / "output.png",
                runtime_dir=runtime,
                model_cache=models,
                receipt_path=receipt_alias,
                force=True,
                command_runner=runner,
            )

    assert called is False
    assert source.read_bytes() == b"source"


def test_runtime_destination_preserves_final_symlink(tmp_path):
    from sneeze.tpn.upscale.image_upscale_runtime import _destination_path

    victim = tmp_path / "victim.png"
    victim.write_bytes(b"victim")
    output = tmp_path / "output.png"
    output.symlink_to(victim)

    destination = _destination_path(output)

    assert destination == output
    assert destination.is_symlink()


def test_run_upscale_rejects_symlink_receipt_before_child(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    output = tmp_path / "output.png"
    runtime, python = _make_trusted_test_runtime(tmp_path)
    models = tmp_path / "models"
    models.mkdir()
    (models / image_upscale.MODEL_FILENAME).write_bytes(b"model")
    receipt_target = tmp_path / "unrelated.json"
    receipt_target.write_text("owned", encoding="utf-8")
    receipt = tmp_path / "receipt.json"
    receipt.symlink_to(receipt_target)
    called = False

    monkeypatch.setattr(
        image_upscale,
        "model_identity",
        lambda *_args, **_kwargs: {},
    )

    def runner(*_args, **_kwargs):
        nonlocal called
        called = True

    with pytest.raises(
        image_upscale.ImageUpscaleError, match="symbolic link"
    ):
        image_upscale.run_upscale(
            source=source,
            output=output,
            runtime_dir=runtime,
            model_cache=models,
            receipt_path=receipt,
            force=True,
            command_runner=runner,
        )

    assert called is False
    assert receipt.is_symlink()
    assert receipt_target.read_text(encoding="utf-8") == "owned"


def test_upscale_command_forwards_options(monkeypatch, tmp_path):
    from sneeze.tpn.upscale import image_upscale
    from sneeze.tpn.upscale.commands import Run

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    captured = {}

    def fake_run_upscale(**kwargs):
        captured.update(kwargs)
        return {"output": {"dimensions": [3840, 2160]}}

    monkeypatch.setattr(image_upscale, "run_upscale", fake_run_upscale)
    stream = StringIO()
    command = Run(None, stream, None)
    command._source = str(source)
    command.output = str(tmp_path / "custom.png")
    command._target_width = 3840
    command.runtime_dir = str(tmp_path / "runtime")
    command.model_cache = str(tmp_path / "models")
    command.cuda_visible_devices = "0"
    command.device = "cuda:0"
    command.dtype = "float32"
    command._tile_size = 512
    command._tile_overlap = 64
    command.receipt = str(tmp_path / "receipt.json")
    command.force = True
    command.run()

    assert captured == {
        "source": str(source),
        "output": str(tmp_path / "custom.png"),
        "target_width": 3840,
        "runtime_dir": str(tmp_path / "runtime"),
        "model_cache": str(tmp_path / "models"),
        "cuda_visible_devices": "0",
        "device": "cuda:0",
        "dtype": "float32",
        "tile_size": 512,
        "tile_overlap": 64,
        "force": True,
        "receipt_path": str(tmp_path / "receipt.json"),
    }
    assert '"dimensions": [' in stream.getvalue()


def test_setup_command_forwards_dry_run(monkeypatch, tmp_path):
    from sneeze.tpn.upscale import image_upscale
    from sneeze.tpn.upscale.commands import Setup

    captured = {}

    def fake_setup(**kwargs):
        captured.update(kwargs)
        return {"dry_run": kwargs["dry_run"]}

    monkeypatch.setattr(image_upscale, "setup_upscaler", fake_setup)
    stream = StringIO()
    command = Setup(None, stream, None)
    command.runtime_dir = str(tmp_path / "runtime")
    command.model_cache = str(tmp_path / "models")
    command.uv_executable = "uv-test"
    command.cuda_visible_devices = "0"
    command.dry_run = True
    command.run()

    assert captured == {
        "runtime_dir": str(tmp_path / "runtime"),
        "model_cache": str(tmp_path / "models"),
        "uv_executable": "uv-test",
        "cuda_visible_devices": "0",
        "dry_run": True,
    }
    assert json.loads(stream.getvalue()) == {"dry_run": True}


def test_upscale_commands_are_discovered_without_runtime_import():
    import sys

    from sneeze import cli as sneeze_cli

    sys.modules.pop("sneeze.tpn.upscale.image_upscale_runtime", None)
    cli = sneeze_cli.CLI(
        program_name="sne",
        module_names=["-sneeze", "sneeze.tpn.upscale"],
        introspect=True,
        auto_plugins=False,
    )

    assert set(cli._commands_by_name) == {"run", "setup", "version"}
    assert "sneeze.tpn.upscale.image_upscale_runtime" not in sys.modules


def test_branded_entry_point_runs_real_help(capfd):
    from sneeze.tpn.upscale import cli

    with pytest.raises(SystemExit) as exc_info:
        cli.main(["help"])

    assert exc_info.value.code == 0
    captured = capfd.readouterr()
    output = captured.out + captured.err
    assert "run (run)" in output
    assert "setup (setup)" in output


def test_upscale_has_a_branded_sneeze_entry_point():
    import tomllib
    from pathlib import Path

    project = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(
            encoding="utf-8"
        )
    )

    assert project["project"]["scripts"]["sne-tpn-upscale"] == (
        "sneeze.tpn.upscale.cli:main"
    )


def test_branded_entry_point_loads_only_upscale_commands(monkeypatch):
    from sneeze.tpn.upscale import cli

    captured = {}

    def fake_branded_main(program_name, modules, argv, *, auto_plugins):
        captured.update(
            {
                "program_name": program_name,
                "modules": modules,
                "argv": argv,
                "auto_plugins": auto_plugins,
            }
        )

    monkeypatch.setattr(cli, "branded_main", fake_branded_main)
    cli.main(["setup", "--dry-run"])

    assert captured == {
        "program_name": "sne-tpn-upscale",
        "modules": ["-sneeze", "sneeze.tpn.upscale"],
        "argv": ["setup", "--dry-run"],
        "auto_plugins": False,
    }
