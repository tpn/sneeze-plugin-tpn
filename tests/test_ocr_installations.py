import copy
import json
import os
import stat
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _stub_distribution_probe(monkeypatch):
    from sneeze.tpn import ocr_installations
    from sneeze.tpn.ocr_backend_setup import sha256_json

    distributions = ["synthetic-runtime==1.0"]
    original = ocr_installations._probe_installed_distributions
    monkeypatch.setattr(
        ocr_installations,
        "_probe_installed_distributions",
        lambda _python: {
            "count": len(distributions),
            "sha256": sha256_json(distributions),
        },
    )
    return original


def _make_executable(tmp_path, name="python"):
    executable = tmp_path / f"{name}-environment" / "bin" / name
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    return executable.resolve()


def test_build_rejects_group_writable_environment_ancestry(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    executable.parent.parent.chmod(0o775)
    cache, _ = _make_snapshot(tmp_path, backend)

    with pytest.raises(InstallationRegistryError, match="writable by group"):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )


def test_build_rejects_group_writable_package_file(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    package = executable.parent.parent / "lib" / "site-packages"
    package.mkdir(parents=True)
    module = package / "runtime.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    module.chmod(0o664)
    cache, _ = _make_snapshot(tmp_path, backend)

    with pytest.raises(InstallationRegistryError, match="writable by group"):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )


def test_build_validates_external_pth_import_root(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    site_packages = (
        executable.parent.parent / "lib" / "python3.12" / "site-packages"
    )
    site_packages.mkdir(parents=True)
    external = tmp_path / "editable-source"
    external.mkdir()
    module = external / "runtime.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (site_packages / "editable.pth").write_text(
        f"{external}\n",
        encoding="utf-8",
    )
    cache, _ = _make_snapshot(tmp_path, backend)

    build_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
    )

    module.chmod(0o664)
    with pytest.raises(InstallationRegistryError, match="writable by group"):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )


def test_build_rejects_unallowlisted_executable_pth_line(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_installations as ocr_installations
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    site_packages = (
        executable.parent.parent / "lib" / "python3.12" / "site-packages"
    )
    site_packages.mkdir(parents=True)
    (site_packages / "dynamic.pth").write_text(
        "import arbitrary_runtime_hook\n",
        encoding="utf-8",
    )
    cache, _ = _make_snapshot(tmp_path, backend)
    probe_called = False

    def unexpected_probe(_python):
        nonlocal probe_called
        probe_called = True
        raise AssertionError("runtime probe must not run before .pth audit")

    monkeypatch.setattr(
        ocr_installations,
        "_probe_installed_distributions",
        unexpected_probe,
    )

    with pytest.raises(InstallationRegistryError, match="not allowlisted"):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )
    assert probe_called is False


def test_build_rejects_bom_prefixed_executable_pth_line(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    site_packages = (
        executable.parent.parent / "lib" / "python3.12" / "site-packages"
    )
    site_packages.mkdir(parents=True)
    (site_packages / "dynamic.pth").write_bytes(
        b"\xef\xbb\xbfimport arbitrary_runtime_hook\n"
    )
    cache, _ = _make_snapshot(tmp_path, backend)

    with pytest.raises(InstallationRegistryError, match="not allowlisted"):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )


def test_build_rejects_external_pth_symlink_spelling(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    site_packages = (
        executable.parent.parent / "lib" / "python3.12" / "site-packages"
    )
    site_packages.mkdir(parents=True)
    external = tmp_path / "external-source"
    external.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    shared.chmod(0o777)
    link = shared / "editable-source"
    link.symlink_to(external, target_is_directory=True)
    (site_packages / "editable.pth").write_text(
        f"{link}\n",
        encoding="utf-8",
    )
    cache, _ = _make_snapshot(tmp_path, backend)

    with pytest.raises(
        InstallationRegistryError,
        match="must be a file or directory",
    ):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )


def test_build_rejects_absent_external_pth_path(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    site_packages = (
        executable.parent.parent / "lib" / "python3.12" / "site-packages"
    )
    site_packages.mkdir(parents=True)
    missing = tmp_path / "shared" / "missing-source"
    missing.parent.mkdir()
    missing.parent.chmod(0o777)
    (site_packages / "editable.pth").write_text(
        f"{missing}\n",
        encoding="utf-8",
    )
    cache, _ = _make_snapshot(tmp_path, backend)

    with pytest.raises(InstallationRegistryError, match="does not exist"):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )


def test_build_accepts_standard_distutils_pth_shim(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_installations import build_backend_installation

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    site_packages = (
        executable.parent.parent / "lib" / "python3.12" / "site-packages"
    )
    site_packages.mkdir(parents=True)
    (site_packages.parent.parent / "python3.1").symlink_to(
        "python3.12",
        target_is_directory=True,
    )
    (site_packages / "distutils-precedence.pth").write_text(
        "import os; var = 'SETUPTOOLS_USE_DISTUTILS'; enabled = "
        "os.environ.get(var, 'local') == 'local'; enabled and "
        "__import__('_distutils_hack').add_shim(); \n",
        encoding="utf-8",
    )
    distutils_hack = site_packages / "_distutils_hack"
    distutils_hack.mkdir()
    (distutils_hack / "__init__.py").write_text(
        "def add_shim():\n    return None\n",
        encoding="utf-8",
    )
    cache, _ = _make_snapshot(tmp_path, backend)

    build_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
    )


def test_build_rejects_group_writable_snapshot_file(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, snapshot = _make_snapshot(tmp_path, backend)
    (snapshot / "config.json").chmod(0o664)

    with pytest.raises(InstallationRegistryError, match="writable by group"):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )


def test_distribution_probe_routes_through_bounded_subprocess(
    tmp_path,
    monkeypatch,
    _stub_distribution_probe,
):
    import sneeze.tpn.ocr_execution as ocr_execution
    from sneeze.tpn.ocr_backend_setup import sha256_json

    captured = {}
    expected = {
        "count": 2,
        "sha256": sha256_json(["one==1", "two==2"]),
    }

    def bounded_runner(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": json.dumps(expected) + "\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(
        ocr_execution,
        "_run_bounded_runtime_probe",
        bounded_runner,
    )
    executable = _make_executable(tmp_path)

    assert _stub_distribution_probe(executable) == expected
    assert captured["argv"][:4] == [
        str(executable),
        "-I",
        "-B",
        "-c",
    ]
    assert captured["kwargs"]["timeout_seconds"] == 60
    assert captured["kwargs"]["runner"] is subprocess.run
    assert captured["kwargs"]["environment"]["PYTHONNOUSERSITE"] == "1"


def _make_snapshot(tmp_path, backend, *, cache_name="hub"):
    from sneeze.tpn.ocr_backend_setup import expected_snapshot_path

    cache = (tmp_path / cache_name).resolve()
    snapshot = expected_snapshot_path(backend, cache)
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    (snapshot / "model.safetensors").write_bytes(b"test-weights")
    return cache, snapshot


def _install_unqualified_backends(monkeypatch):
    from sneeze.tpn import ocr_backends
    from sneeze.tpn.ocr_backend_setup import file_sha256
    from sneeze.tpn.ocr_implementation import backend_implementation_record

    deepseek = replace(
        ocr_backends.get_backend("deepseek"),
        snapshot_identity_sha256=None,
        weights_sha256=None,
    )
    mineru_base = ocr_backends.get_backend("mineru25")
    adapter = (
        Path(ocr_backends.__file__).resolve().parent
        / mineru_base.adapter_relative_path
    ).resolve()
    mineru = replace(
        mineru_base,
        qualified_adapter_sha256=file_sha256(adapter),
        snapshot_identity_sha256=None,
        weights_sha256=None,
    )
    by_id = {
        deepseek.backend_id: deepseek,
        mineru.backend_id: mineru,
    }
    real_get_backend = ocr_backends.get_backend

    def fake_get_backend(name):
        canonical = real_get_backend(name).backend_id
        return by_id[canonical]

    monkeypatch.setattr(ocr_backends, "get_backend", fake_get_backend)
    deepseek = replace(
        deepseek,
        qualified_implementation_sha256=backend_implementation_record(
            deepseek.backend_id
        )["sha256"],
    )
    mineru = replace(
        mineru,
        qualified_implementation_sha256=backend_implementation_record(
            mineru.backend_id
        )["sha256"],
    )
    by_id[deepseek.backend_id] = deepseek
    by_id[mineru.backend_id] = mineru
    return deepseek, mineru


def test_default_registry_path_obeys_xdg_and_home():
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        default_installation_registry_path,
    )

    assert default_installation_registry_path(
        {"XDG_CONFIG_HOME": "/srv/config", "HOME": "/ignored"}
    ) == Path("/srv/config/sneeze/tpn/ocr-installations.json")
    assert default_installation_registry_path(
        {"HOME": "/home/example"}
    ) == Path("/home/example/.config/sneeze/tpn/ocr-installations.json")
    with pytest.raises(
        InstallationRegistryError,
        match="XDG_CONFIG_HOME must be an absolute path",
    ):
        default_installation_registry_path({"XDG_CONFIG_HOME": "relative"})


def test_build_records_exact_machine_artifacts(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_backend_setup import file_sha256, sha256_json
    from sneeze.tpn.ocr_installations import build_backend_installation

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, snapshot = _make_snapshot(tmp_path, backend)
    record = build_backend_installation(
        "deepseek",
        python_executable=executable,
        hf_cache=cache,
        cuda_visible_devices="GPU-deadbeef",
        device="cuda:0",
        attention_backend=None,
    )

    assert record["backend_id"] == backend.backend_id
    assert record["model"] == {
        "id": backend.model_id,
        "revision": backend.model_revision,
    }
    assert record["python_executable"] == {
        "path": str(executable),
        "sha256": file_sha256(executable),
        "installed_distributions_count": 1,
        "installed_distributions_sha256": sha256_json(
            ["synthetic-runtime==1.0"]
        ),
    }
    assert record["hf_cache"]["path"] == str(cache)
    assert record["hf_cache"]["identity_sha256"]
    assert record["snapshot"]["path"] == str(snapshot)
    assert record["snapshot"]["file_count"] == 2
    assert record["snapshot"]["files_sha256"]
    assert record["snapshot"]["weights_sha256"] == file_sha256(
        snapshot / "model.safetensors"
    )
    assert record["adapter"] is None
    assert record["cuda"] == {
        "visible_devices": "GPU-deadbeef",
        "device": "cuda:0",
        "attention_backend": None,
    }
    assert len(record["record_sha256"]) == 64


@pytest.mark.parametrize("visible_devices", ["1", "GPU-deadbeef"])
def test_single_gpu_binding_accepts_one_index_or_uuid(visible_devices):
    from sneeze.tpn.ocr_installations import (
        validate_single_gpu_cuda_binding,
    )

    assert validate_single_gpu_cuda_binding(
        {
            "visible_devices": visible_devices,
            "device": "cuda:0",
        }
    ) == (visible_devices, "cuda:0")


@pytest.mark.parametrize(
    ("routing", "message"),
    [
        (
            {"cuda_visible_devices": "0,1", "device": "cuda:0"},
            "must select exactly one GPU",
        ),
        *[
            (
                {"cuda_visible_devices": selector, "device": "cuda:0"},
                "must be one decimal ordinal or GPU UUID selector",
            )
            for selector in ("-1", "MIG-deadbeef", "0-1", "arbitrary")
        ],
        (
            {"cuda_visible_devices": "1", "device": "cuda:1"},
            "must be cuda:0 for single-GPU OCR pipelines",
        ),
    ],
)
def test_build_rejects_non_single_gpu_routing(
    tmp_path,
    monkeypatch,
    routing,
    message,
):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, _ = _make_snapshot(tmp_path, backend)

    with pytest.raises(InstallationRegistryError, match=message):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
            **routing,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("visible_devices", "GPU-one,GPU-two", "exactly one GPU"),
        *[
            (
                "visible_devices",
                selector,
                "must be one decimal ordinal or GPU UUID selector",
            )
            for selector in ("-1", "MIG-deadbeef", "0-1", "arbitrary")
        ],
        (
            "device",
            "cuda:1",
            "must be cuda:0 for single-GPU OCR pipelines",
        ),
    ],
)
def test_validation_rejects_resealed_non_single_gpu_routing(
    tmp_path,
    monkeypatch,
    field,
    value,
    message,
):
    from sneeze.tpn import ocr_backends, ocr_installations
    from sneeze.tpn.ocr_implementation import backend_implementation_record
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
        validate_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    backend = replace(
        backend,
        qualified_implementation_sha256=backend_implementation_record(
            backend.backend_id
        )["sha256"],
    )
    monkeypatch.setattr(ocr_backends, "get_backend", lambda _name: backend)
    executable = _make_executable(tmp_path)
    cache, _ = _make_snapshot(tmp_path, backend)
    record = build_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
    )
    tampered = copy.deepcopy(record)
    tampered["cuda"][field] = value
    unsigned = dict(tampered)
    unsigned.pop("record_sha256")
    tampered["record_sha256"] = ocr_installations._sha256_json(unsigned)

    with pytest.raises(InstallationRegistryError, match=message):
        validate_backend_installation(tampered, validate_current=False)


def test_atomic_upsert_preserves_other_backends(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_installations import (
        build_backend_installation,
        get_backend_installation,
        load_installation_registry,
        upsert_backend_installation,
    )

    deepseek, mineru = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    deepseek_cache, _ = _make_snapshot(
        tmp_path,
        deepseek,
        cache_name="deepseek-cache",
    )
    mineru_cache, _ = _make_snapshot(
        tmp_path,
        mineru,
        cache_name="mineru-cache",
    )
    registry_path = (tmp_path / "config" / "installations.json").resolve()
    deepseek_record = build_backend_installation(
        deepseek.backend_id,
        python_executable=executable,
        hf_cache=deepseek_cache,
        cuda_visible_devices="0",
        device="cuda:0",
    )
    first = upsert_backend_installation(
        deepseek_record,
        path=registry_path,
    )
    original_deepseek = first["installations"][deepseek.backend_id]

    mineru_record = build_backend_installation(
        mineru.backend_id,
        python_executable=executable,
        hf_cache=mineru_cache,
        cuda_visible_devices="1",
        device="cuda:0",
        attention_backend="sdpa",
    )
    second = upsert_backend_installation(
        mineru_record,
        path=registry_path,
    )

    assert sorted(second["installations"]) == [
        deepseek.backend_id,
        mineru.backend_id,
    ]
    assert second["installations"][deepseek.backend_id] == original_deepseek
    assert load_installation_registry(registry_path) == second
    assert (
        get_backend_installation(
            "mineru25",
            path=registry_path,
        )
        == mineru_record
    )
    assert stat.S_IMODE(registry_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(registry_path.parent.stat().st_mode) == 0o700
    assert not list(registry_path.parent.glob("*.candidate"))


def test_registry_load_accepts_private_ancestry_and_leaf(tmp_path):
    from sneeze.tpn.ocr_installations import (
        empty_installation_registry,
        load_installation_registry,
    )

    parent = tmp_path / "private-config"
    parent.mkdir(mode=0o700)
    parent.chmod(0o700)
    path = parent / "installations.json"
    expected = empty_installation_registry()
    path.write_text(json.dumps(expected), encoding="utf-8")
    path.chmod(0o600)

    assert (
        load_installation_registry(
            path,
            validate_current=False,
        )
        == expected
    )


def test_registry_load_rejects_leaf_and_ancestor_symlinks(tmp_path):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        empty_installation_registry,
        load_installation_registry,
    )

    target_parent = tmp_path / "target-parent"
    target_parent.mkdir(mode=0o700)
    target = target_parent / "installations.json"
    target.write_text(
        json.dumps(empty_installation_registry()),
        encoding="utf-8",
    )
    target.chmod(0o600)

    leaf_link = tmp_path / "leaf-link.json"
    leaf_link.symlink_to(target)
    with pytest.raises(
        InstallationRegistryError,
        match="regular, non-symlink file",
    ):
        load_installation_registry(leaf_link, validate_current=False)

    parent_link = tmp_path / "parent-link"
    parent_link.symlink_to(target_parent, target_is_directory=True)
    with pytest.raises(
        InstallationRegistryError,
        match="ancestry cannot contain symlinks",
    ):
        load_installation_registry(
            parent_link / "installations.json",
            validate_current=False,
        )


def test_registry_rejects_insecure_ancestry_and_leaf_modes(tmp_path):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        empty_installation_registry,
        load_installation_registry,
    )

    unsafe_parent = tmp_path / "shared-config"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o770)
    with pytest.raises(
        InstallationRegistryError,
        match="ancestry must not be writable",
    ):
        load_installation_registry(
            unsafe_parent / "missing.json",
            validate_current=False,
        )

    private_parent = tmp_path / "private-config"
    private_parent.mkdir(mode=0o700)
    path = private_parent / "installations.json"
    path.write_text(
        json.dumps(empty_installation_registry()),
        encoding="utf-8",
    )
    path.chmod(0o640)
    with pytest.raises(
        InstallationRegistryError,
        match=r"private permissions \(0600\)",
    ):
        load_installation_registry(path, validate_current=False)


def test_registry_owner_checks_and_root_sticky_exception(
    tmp_path,
    monkeypatch,
):
    from types import SimpleNamespace

    from sneeze.tpn import ocr_installations
    from sneeze.tpn.ocr_installations import InstallationRegistryError

    current_uid = os.geteuid()
    foreign_uid = current_uid + 1
    monkeypatch.setattr(
        ocr_installations,
        "_effective_uid",
        lambda: current_uid,
    )
    foreign_directory = SimpleNamespace(
        st_mode=stat.S_IFDIR | 0o700,
        st_uid=foreign_uid,
    )
    with pytest.raises(
        InstallationRegistryError,
        match="owned by root or the current user",
    ):
        ocr_installations._validate_registry_directory_stat(
            foreign_directory,
            label=str(tmp_path / "foreign"),
        )

    root_sticky = SimpleNamespace(
        st_mode=stat.S_IFDIR | stat.S_ISVTX | 0o777,
        st_uid=0,
    )
    ocr_installations._validate_registry_directory_stat(
        root_sticky,
        label="/tmp",
    )

    leaf = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_uid=foreign_uid)
    with pytest.raises(
        InstallationRegistryError,
        match="registry must be owned by the current user",
    ):
        ocr_installations._validate_registry_leaf_stat(leaf)


def test_registry_read_rejects_leaf_replacement_race(tmp_path, monkeypatch):
    from sneeze.tpn import ocr_installations
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        empty_installation_registry,
        load_installation_registry,
    )

    path = tmp_path / "installations.json"
    replacement = tmp_path / "replacement.json"
    payload = json.dumps(empty_installation_registry())
    path.write_text(payload, encoding="utf-8")
    replacement.write_text(payload, encoding="utf-8")
    path.chmod(0o600)
    replacement.chmod(0o600)
    real_stat = os.stat
    real_replace = os.replace
    raced = False

    def racing_stat(name, *args, dir_fd=None, follow_symlinks=True, **kwargs):
        nonlocal raced
        if name == path.name and dir_fd is not None and not raced:
            raced = True
            real_replace(
                replacement.name,
                path.name,
                src_dir_fd=dir_fd,
                dst_dir_fd=dir_fd,
            )
        return real_stat(
            name,
            *args,
            dir_fd=dir_fd,
            follow_symlinks=follow_symlinks,
            **kwargs,
        )

    monkeypatch.setattr(ocr_installations.os, "stat", racing_stat)
    with pytest.raises(
        InstallationRegistryError,
        match="identity changed while opening",
    ):
        load_installation_registry(path, validate_current=False)
    assert raced is True


def test_registry_atomic_write_remains_anchored_after_parent_replacement(
    tmp_path,
):
    from sneeze.tpn.ocr_installations import (
        _atomic_write_registry,
        _open_secure_registry_parent,
        empty_installation_registry,
    )

    container = tmp_path / "container"
    original_parent = container / "registry-parent"
    moved_parent = container / "moved-parent"
    replacement_parent = container / "replacement-parent"
    original_parent.mkdir(parents=True, mode=0o700)
    replacement_parent.mkdir(mode=0o700)
    registry_path = original_parent / "installations.json"
    descriptor = _open_secure_registry_parent(registry_path, create=False)
    assert descriptor is not None
    original_parent.rename(moved_parent)
    original_parent.symlink_to(replacement_parent, target_is_directory=True)
    try:
        _atomic_write_registry(
            registry_path,
            empty_installation_registry(),
            parent_descriptor=descriptor,
            expected_identity=None,
        )
    finally:
        os.close(descriptor)

    assert (moved_parent / registry_path.name).is_file()
    assert (
        stat.S_IMODE((moved_parent / registry_path.name).stat().st_mode)
        == 0o600
    )
    assert not (replacement_parent / registry_path.name).exists()


def test_registry_atomic_create_never_replaces_unexpected_leaf(tmp_path):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        _atomic_write_registry,
        _open_secure_registry_parent,
        empty_installation_registry,
    )

    path = tmp_path / "installations.json"
    descriptor = _open_secure_registry_parent(path, create=False)
    assert descriptor is not None
    path.write_text("preserve this file\n", encoding="utf-8")
    path.chmod(0o600)
    try:
        with pytest.raises(
            InstallationRegistryError,
            match="appeared while writing",
        ):
            _atomic_write_registry(
                path,
                empty_installation_registry(),
                parent_descriptor=descriptor,
                expected_identity=None,
            )
    finally:
        os.close(descriptor)
    assert path.read_text(encoding="utf-8") == "preserve this file\n"


def test_registry_creation_ignores_restrictive_umask(tmp_path):
    from sneeze.tpn.ocr_installations import (
        _atomic_write_registry,
        _open_secure_registry_parent,
        empty_installation_registry,
    )

    path = tmp_path / "new-config" / "nested" / "installations.json"
    previous_umask = os.umask(0o777)
    descriptor = None
    try:
        descriptor = _open_secure_registry_parent(path, create=True)
        assert descriptor is not None
        _atomic_write_registry(
            path,
            empty_installation_registry(),
            parent_descriptor=descriptor,
            expected_identity=None,
        )
    finally:
        os.umask(previous_umask)
        if descriptor is not None:
            os.close(descriptor)

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_registry_lock_rejects_symlink_without_changing_target(tmp_path):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        _exclusive_registry_lock,
    )

    registry_path = (tmp_path / "installations.json").resolve()
    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    target = tmp_path / "lock-target"
    target.write_text("do not open through the symlink\n", encoding="utf-8")
    target.chmod(0o640)
    lock_path.symlink_to(target)

    with pytest.raises(
        InstallationRegistryError,
        match="could not open installation registry lock",
    ):
        with _exclusive_registry_lock(registry_path):
            pytest.fail("symlink lock unexpectedly acquired")

    assert target.read_text(encoding="utf-8") == (
        "do not open through the symlink\n"
    )
    assert stat.S_IMODE(target.stat().st_mode) == 0o640


def test_registry_lock_rejects_fifo_without_blocking_or_chmod(tmp_path):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        _exclusive_registry_lock,
    )

    registry_path = (tmp_path / "installations.json").resolve()
    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    os.mkfifo(lock_path, mode=0o640)
    original_mode = stat.S_IMODE(lock_path.stat().st_mode)

    with pytest.raises(
        InstallationRegistryError,
        match="lock must be a regular file",
    ):
        with _exclusive_registry_lock(registry_path):
            pytest.fail("FIFO lock unexpectedly acquired")

    assert stat.S_IMODE(lock_path.stat().st_mode) == original_mode


def test_registry_lock_checks_owner_before_chmod_or_flock(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_installations
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        _open_registry_lock,
    )

    registry_path = (tmp_path / "installations.json").resolve()
    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    lock_path.write_text("", encoding="utf-8")
    lock_path.chmod(0o640)
    calls = []
    parent_descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    monkeypatch.setattr(
        ocr_installations,
        "_effective_uid",
        lambda: lock_path.stat().st_uid + 1,
    )
    monkeypatch.setattr(
        ocr_installations.os,
        "fchmod",
        lambda *_args: calls.append("fchmod"),
    )
    try:
        with pytest.raises(
            InstallationRegistryError,
            match="lock must be owned by the current user",
        ):
            _open_registry_lock(parent_descriptor, lock_path.name)
    finally:
        os.close(parent_descriptor)

    assert calls == []
    assert stat.S_IMODE(lock_path.stat().st_mode) == 0o640


def test_registry_lock_reuses_existing_regular_file(tmp_path):
    from sneeze.tpn.ocr_installations import _exclusive_registry_lock

    registry_path = (tmp_path / "installations.json").resolve()
    lock_path = registry_path.with_name(f".{registry_path.name}.lock")
    observed_inodes = []
    for _ in range(2):
        with _exclusive_registry_lock(registry_path):
            metadata = lock_path.stat()
            observed_inodes.append((metadata.st_dev, metadata.st_ino))
            assert stat.S_IMODE(metadata.st_mode) == 0o600

    assert observed_inodes[0] == observed_inodes[1]


def test_registry_rejects_content_and_record_hash_tampering(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_installations
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
        load_installation_registry,
        upsert_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, _ = _make_snapshot(tmp_path, backend)
    registry_path = (tmp_path / "installations.json").resolve()
    record = build_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
    )
    upsert_backend_installation(record, path=registry_path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["installations"][backend.backend_id]["cuda"]["device"] = "cuda:1"
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        InstallationRegistryError,
        match="content hash mismatch",
    ):
        load_installation_registry(registry_path)

    unsigned = dict(payload)
    unsigned.pop("content_sha256")
    payload["content_sha256"] = ocr_installations._sha256_json(unsigned)
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        InstallationRegistryError,
        match="record hash mismatch",
    ):
        load_installation_registry(registry_path)


def test_registry_rejects_stale_executable(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        load_installation_registry,
        register_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, _ = _make_snapshot(tmp_path, backend)
    registry_path = (tmp_path / "installations.json").resolve()
    register_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
        path=registry_path,
    )
    executable.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
    with pytest.raises(
        InstallationRegistryError,
        match="stale Python executable hash",
    ):
        load_installation_registry(registry_path)


def test_installation_rejects_installed_distribution_drift(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import ocr_installations
    from sneeze.tpn.ocr_backend_setup import sha256_json
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
        validate_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, _ = _make_snapshot(tmp_path, backend)
    record = build_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
    )
    changed = ["synthetic-runtime==2.0"]
    monkeypatch.setattr(
        ocr_installations,
        "_probe_installed_distributions",
        lambda _python: {
            "count": len(changed),
            "sha256": sha256_json(changed),
        },
    )

    with pytest.raises(
        InstallationRegistryError,
        match="stale Python distributions",
    ):
        validate_backend_installation(record, validate_current=True)


def test_registry_rejects_stale_snapshot(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        load_installation_registry,
        register_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, snapshot = _make_snapshot(tmp_path, backend)
    registry_path = (tmp_path / "installations.json").resolve()
    register_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
        path=registry_path,
    )
    (snapshot / "config.json").write_text(
        '{"tampered": true}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        InstallationRegistryError,
        match="stale or tampered model snapshot",
    ):
        load_installation_registry(registry_path)


def test_registry_rejects_stale_backend_revision(tmp_path, monkeypatch):
    from sneeze.tpn import ocr_backends
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
        validate_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, _ = _make_snapshot(tmp_path, backend)
    record = build_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
    )
    changed = replace(backend, model_revision="0" * 40)
    monkeypatch.setattr(ocr_backends, "get_backend", lambda _name: changed)
    with pytest.raises(
        InstallationRegistryError,
        match="stale installation model revision",
    ):
        validate_backend_installation(record, validate_current=False)


def test_registry_rejects_stale_adapter(tmp_path, monkeypatch):
    from sneeze.tpn import ocr_backends, ocr_installations
    from sneeze.tpn.ocr_backend_setup import file_sha256
    from sneeze.tpn.ocr_implementation import backend_implementation_record
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        load_installation_registry,
        register_backend_installation,
    )

    base = ocr_backends.get_backend("mineru25")
    adapter = tmp_path / "adapter.py"
    adapter.write_text("# adapter v1\n", encoding="utf-8")
    backend = replace(
        base,
        qualified_adapter_sha256=file_sha256(adapter),
        snapshot_identity_sha256=None,
        weights_sha256=None,
    )
    monkeypatch.setattr(ocr_backends, "get_backend", lambda _name: backend)
    backend = replace(
        backend,
        qualified_implementation_sha256=backend_implementation_record(
            backend.backend_id
        )["sha256"],
    )
    monkeypatch.setattr(
        ocr_installations,
        "_adapter_path",
        lambda _backend: adapter.resolve(),
    )
    executable = _make_executable(tmp_path)
    cache, _ = _make_snapshot(tmp_path, backend)
    registry_path = (tmp_path / "installations.json").resolve()
    register_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
        path=registry_path,
    )
    adapter.write_text("# adapter v2\n", encoding="utf-8")
    with pytest.raises(
        InstallationRegistryError,
        match="adapter is not qualified",
    ):
        load_installation_registry(registry_path)


def test_nonvalidating_load_still_checks_schema_and_hashes(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        load_installation_registry,
        register_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, snapshot = _make_snapshot(tmp_path, backend)
    registry_path = (tmp_path / "installations.json").resolve()
    register_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
        path=registry_path,
    )
    executable.unlink()
    for path in snapshot.iterdir():
        path.unlink()
    snapshot.rmdir()
    assert load_installation_registry(
        registry_path,
        validate_current=False,
    )["installations"][backend.backend_id]["model"]["revision"] == (
        backend.model_revision
    )

    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    payload["unexpected"] = True
    registry_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(
        InstallationRegistryError,
        match="unexpected or missing fields",
    ):
        load_installation_registry(
            registry_path,
            validate_current=False,
        )


def test_registry_rejects_secret_bearing_paths(tmp_path, monkeypatch):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache, _ = _make_snapshot(
        tmp_path,
        backend,
        cache_name="api_key=do-not-record-this",
    )
    with pytest.raises(
        InstallationRegistryError,
        match="credential-like",
    ):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )


def test_registry_rejects_relative_and_missing_artifact_paths(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
        load_installation_registry,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    executable = _make_executable(tmp_path)
    cache = (tmp_path / "empty-cache").resolve()
    cache.mkdir()
    with pytest.raises(
        InstallationRegistryError,
        match="Python executable must be an absolute path",
    ):
        build_backend_installation(
            backend.backend_id,
            python_executable="python",
            hf_cache=cache,
        )
    with pytest.raises(
        InstallationRegistryError,
        match="pinned model snapshot is missing",
    ):
        build_backend_installation(
            backend.backend_id,
            python_executable=executable,
            hf_cache=cache,
        )
    with pytest.raises(
        InstallationRegistryError,
        match="registry path must be absolute",
    ):
        load_installation_registry("relative.json")


@pytest.mark.parametrize("navigation_component", [".", ".."])
def test_validation_rejects_lexical_navigation_in_stored_paths(
    tmp_path,
    monkeypatch,
    navigation_component,
):
    from sneeze.tpn import ocr_backends
    from sneeze.tpn.ocr_implementation import backend_implementation_record
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        build_backend_installation,
        validate_backend_installation,
    )

    backend, _ = _install_unqualified_backends(monkeypatch)
    backend = replace(
        backend,
        qualified_implementation_sha256=backend_implementation_record(
            backend.backend_id
        )["sha256"],
    )
    monkeypatch.setattr(ocr_backends, "get_backend", lambda _name: backend)
    executable = _make_executable(tmp_path)
    cache, _ = _make_snapshot(tmp_path, backend)
    record = build_backend_installation(
        backend.backend_id,
        python_executable=executable,
        hf_cache=cache,
    )
    tampered = copy.deepcopy(record)
    tampered["python_executable"]["path"] = (
        f"{executable.parent}/{navigation_component}/{executable.name}"
    )

    with pytest.raises(
        InstallationRegistryError,
        match="must be a canonical absolute path",
    ):
        validate_backend_installation(tampered, validate_current=False)


def test_registry_file_rejects_duplicate_json_fields(tmp_path):
    from sneeze.tpn.ocr_installations import (
        InstallationRegistryError,
        load_installation_registry,
    )

    path = (tmp_path / "installations.json").resolve()
    path.write_text(
        '{"schema_version":1,"schema_version":1}\n',
        encoding="utf-8",
    )
    path.chmod(0o600)
    with pytest.raises(
        InstallationRegistryError,
        match="duplicate JSON field",
    ):
        load_installation_registry(path, validate_current=False)
