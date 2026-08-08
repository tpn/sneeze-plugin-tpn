import os
import stat

import pytest


def test_private_directory_accepts_sticky_tmp_ancestor_and_creates_0700(
    tmp_path,
):
    from sneeze.tpn.ocr_path_security import prepare_private_directory

    destination = tmp_path / "private" / "run"
    assert prepare_private_directory(
        destination,
        label="test work directory",
    ) == destination
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert destination.stat().st_uid == os.geteuid()


def test_private_directory_creation_ignores_restrictive_umask(tmp_path):
    from sneeze.tpn.ocr_path_security import prepare_private_directory

    destination = tmp_path / "private"
    previous_umask = os.umask(0o777)
    try:
        assert prepare_private_directory(
            destination,
            label="test work directory",
        ) == destination
    finally:
        os.umask(previous_umask)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o700


def test_private_directory_rejects_group_writable_ancestor(tmp_path):
    from sneeze.tpn.ocr_path_security import (
        TrustedPathError,
        prepare_private_directory,
    )

    unsafe = tmp_path / "shared"
    unsafe.mkdir()
    unsafe.chmod(0o775)

    with pytest.raises(TrustedPathError, match="writable by group"):
        prepare_private_directory(
            unsafe / "run",
            label="test work directory",
        )


def test_private_directory_requires_current_owned_0700_leaf(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_path_security as path_security

    destination = tmp_path / "existing"
    destination.mkdir(mode=0o755)
    destination.chmod(0o755)
    with pytest.raises(path_security.TrustedPathError, match="mode 0700"):
        path_security.prepare_private_directory(
            destination,
            label="test work directory",
        )

    destination.chmod(0o700)
    assert path_security.prepare_private_directory(
        destination,
        label="test work directory",
    ) == destination

    monkeypatch.setattr(
        path_security,
        "_effective_uid",
        lambda: os.geteuid() + 1,
    )
    with pytest.raises(path_security.TrustedPathError, match="owned by"):
        path_security.prepare_private_directory(
            destination,
            label="test work directory",
        )


def test_private_directory_rejects_symlink_leaf(tmp_path):
    from sneeze.tpn.ocr_path_security import (
        TrustedPathError,
        prepare_private_directory,
    )

    target = tmp_path / "target"
    target.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(TrustedPathError, match="symlink"):
        prepare_private_directory(link, label="test work directory")


def test_trusted_file_rejects_group_writable_leaf(tmp_path):
    from sneeze.tpn.ocr_path_security import (
        TrustedPathError,
        validate_trusted_file,
    )

    executable = tmp_path / "python"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o775)

    with pytest.raises(TrustedPathError, match="writable by group"):
        validate_trusted_file(
            executable,
            label="Python executable",
            executable=True,
        )

    executable.chmod(0o755)
    assert validate_trusted_file(
        executable,
        label="Python executable",
        executable=True,
    ) == executable


def test_trusted_tree_rejects_writable_package_file_and_directory(tmp_path):
    from sneeze.tpn.ocr_path_security import (
        TrustedPathError,
        validate_trusted_tree,
    )

    environment = tmp_path / "environment"
    package = environment / "lib" / "site-packages" / "example"
    package.mkdir(parents=True)
    module = package / "__init__.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")

    module.chmod(0o664)
    with pytest.raises(TrustedPathError, match="writable by group"):
        validate_trusted_tree(environment, label="Python environment")

    module.chmod(0o644)
    package.chmod(0o775)
    with pytest.raises(TrustedPathError, match="writable by group"):
        validate_trusted_tree(environment, label="Python environment")

    package.chmod(0o755)
    assert validate_trusted_tree(
        environment,
        label="Python environment",
    ) == environment


def test_trusted_tree_allows_internal_file_symlink_only(tmp_path):
    from sneeze.tpn.ocr_path_security import (
        TrustedPathError,
        validate_trusted_tree,
    )

    cache = tmp_path / "cache"
    snapshot = cache / "snapshots" / ("a" * 40)
    blobs = cache / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    blob = blobs / "weights"
    blob.write_bytes(b"weights")
    (snapshot / "model.safetensors").symlink_to(blob)

    assert validate_trusted_tree(
        snapshot,
        label="model snapshot",
        symlink_root=cache,
    ) == snapshot

    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    (snapshot / "model.safetensors").unlink()
    (snapshot / "model.safetensors").symlink_to(outside)
    with pytest.raises(TrustedPathError, match="outside its trusted root"):
        validate_trusted_tree(
            snapshot,
            label="model snapshot",
            symlink_root=cache,
        )


def test_trusted_file_rejects_symlink_loop_as_trust_error(tmp_path):
    from sneeze.tpn.ocr_path_security import (
        TrustedPathError,
        validate_trusted_file,
    )

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.symlink_to(second)
    second.symlink_to(first)

    with pytest.raises(TrustedPathError, match="does not exist"):
        validate_trusted_file(first, label="looped file")


def test_trusted_tree_rejects_internal_symlink_loop_as_trust_error(tmp_path):
    from sneeze.tpn.ocr_path_security import (
        TrustedPathError,
        validate_trusted_tree,
    )

    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "loop").symlink_to("loop")

    with pytest.raises(TrustedPathError, match="outside its trusted root"):
        validate_trusted_tree(tree, label="looped tree")


def test_private_output_path_requires_current_owned_0600_file(tmp_path):
    from sneeze.tpn.ocr_path_security import (
        TrustedPathError,
        prepare_private_output_path,
    )

    output = tmp_path / "profiles" / "profile.json"
    assert prepare_private_output_path(
        output,
        label="profile output",
    ) == output
    output.write_text("{}\n", encoding="utf-8")
    output.chmod(0o644)
    with pytest.raises(TrustedPathError, match="mode 0600"):
        prepare_private_output_path(output, label="profile output")
    output.chmod(0o600)
    assert prepare_private_output_path(
        output,
        label="profile output",
    ) == output
