"""POSIX trust checks for local OCR execution artifacts and work trees.

Local OCR backends execute Python environments and model code supplied by
machine-local paths.  Content hashes detect drift, but they do not prevent a
different user from replacing a path between validation and execution.  This
module establishes the complementary filesystem boundary: every reachable
component must be controlled by root or the current effective user, and no
other user may have write permission.

The sole writable-ancestor exception is a root-owned sticky directory such as
``/tmp``.  A private work directory beneath it must still be owned by the
current user and have mode ``0700``.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from pathlib import Path


class TrustedPathError(ValueError):
    """Raised when a local execution path crosses an unsafe trust boundary."""


_DIRECTORY_MODE = 0o700
_OUTPUT_FILE_MODE = 0o600
_PROTECTED_CORPUS_POLICY_RELATIVE_PATH = (
    Path("sneeze") / "tpn" / "ocr-protected-roots"
)


def _effective_uid() -> int:
    getter = getattr(os, "geteuid", None)
    if getter is None:  # pragma: no cover - supported OCR hosts are POSIX
        raise TrustedPathError("trusted OCR paths require POSIX user IDs")
    return int(getter())


def _directory_open_flags() -> int:
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, name) for name in required):
        raise TrustedPathError(
            "trusted OCR paths require POSIX openat controls"
        )
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
    )


def _file_open_flags() -> int:
    if not hasattr(os, "O_NOFOLLOW"):
        raise TrustedPathError(
            "trusted OCR paths require POSIX no-follow opens"
        )
    return (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _canonical_absolute_path(
    raw_path: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        raise TrustedPathError(f"{label} must be an absolute path")
    canonical = Path(os.path.abspath(os.fspath(path)))
    if path != canonical or any(part in {".", ".."} for part in path.parts):
        raise TrustedPathError(f"{label} must be a canonical absolute path")
    return canonical


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _validate_owner(metadata: os.stat_result, *, label: str) -> None:
    if metadata.st_uid not in {0, _effective_uid()}:
        raise TrustedPathError(
            f"{label} must be owned by root or the current user"
        )


def _validate_directory_metadata(
    metadata: os.stat_result,
    *,
    label: str,
    allow_root_sticky: bool,
    require_current_owner: bool = False,
    exact_mode: int | None = None,
) -> None:
    if not stat.S_ISDIR(metadata.st_mode):
        raise TrustedPathError(f"{label} must contain only directories")
    _validate_owner(metadata, label=label)
    if require_current_owner and metadata.st_uid != _effective_uid():
        raise TrustedPathError(f"{label} must be owned by the current user")
    mode = stat.S_IMODE(metadata.st_mode)
    writable_by_others = mode & 0o022
    root_sticky = metadata.st_uid == 0 and bool(
        metadata.st_mode & stat.S_ISVTX
    )
    if writable_by_others and not (allow_root_sticky and root_sticky):
        raise TrustedPathError(
            f"{label} must not be writable by group or other users"
        )
    if exact_mode is not None and mode != exact_mode:
        raise TrustedPathError(f"{label} must have mode {exact_mode:04o}")


def _validate_regular_file_metadata(
    metadata: os.stat_result,
    *,
    label: str,
    executable: bool,
    require_current_owner: bool = False,
    exact_mode: int | None = None,
) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise TrustedPathError(f"{label} must be a regular file")
    _validate_owner(metadata, label=label)
    if require_current_owner and metadata.st_uid != _effective_uid():
        raise TrustedPathError(f"{label} must be owned by the current user")
    mode = stat.S_IMODE(metadata.st_mode)
    if mode & 0o022:
        raise TrustedPathError(
            f"{label} must not be writable by group or other users"
        )
    if exact_mode is not None and mode != exact_mode:
        raise TrustedPathError(f"{label} must have mode {exact_mode:04o}")
    if executable and not mode & 0o111:
        raise TrustedPathError(f"{label} must be executable")


def _open_directory_chain(
    path: Path,
    *,
    label: str,
    create: bool,
    private_leaf: bool,
) -> int:
    """Return an fd for ``path`` after an fd-anchored, no-follow walk."""

    flags = _directory_open_flags()
    try:
        descriptor = os.open(path.anchor, flags)
    except OSError as exc:
        raise TrustedPathError(
            f"could not open {label} filesystem root"
        ) from exc
    try:
        root_metadata = os.fstat(descriptor)
        _validate_directory_metadata(
            root_metadata,
            label=path.anchor,
            allow_root_sticky=True,
        )
        current = Path(path.anchor)
        components = path.parts[1:]
        if not components and private_leaf:
            raise TrustedPathError(
                f"{label} must not be the filesystem root"
            )
        for index, component in enumerate(components):
            current /= component
            leaf = index == len(components) - 1
            created = False
            try:
                path_metadata = os.stat(
                    component,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                if not create:
                    raise TrustedPathError(
                        f"{label} does not exist"
                    ) from None
                try:
                    os.mkdir(component, _DIRECTORY_MODE, dir_fd=descriptor)
                    created = True
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise TrustedPathError(
                        f"could not create {label}"
                    ) from exc
                try:
                    path_metadata = os.stat(
                        component,
                        dir_fd=descriptor,
                        follow_symlinks=False,
                    )
                except OSError as exc:
                    raise TrustedPathError(
                        f"could not inspect {label}"
                    ) from exc
                if created:
                    try:
                        os.chmod(
                            component,
                            _DIRECTORY_MODE,
                            dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                        chmod_metadata = os.stat(
                            component,
                            dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                    except OSError as exc:
                        raise TrustedPathError(
                            f"could not secure {label}"
                        ) from exc
                    if not _same_identity(path_metadata, chmod_metadata):
                        raise TrustedPathError(
                            f"{label} changed while securing"
                        ) from None
                    path_metadata = chmod_metadata
            except OSError as exc:
                raise TrustedPathError(f"could not inspect {label}") from exc

            if stat.S_ISLNK(path_metadata.st_mode):
                raise TrustedPathError(f"{label} must not traverse symlinks")
            try:
                child = os.open(component, flags, dir_fd=descriptor)
            except OSError as exc:
                raise TrustedPathError(
                    f"could not securely open {label}"
                ) from exc
            try:
                descriptor_metadata = os.fstat(child)
                if not _same_identity(path_metadata, descriptor_metadata):
                    raise TrustedPathError(f"{label} changed while opening")
                _validate_directory_metadata(
                    descriptor_metadata,
                    label=str(current),
                    allow_root_sticky=not leaf,
                    require_current_owner=leaf and private_leaf,
                    exact_mode=(
                        _DIRECTORY_MODE if leaf and private_leaf else None
                    ),
                )
                if created:
                    _validate_directory_metadata(
                        descriptor_metadata,
                        label=str(current),
                        allow_root_sticky=False,
                        require_current_owner=True,
                        exact_mode=_DIRECTORY_MODE,
                    )
            except Exception:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        return descriptor
    except Exception:
        os.close(descriptor)
        raise


def prepare_private_directory(
    raw_path: str | os.PathLike[str],
    *,
    label: str,
    require_empty: bool = False,
) -> Path:
    """Create or validate a current-user-owned ``0700`` work directory."""

    path = _canonical_absolute_path(raw_path, label=label)
    descriptor = _open_directory_chain(
        path,
        label=label,
        create=True,
        private_leaf=True,
    )
    try:
        if require_empty and os.listdir(descriptor):
            raise TrustedPathError(f"{label} must be empty")
        path_metadata = path.lstat()
        descriptor_metadata = os.fstat(descriptor)
        if not _same_identity(path_metadata, descriptor_metadata):
            raise TrustedPathError(f"{label} changed after validation")
    finally:
        os.close(descriptor)
    return path


def prepare_trusted_directory(
    raw_path: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    """Create missing parents privately and validate a non-writable leaf."""

    path = _canonical_absolute_path(raw_path, label=label)
    descriptor = _open_directory_chain(
        path,
        label=label,
        create=True,
        private_leaf=False,
    )
    try:
        path_metadata = path.lstat()
        descriptor_metadata = os.fstat(descriptor)
        if not _same_identity(path_metadata, descriptor_metadata):
            raise TrustedPathError(f"{label} changed after validation")
    finally:
        os.close(descriptor)
    return path


def validate_trusted_directory(
    raw_path: str | os.PathLike[str],
    *,
    label: str,
    resolve_symlinks: bool = True,
) -> Path:
    """Resolve and validate one existing trusted directory."""

    source = _canonical_absolute_path(raw_path, label=label)
    if resolve_symlinks:
        try:
            path = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrustedPathError(f"{label} does not exist") from exc
    else:
        path = source
    descriptor = _open_directory_chain(
        path,
        label=label,
        create=False,
        private_leaf=False,
    )
    try:
        path_metadata = path.lstat()
        descriptor_metadata = os.fstat(descriptor)
        if not _same_identity(path_metadata, descriptor_metadata):
            raise TrustedPathError(f"{label} changed after validation")
    finally:
        os.close(descriptor)
    return path


def validate_trusted_file(
    raw_path: str | os.PathLike[str],
    *,
    label: str,
    executable: bool = False,
    resolve_symlinks: bool = True,
) -> Path:
    """Resolve and fd-validate a trusted regular file."""

    source = _canonical_absolute_path(raw_path, label=label)
    if resolve_symlinks:
        try:
            path = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise TrustedPathError(f"{label} does not exist") from exc
    else:
        path = source
    parent = validate_trusted_directory(
        path.parent,
        label=f"{label} ancestry",
        resolve_symlinks=resolve_symlinks,
    )
    try:
        parent_descriptor = os.open(parent, _directory_open_flags())
    except OSError as exc:
        raise TrustedPathError(f"could not open {label} parent") from exc
    try:
        try:
            descriptor = os.open(
                path.name,
                _file_open_flags(),
                dir_fd=parent_descriptor,
            )
        except OSError as exc:
            raise TrustedPathError(
                f"could not securely open {label}"
            ) from exc
        try:
            descriptor_metadata = os.fstat(descriptor)
            path_metadata = os.stat(
                path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            if not _same_identity(path_metadata, descriptor_metadata):
                raise TrustedPathError(f"{label} changed while opening")
            _validate_regular_file_metadata(
                descriptor_metadata,
                label=label,
                executable=executable,
            )
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)
    return path


def _protected_corpus_policy_path(
    environment: Mapping[str, str],
) -> Path:
    raw_config_home = environment.get("XDG_CONFIG_HOME")
    if raw_config_home:
        config_home = Path(raw_config_home)
    else:
        raw_home = environment.get("HOME")
        config_home = (
            Path(raw_home) if raw_home else Path.home()
        ) / ".config"
    try:
        return _canonical_absolute_path(
            config_home / _PROTECTED_CORPUS_POLICY_RELATIVE_PATH,
            label="protected corpus policy",
        )
    except TrustedPathError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrustedPathError(
            "could not resolve protected corpus policy"
        ) from exc


def load_protected_corpus_roots(
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, ...]:
    """Load private machine-local corpus roots without exposing them."""

    policy_path = _protected_corpus_policy_path(
        os.environ if environment is None else environment
    )
    try:
        policy_path.lstat()
    except FileNotFoundError:
        return ()
    except (OSError, ValueError) as exc:
        raise TrustedPathError(
            "could not inspect protected corpus policy"
        ) from exc
    policy_path = validate_trusted_file(
        policy_path,
        label="protected corpus policy",
        resolve_symlinks=False,
    )
    try:
        metadata = policy_path.lstat()
    except OSError as exc:
        raise TrustedPathError(
            "could not inspect protected corpus policy"
        ) from exc
    _validate_regular_file_metadata(
        metadata,
        label="protected corpus policy",
        executable=False,
        require_current_owner=True,
        exact_mode=_OUTPUT_FILE_MODE,
    )
    try:
        contents = policy_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise TrustedPathError(
            "could not read protected corpus policy"
        ) from exc

    roots: dict[str, Path] = {}
    for line_number, raw_line in enumerate(contents.splitlines(), start=1):
        value = raw_line.strip()
        if not value or value.startswith("#"):
            continue
        try:
            root = _canonical_absolute_path(
                value,
                label="protected corpus root",
            )
            root = validate_trusted_directory(
                root,
                label="protected corpus root",
            )
        except (OSError, RuntimeError, TrustedPathError, ValueError) as exc:
            raise TrustedPathError(
                "protected corpus policy contains an invalid root at line "
                f"{line_number}"
            ) from exc
        roots[str(root)] = root
    return tuple(roots[key] for key in sorted(roots))


def reject_protected_corpus_overlap(
    raw_path: str | os.PathLike[str],
    *,
    label: str,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Reject a path that contains or is contained by a protected corpus."""

    try:
        path = _canonical_absolute_path(raw_path, label=label)
        resolved = Path(os.path.realpath(path))
    except TrustedPathError:
        raise
    except (OSError, RuntimeError, ValueError) as exc:
        raise TrustedPathError(f"{label} is invalid") from exc
    for root in load_protected_corpus_roots(environment=environment):
        if (
            resolved == root
            or resolved.is_relative_to(root)
            or root.is_relative_to(resolved)
        ):
            raise TrustedPathError(
                f"{label} must be separate from protected corpora"
            )
    return resolved


def prepare_private_output_path(
    raw_path: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    """Validate a new or current-user-owned ``0600`` output-file path."""

    path = _canonical_absolute_path(raw_path, label=label)
    prepare_trusted_directory(path.parent, label=f"{label} parent")
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return path
    except OSError as exc:
        raise TrustedPathError(f"could not inspect {label}") from exc
    if stat.S_ISLNK(metadata.st_mode):
        raise TrustedPathError(f"{label} must not be a symlink")
    _validate_regular_file_metadata(
        metadata,
        label=label,
        executable=False,
        require_current_owner=True,
        exact_mode=_OUTPUT_FILE_MODE,
    )
    return path


def validate_trusted_tree(
    raw_root: str | os.PathLike[str],
    *,
    label: str,
    symlink_root: str | os.PathLike[str] | None = None,
    resolve_root_symlinks: bool = True,
) -> Path:
    """Recursively reject cross-user-writable executable or model content."""

    root = validate_trusted_directory(
        raw_root,
        label=label,
        resolve_symlinks=resolve_root_symlinks,
    )
    allowed = (
        root
        if symlink_root is None
        else validate_trusted_directory(
            symlink_root,
            label=f"{label} symlink root",
        )
    )
    if not root.is_relative_to(allowed):
        raise TrustedPathError(f"{label} must remain within its symlink root")

    stack = [root]
    visited: set[tuple[int, int]] = set()
    while stack:
        directory = stack.pop()
        try:
            directory_metadata = directory.stat()
        except OSError as exc:
            raise TrustedPathError(f"could not inspect {label}") from exc
        _validate_directory_metadata(
            directory_metadata,
            label=str(directory),
            allow_root_sticky=False,
        )
        identity = (directory_metadata.st_dev, directory_metadata.st_ino)
        if identity in visited:
            continue
        visited.add(identity)
        try:
            entries = list(os.scandir(directory))
        except OSError as exc:
            raise TrustedPathError(f"could not scan {label}") from exc
        for entry in entries:
            entry_path = Path(entry.path)
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise TrustedPathError(f"could not inspect {label}") from exc
            if stat.S_ISLNK(metadata.st_mode):
                _validate_owner(metadata, label=str(entry_path))
                try:
                    target = entry_path.resolve(strict=True)
                    target.relative_to(allowed)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise TrustedPathError(
                        f"{label} contains a symlink outside its trusted root"
                    ) from exc
                if target.is_dir():
                    validate_trusted_directory(
                        target,
                        label=f"{label} symlink target",
                    )
                    stack.append(target)
                else:
                    validate_trusted_file(
                        target,
                        label=f"{label} symlink target",
                    )
                continue
            if stat.S_ISDIR(metadata.st_mode):
                _validate_directory_metadata(
                    metadata,
                    label=str(entry_path),
                    allow_root_sticky=False,
                )
                stack.append(entry_path)
                continue
            _validate_regular_file_metadata(
                metadata,
                label=str(entry_path),
                executable=False,
            )
    return root


__all__ = [
    "TrustedPathError",
    "load_protected_corpus_roots",
    "prepare_private_directory",
    "prepare_private_output_path",
    "prepare_trusted_directory",
    "reject_protected_corpus_overlap",
    "validate_trusted_directory",
    "validate_trusted_file",
    "validate_trusted_tree",
]
