import builtins
import importlib
import json
import os
import subprocess
import sys

import pytest


def _trusted_executable(tmp_path, name):
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return str(path)


def _completed(argv, *, stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        argv,
        returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _smi_row(
    *,
    index=0,
    uuid="GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
    bus="00000000:01:00.0",
    device_id="0x223010DE",
    name="NVIDIA RTX A6000",
    capability="8.6",
    total_mib="24576",
    free_mib="20480",
    driver="590.00",
):
    return ", ".join(
        str(value)
        for value in (
            index,
            uuid,
            bus,
            device_id,
            name,
            capability,
            total_mib,
            free_mib,
            driver,
        )
    )


def _torch_result(
    *,
    name="NVIDIA RTX A6000",
    uuid="GPU-AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE",
    capability=(8, 6),
    total_bytes=24 * 1024**3,
    free_bytes=20 * 1024**3,
    integrated=False,
):
    return {
        "ok": True,
        "python": "3.12.13",
        "torch": "2.11.0+cu130",
        "torch_cuda": "13.0",
        "visible_device_count": 1,
        "device": {
            "name": name,
            "uuid": uuid,
            "compute_capability": list(capability),
            "total_memory_bytes": total_bytes,
            "free_memory_bytes": free_bytes,
            "integrated": integrated,
        },
    }


def _runner_for(smi_output, torch_result=None, observed_environments=None):
    def run(argv, **kwargs):
        if observed_environments is not None:
            observed_environments.append(dict(kwargs["env"]))
        if any(argument.startswith("--query-gpu=") for argument in argv):
            return _completed(argv, stdout=smi_output + "\n")
        if "-c" in argv:
            result = torch_result
            if result is None:
                result = {"ok": False, "error_type": "ImportError"}
            return _completed(
                argv,
                stdout=json.dumps(result) + "\n",
                returncode=0 if result.get("ok") else 1,
            )
        raise AssertionError(f"unexpected command: {argv!r}")

    return run


def _write_meminfo(path, *, total_gib=64, available_gib=48):
    path.write_text(
        f"MemTotal:       {total_gib * 1024**2} kB\n"
        f"MemAvailable:   {available_gib * 1024**2} kB\n",
        encoding="utf-8",
    )


def _assert_receipt_hash(receipt):
    from sneeze.tpn.ocr_hardware import content_sha256

    payload = dict(receipt)
    digest = payload.pop("receipt_sha256")
    assert digest == content_sha256(payload)


def test_module_import_is_stdlib_only(monkeypatch):
    blocked = {"torch", "transformers", "vllm", "pynvml"}
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in blocked:
            raise AssertionError(f"heavy import during discovery: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    sys.modules.pop("sneeze.tpn.ocr_hardware", None)
    module = importlib.import_module("sneeze.tpn.ocr_hardware")
    assert module.SCHEMA_VERSION == 1


def test_dedicated_probe_keeps_observation_identity_but_portable_selector(
    tmp_path,
):
    from sneeze.tpn.ocr_hardware import probe_nvidia_hardware

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    environments = []
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
            "PATH": "/usr/bin:/bin",
            "OPENAI_API_KEY": "synthetic-do-not-copy",
        },
        meminfo_path=meminfo,
        hostname="test-host",
        runner=_runner_for(
            _smi_row(),
            _torch_result(),
            environments,
        ),
    )

    assert observation["hostname"] == "test-host"
    assert observation["device"]["uuid"].startswith("GPU-")
    assert observation["device"]["pci_bus_id"] == "00000000:01:00.0"
    assert observation["memory"]["model"] == "dedicated"
    assert observation["memory"]["total_memory_bytes"] == 24 * 1024**3
    selector = observation["selector"]
    assert set(selector) == {
        "schema_version",
        "vendor",
        "product_name",
        "pci_device_id",
        "compute_capability",
        "memory_model",
        "total_memory_bytes",
    }
    assert selector == {
        "schema_version": 1,
        "vendor": "nvidia",
        "product_name": "NVIDIA RTX A6000",
        "pci_device_id": "0x223010de",
        "compute_capability": [8, 6],
        "memory_model": "dedicated",
        "total_memory_bytes": 24 * 1024**3,
    }
    assert all("OPENAI_API_KEY" not in item for item in environments)


def test_torch_probe_is_isolated_and_receives_no_ambient_secret(tmp_path):
    from sneeze.tpn.ocr_hardware import probe_nvidia_hardware

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    calls = []
    base_runner = _runner_for(_smi_row(), _torch_result())

    def recording_runner(argv, **kwargs):
        calls.append((list(argv), dict(kwargs["env"])))
        return base_runner(argv, **kwargs)

    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={
            "CUDA_VISIBLE_DEVICES": "0",
            "PATH": "/usr/bin:/bin",
            "ANTHROPIC_API_KEY": "synthetic-do-not-copy",
        },
        meminfo_path=meminfo,
        runner=recording_runner,
    )

    torch_argv, torch_env = next(
        (argv, env) for argv, env in calls if "-c" in argv
    )
    assert torch_argv[1:3] == ["-I", "-B"]
    assert "-S" not in torch_argv
    assert "ANTHROPIC_API_KEY" not in torch_env
    assert observation["torch_probe"]["status"] == "available"


def test_default_probe_runner_streams_successfully(tmp_path):
    from sneeze.tpn.ocr_hardware import probe_nvidia_hardware

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text(
        f"#!{sys.executable}\nprint({_smi_row()!r})\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)

    observation = probe_nvidia_hardware(
        nvidia_smi=str(nvidia_smi),
        environment={"CUDA_VISIBLE_DEVICES": "0", "PATH": os.defpath},
        meminfo_path=meminfo,
    )

    assert observation["memory"]["model"] == "unknown"
    assert observation["memory"]["total_memory_bytes"] is None
    assert observation["nvidia_smi"]["returncode"] == 0
    assert observation["nvidia_smi"]["discovered_device_count"] == 1


def test_probe_rejects_group_writable_nvidia_smi(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        HardwareProbeError,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    nvidia_smi = tmp_path / "nvidia-smi"
    nvidia_smi.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    nvidia_smi.chmod(0o775)

    with pytest.raises(HardwareProbeError, match="writable by group"):
        probe_nvidia_hardware(
            nvidia_smi=str(nvidia_smi),
            environment={"CUDA_VISIBLE_DEVICES": "0"},
            meminfo_path=meminfo,
        )


def test_probe_rejects_group_writable_torch_interpreter(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        HardwareProbeError,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    torch_python = tmp_path / "python"
    torch_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    torch_python.chmod(0o775)

    with pytest.raises(HardwareProbeError, match="writable by group"):
        probe_nvidia_hardware(
            nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
            torch_python=str(torch_python),
            environment={"CUDA_VISIBLE_DEVICES": "0"},
            meminfo_path=meminfo,
            runner=_runner_for(_smi_row()),
        )


def test_nvml_capacity_without_torch_memory_topology_stays_unknown(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        calculate_memory_tiers,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        runner=_runner_for(_smi_row()),
    )
    budgets = calculate_memory_tiers(observation)

    assert observation["memory"]["model"] == "unknown"
    assert observation["memory"]["total_memory_bytes"] is None
    assert observation["memory"]["total_sources"] == {
        "nvidia_smi": 24 * 1024**3
    }
    assert all(tier["supported"] is False for tier in budgets["tiers"])


def test_visible_device_selection_fails_closed(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        HardwareProbeError,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    rows = "\n".join(
        (
            _smi_row(index=0, bus="00000000:65:00.0"),
            _smi_row(
                index=1,
                uuid="GPU-BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF",
                bus="00000000:17:00.0",
            ),
        )
    )
    runner = _runner_for(rows)

    with pytest.raises(HardwareProbeError, match="multiple NVIDIA GPUs"):
        probe_nvidia_hardware(
            nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
            environment={},
            meminfo_path=meminfo,
            runner=runner,
        )
    with pytest.raises(HardwareProbeError, match="numeric.*ambiguous"):
        probe_nvidia_hardware(
            nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
            environment={"CUDA_VISIBLE_DEVICES": "0"},
            meminfo_path=meminfo,
            runner=runner,
        )
    with pytest.raises(HardwareProbeError, match="did not resolve uniquely"):
        probe_nvidia_hardware(
            nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
            environment={"CUDA_VISIBLE_DEVICES": "GPU-NOT-FOUND"},
            meminfo_path=meminfo,
            runner=runner,
        )
    with pytest.raises(HardwareProbeError, match="exactly one"):
        probe_nvidia_hardware(
            nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
            environment={"CUDA_VISIBLE_DEVICES": "0,1"},
            meminfo_path=meminfo,
            runner=runner,
        )


def test_pci_order_and_unique_uuid_resolve_exact_device(tmp_path):
    from sneeze.tpn.ocr_hardware import probe_nvidia_hardware

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    second_uuid = "GPU-BBBBBBBB-CCCC-DDDD-EEEE-FFFFFFFFFFFF"
    rows = "\n".join(
        (
            _smi_row(index=0, bus="00000000:65:00.0"),
            _smi_row(index=1, uuid=second_uuid, bus="00000000:17:00.0"),
        )
    )
    runner = _runner_for(rows)
    by_ordinal = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        environment={
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": "0",
        },
        meminfo_path=meminfo,
        runner=runner,
    )
    assert by_ordinal["device"]["uuid"] == second_uuid
    assert by_ordinal["selection"]["method"] == "cuda_pci_bus_ordinal"

    by_uuid = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        environment={"CUDA_VISIBLE_DEVICES": "GPU-BBBBBBBB"},
        meminfo_path=meminfo,
        runner=runner,
    )
    assert by_uuid["device"]["uuid"] == second_uuid
    assert by_uuid["selection"]["method"] == "cuda_gpu_uuid"


def test_unified_gpu_ignores_small_torch_ats_window_for_capacity(tmp_path):
    from sneeze.tpn.ocr_hardware import probe_nvidia_hardware

    gib = 1024**3
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_gib=128, available_gib=90)
    torch_result = _torch_result(
        name="NVIDIA GB10",
        capability=(12, 1),
        total_bytes=512 * 1024**2,
        free_bytes=800 * 1024**2,
        integrated=True,
    )
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        runner=_runner_for(
            _smi_row(
                name="NVIDIA GB10",
                capability="12.1",
                total_mib="N/A",
                free_mib="N/A",
            ),
            torch_result,
        ),
    )

    assert observation["memory"]["model"] == "unified"
    assert observation["memory"]["nvml_memory_available"] is False
    assert observation["memory"]["total_memory_bytes"] == 128 * gib
    assert observation["memory"]["free_memory_bytes"] == 90 * gib
    assert observation["memory"]["total_sources"] == {
        "system": 128 * gib,
        "torch_cuda": 512 * 1024**2,
    }
    assert observation["memory"]["free_sources"]["torch_cuda"] == (
        800 * 1024**2
    )
    assert observation["selector"]["memory_model"] == "unified"


def test_nvml_na_uses_torch_memory_for_explicit_discrete_gpu(tmp_path):
    from sneeze.tpn.ocr_hardware import probe_nvidia_hardware

    gib = 1024**3
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_gib=128, available_gib=90)
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        runner=_runner_for(
            _smi_row(total_mib="N/A", free_mib="N/A"),
            _torch_result(
                total_bytes=24 * gib,
                free_bytes=20 * gib,
                integrated=False,
            ),
        ),
    )

    assert observation["memory"]["model"] == "dedicated"
    assert observation["memory"]["total_memory_bytes"] == 24 * gib
    assert observation["memory"]["free_memory_bytes"] == 20 * gib
    assert observation["memory"]["total_sources"] == {"torch_cuda": 24 * gib}
    assert observation["memory"]["free_sources"] == {"torch_cuda": 20 * gib}
    assert observation["selector"]["memory_model"] == "dedicated"


def test_nvml_na_without_explicit_torch_integration_stays_unknown(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        calculate_memory_tiers,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_gib=128, available_gib=90)
    torch_result = _torch_result()
    del torch_result["device"]["integrated"]
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        runner=_runner_for(
            _smi_row(total_mib="N/A", free_mib="N/A"),
            torch_result,
        ),
    )
    budgets = calculate_memory_tiers(observation)

    assert observation["memory"]["model"] == "unknown"
    assert observation["memory"]["total_memory_bytes"] is None
    assert observation["memory"]["free_memory_bytes"] is None
    assert observation["memory"]["total_sources"] == {
        "torch_cuda": 24 * 1024**3
    }
    assert "system" not in observation["memory"]["total_sources"]
    assert all(tier["supported"] is False for tier in budgets["tiers"])


def test_unified_gpu_capacity_is_capped_by_current_cgroup_v2(tmp_path):
    from sneeze.tpn.ocr_hardware import probe_nvidia_hardware

    gib = 1024**3
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_gib=128, available_gib=90)
    cgroup_root = tmp_path / "cgroup"
    current_group = cgroup_root / "users" / "job"
    current_group.mkdir(parents=True)
    (current_group / "memory.max").write_text(str(80 * gib), encoding="utf-8")
    (current_group / "memory.current").write_text(
        str(20 * gib), encoding="utf-8"
    )
    (current_group / "memory.peak").write_text(
        str(30 * gib), encoding="utf-8"
    )
    membership = tmp_path / "self-cgroup"
    membership.write_text("0::/users/job\n", encoding="utf-8")

    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        cgroup_root=cgroup_root,
        cgroup_membership_path=membership,
        runner=_runner_for(
            _smi_row(
                name="NVIDIA GB10",
                capability="12.1",
                total_mib="N/A",
                free_mib="N/A",
            ),
            _torch_result(
                name="NVIDIA GB10",
                capability=(12, 1),
                total_bytes=128 * gib,
                free_bytes=800 * 1024**2,
                integrated=True,
            ),
        ),
    )

    assert observation["memory"]["total_memory_bytes"] == 80 * gib
    assert observation["memory"]["free_memory_bytes"] == 60 * gib
    assert observation["memory"]["total_sources"]["cgroup_v2"] == 80 * gib
    assert observation["memory"]["free_sources"]["cgroup_v2"] == 60 * gib
    assert observation["memory"]["cgroup_v2"] == {
        "status": "available",
        "unlimited": False,
        "limit_bytes": 80 * gib,
        "current_bytes": 20 * gib,
        "peak_bytes": 30 * gib,
        "available_bytes": 60 * gib,
    }
    assert observation["selector"]["total_memory_bytes"] == 80 * gib


def test_unified_gpu_capacity_uses_tightest_cgroup_v2_ancestor(tmp_path):
    from sneeze.tpn.ocr_hardware import probe_nvidia_hardware

    gib = 1024**3
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_gib=128, available_gib=90)
    cgroup_root = tmp_path / "cgroup"
    parent_group = cgroup_root / "users"
    current_group = parent_group / "job"
    current_group.mkdir(parents=True)
    (current_group / "memory.max").write_text("max", encoding="utf-8")
    (current_group / "memory.current").write_text(
        str(5 * gib), encoding="utf-8"
    )
    (current_group / "memory.peak").write_text(
        str(6 * gib), encoding="utf-8"
    )
    (parent_group / "memory.max").write_text(
        str(80 * gib), encoding="utf-8"
    )
    (parent_group / "memory.current").write_text(
        str(30 * gib), encoding="utf-8"
    )
    (cgroup_root / "memory.max").write_text(
        str(96 * gib), encoding="utf-8"
    )
    (cgroup_root / "memory.current").write_text(
        str(90 * gib), encoding="utf-8"
    )
    membership = tmp_path / "self-cgroup"
    membership.write_text("0::/users/job\n", encoding="utf-8")

    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        cgroup_root=cgroup_root,
        cgroup_membership_path=membership,
        runner=_runner_for(
            _smi_row(
                name="NVIDIA GB10",
                capability="12.1",
                total_mib="N/A",
                free_mib="N/A",
            ),
            _torch_result(
                name="NVIDIA GB10",
                capability=(12, 1),
                total_bytes=128 * gib,
                free_bytes=800 * 1024**2,
                integrated=True,
            ),
        ),
    )

    assert observation["memory"]["total_memory_bytes"] == 80 * gib
    assert observation["memory"]["free_memory_bytes"] == 6 * gib
    assert observation["memory"]["cgroup_v2"] == {
        "status": "available",
        "unlimited": False,
        "limit_bytes": 80 * gib,
        "current_bytes": 5 * gib,
        "peak_bytes": 6 * gib,
        "available_bytes": 6 * gib,
    }


def test_cgroup_v2_diagnostics_stay_bound_to_process_leaf(tmp_path):
    from sneeze.tpn.ocr_hardware import _read_cgroup_v2_memory

    root = tmp_path / "cgroup"
    parent = root / "users"
    leaf = parent / "job"
    leaf.mkdir(parents=True)
    (leaf / "memory.current").write_text("10\n", encoding="utf-8")
    (leaf / "memory.peak").write_text("20\n", encoding="utf-8")
    (parent / "memory.max").write_text("100\n", encoding="utf-8")
    (parent / "memory.current").write_text("30\n", encoding="utf-8")
    (root / "memory.max").write_text("200\n", encoding="utf-8")
    (root / "memory.current").write_text("150\n", encoding="utf-8")
    membership = tmp_path / "self-cgroup"
    membership.write_text("0::/users/job\n", encoding="utf-8")

    assert _read_cgroup_v2_memory(root, membership) == {
        "status": "available",
        "unlimited": False,
        "limit_bytes": 100,
        "current_bytes": 10,
        "peak_bytes": 20,
        "available_bytes": 50,
    }


@pytest.mark.parametrize(
    "membership_contents",
    [None, "1:memory:/workload\n"],
)
def test_cgroup_v2_memory_requires_known_process_membership(
    tmp_path,
    membership_contents,
):
    from sneeze.tpn.ocr_hardware import _read_cgroup_v2_memory

    cgroup_root = tmp_path / "cgroup"
    cgroup_root.mkdir()
    (cgroup_root / "memory.max").write_text("1024\n", encoding="utf-8")
    (cgroup_root / "memory.current").write_text("512\n", encoding="utf-8")
    membership = tmp_path / "self-cgroup"
    if membership_contents is not None:
        membership.write_text(membership_contents, encoding="utf-8")

    assert _read_cgroup_v2_memory(cgroup_root, membership) == {
        "status": "unavailable",
        "unlimited": None,
        "limit_bytes": None,
        "current_bytes": None,
        "peak_bytes": None,
        "available_bytes": None,
    }


def test_unified_gpu_headroom_is_unknown_when_finite_cgroup_usage_is_missing(
    tmp_path,
):
    from sneeze.tpn.ocr_hardware import (
        calculate_memory_tiers,
        probe_nvidia_hardware,
    )

    gib = 1024**3
    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_gib=128, available_gib=90)
    cgroup_root = tmp_path / "cgroup"
    parent_group = cgroup_root / "users"
    current_group = parent_group / "job"
    current_group.mkdir(parents=True)
    (current_group / "memory.max").write_text("max\n", encoding="utf-8")
    (current_group / "memory.current").write_text(
        str(5 * gib), encoding="utf-8"
    )
    (parent_group / "memory.max").write_text(
        str(80 * gib), encoding="utf-8"
    )
    membership = tmp_path / "self-cgroup"
    membership.write_text("0::/users/job\n", encoding="utf-8")

    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        cgroup_root=cgroup_root,
        cgroup_membership_path=membership,
        runner=_runner_for(
            _smi_row(
                name="NVIDIA GB10",
                capability="12.1",
                total_mib="N/A",
                free_mib="N/A",
            ),
            _torch_result(
                name="NVIDIA GB10",
                capability=(12, 1),
                total_bytes=128 * gib,
                free_bytes=800 * 1024**2,
                integrated=True,
            ),
        ),
    )
    budgets = calculate_memory_tiers(observation)

    assert observation["memory"]["total_memory_bytes"] == 80 * gib
    assert observation["memory"]["free_memory_bytes"] is None
    assert observation["memory"]["cgroup_v2"]["available_bytes"] is None
    assert all(
        "live_free_memory_unknown" in tier["reason_codes"]
        for tier in budgets["tiers"]
    )
    assert all(
        tier["full_budget_available"] is False for tier in budgets["tiers"]
    )


def test_nvml_na_without_torch_has_explicit_unsupported_budgets(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        calculate_memory_tiers,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_gib=128, available_gib=90)
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        runner=_runner_for(
            _smi_row(
                name="NVIDIA GB10",
                capability="12.1",
                total_mib="N/A",
                free_mib="N/A",
            )
        ),
    )
    budgets = calculate_memory_tiers(observation)

    assert observation["memory"]["model"] == "unknown"
    assert all(tier["supported"] is False for tier in budgets["tiers"])
    assert all(
        "total_memory_unknown" in tier["reason_codes"]
        for tier in budgets["tiers"]
    )
    assert all(
        "live_free_memory_unknown" in tier["reason_codes"]
        for tier in budgets["tiers"]
    )
    _assert_receipt_hash(budgets)


def test_dedicated_tiers_report_support_and_live_availability(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        GIB,
        MemoryBudgetPolicy,
        calculate_memory_tiers,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        runner=_runner_for(
            _smi_row(free_mib="10240"),
            _torch_result(),
        ),
    )
    budgets = calculate_memory_tiers(
        observation,
        minimum_required_bytes=7 * GIB,
        policy=MemoryBudgetPolicy(dedicated_reserve_bytes=2 * GIB),
    )
    by_id = {tier["tier"]: tier for tier in budgets["tiers"]}

    assert by_id["25"]["capacity_budget_bytes"] == 6 * GIB
    assert by_id["25"]["supported"] is False
    assert "budget_below_minimum_required" in by_id["25"]["reason_codes"]
    assert by_id["50"]["capacity_budget_bytes"] == 12 * GIB
    assert by_id["50"]["supported"] is True
    assert by_id["50"]["minimum_fit_available"] is True
    assert by_id["50"]["full_budget_available"] is False
    assert by_id["max"]["capacity_budget_bytes"] == int(24 * GIB * 0.9)
    assert by_id["max"]["full_budget_available"] is False
    _assert_receipt_hash(budgets)


def test_unified_memory_tiers_reserve_host_capacity(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        GIB,
        MemoryBudgetPolicy,
        calculate_memory_tiers,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo, total_gib=128, available_gib=90)
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        runner=_runner_for(
            _smi_row(
                name="NVIDIA GB10",
                capability="12.1",
                total_mib="N/A",
                free_mib="N/A",
            ),
            _torch_result(
                name="NVIDIA GB10",
                capability=(12, 1),
                total_bytes=128 * GIB,
                free_bytes=100 * GIB,
                integrated=True,
            ),
        ),
    )
    budgets = calculate_memory_tiers(
        observation,
        policy=MemoryBudgetPolicy(unified_reserve_bytes=16 * GIB),
    )
    by_id = {tier["tier"]: tier for tier in budgets["tiers"]}

    assert by_id["25"]["capacity_budget_bytes"] == 32 * GIB
    assert by_id["25"]["full_budget_available"] is True
    assert by_id["50"]["capacity_budget_bytes"] == 64 * GIB
    assert by_id["50"]["full_budget_available"] is True
    assert by_id["max"]["capacity_budget_bytes"] == 96 * GIB
    assert by_id["max"]["live_memory_after_reserve_bytes"] == 74 * GIB
    assert by_id["max"]["full_budget_available"] is False


def test_reserve_can_make_a_tier_explicitly_unsupported(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        GIB,
        MemoryBudgetPolicy,
        calculate_memory_tiers,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        torch_python=_trusted_executable(tmp_path, "torch-python"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        runner=_runner_for(_smi_row(), _torch_result()),
    )
    budgets = calculate_memory_tiers(
        observation,
        policy=MemoryBudgetPolicy(dedicated_reserve_bytes=24 * GIB),
    )

    assert all(tier["supported"] is False for tier in budgets["tiers"])
    assert all(
        "budget_zero_after_reserve" in tier["reason_codes"]
        for tier in budgets["tiers"]
    )


def test_receipts_are_deterministic_hashed_and_reject_secrets(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        HardwareProbeError,
        calculate_memory_tiers,
        make_hardware_receipt,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    observation = probe_nvidia_hardware(
        nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
        environment={"CUDA_VISIBLE_DEVICES": "0"},
        meminfo_path=meminfo,
        hostname="deterministic-host",
        runner=_runner_for(_smi_row()),
    )
    budgets = calculate_memory_tiers(observation)
    first = make_hardware_receipt(observation, memory_budgets=budgets)
    second = make_hardware_receipt(observation, memory_budgets=budgets)

    assert first == second
    _assert_receipt_hash(first)
    compromised = dict(observation)
    compromised["api_key"] = "sk-synthetic-not-a-real-credential"
    with pytest.raises(HardwareProbeError, match="secret-bearing"):
        make_hardware_receipt(compromised)
    for value in ("hf_" + "x" * 20, "ghp_" + "x" * 20):
        compromised = dict(observation)
        compromised["safe_label"] = value
        with pytest.raises(HardwareProbeError, match="credential-like"):
            make_hardware_receipt(compromised)


def test_torch_identity_mismatch_is_rejected(tmp_path):
    from sneeze.tpn.ocr_hardware import (
        HardwareProbeError,
        probe_nvidia_hardware,
    )

    meminfo = tmp_path / "meminfo"
    _write_meminfo(meminfo)
    with pytest.raises(HardwareProbeError, match="compute capabilities"):
        probe_nvidia_hardware(
            nvidia_smi=_trusted_executable(tmp_path, "nvidia-smi"),
            torch_python=_trusted_executable(tmp_path, "torch-python"),
            environment={"CUDA_VISIBLE_DEVICES": "0"},
            meminfo_path=meminfo,
            runner=_runner_for(
                _smi_row(capability="8.6"),
                _torch_result(capability=(12, 0)),
            ),
        )


@pytest.mark.parametrize("stream_name", ("stdout", "stderr"))
def test_run_bounded_stops_producer_at_per_stream_limit(
    tmp_path,
    stream_name,
):
    from sneeze.tpn.ocr_hardware import HardwareProbeError, _run_bounded

    completed_marker = tmp_path / "producer-completed"
    producer = "\n".join(
        (
            "import pathlib",
            "import sys",
            f"stream = sys.{stream_name}.buffer",
            "for _ in range(4096):",
            "    stream.write(b'x' * 1024)",
            "    stream.flush()",
            "pathlib.Path(sys.argv[1]).write_text('completed')",
        )
    )

    with pytest.raises(
        HardwareProbeError,
        match=rf"hardware probe {stream_name} exceeded 64 KiB",
    ):
        _run_bounded(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                producer,
                str(completed_marker),
            ],
            environment={"PATH": os.defpath},
            timeout_seconds=10,
            runner=subprocess.run,
        )

    assert not completed_marker.exists()


def test_run_bounded_timeout_reaps_process(tmp_path):
    from sneeze.tpn.ocr_hardware import HardwareProbeError, _run_bounded

    pid_path = tmp_path / "producer.pid"
    producer = "\n".join(
        (
            "import os",
            "import pathlib",
            "import sys",
            "import time",
            "pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))",
            "time.sleep(30)",
        )
    )

    with pytest.raises(HardwareProbeError, match="TimeoutExpired"):
        _run_bounded(
            [sys.executable, "-I", "-B", "-c", producer, str(pid_path)],
            environment={"PATH": os.defpath},
            timeout_seconds=0.5,
            runner=subprocess.run,
        )

    producer_pid = int(pid_path.read_text(encoding="utf-8"))
    with pytest.raises(ProcessLookupError):
        os.kill(producer_pid, 0)
