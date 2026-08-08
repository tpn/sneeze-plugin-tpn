from __future__ import annotations

import hashlib
import json
from io import StringIO

import pytest


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_registry_and_profile_commands_are_discoverable_and_long_only():
    from sneeze import cli as sneeze_cli

    cli = sneeze_cli.CLI(
        program_name="sne",
        module_names=["sneeze", "sneeze.tpn"],
        introspect=True,
        auto_plugins=False,
    )
    expected = {
        "ocr-installation-register",
        "ocr-installation-list",
        "ocr-hardware-probe",
        "ocr-pipeline-list",
        "ocr-profile-list",
        "ocr-profile-qualify",
        "ocr-profile-show",
        "ocr-profile-tune",
    }
    assert expected <= set(cli._commands_by_name)
    assert cli._commands_by_name["ocr-pipeline-list"].shortname == "opl"
    assert cli._commands_by_name["ocr-profile-list"].shortname == "oprl"
    assert cli._commands_by_shortname["opl"].name == "ocr-pipeline-list"
    assert cli._commands_by_shortname["oprl"].name == "ocr-profile-list"

    register = cli._commands_by_name["ocr-installation-register"].command
    for name in (
        "backend",
        "python_executable",
        "cache_dir",
        "installation_registry",
        "cuda_visible_devices",
        "device",
        "attention_backend",
    ):
        assert register._invariants[name]._opt_short is None


def test_installation_register_forwards_exact_binding_and_emits_receipt(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_installations as installations
    from sneeze.tpn.commands import OcrInstallationRegister

    registry_path = (tmp_path / "installations.json").resolve()
    python_path = (tmp_path / "python").resolve()
    cache_path = (tmp_path / "cache").resolve()
    cache_path.mkdir()
    captured = {}
    record = {
        "backend_id": "deepseek-vllm",
        "record_sha256": _digest("record"),
    }

    def fake_register(backend_id, **kwargs):
        captured["backend_id"] = backend_id
        captured.update(kwargs)
        return {
            "content_sha256": _digest("registry"),
            "installations": {backend_id: record},
        }

    monkeypatch.setattr(
        installations,
        "register_backend_installation",
        fake_register,
    )
    output = StringIO()
    command = OcrInstallationRegister(None, output, None)
    command.backend = "deepseek"
    command.python_executable = str(python_path)
    command.cache_dir = str(cache_path)
    command.installation_registry = str(registry_path)
    command.cuda_visible_devices = "GPU-ONE"
    command.device = "cuda:0"
    command.attention_backend = "FLASH_ATTN"

    command.run()

    assert captured == {
        "backend_id": "deepseek-vllm",
        "python_executable": str(python_path),
        "hf_cache": str(cache_path),
        "cuda_visible_devices": "GPU-ONE",
        "device": "cuda:0",
        "attention_backend": "FLASH_ATTN",
        "path": str(registry_path),
    }
    receipt = json.loads(output.getvalue())
    assert receipt["kind"] == "ocr_backend_installation_registration"
    assert receipt["registry_path"] == str(registry_path)
    assert receipt["installation"] == record


def test_installation_list_revalidates_current_artifacts(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_installations as installations
    from sneeze.tpn.commands import OcrInstallationList

    registry_path = (tmp_path / "installations.json").resolve()
    captured = {}

    def fake_load(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return {
            "content_sha256": _digest("registry"),
            "installations": {},
        }

    monkeypatch.setattr(
        installations,
        "load_installation_registry",
        fake_load,
    )
    output = StringIO()
    command = OcrInstallationList(None, output, None)
    command.installation_registry = str(registry_path)
    command.format = "json"

    command.run()

    assert captured == {
        "path": str(registry_path),
        "validate_current": True,
        "missing_ok": True,
    }
    receipt = json.loads(output.getvalue())
    assert receipt["registry_path"] == str(registry_path)
    assert receipt["installations"] == []


def test_hardware_probe_rejects_protected_output_before_probe(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    import sneeze.tpn.ocr_hardware as hardware
    from sneeze.tpn.commands import OcrHardwareProbe

    protected = tmp_path / "canonical" / "papers"
    protected.mkdir(parents=True)
    config_home = tmp_path / "config"
    policy = config_home / "sneeze" / "tpn" / "ocr-protected-roots"
    policy.parent.mkdir(parents=True)
    policy.write_text(f"{protected}\n", encoding="utf-8")
    policy.chmod(0o600)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    def unexpected_probe(**_kwargs):
        raise AssertionError("hardware probe must not start")

    monkeypatch.setattr(hardware, "probe_nvidia_hardware", unexpected_probe)
    output_path = protected / "hardware" / "receipt.json"
    command = OcrHardwareProbe(None, StringIO(), None)
    command.output = str(output_path)

    with pytest.raises(CommandError, match="protected corpora"):
        command.run()

    assert not output_path.parent.exists()


def test_hardware_probe_rejects_symlinked_output_parent_before_probe(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    import sneeze.tpn.ocr_hardware as hardware
    from sneeze.tpn.commands import OcrHardwareProbe

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias = tmp_path / "parent-alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    def unexpected_probe(**_kwargs):
        raise AssertionError("hardware probe must not start")

    monkeypatch.setattr(hardware, "probe_nvidia_hardware", unexpected_probe)
    command = OcrHardwareProbe(None, StringIO(), None)
    command.output = str(alias / "receipt.json")

    with pytest.raises(CommandError, match="must not traverse symlinks"):
        command.run()

    assert list(real_parent.iterdir()) == []


def test_pipeline_list_marks_only_local_mineru_structured():
    from sneeze.tpn.commands import OcrPipelineList

    output = StringIO()
    command = OcrPipelineList(None, output, None)
    command.format = "json"
    command.run()

    pipelines = {
        value["pipeline_id"]: value
        for value in json.loads(output.getvalue())["pipelines"]
    }
    assert pipelines["local-mineru25"]["structured_execution"][
        "enabled"
    ]
    assert not pipelines["deepseek-vllm-full"]["structured_execution"][
        "enabled"
    ]


def test_profile_list_and_show_use_validated_profile_apis(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_execution as execution
    import sneeze.tpn.ocr_profiles as profiles
    from sneeze.tpn.commands import OcrProfileList, OcrProfileShow

    profile_dir = tmp_path.resolve()
    provisional = {
        "profile_id": "provisional-profile",
        "pipeline_id": "local-mineru25",
        "qualification": {"state": "provisional"},
        "memory_tiers": {
            "25": {"supported": False},
            "50": {"supported": True},
            "max": {"supported": True},
        },
        "profile_sha256": _digest("provisional"),
    }
    trusted = {
        "profile_id": "trusted-profile",
        "pipeline_id": "local-mineru25",
        "qualification": {"state": "qualified"},
        "memory_tiers": {
            "25": {"supported": True},
            "50": {"supported": True},
            "max": {"supported": True},
        },
        "profile_sha256": _digest("trusted"),
    }
    untrusted = {
        "profile_id": "untrusted-profile",
        "pipeline_id": "local-mineru25",
        "qualification": {"state": "qualified"},
        "memory_tiers": {
            "25": {"supported": True},
            "50": {"supported": True},
            "max": {"supported": True},
        },
        "profile_sha256": _digest("untrusted"),
    }
    captured = []

    def fake_list(*, paths):
        captured.append(("list", tuple(paths)))
        return (provisional, untrusted, trusted)

    def fake_show(profile_id, *, paths):
        captured.append(("show", profile_id, tuple(paths)))
        return untrusted

    monkeypatch.setattr(
        execution,
        "validate_bundled_profile_registry",
        lambda: (),
    )
    monkeypatch.setattr(
        profiles,
        "TRUSTED_QUALIFIED_PROFILE_SHA256S",
        frozenset({trusted["profile_sha256"]}),
    )
    monkeypatch.setattr(profiles, "list_execution_profiles", fake_list)
    monkeypatch.setattr(profiles, "show_execution_profile", fake_show)

    list_output = StringIO()
    list_command = OcrProfileList(None, list_output, None)
    list_command.profile_dir = str(profile_dir)
    list_command.format = "text"
    list_command.run()
    text = list_output.getvalue()
    assert (
        "provisional-profile: pipeline=local-mineru25; "
        "qualification=provisional; tiers=50,max; "
        "trusted_by_current_plugin=no; production_eligible=no;"
    ) in text
    assert (
        "untrusted-profile: pipeline=local-mineru25; "
        "qualification=qualified; tiers=25,50,max; "
        "trusted_by_current_plugin=no; production_eligible=no;"
    ) in text
    assert (
        "trusted-profile: pipeline=local-mineru25; "
        "qualification=qualified; tiers=25,50,max; "
        "trusted_by_current_plugin=yes; production_eligible=yes;"
    ) in text

    json_output = StringIO()
    json_command = OcrProfileList(None, json_output, None)
    json_command.profile_dir = str(profile_dir)
    json_command.format = "json"
    json_command.run()
    registry = json.loads(json_output.getvalue())
    assert registry["profiles"] == [provisional, untrusted, trusted]
    assert registry["profile_statuses"] == [
        {
            "profile_id": "provisional-profile",
            "profile_sha256": provisional["profile_sha256"],
            "qualification_state": "provisional",
            "trusted_by_current_plugin": False,
            "production_eligible": False,
            "production_ineligibility_reason": (
                "profile qualification is provisional"
            ),
        },
        {
            "profile_id": "untrusted-profile",
            "profile_sha256": untrusted["profile_sha256"],
            "qualification_state": "qualified",
            "trusted_by_current_plugin": False,
            "production_eligible": False,
            "production_ineligibility_reason": (
                "profile hash is not trusted by this plugin build"
            ),
        },
        {
            "profile_id": "trusted-profile",
            "profile_sha256": trusted["profile_sha256"],
            "qualification_state": "qualified",
            "trusted_by_current_plugin": True,
            "production_eligible": True,
            "production_ineligibility_reason": None,
        },
    ]

    show_output = StringIO()
    show_command = OcrProfileShow(None, show_output, None)
    show_command.profile_id = "untrusted-profile"
    show_command.profile_dir = str(profile_dir)
    show_command.run()
    detail = json.loads(show_output.getvalue())
    assert detail["kind"] == "ocr_execution_profile_detail"
    assert detail["profile"] == untrusted
    assert detail["profile_status"] == registry["profile_statuses"][1]
    assert "trusted_by_current_plugin" not in untrusted
    assert "production_eligible" not in untrusted
    assert captured == [
        ("list", (profile_dir,)),
        ("list", (profile_dir,)),
        ("show", "untrusted-profile", (profile_dir,)),
    ]


def test_profile_help_describes_bundled_then_xdg_default(capsys):
    import pytest

    from sneeze import cli as sneeze_cli

    cli = sneeze_cli.CLI(
        program_name="sne",
        module_names=["sneeze", "sneeze.tpn"],
        introspect=True,
        auto_plugins=False,
    )

    for name in ("ocr-profile-list", "ocr-profile-show"):
        with pytest.raises(SystemExit) as exc_info:
            cli._commands_by_name[name].run(["--help"])
        assert exc_info.value.code == 0

    help_text = " ".join(capsys.readouterr().out.split())
    assert help_text.count(
        "defaults to bundled profiles first, then the XDG directory"
    ) == 2


def test_profile_show_default_skips_missing_bundled_directory(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_execution as execution
    import sneeze.tpn.ocr_profiles as profiles
    from sneeze.tpn.commands import OcrProfileShow

    bundled = tmp_path / "missing-bundled"
    machine = tmp_path / "machine-profiles"
    machine.mkdir()
    captured = {}
    profile = {
        "profile_id": "machine-profile",
        "profile_sha256": _digest("machine-profile"),
        "qualification": {"state": "provisional"},
    }

    monkeypatch.setattr(
        execution,
        "bundled_profile_directory",
        lambda: bundled,
    )
    monkeypatch.setattr(
        execution,
        "default_profile_directory",
        lambda: machine,
    )
    monkeypatch.setattr(
        profiles,
        "TRUSTED_QUALIFIED_PROFILE_SHA256S",
        frozenset(),
    )

    def fake_show(profile_id, *, paths):
        captured["profile_id"] = profile_id
        captured["paths"] = tuple(paths)
        return profile

    monkeypatch.setattr(profiles, "show_execution_profile", fake_show)
    output = StringIO()
    command = OcrProfileShow(None, output, None)
    command.profile_id = "machine-profile"

    command.run()

    assert captured == {
        "profile_id": "machine-profile",
        "paths": (machine,),
    }
    detail = json.loads(output.getvalue())
    assert detail["profile"] == profile
    assert detail["profile_status"]["trusted_by_current_plugin"] is False
    assert detail["profile_status"]["production_eligible"] is False


def test_profile_qualify_requires_review_and_emits_untrusted_candidate(
    tmp_path,
    monkeypatch,
):
    import pytest
    from sneeze.command import CommandError

    import sneeze.tpn.ocr_profiles as profiles
    from sneeze.tpn.commands import OcrProfileQualify

    source = (tmp_path / "provisional.json").resolve()
    output_path = (tmp_path / "qualified.json").resolve()
    source.write_text("{}\n", encoding="utf-8")
    provisional = {"profile_sha256": _digest("provisional")}
    qualified = {"profile_sha256": _digest("qualified")}
    captured = {}

    monkeypatch.setattr(
        profiles,
        "load_execution_profile",
        lambda path: provisional,
    )
    monkeypatch.setattr(
        profiles,
        "qualify_execution_profile",
        lambda profile: qualified,
    )

    def fake_write(profile, path):
        captured["profile"] = profile
        captured["path"] = path
        output_path.write_text("{}\n", encoding="utf-8")
        return qualified

    monkeypatch.setattr(profiles, "write_execution_profile", fake_write)

    rejected = OcrProfileQualify(None, StringIO(), None)
    rejected.profile = str(source)
    rejected.output = str(output_path)
    rejected.reviewed_evidence = False
    with pytest.raises(CommandError, match="reviewed-evidence"):
        rejected.run()

    command_output = StringIO()
    command = OcrProfileQualify(None, command_output, None)
    command.profile = str(source)
    command.output = str(output_path)
    command.reviewed_evidence = True
    command.run()

    receipt = json.loads(command_output.getvalue())
    assert captured == {"profile": qualified, "path": output_path}
    assert receipt["qualified_profile_sha256"] == _digest("qualified")
    assert receipt["trusted_by_current_plugin"] is False
    assert "add its exact hash" in receipt["next_gate"]
