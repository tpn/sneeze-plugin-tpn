# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
from io import StringIO

import pytest


def test_setup_dry_run_is_reproducible_and_does_not_write(tmp_path):
    from sneeze.tpn.upscale.image_upscale import (
        MODEL_SHA256,
        MODEL_URL,
        PYTORCH_INDEX_URL,
        setup_upscaler,
    )

    runtime = tmp_path / "runtime"
    models = tmp_path / "models"
    receipt = setup_upscaler(
        runtime_dir=runtime,
        model_cache=models,
        uv_executable="uv-test",
        dry_run=True,
    )

    assert receipt["dry_run"] is True
    assert receipt["model"]["sha256"] == MODEL_SHA256
    assert receipt["model"]["download_mirror"]["url"] == MODEL_URL
    assert receipt["model"]["upstream"]["repository"] == "XPixelGroup/HAT"
    assert receipt["runtime"]["pytorch_index_url"] == PYTORCH_INDEX_URL
    assert receipt["commands"][0] == [
        "uv-test",
        "venv",
        "--python",
        "3.12",
        str(runtime),
    ]
    assert "torch==2.11.0" in receipt["commands"][1]
    assert "spandrel==0.4.2" in receipt["commands"][2]
    assert not runtime.exists()
    assert not models.exists()


def test_runtime_probe_requires_pinned_cuda_bf16_tuple():
    from sneeze.tpn.upscale.image_upscale import (
        ImageUpscaleError,
        _validate_runtime_probe,
    )

    qualified = {
        "cuda_available": True,
        "gpu": {"bfloat16_supported": True},
        "python": "3.12.8",
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


def test_default_output_is_non_destructive_sibling(tmp_path):
    from sneeze.tpn.upscale.image_upscale import default_output_path

    source = tmp_path / "slide.png"
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


def test_run_upscale_validates_model_hash_before_child(tmp_path):
    from sneeze.tpn.upscale.image_upscale import (
        ImageUpscaleError,
        run_upscale,
    )

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    runtime = tmp_path / "runtime"
    python = runtime / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    (models / "HAT-L_SRx2_ImageNet-pretrain.pth").write_bytes(b"wrong")

    with pytest.raises(ImageUpscaleError, match="Model SHA-256 mismatch"):
        run_upscale(
            source=source,
            runtime_dir=runtime,
            model_cache=models,
        )


def test_run_upscale_launches_isolated_runtime_and_writes_receipt(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.upscale import image_upscale

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    output = tmp_path / "source-4k.png"
    runtime = tmp_path / "runtime"
    python = runtime / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    models = tmp_path / "models"
    models.mkdir()
    model = models / image_upscale.MODEL_FILENAME
    model.write_bytes(b"model")
    receipt_path = tmp_path / "receipt.json"
    calls = []
    for name in ("PYTHONPATH", "PYTHONHOME", "__PYVENV_LAUNCHER__"):
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
            "source": {
                "path": str(source),
                "sha256": image_upscale.file_sha256(source),
            },
            "output": {
                "path": str(output),
                "dimensions": [3840, 2160],
                "sha256": image_upscale.file_sha256(output),
            },
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
    assert argv[:2] == [str(python), str(image_upscale.runtime_script())]
    assert argv[argv.index("--target-width") + 1] == "3840"
    assert argv[argv.index("--dtype") + 1] == "bfloat16"
    assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    for name in ("PYTHONPATH", "PYTHONHOME", "__PYVENV_LAUNCHER__"):
        assert name not in kwargs["env"]
    assert receipt["receipt_path"] == str(receipt_path)
    persisted = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert persisted == receipt


def test_run_upscale_preserves_receipt_that_appears_during_inference(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.upscale import image_upscale

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    output = tmp_path / "source-4k.png"
    receipt_path = tmp_path / "receipt.json"
    runtime = tmp_path / "runtime"
    python = runtime / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
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
            "source": {
                "path": str(source),
                "sha256": image_upscale.file_sha256(source),
            },
            "output": {
                "path": str(output),
                "dimensions": [3840, 2160],
                "sha256": image_upscale.file_sha256(output),
            },
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
    ):
        image_upscale.run_upscale(
            source=source,
            output=output,
            runtime_dir=runtime,
            model_cache=models,
            receipt_path=receipt_path,
            command_runner=runner,
        )

    assert receipt_path.read_text(encoding="utf-8") == "owned"


def test_runtime_png_publish_honors_umask_and_is_atomic(
    tmp_path, monkeypatch
):
    import stat

    from sneeze.tpn.upscale import image_upscale_runtime

    temporary = tmp_path / "temporary.png"
    output = tmp_path / "output.png"
    temporary.write_bytes(b"complete")
    output.write_bytes(b"owned")
    observed_umasks = []

    def fake_umask(value):
        observed_umasks.append(value)
        return 0o027

    monkeypatch.setattr(image_upscale_runtime.os, "umask", fake_umask)

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
    assert observed_umasks == [0, 0o027, 0, 0o027]
    assert not temporary.exists()


def test_run_upscale_refuses_receipt_alias_before_child(
    tmp_path, monkeypatch
):
    from sneeze.tpn.upscale import image_upscale

    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    runtime = tmp_path / "runtime"
    python = runtime / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
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

    with pytest.raises(
        image_upscale.ImageUpscaleError,
        match="Receipt path must be separate",
    ):
        image_upscale.run_upscale(
            source=source,
            output=tmp_path / "output.png",
            runtime_dir=runtime,
            model_cache=models,
            receipt_path=source,
            force=True,
            command_runner=runner,
        )

    assert called is False
    assert source.read_bytes() == b"source"


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
        module_names=["sneeze", "sneeze.tpn.upscale"],
        introspect=True,
        auto_plugins=False,
    )

    assert "run" in cli._commands_by_name
    assert "setup" in cli._commands_by_name
    assert "sneeze.tpn.upscale.image_upscale_runtime" not in sys.modules


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
