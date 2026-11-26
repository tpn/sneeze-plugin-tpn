# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

from sneeze.tpn.video_upscale import runner


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "example input.mov"
    path.touch()
    return path


def test_defaults_and_model_order(source, tmp_path):
    other = tmp_path / "second.mp4"
    other.touch()
    (default,) = runner.plan([source], {})
    assert default.model == "turbo"
    assert default.output == source.with_name("example input-turbo-2x.mp4")
    output_dir = tmp_path / "4K"
    jobs = runner.plan(
        [source, other], {"model": "both", "output_dir": output_dir}
    )
    assert [(j.source, j.model, j.output) for j in jobs] == [
        (path, model, output_dir / f"{path.stem}-{model}-2x.mp4")
        for path in [source, other]
        for model in ["turbo", "seedvr2"]
    ]
    assert (
        default.settings
        == jobs[0].settings
        == {
            "encoder": "nvenc",
            "decoder": "cpu",
            "seed": 1,
            "audio": "auto",
            "turbo_input": "prefetch",
            "turbo_lookahead": 2,
            "turbo_decoder": "fast",
            "turbo_microbatch": 2,
        }
    )
    assert jobs[1].settings == {
        "encoder": "cpu",
        "decoder": "cpu",
        "seed": 42,
        "audio": "auto",
        "seed_lookahead": 0,
        "seed_batch": 17,
        "seed_chunk": 53,
        "seed_overlap": 4,
    }


def test_overrides_resolved_for_each_model(source):
    shared = {
        "encoder": "cpu",
        "decoder": "nvdec",
        "seed": 0,
        "audio": "drop",
    }
    turbo = {
        "turbo_decoder": "exact",
        "turbo_microbatch": 1,
        "turbo_input": "serial",
    }
    seed = {"seed_batch": 9, "seed_chunk": 33, "seed_overlap": 0}
    jobs = runner.plan([source], {"model": "both", **shared, **turbo, **seed})
    assert jobs[0].settings == shared | turbo | {"turbo_lookahead": 0}
    assert jobs[1].settings == shared | seed | {"seed_lookahead": 0}


def test_all_includes_ltx_with_its_measured_defaults(source):
    turbo, seed, ltx = runner.plan(
        [source], {"model": "all", "compare": True}
    )
    assert (turbo.model, seed.model, ltx.model) == ("turbo", "seedvr2", "ltx")
    assert turbo.preview is None and seed.preview is None
    assert ltx.preview == source.with_name("example input-all-compare.mp4")
    assert ltx.output == source.with_name("example input-ltx-2x.mp4")
    assert ltx.settings == {
        "audio": "auto",
        "encoder": "cpu",
        "decoder": "cpu",
        "seed": 42,
        "ltx_prompt": runner.LTX_PROMPT,
        "ltx_strength": 1.0,
        "ltx_vae_tile": 2048,
        "ltx_chunk": 241,
    }


def test_render_needs_a_configured_runtime(source, tmp_path, monkeypatch):
    installation = runner.read_installation(tmp_path / "absent.toml")
    monkeypatch.setattr(runner, "INSTALLATION", installation)
    (job,) = runner.plan([source], {})
    with pytest.raises(ValueError, match="absolute runtime root"):
        runner.container_command(job, tmp_path, "test-worker")


def test_ltx_overrides_reach_its_offline_container(
    source, tmp_path, monkeypatch
):
    base = tmp_path / "model-runtime"
    (base / "cache").mkdir(parents=True)
    assets = tmp_path / "model-weights"
    (assets / "models").mkdir(parents=True)
    (base / "models").symlink_to(assets / "models")
    config = tmp_path / "video-upscale.toml"
    config.write_text(
        f'root = "{tmp_path.as_posix()}"\n'
        'gpu = "example-device"\n'
        '[paths]\nltx = "model-runtime"\nltx_assets = "model-weights"\n'
    )
    installation = runner.read_installation(config)
    monkeypatch.setattr(runner, "INSTALLATION", installation)
    monkeypatch.setattr(runner, "GB300", installation["gpu"])
    opts = {
        "ltx_prompt": 'A reflection in "water".',
        "ltx_strength": "0.25",
        "ltx_vae_tile": 1024,
        "ltx_chunk": 17,
        "seed": 0,
    }
    (ltx,) = runner.plan([source], {"model": "ltx", **opts})
    cmd = runner.container_command(ltx, tmp_path, "ltx-test")
    assert cmd[-5:] == [
        runner.LTX_IMAGE,
        "-u",
        "/worker/media.py",
        "ltx",
        "/input/source.mov",
    ]
    assert cmd[cmd.index("--gpus") + 1] == f"device={runner.GB300}"
    assert cmd[cmd.index("-w") + 1] == str(base)
    for path in (assets, base):
        assert runner.mount(path, path)[1] in cmd
    assert "HF_HUB_OFFLINE=1" in cmd and "--network=none" in cmd
    for key, value in opts.items():
        assert f"VSR_{key.upper()}={value}" in cmd


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("encoder", "nvenc"),
        ("decoder", "nvdec"),
        ("lookahead", 0),
        ("ltx_strength", "nan"),
        ("ltx_strength", "inf"),
        ("ltx_strength", "-0.1"),
        ("ltx_strength", "bad"),
        ("ltx_strength", "1.01"),
        ("ltx_vae_tile", 600),
        ("ltx_chunk", 8),
        ("ltx_chunk", 242),
        ("ltx_chunk", 16),
        ("ltx_prompt", "  "),
    ],
)
def test_ltx_invalid_settings_report_the_option(source, key, value):
    with pytest.raises(ValueError, match="--" + key.replace("_", "-")):
        runner.plan([source], {"model": "ltx", key: value})


@pytest.mark.parametrize(
    ("model", "key", "value"),
    [
        ("turbo", "seed_batch", 17),
        ("turbo", "seed_chunk", 53),
        ("turbo", "seed_overlap", 0),
        ("seedvr2", "turbo_input", "prefetch"),
        ("seedvr2", "turbo_lookahead", 2),
        ("turbo", "seed_lookahead", 3),
        ("seedvr2", "turbo_decoder", "fast"),
        ("seedvr2", "turbo_microbatch", 2),
        ("seedvr2", "ltx_prompt", "water"),
        ("both", "ltx_strength", "0.5"),
        ("turbo", "ltx_chunk", 17),
        ("ltx", "seed_chunk", 17),
    ],
)
def test_unused_model_option_is_an_error(source, model, key, value):
    with pytest.raises(ValueError, match="--" + key.replace("_", "-")):
        runner.plan([source], {"model": model, key: value})


def test_duplicate_outputs_even_with_force(source, tmp_path):
    other = tmp_path / "other" / source.name
    other.parent.mkdir()
    other.touch()
    with pytest.raises(ValueError, match="Two results"):
        runner.plan(
            [source, other], {"output_dir": tmp_path / "out", "force": True}
        )


def test_input_is_protected_even_with_force_and_hard_links(tmp_path):
    source = tmp_path / "camera.mp4"
    source.write_bytes(b"input")
    alias = tmp_path / "another-name.mp4"
    alias.hardlink_to(source)
    for path in (source, alias):
        with pytest.raises(ValueError, match="overwrite an input"):
            runner.plan([source], {"output": path, "force": True})
    assert source.read_bytes() == b"input"


def test_dry_run_has_no_worker_or_filesystem_side_effects(
    source, tmp_path, monkeypatch, capsys
):
    def unexpected_worker(*args):
        pytest.fail("Dry run tried to start a worker")

    lock = tmp_path / "gpu.lock"
    dest = tmp_path / "missing" / "result.mp4"
    monkeypatch.setattr(runner, "container_command", unexpected_worker)
    monkeypatch.setattr(runner, "LOCK", lock)
    runner.execute(
        [source], {"output": dest, "dry_run": True}, sys.stdout, sys.stderr
    )
    out, err = capsys.readouterr()
    assert str(dest) in out and "encoder=nvenc" in out
    assert not err
    assert not dest.parent.exists()
    assert not lock.exists()


@pytest.mark.parametrize("fail", [False, True])
def test_execute_progress_and_atomic_replacement(
    source, tmp_path, monkeypatch, capsys, fail
):
    dest = tmp_path / "result.mp4"
    dest.write_bytes(b"original")
    monkeypatch.setattr(runner, "LOCK", tmp_path / "gpu.lock")

    def worker(job, stage, name):
        script = (
            "from pathlib import Path\n"
            "import sys\n"
            "Path(sys.argv[1]).write_bytes(b'candidate')\n"
            "print('VSR_PROGRESS 2 4', flush=True)\n"
        )
        if fail:
            script += "print('encoder failed')\nsys.exit(7)\n"
        else:
            script += "print('VSR_PROGRESS 4 4', flush=True)\n"
        return [
            sys.executable,
            "-c",
            script,
            str(Path(stage) / "verified.mp4"),
        ]

    monkeypatch.setattr(runner, "container_command", worker)
    opts = {"output": dest, "force": True}
    if fail:
        with pytest.raises(RuntimeError, match="encoder failed"):
            runner.execute([source], opts, sys.stdout, sys.stderr)
    else:
        runner.execute([source], opts, sys.stdout, sys.stderr)
    out, err = capsys.readouterr()
    assert "2/4 frames (50%)" in err
    assert "fps" in err and "ETA" in err
    if fail:
        assert not out
        assert dest.read_bytes() == b"original"
    else:
        assert "4/4 frames (100%)" in err
        assert out.strip() == str(dest)
        assert dest.read_bytes() == b"candidate"
    assert not list(tmp_path.glob(".sne-upscale-*"))


def test_comparison_reserved_before_rendering(source, tmp_path):
    out = tmp_path / "renders"
    options = {"model": "both", "compare": True, "output_dir": out}
    first, second = runner.plan([source], options)
    assert first.preview is None
    assert second.preview == out / "example input-both-compare.mp4"
    out.mkdir()
    second.preview.touch()
    with pytest.raises(ValueError, match="Output exists"):
        runner.plan([source], options)
    collision = source.with_name("example input-turbo-compare.mp4")
    with pytest.raises(ValueError, match="Two results"):
        runner.plan([source], {"compare": True, "output": collision})


def test_preview_failure_preserves_completed_renders(
    source, tmp_path, monkeypatch, capsys
):
    from sneeze.tpn.video_upscale import compare

    def worker(job, stage, name):
        script = (
            "from pathlib import Path\n"
            "import sys\n"
            "Path(sys.argv[1]).write_bytes(b'render')\n"
            "print('VSR_PROGRESS 1 1')"
        )
        return [sys.executable, "-c", script, str(stage / "verified.mp4")]

    def failing_preview(path, renders, candidate, report):
        assert path == source
        assert all(output.is_file() for _, output in renders)
        candidate.write_bytes(b"incomplete")
        raise RuntimeError("Preview failed")

    monkeypatch.setattr(runner, "LOCK", tmp_path / "gpu.lock")
    monkeypatch.setattr(runner, "container_command", worker)
    monkeypatch.setattr(compare, "render_preview", failing_preview)
    opts = {"model": "both", "compare": True, "force": True, "quiet": True}
    jobs = runner.plan([source], opts)
    jobs[-1].preview.write_bytes(b"original")
    with pytest.raises(RuntimeError, match="Preview failed"):
        runner.execute([source], opts, sys.stdout, sys.stderr)
    out, err = capsys.readouterr()
    assert out.splitlines() == [str(job.output) for job in jobs]
    assert not err
    assert jobs[-1].preview.read_bytes() == b"original"


@pytest.mark.parametrize("depth", range(5))
def test_shared_lookahead_selects_each_worker(source, depth):
    turbo, seed = runner.plan([source], {"model": "both", "lookahead": depth})
    assert turbo.settings["turbo_input"] == (
        "prefetch" if depth else "serial"
    )
    assert (
        turbo.settings["turbo_lookahead"]
        == seed.settings["seed_lookahead"]
        == depth
    )
    assert seed.settings["seed_overlap"] == 4


def test_specific_lookahead_overrides_shared_depth(source):
    first, second = runner.plan(
        [source],
        {
            "model": "both",
            "lookahead": 3,
            "turbo_lookahead": 0,
            "seed_lookahead": 2,
        },
    )
    assert first.settings["turbo_input"] == "serial"
    assert first.settings["turbo_lookahead"] == 0
    assert second.settings["seed_lookahead"] == 2


@pytest.mark.parametrize(
    "settings",
    [
        {"model": "turbo", "lookahead": -1},
        {"model": "both", "lookahead": 5},
        {"model": "turbo", "turbo_lookahead": "two"},
        {"model": "seedvr2", "seed_lookahead": 5},
        {"turbo_input": "serial", "lookahead": 2},
        {"turbo_input": "prefetch", "turbo_lookahead": 0},
    ],
)
def test_invalid_lookahead_rejected(source, settings):
    with pytest.raises(ValueError, match="lookahead"):
        runner.plan([source], settings)
