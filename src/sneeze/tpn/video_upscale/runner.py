# Copyright (c) 2026 Trent Nelson
# SPDX-License-Identifier: Apache-2.0
"""Plan serial 2× video jobs and run the pinned inference containers."""

import csv
import fcntl
import io
import os
import signal
import subprocess
import tempfile
import time
import tomllib
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from threading import current_thread, main_thread

WORKERS = Path(__file__).resolve().parent
IMAGE = (
    "vllm/vllm-openai@sha256:"
    "e3481489818d3e563cae517b84ab959cc8b69f85a9fbb771c189140131ffc65c"
)
LTX_IMAGE = (
    "lmsysorg/sglang@sha256:"
    "7f132b890eb6d55f6a8ded87f85e359434878d0fb7f6758582cf759ffb35b5d8"
)
LTX_PROMPT = (
    "Preserve the original scene, motion, colors and lighting. "
    "Refine natural details and textures."
)
CONFIG_PATH = (
    Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    / "sneeze/video-upscale.toml"
)


def read_installation(path):
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except FileNotFoundError:
        return {}


INSTALLATION = read_installation(CONFIG_PATH)
GB300 = INSTALLATION.get("gpu")
LOCK = INSTALLATION.get("lock", "/tmp/sneeze-video-upscale.lock")
DOCKER = ["sudo", "-n", "docker"]


@dataclass
class Job:
    model: str
    source: Path
    output: Path
    settings: dict
    preview: Path | None = None


def choice(options, key, default, allowed):
    value = options.get(key) or default
    if value not in allowed:
        names = ", ".join(allowed)
        raise ValueError(f"--{key.replace('_', '-')} must be one of {names}")
    return value


def number(options, key, default, low, high=None):
    value = options.get(key)
    if value is None:
        return default
    flag = "--" + key.replace("_", "-")
    try:
        value = int(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{flag} expects an integer") from exc
    if value < low or (high is not None and value > high):
        bounds = f"{low}..{high}" if high is not None else f">= {low}"
        raise ValueError(f"{flag} must be {bounds}")
    return value


def settings(model, opts):
    turbo = model == "turbo"
    result = {
        "encoder": choice(opts, "encoder", "auto", ("auto", "cpu", "nvenc")),
        "decoder": choice(opts, "decoder", "auto", ("auto", "cpu", "nvdec")),
        "audio": choice(
            opts, "audio", "auto", ("auto", "copy", "aac", "drop")
        ),
        "seed": number(
            opts,
            "seed",
            1 if turbo else 42,
            0,
            2**32 - 1 - (1_000_000 if model == "seedvr2" else 0),
        ),
    }
    if result["encoder"] == "auto":
        result["encoder"] = "nvenc" if turbo else "cpu"
    if result["decoder"] == "auto":
        result["decoder"] = "cpu"
    shared_depth = number(opts, "lookahead", None, 0, 4)
    if turbo:
        input_mode = choice(
            opts, "turbo_input", "prefetch", ("prefetch", "serial")
        )
        default_depth = 0 if input_mode == "serial" else 2
        depth = number(
            opts,
            "turbo_lookahead",
            default_depth if shared_depth is None else shared_depth,
            0,
            4,
        )
        mode = "prefetch" if depth else "serial"
        if opts.get("turbo_input") is not None and input_mode != mode:
            raise ValueError(
                f"Turbo lookahead {depth} requires --turbo-input {mode}. "
                "You can omit --turbo-input when setting the depth."
            )
        result.update(turbo_input=mode, turbo_lookahead=depth)
        result["turbo_decoder"] = choice(
            opts, "turbo_decoder", "fast", ("fast", "exact")
        )
        micro = number(opts, "turbo_microbatch", 2, 1, 4)
        if micro not in (1, 2, 4):
            raise ValueError("--turbo-microbatch accepts 1, 2 or 4")
        result["turbo_microbatch"] = micro
    elif model == "seedvr2":
        result["seed_lookahead"] = number(
            opts, "seed_lookahead", shared_depth or 0, 0, 4
        )
        batch = number(opts, "seed_batch", 17, 1)
        chunk = number(opts, "seed_chunk", 53, 1)
        overlap = number(opts, "seed_overlap", 4, 0)
        if (batch - 1) % 4:
            raise ValueError("--seed-batch must be 4n+1 (e.g. 9 or 17)")
        if overlap >= batch or overlap > chunk:
            raise ValueError(
                "Seed overlap must be below batch size and at most chunk size"
            )
        result.update(
            seed_batch=batch, seed_chunk=chunk, seed_overlap=overlap
        )
    elif model == "ltx":
        for key in ("encoder", "decoder"):
            if result[key] != "cpu":
                raise ValueError(f"LTX requires --{key} auto or cpu")
        prompt = opts.get("ltx_prompt")
        if prompt is not None and not prompt.strip():
            raise ValueError("--ltx-prompt requires nonempty text")
        strength = opts.get("ltx_strength")
        try:
            strength = float(strength) if strength is not None else 1.0
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "--ltx-strength expects a number from 0 to 1"
            ) from exc
        if not 0 <= strength <= 1:
            raise ValueError("--ltx-strength must be from 0 to 1")
        vae_tile = number(opts, "ltx_vae_tile", 2048, 512, 2048)
        if vae_tile not in (512, 1024, 1536, 2048):
            raise ValueError("--ltx-vae-tile accepts 512, 1024, 1536 or 2048")
        chunk = number(opts, "ltx_chunk", 241, 9, 241)
        if (chunk - 1) % 8:
            raise ValueError("--ltx-chunk must be 8n+1 (e.g. 17 or 241)")
        result.update(
            ltx_prompt=prompt or LTX_PROMPT,
            ltx_strength=strength,
            ltx_vae_tile=vae_tile,
            ltx_chunk=chunk,
        )
    return result


def check_destination(path, force):
    if path.suffix.lower() != ".mp4":
        raise ValueError(f"Output must have a .mp4 extension: {path}")
    if os.path.lexists(path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Output must be a regular file: {path}")
        if not force:
            raise ValueError(
                f"Output exists: {path}. Choose another path or use --force."
            )


def plan(args, opts):
    if not args:
        raise ValueError(
            "Supply at least one input video. See sne upscale-video --help."
        )
    groups = {
        "both": ("turbo", "seedvr2"),
        "all": ("turbo", "seedvr2", "ltx"),
    }
    selected = choice(opts, "model", "turbo", (*groups["all"], *groups))
    models = groups.get(selected, (selected,))
    if selected == "ltx" and opts.get("lookahead") is not None:
        raise ValueError("--lookahead applies only to Turbo and SeedVR2")
    specific = {
        "turbo": (
            "turbo_decoder",
            "turbo_microbatch",
            "turbo_input",
            "turbo_lookahead",
        ),
        "seedvr2": (
            "seed_batch",
            "seed_chunk",
            "seed_overlap",
            "seed_lookahead",
        ),
        "ltx": ("ltx_prompt", "ltx_strength", "ltx_vae_tile", "ltx_chunk"),
    }
    for model, keys in specific.items():
        for key in keys:
            if model not in models and opts.get(key) is not None:
                raise ValueError(
                    f"--{key.replace('_', '-')} requires {model} "
                    "among the selected models"
                )
    output = opts.get("output")
    directory = opts.get("output_dir")
    if output and (directory or len(args) * len(models) != 1):
        raise ValueError(
            "--output selects one result. Use --output-dir for a batch."
        )
    if directory:
        directory = Path(directory).expanduser().resolve()
        if directory.exists() and not directory.is_dir():
            raise ValueError(f"Output directory is a file: {directory}")
    sources = [Path(arg).expanduser().resolve() for arg in args]
    identities = set()
    for source in sources:
        if not source.is_file() or not os.access(source, os.R_OK):
            raise ValueError(f"Missing or unreadable input: {source}")
        stat = source.stat()
        identities.add((stat.st_dev, stat.st_ino))
    outputs = set()

    def reserve(dest):
        if dest.exists():
            st = dest.stat()
            if (st.st_dev, st.st_ino) in identities:
                raise ValueError(f"Output would overwrite an input: {dest}")
        if dest in outputs:
            raise ValueError(
                f"Two results would use {dest}. Choose distinct input stems."
            )
        check_destination(dest, opts.get("force"))
        outputs.add(dest)

    jobs = []
    for source in sources:
        for model in models:
            dest = (
                Path(output).expanduser()
                if output
                else (directory or source.parent)
                / f"{source.stem}-{model}-2x.mp4"
            )
            # Resolve the directory only so a final symlink is visible.
            dest = dest.parent.resolve() / dest.name
            reserve(dest)
            jobs.append(Job(model, source, dest, settings(model, opts)))
        if opts.get("compare"):
            preview = dest.with_name(f"{source.stem}-{selected}-compare.mp4")
            reserve(preview)
            jobs[-1].preview = preview
    return jobs


def mount(path, target, writable=False):
    fields = ["type=bind", f"src={path}", f"dst={target}"]
    if not writable:
        fields.append("readonly")
    buf = io.StringIO()
    csv.writer(buf).writerow(fields)
    return ["--mount", buf.getvalue().rstrip("\r\n")]


def container_command(job, stage, name):
    root = INSTALLATION.get("root")
    if not isinstance(root, str) or not Path(root).is_absolute():
        raise ValueError(f"Set an absolute runtime root in {CONFIG_PATH}")
    if not isinstance(GB300, str) or not GB300.strip():
        raise ValueError(f"Set the inference GPU UUID in {CONFIG_PATH}")
    root = Path(root)
    paths = INSTALLATION.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"Expected a [paths] table in {CONFIG_PATH}")

    def asset(key, default):
        value = paths.get(key, default)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Invalid [paths].{key} in {CONFIG_PATH}")
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"[paths].{key} must be relative to root")
        return root / path

    base = asset(job.model, job.model)
    ltx = job.model == "ltx"
    cache = base / "cache"
    if not cache.is_dir() or not (base / "models").is_dir():
        raise ValueError(f"Missing {job.model} model or cache at {base}")
    imports = [asset("codecs", "codecs"), base, base / "fa4"]
    cwd = "/output"
    image = IMAGE
    if ltx:
        imports = [
            base,
            base / "comfy-runtime",
            base / "runtime",
            base / "comfy",
        ]
        cwd = str(base)
        image = LTX_IMAGE
    elif job.model == "turbo":
        src = asset("flashvsr", "flashvsr/source")
        imports.extend(
            [src.parent / "pinned-deps", src, src / "operators/flashvsr_attn"]
        )
    else:
        imports.extend([base / "deps", base / "source"])
        cwd = str(base / "source")
    codecs = job.settings
    rtx = codecs["encoder"] == "nvenc" or codecs["decoder"] == "nvdec"
    command = [
        *DOCKER,
        "run",
        "--rm",
        "--pull=never",
        "--init",
        "--name",
        name,
        "--network=none",
        "--shm-size=8g" if ltx else "--shm-size=4g",
        "--gpus",
        "all" if rtx else f"device={GB300}",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
    ]
    env = {
        "CUDA_VISIBLE_DEVICES": GB300,
        "NVIDIA_DRIVER_CAPABILITIES": "compute,video,utility",
        "OMP_NUM_THREADS": "8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "IMAGEIO_FFMPEG_EXE": "/usr/bin/ffmpeg",
        "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        "HOME": cache,
        "TRITON_CACHE_DIR": cache / "triton",
        "PYTHONPATH": os.pathsep.join(map(str, imports)),
    }
    if ltx:
        env.update(HF_HUB_OFFLINE="1", HF_HOME=cache / "huggingface")
    env.update({"VSR_" + k.upper(): v for k, v in job.settings.items()})
    for key, value in env.items():
        command.extend(["-e", f"{key}={value}"])
    source = "/input/source" + job.source.suffix.lower()
    if ltx:
        assets = asset("ltx_assets", "ltx-assets")
        command += mount(base, base) + mount(assets, assets)
    else:
        command += mount(root, root)
    command += mount(cache, cache, True)
    command += mount(WORKERS, "/worker") + mount(job.source, source)
    command += mount(stage, "/output", True)
    return command + [
        "-w",
        cwd,
        "--entrypoint",
        "python3",
        image,
        "-u",
        "/worker/media.py",
        job.model,
        source,
    ]


def duration(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{minutes}:{seconds:02d}"


def monitor(command, name, report, verbose):
    tail = deque(maxlen=20)
    start = time.monotonic()
    frames = expected = 0
    last_count = None
    proc = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
    )
    try:
        for raw in proc.stdout:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("VSR_PROGRESS "):
                _, completed, expected = line.split()
                frames, expected = int(completed), int(expected)
                if frames == last_count:
                    continue
                last_count = frames
                elapsed = time.monotonic() - start
                speed = frames / elapsed
                eta = (
                    duration((expected - frames) / speed)
                    if speed
                    else "pending"
                )
                report(
                    f"{frames}/{expected} frames ({frames / expected:.0%})"
                    f"  {speed:.2f} fps  elapsed {duration(elapsed)}"
                    f"  ETA {eta}"
                )
            else:
                tail.append(line)
                if (
                    verbose
                    or line.startswith(
                        (
                            "Input ",
                            "Verified ",
                            "Loading ",
                            "Finalizing ",
                            "Audio stream copy failed",
                        )
                    )
                    or "Chunk " in line
                    or " Phase " in line
                ):
                    report(line)
        status = proc.wait()
    except BaseException:
        # The model container can outlive its attached Docker client.
        # Stop creation/attachment first so a late Docker start cannot race
        # cleanup. sudo forwards TERM to its child.
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass
        subprocess.run([*DOCKER, "rm", "-f", name], capture_output=True)
        raise
    finally:
        proc.stdout.close()
        proc.wait()
    if status:
        raise RuntimeError(
            f"Video worker exited {status}:\n" + "\n".join(tail)
        )
    if not frames or frames != expected:
        raise RuntimeError(
            f"Incomplete video worker progress ({frames}/{expected})"
        )
    return frames


@contextmanager
def cleanup_on_termination():
    if current_thread() is not main_thread():
        yield
        return

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@contextmanager
def gpu_lock(report):
    with open(LOCK, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report("Waiting for the current GB300 upscale job to finish…")
            fcntl.flock(lock, fcntl.LOCK_EX)
        yield


@contextmanager
def staged_output(destination, force):
    check_destination(destination, force)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".sne-upscale-", dir=destination.parent
    ) as stage:
        candidate = Path(stage) / "verified.mp4"
        yield candidate
        if force:
            os.replace(candidate, destination)
        else:
            os.link(candidate, destination)


def describe(job, index, count):
    config = "  ".join(
        f"{k.replace('_', '-')}={v}" for k, v in job.settings.items()
    )
    title = f"[{index}/{count}] {job.model}  {job.source} → {job.output}"
    return f"{title}\n  {config}"


def execute(args, opts, stdout, stderr):
    jobs = plan(args, opts)
    if opts.get("quiet") and opts.get("verbose"):
        raise ValueError("Choose either --quiet or --verbose")

    tty_progress = stderr.isatty() and not opts.get("verbose")
    active_line = False

    def report(line):
        nonlocal active_line
        if opts.get("quiet"):
            return
        if tty_progress and " frames (" in line and "  ETA " in line:
            # On a terminal replace the previous count; keep 100% visible.
            complete = "(100%)" in line
            active_line = not complete
            print(
                "\r\x1b[K" + line,
                end="\n" if complete else "\r",
                file=stderr,
                flush=True,
            )
        else:
            prefix = "\r\x1b[K" if active_line else ""
            print(prefix + line, file=stderr, flush=True)
            active_line = False

    if opts.get("dry_run"):
        for index, job in enumerate(jobs, 1):
            print(describe(job, index, len(jobs)), file=stdout)
            if job.preview:
                print(f"  Side-by-side → {job.preview}", file=stdout)
        return

    force = opts.get("force")
    for _, group in groupby(enumerate(jobs, 1), lambda item: item[1].source):
        renders = []
        with cleanup_on_termination(), gpu_lock(report):
            for index, job in group:
                report(describe(job, index, len(jobs)))
                start = time.monotonic()
                with staged_output(job.output, force) as candidate:
                    name = f"sne-video-{uuid.uuid4().hex}"
                    command = container_command(job, candidate.parent, name)
                    report("Starting the container and checking the input…")
                    frames = monitor(
                        command, name, report, opts.get("verbose")
                    )
                elapsed = time.monotonic() - start
                report(
                    f"Finished in {elapsed:.1f}s · "
                    f"{frames / elapsed:.2f} end-to-end fps"
                )
                print(job.output, file=stdout, flush=True)
                renders.append((job.model, job.output))
                if job.preview:
                    from .compare import render_preview

                    with staged_output(job.preview, force) as candidate:
                        render_preview(job.source, renders, candidate, report)
                    print(job.preview, file=stdout, flush=True)
