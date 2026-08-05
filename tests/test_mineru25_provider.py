from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from sneeze.tpn.providers import mineru25


def _request(tmp_path):
    image_path = tmp_path / "page.png"
    image_path.write_bytes(b"png")
    return {
        "schema_version": 1,
        "attempt_id": "a" * 64,
        "lane": "best",
        "settings": dict(mineru25.QUALIFIED_REQUEST_SETTINGS),
        "input": {
            "image": {
                "path": str(image_path),
                "sha256": hashlib.sha256(b"png").hexdigest(),
                "dimensions": {"w": 1, "h": 1},
            }
        },
    }


def test_validate_request_requires_pinned_native_lane(tmp_path):
    request = _request(tmp_path)
    mineru25.validate_request(request)

    request["lane"] = "parity"
    with pytest.raises(ValueError, match="only supports lane=best"):
        mineru25.validate_request(request)

    request["lane"] = "best"
    request["settings"]["backend"] = "direct-transformers"
    with pytest.raises(ValueError, match="exactly match"):
        mineru25.validate_request(request)

    request["settings"] = {
        **mineru25.QUALIFIED_REQUEST_SETTINGS,
        "batch_size": 8,
    }
    with pytest.raises(ValueError, match="exactly match"):
        mineru25.validate_request(request)

    request["settings"] = {
        **mineru25.QUALIFIED_REQUEST_SETTINGS,
        "unsupported_control": True,
    }
    with pytest.raises(ValueError, match="exactly match"):
        mineru25.validate_request(request)


def test_generation_tracer_does_not_mark_eos_at_limit_as_max_tokens():
    class FakeSequence:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.values

    class FakeInputIds:
        shape = (1, 2)

    class FakeProcessor:
        def apply_chat_template(self, *args, **kwargs):
            return "rendered prompt"

        def decode(self, token_ids, **kwargs):
            return "decoded"

    class FakeModel:
        generation_config = SimpleNamespace(eos_token_id=9)

        def generate(self, *args, **kwargs):
            return [FakeSequence([4, 5, 9])]

    model = FakeModel()
    processor = FakeProcessor()
    tracer = mineru25.GenerationTracer(model, processor)
    tracer.install()

    processor.apply_chat_template([])
    model.generate(input_ids=FakeInputIds(), max_length=3)

    assert tracer.prompts == ["rendered prompt"]
    assert tracer.traces[0]["eos"] is True
    assert tracer.traces[0]["max_tokens"] is False
    assert tracer.traces[0]["finish_reason"] == "stop"
    assert tracer.traces[0]["stop_reason"] == "eos"


def test_snapshot_identity_hashes_critical_files(tmp_path):
    revision = "b" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    filenames = (
        "model.safetensors",
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "chat_template.jinja",
        "added_tokens.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "README.md",
    )
    for filename in filenames:
        (snapshot / filename).write_text(filename, encoding="utf-8")

    identity = mineru25.snapshot_identity(snapshot, revision)

    assert identity["commit"] == revision
    assert set(identity["files"]) == set(filenames)
    assert "added_tokens.json" in identity["files"]
    assert "README.md" in identity["files"]
    assert identity["files"]["config.json"]["sha256"] == hashlib.sha256(
        b"config.json"
    ).hexdigest()
    assert len(identity["files_sha256"]) == 64


def test_error_response_is_structured_and_preserves_trace(tmp_path):
    request = _request(tmp_path)
    tracer = SimpleNamespace(
        prompts=["prompt"],
        traces=[{"output_tokens": 1}],
    )
    error = RuntimeError("model initialization failed")

    response = mineru25.error_response(
        request,
        artifacts_dir=tmp_path,
        exc=error,
        phase="initialization",
        traceback_text="traceback",
        tracer=tracer,
    )

    assert response["status"] == "error"
    assert response["error"] == {
        "type": "RuntimeError",
        "message": "model initialization failed",
        "phase": "initialization",
    }
    artifact = tmp_path / response["artifacts"][0]["relative_path"]
    assert artifact.is_file()
    assert mineru25.sha256_path(artifact) == response["artifacts"][0][
        "sha256"
    ]
