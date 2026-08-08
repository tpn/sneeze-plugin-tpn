import base64
import hashlib
import json
import os
import sys
from io import StringIO
from types import SimpleNamespace

import pytest

VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+Xk6cAAAAASUVORK5CYII="
)


def _structured_reuse_receipt(profile_id="profile-a"):
    from sneeze.tpn.ocr_backend_setup import make_receipt
    from sneeze.tpn.ocr_profiles import get_pipeline_definition

    pipeline = get_pipeline_definition("local-mineru25")
    profile_sha256 = hashlib.sha256(
        profile_id.encode("utf-8")
    ).hexdigest()
    return make_receipt(
        "ocr_structured_execution",
        {
            "pipeline_id": pipeline["pipeline_id"],
            "pipeline_definition_sha256": pipeline[
                "definition_sha256"
            ],
            "profile_id": profile_id,
            "profile_sha256": profile_sha256,
            "qualification_state": "qualified",
            "allow_provisional_profile": False,
            "memory_tier": "max",
            "memory_budget_bytes": 40 * 1024**3,
            "reserve_bytes": 2 * 1024**3,
            "hardware_selector": {
                "schema_version": 1,
                "vendor": "nvidia",
                "product_name": "Synthetic GPU",
                "pci_device_id": "0x1234",
                "compute_capability": [9, 0],
                "memory_model": "dedicated",
                "total_memory_bytes": 48 * 1024**3,
            },
            "match": {
                "schema_version": 1,
                "profile_id": profile_id,
                "pipeline_id": pipeline["pipeline_id"],
                "memory_tier": "max",
                "matched": True,
                "reasons": [],
                "profile_sha256": profile_sha256,
            },
            "stage_settings": {
                "grounded_ocr": {"batch_size": 1},
                "markdown": {"batch_size": 1},
            },
            "attention_backend": {
                "requested": "FLASH_ATTN",
                "qualified_selected": "FLASH_ATTN",
                "selected_verification": (
                    "deferred_until_engine_initialization"
                ),
            },
            "installation_record_sha256": {
                "deepseek-vllm": hashlib.sha256(b"deepseek").hexdigest(),
                "mineru25-pro-2605": hashlib.sha256(b"mineru").hexdigest(),
            },
            "deepseek_snapshot": {
                "revision": "9" * 40,
                "files_sha256": hashlib.sha256(b"snapshot").hexdigest(),
            },
        },
    )


def _reseal_structured_reuse_receipt(receipt):
    from sneeze.tpn.ocr_backend_setup import make_receipt

    payload = {
        key: value
        for key, value in receipt.items()
        if key not in {"schema_version", "kind", "receipt_sha256"}
    }
    return make_receipt("ocr_structured_execution", payload)


def _write_structured_reuse_bundle(tmp_path, receipt, stem="page-1"):
    pages = tmp_path / "pages"
    pages.mkdir(exist_ok=True)
    image_path = pages / f"{stem}.png"
    ocr_path = pages / f"{stem}.ocr"
    markdown_path = pages / f"{stem}.md"
    json_path = pages / f"{stem}.json"
    image_path.write_bytes(VALID_PNG_BYTES)
    ocr_path.write_text("grounded evidence\n", encoding="utf-8")
    markdown_path.write_text("structured markdown\n", encoding="utf-8")
    image_sha256 = hashlib.sha256(VALID_PNG_BYTES).hexdigest()

    def stage_metadata(output_path):
        output_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()
        return {
            "status": "completed",
            "image": {"sha256": image_sha256},
            "execution_profile": receipt,
            "finish_reason": "stop",
            "termination": {
                "eos": True,
                "length": False,
                "max_tokens": False,
            },
            "validation": {"status": "pass", "flags": []},
            "promotion": {
                "promoted": True,
                "candidate_sha256": output_sha256,
                "persisted_output_sha256": output_sha256,
            },
        }

    payload = {
        "image_sha256": image_sha256,
        "ocr_text": ocr_path.read_text(encoding="utf-8"),
        "markdown_text": markdown_path.read_text(encoding="utf-8"),
        "generation_metadata": {
            "ocr": stage_metadata(ocr_path),
            "markdown": stage_metadata(markdown_path),
        },
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "pages": pages,
        "image": image_path,
        "ocr": ocr_path,
        "markdown": markdown_path,
        "json": json_path,
        "payload": payload,
    }


def test_plugin_package_imports():
    import importlib

    assert importlib.import_module("sneeze.tpn")


def test_pdf_page_tasks_split_existing_outputs(tmp_path, monkeypatch):
    from sneeze.tpn import pdf_pages

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    dest_dir = tmp_path / "pages"
    dest_dir.mkdir()
    pdf_path = source_dir / "alpha.pdf"
    pdf_path.write_text("", encoding="utf-8")

    tasks = pdf_pages.build_pdf_pages_tasks(
        str(source_dir),
        str(dest_dir),
        recursive=False,
        text_root=str(dest_dir),
    )
    assert len(tasks) == 1

    page_dir = dest_dir / "alpha"
    page_dir.mkdir()
    (page_dir / "page-01.png").write_bytes(VALID_PNG_BYTES)
    (page_dir / "page-01.txt").write_text("", encoding="utf-8")
    (page_dir / "page-02.png").write_bytes(VALID_PNG_BYTES)

    monkeypatch.setattr(pdf_pages, "get_pdf_page_count", lambda _path: 12)

    page_tasks = pdf_pages.build_pdf_page_tasks(
        tasks,
        extract_text=True,
        force=False,
    )
    by_page = {task.page_num: task for task in page_tasks}

    assert 1 not in by_page
    assert by_page[2].render_image is False
    assert by_page[2].render_text is True
    assert by_page[2].text_output_path == str(page_dir / "page-02.txt")
    assert by_page[3].page_stem == "page-03"


def test_extract_pdf_page_and_text_invokes_poppler(tmp_path, monkeypatch):
    from sneeze.tpn import pdf_pages

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_text("", encoding="utf-8")
    image_dir = tmp_path / "images"
    text_dir = tmp_path / "text"
    task = pdf_pages.PdfPageTask(
        source_path=str(pdf_path),
        dest_dir=str(image_dir),
        page_num=2,
        page_count=12,
        image_output_path=str(image_dir / "page-02.png"),
        render_image=True,
        render_text=True,
        text_output_path=str(text_dir / "page-02.txt"),
        dpi=300,
    )
    calls = []

    def fake_run(args, check, stdout, stderr, text):
        calls.append(args)
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(pdf_pages.subprocess, "run", fake_run)

    assert pdf_pages.extract_pdf_page_and_text(task) is True
    assert calls == [
        [
            "pdftocairo",
            "-png",
            "-r",
            "300",
            "-singlefile",
            "-f",
            "2",
            "-l",
            "2",
            str(pdf_path),
            str(image_dir / "page-02"),
        ],
        [
            "pdftotext",
            "-f",
            "2",
            "-l",
            "2",
            str(pdf_path),
            str(text_dir / "page-02.txt"),
        ],
    ]


def test_pdf_to_pages_uses_public_cli_dest_dir(tmp_path, monkeypatch):
    from sneeze.tpn import pdf_pages
    from sneeze.tpn.commands import PdfToPages

    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "pages"
    source_dir.mkdir()
    captured = {}

    def fake_build_pdf_pages_tasks(source, dest, **kwargs):
        captured["source"] = source
        captured["dest"] = dest
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr(
        pdf_pages,
        "build_pdf_pages_tasks",
        fake_build_pdf_pages_tasks,
    )

    out = StringIO()
    command = PdfToPages(None, out, None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)

    command.run()

    assert captured["source"] == str(source_dir)
    assert captured["dest"] == str(dest_dir)


def test_pdf_to_pages_forwards_shared_renderer_options(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.commands as commands
    from sneeze.tpn.commands import PdfToPages

    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "pages"
    source_dir.mkdir()
    captured = {}

    def fake_run_pdf_to_pages(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        commands,
        "_run_pdf_to_pages",
        fake_run_pdf_to_pages,
    )
    command = PdfToPages(None, StringIO(), None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.recursive = True
    command._parallelism = 3
    command._limit = 2
    command._dpi = 300
    command.force = True
    command.no_text = True

    command.run()

    assert captured == {
        "source_dir": str(source_dir),
        "dest_dir": str(dest_dir),
        "recursive": True,
        "parallelism": 3,
        "limit": 2,
        "dpi": 300,
        "force": True,
        "no_text": True,
        "record_encrypted_sentinels": True,
        "out": command._out,
    }


def test_combined_render_rejects_different_text_source_before_writes(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    import sneeze.tpn.commands as commands
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pdfs"
    dest_dir = tmp_path / "pages"
    text_source_dir = tmp_path / "stale-text"
    source_dir.mkdir()
    text_source_dir.mkdir()
    called = False

    def unexpected_render(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(commands, "_run_pdf_to_pages", unexpected_render)
    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.text_source_dir = str(text_source_dir)
    command.pdf_to_pages = True

    with pytest.raises(
        CommandError,
        match="--text-source-dir must equal the effective destination",
    ):
        command.run()

    assert called is False
    assert not dest_dir.exists()


def test_combined_renderer_does_not_write_source_encrypted_sentinel(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn import pdf_pages
    from sneeze.tpn.commands import _run_pdf_to_pages

    source_dir = tmp_path / "source"
    dest_dir = tmp_path / "pages"
    source_dir.mkdir()
    readable_pdf = source_dir / "readable.pdf"
    encrypted_pdf = source_dir / "encrypted.pdf"
    readable_pdf.write_bytes(b"readable")
    encrypted_pdf.write_bytes(b"encrypted")
    page_dir = dest_dir / "readable"
    image_path = page_dir / "page-1.png"
    prepared = pdf_pages.PdfPagesTask(
        source_path=str(readable_pdf),
        dest_dir=str(page_dir),
        text_dest_dir=str(page_dir),
        page_count=1,
    )
    skipped = pdf_pages.PdfPreparationError(
        source_path=str(encrypted_pdf),
        error="synthetic encrypted PDF",
        kind="encrypted",
        sentinel_path=f"{encrypted_pdf}.encrypted",
    )
    page_task = pdf_pages.PdfPageTask(
        source_path=str(readable_pdf),
        dest_dir=str(page_dir),
        page_num=1,
        page_count=1,
        image_output_path=str(image_path),
        render_image=True,
    )
    sentinel_writer_called = False

    monkeypatch.setattr(
        pdf_pages,
        "build_pdf_pages_tasks",
        lambda *_args, **_kwargs: [prepared],
    )
    monkeypatch.setattr(
        pdf_pages,
        "prepare_pdf_pages_tasks",
        lambda *_args, **_kwargs: ([prepared], [skipped]),
    )
    monkeypatch.setattr(
        pdf_pages,
        "build_pdf_page_tasks",
        lambda *_args, **_kwargs: [page_task],
    )

    def fake_run_tasks(tasks, _fn, **_kwargs):
        assert tasks == [page_task]
        page_dir.mkdir(parents=True)
        image_path.write_bytes(VALID_PNG_BYTES)

    def fake_write_sentinels(_skipped):
        nonlocal sentinel_writer_called
        sentinel_writer_called = True
        return []

    monkeypatch.setattr(pdf_pages, "run_tasks_in_threads", fake_run_tasks)
    monkeypatch.setattr(
        pdf_pages,
        "write_encrypted_sentinels",
        fake_write_sentinels,
    )

    result = _run_pdf_to_pages(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        recursive=False,
        parallelism=1,
        limit=None,
        dpi=400,
        force=False,
        no_text=False,
        record_encrypted_sentinels=False,
        out=lambda _message: None,
    )

    assert result.image_paths == (str(image_path),)
    assert sentinel_writer_called is False
    assert not (source_dir / "encrypted.pdf.encrypted").exists()


def test_ocr_task_collection_skips_existing_outputs(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_tasks

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    (source_dir / "page-2.png").write_bytes(VALID_PNG_BYTES)
    (dest_dir / "page-1.md").write_text("done", encoding="utf-8")

    tasks, pdf_count = collect_ocr_tasks(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        output_ext="md",
        force=False,
        scan_parallelism=1,
    )

    assert pdf_count == 1
    assert len(tasks) == 1
    assert tasks[0].image_path == str(source_dir / "page-2.png")
    assert tasks[0].output_path == str(dest_dir / "page-2.md")


def test_ocr_task_collection_refreshes_rendered_image_output(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_tasks

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    image_path = source_dir / "page-1.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    (dest_dir / "page-1.md").write_text("stale", encoding="utf-8")

    tasks, _pdf_count = collect_ocr_tasks(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        output_ext="md",
        force=False,
        refresh_image_paths=(str(image_path),),
        scan_parallelism=1,
    )

    assert [task.image_path for task in tasks] == [str(image_path)]


def test_ocr_task_collection_refreshes_output_newer_image_after_retry(
    tmp_path,
):
    from sneeze.tpn.ocr_vllm import collect_ocr_tasks

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    image_path = source_dir / "page-1.png"
    output_path = dest_dir / "page-1.md"
    output_path.write_text("stale", encoding="utf-8")
    image_path.write_bytes(VALID_PNG_BYTES)
    os.utime(output_path, ns=(1_000_000_000, 1_000_000_000))
    os.utime(image_path, ns=(2_000_000_000, 2_000_000_000))

    tasks, _pdf_count = collect_ocr_tasks(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        output_ext="md",
        force=False,
        verify_input_freshness=True,
        scan_parallelism=1,
    )

    assert [task.image_path for task in tasks] == [str(image_path)]


def test_ocr_full_collection_skips_only_complete_bundles(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    (source_dir / "page-2.png").write_bytes(VALID_PNG_BYTES)
    for suffix in ("ocr", "md", "json"):
        (dest_dir / f"page-1.{suffix}").write_text(
            "done",
            encoding="utf-8",
        )
    (dest_dir / "page-2.ocr").write_text("ocr", encoding="utf-8")

    items, pdf_count = collect_ocr_full_items(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        force=False,
        scan_parallelism=1,
    )

    assert pdf_count == 1
    assert len(items) == 1
    assert items[0].image_path == str(source_dir / "page-2.png")
    assert items[0].ocr_path == str(dest_dir / "page-2.ocr")
    assert items[0].markdown_path == str(dest_dir / "page-2.md")
    assert items[0].json_path == str(dest_dir / "page-2.json")
    assert items[0].need_ocr is False
    assert items[0].need_markdown is True
    assert items[0].need_json is True


def test_complete_ocr_bundle_skips_image_hashing(tmp_path, monkeypatch):
    import sneeze.tpn.ocr_vllm as ocr_vllm

    image_path = tmp_path / "page-1.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    for suffix, content in (
        ("ocr", "grounded OCR"),
        ("md", "markdown"),
        ("json", '{"markdown_quality":{"status":"pass"}}'),
    ):
        (tmp_path / f"page-1.{suffix}").write_text(
            content,
            encoding="utf-8",
        )
    monkeypatch.setattr(
        ocr_vllm,
        "file_sha256",
        lambda _path: pytest.fail("complete bundles must not hash the PNG"),
    )

    item = ocr_vllm.ocr_full_item_from_candidate(
        ocr_vllm.OcrScanCandidate(
            image_path=str(image_path),
            base="page-1",
        ),
        str(tmp_path),
        force=False,
    )

    assert item is None


def test_ocr_full_collection_refreshes_rendered_image_bundle(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    image_path = source_dir / "page-1.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    for suffix in ("ocr", "md", "json"):
        (dest_dir / f"page-1.{suffix}").write_text(
            "stale",
            encoding="utf-8",
        )

    items, _pdf_count = collect_ocr_full_items(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        force=False,
        refresh_image_paths=(str(image_path),),
        scan_parallelism=1,
    )

    assert len(items) == 1
    assert items[0].need_ocr is True
    assert items[0].need_markdown is True
    assert items[0].need_json is True


def test_ocr_full_collection_refreshes_only_metadata_for_rendered_text(
    tmp_path,
):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    image_path = source_dir / "page-1.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    for suffix in ("ocr", "md", "json"):
        (dest_dir / f"page-1.{suffix}").write_text(
            "existing",
            encoding="utf-8",
        )

    items, _pdf_count = collect_ocr_full_items(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        force=False,
        refresh_metadata_image_paths=(str(image_path),),
        scan_parallelism=1,
    )

    assert len(items) == 1
    assert items[0].need_ocr is False
    assert items[0].need_markdown is False
    assert items[0].need_json is True


def test_ocr_full_collection_uses_input_freshness_after_retry(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    image_path = source_dir / "page-1.png"
    text_path = source_dir / "page-1.txt"
    image_path.write_bytes(VALID_PNG_BYTES)
    text_path.write_text("new raw text", encoding="utf-8")
    output_paths = []
    for suffix in ("ocr", "md", "json"):
        output_path = dest_dir / f"page-1.{suffix}"
        output_path.write_text("stale", encoding="utf-8")
        os.utime(output_path, ns=(1_000_000_000, 1_000_000_000))
        output_paths.append(output_path)
    os.utime(image_path, ns=(2_000_000_000, 2_000_000_000))
    os.utime(text_path, ns=(3_000_000_000, 3_000_000_000))

    items, _pdf_count = collect_ocr_full_items(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=str(source_dir),
        force=False,
        verify_input_freshness=True,
        scan_parallelism=1,
    )

    assert len(items) == 1
    assert items[0].need_ocr is True
    assert items[0].need_markdown is True
    assert items[0].need_json is True

    for output_path in output_paths:
        os.utime(output_path, ns=(4_000_000_000, 4_000_000_000))
    os.utime(text_path, ns=(5_000_000_000, 5_000_000_000))
    items, _pdf_count = collect_ocr_full_items(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=str(source_dir),
        force=False,
        verify_input_freshness=True,
        scan_parallelism=1,
    )

    assert len(items) == 1
    assert items[0].need_ocr is False
    assert items[0].need_markdown is False
    assert items[0].need_json is True


def test_ocr_full_freshness_rebuilds_only_missing_bundle_member(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    image_path = source_dir / "page-1.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    os.utime(image_path, ns=(1_000_000_000, 1_000_000_000))
    for suffix in ("ocr", "md"):
        output_path = dest_dir / f"page-1.{suffix}"
        output_path.write_text("existing", encoding="utf-8")
        os.utime(output_path, ns=(2_000_000_000, 2_000_000_000))

    items, _pdf_count = collect_ocr_full_items(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        force=False,
        verify_input_freshness=True,
        scan_parallelism=1,
    )

    assert len(items) == 1
    assert items[0].need_ocr is False
    assert items[0].need_markdown is False
    assert items[0].need_json is True


def test_ocr_full_collection_rechecks_persisted_quality_failure(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    dest_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    (dest_dir / "page-1.ocr").write_text("ocr", encoding="utf-8")
    (dest_dir / "page-1.md").write_text("markdown", encoding="utf-8")
    (dest_dir / "page-1.json").write_text(
        json.dumps({"markdown_quality": {"status": "fail"}}),
        encoding="utf-8",
    )

    items, _pdf_count = collect_ocr_full_items(
        source_dir=str(source_dir),
        dest_dir=str(dest_dir),
        text_source_dir=None,
        force=False,
        scan_parallelism=1,
    )

    assert len(items) == 1
    assert items[0].need_ocr is False
    assert items[0].need_markdown is False
    assert items[0].need_json is True


def test_structured_full_collection_reuses_exact_provenance(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)

    items, pdf_count = collect_ocr_full_items(
        source_dir=str(bundle["pages"]),
        dest_dir=str(bundle["pages"]),
        text_source_dir=str(bundle["pages"]),
        force=False,
        scan_parallelism=1,
        expected_execution_receipt=receipt,
    )

    assert pdf_count == 1
    assert items == []


@pytest.mark.parametrize(
    ("mismatch", "stage"),
    (
        ("legacy", "grounded_ocr"),
        ("profile", "grounded_ocr"),
        ("image", "grounded_ocr"),
        ("output_hash", "markdown"),
    ),
)
def test_structured_full_collection_rejects_unproven_reuse(
    tmp_path,
    mismatch,
    stage,
):
    from sneeze.tpn.ocr_vllm import (
        StructuredOutputReuseError,
        collect_ocr_full_items,
    )

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)
    if mismatch == "legacy":
        bundle["json"].write_text("{}", encoding="utf-8")
    elif mismatch == "profile":
        different = _structured_reuse_receipt("profile-b")
        for metadata in bundle["payload"]["generation_metadata"].values():
            metadata["execution_profile"] = different
        bundle["json"].write_text(
            json.dumps(bundle["payload"]),
            encoding="utf-8",
        )
    elif mismatch == "image":
        bundle["payload"]["image_sha256"] = "0" * 64
        bundle["json"].write_text(
            json.dumps(bundle["payload"]),
            encoding="utf-8",
        )
    else:
        bundle["markdown"].write_text(
            "structured markdown changed\n",
            encoding="utf-8",
        )

    with pytest.raises(StructuredOutputReuseError) as raised:
        collect_ocr_full_items(
            source_dir=str(bundle["pages"]),
            dest_dir=str(bundle["pages"]),
            text_source_dir=str(bundle["pages"]),
            force=False,
            scan_parallelism=1,
            expected_execution_receipt=receipt,
        )

    message = str(raised.value)
    assert str(bundle["json"]) in message
    assert f"stage {stage}" in message
    assert "fresh --dest-dir" in message
    assert "--force" in message


def test_structured_full_collection_rejects_reused_stage_in_partial_bundle(
    tmp_path,
):
    from sneeze.tpn.ocr_vllm import (
        StructuredOutputReuseError,
        collect_ocr_full_items,
    )

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)
    bundle["ocr"].unlink()
    bundle["payload"]["generation_metadata"].pop("markdown")
    bundle["json"].write_text(
        json.dumps(bundle["payload"]),
        encoding="utf-8",
    )

    with pytest.raises(StructuredOutputReuseError, match="stage markdown"):
        collect_ocr_full_items(
            source_dir=str(bundle["pages"]),
            dest_dir=str(bundle["pages"]),
            text_source_dir=str(bundle["pages"]),
            force=False,
            scan_parallelism=1,
            expected_execution_receipt=receipt,
        )


@pytest.mark.parametrize("refresh", [False, True])
def test_structured_full_collection_explicit_regeneration_bypasses_reuse(
    tmp_path,
    refresh,
):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)
    bundle["json"].write_text("{}", encoding="utf-8")

    items, _pdf_count = collect_ocr_full_items(
        source_dir=str(bundle["pages"]),
        dest_dir=str(bundle["pages"]),
        text_source_dir=str(bundle["pages"]),
        force=not refresh,
        refresh_image_paths=(str(bundle["image"]),) if refresh else None,
        scan_parallelism=1,
        expected_execution_receipt=receipt,
    )

    assert len(items) == 1
    assert items[0].need_ocr is True
    assert items[0].need_markdown is True
    assert items[0].need_json is True


def test_structured_full_collection_rejects_noncanonical_receipt(tmp_path):
    from sneeze.tpn.ocr_vllm import (
        StructuredOutputReuseError,
        collect_ocr_full_items,
    )

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)
    invalid_receipt = {**receipt, "profile_id": "tampered"}

    with pytest.raises(
        StructuredOutputReuseError,
        match="selected execution receipt is invalid",
    ):
        collect_ocr_full_items(
            source_dir=str(bundle["pages"]),
            dest_dir=str(bundle["pages"]),
            text_source_dir=str(bundle["pages"]),
            force=False,
            scan_parallelism=1,
            expected_execution_receipt=invalid_receipt,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (("finish_reason", "length"), ("termination.length", True)),
)
def test_structured_full_collection_rejects_contradictory_termination(
    tmp_path,
    field,
    value,
):
    from sneeze.tpn.ocr_vllm import (
        StructuredOutputReuseError,
        collect_ocr_full_items,
    )

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)
    metadata = bundle["payload"]["generation_metadata"]["ocr"]
    if field == "finish_reason":
        metadata[field] = value
    else:
        metadata["termination"]["length"] = value
    bundle["json"].write_text(
        json.dumps(bundle["payload"]), encoding="utf-8"
    )

    with pytest.raises(
        StructuredOutputReuseError,
        match="stage grounded_ocr",
    ):
        collect_ocr_full_items(
            source_dir=str(bundle["pages"]),
            dest_dir=str(bundle["pages"]),
            text_source_dir=str(bundle["pages"]),
            force=False,
            scan_parallelism=1,
            expected_execution_receipt=receipt,
        )


@pytest.mark.parametrize("candidate", [None, "0" * 64])
def test_structured_full_collection_rejects_candidate_digest_drift(
    tmp_path,
    candidate,
):
    from sneeze.tpn.ocr_vllm import (
        StructuredOutputReuseError,
        collect_ocr_full_items,
    )

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)
    promotion = bundle["payload"]["generation_metadata"]["ocr"][
        "promotion"
    ]
    if candidate is None:
        promotion.pop("candidate_sha256")
    else:
        promotion["candidate_sha256"] = candidate
    bundle["json"].write_text(
        json.dumps(bundle["payload"]), encoding="utf-8"
    )

    with pytest.raises(
        StructuredOutputReuseError,
        match="promoted output hash",
    ):
        collect_ocr_full_items(
            source_dir=str(bundle["pages"]),
            dest_dir=str(bundle["pages"]),
            text_source_dir=str(bundle["pages"]),
            force=False,
            scan_parallelism=1,
            expected_execution_receipt=receipt,
        )


@pytest.mark.parametrize("bundle_text", [None, "stale markdown\n"])
def test_structured_full_collection_rejects_missing_or_stale_bundle_text(
    tmp_path,
    bundle_text,
):
    from sneeze.tpn.ocr_vllm import (
        StructuredOutputReuseError,
        collect_ocr_full_items,
    )

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)
    if bundle_text is None:
        bundle["payload"].pop("markdown_text")
    else:
        bundle["payload"]["markdown_text"] = bundle_text
    bundle["json"].write_text(
        json.dumps(bundle["payload"]), encoding="utf-8"
    )

    with pytest.raises(
        StructuredOutputReuseError,
        match="stage markdown.*bundle markdown_text",
    ):
        collect_ocr_full_items(
            source_dir=str(bundle["pages"]),
            dest_dir=str(bundle["pages"]),
            text_source_dir=str(bundle["pages"]),
            force=False,
            scan_parallelism=1,
            expected_execution_receipt=receipt,
        )


@pytest.mark.parametrize(
    ("mutation", "value"),
    (
        ("missing_pipeline", None),
        ("pipeline_id", "deepseek-vllm-full"),
        ("profile_id", ""),
        ("pipeline_definition_sha256", "0" * 64),
    ),
)
def test_structured_full_collection_rejects_incomplete_validly_hashed_receipt(
    tmp_path,
    mutation,
    value,
):
    from sneeze.tpn.ocr_vllm import (
        StructuredOutputReuseError,
        collect_ocr_full_items,
    )

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)
    mutated = dict(receipt)
    if mutation == "missing_pipeline":
        mutated.pop("pipeline_id")
    else:
        mutated[mutation] = value
    mutated = _reseal_structured_reuse_receipt(mutated)

    with pytest.raises(
        StructuredOutputReuseError,
        match="selected execution receipt is invalid",
    ):
        collect_ocr_full_items(
            source_dir=str(bundle["pages"]),
            dest_dir=str(bundle["pages"]),
            text_source_dir=str(bundle["pages"]),
            force=False,
            scan_parallelism=1,
            expected_execution_receipt=mutated,
        )


def test_structured_full_collection_propagates_parallel_reuse_error(tmp_path):
    from sneeze.tpn.ocr_vllm import (
        StructuredOutputReuseError,
        collect_ocr_full_items,
    )

    receipt = _structured_reuse_receipt()
    first = _write_structured_reuse_bundle(tmp_path, receipt)
    second = _write_structured_reuse_bundle(tmp_path, receipt, stem="page-2")
    second["payload"]["markdown_text"] = "stale markdown\n"
    second["json"].write_text(
        json.dumps(second["payload"]), encoding="utf-8"
    )

    with pytest.raises(
        StructuredOutputReuseError,
        match="page-2.json stage markdown",
    ):
        collect_ocr_full_items(
            source_dir=str(first["pages"]),
            dest_dir=str(first["pages"]),
            text_source_dir=str(first["pages"]),
            force=False,
            scan_parallelism=2,
            expected_execution_receipt=receipt,
        )


def test_unstructured_full_collection_preserves_legacy_reuse(tmp_path):
    from sneeze.tpn.ocr_vllm import collect_ocr_full_items

    receipt = _structured_reuse_receipt()
    bundle = _write_structured_reuse_bundle(tmp_path, receipt)
    bundle["json"].write_text("{}", encoding="utf-8")

    items, _pdf_count = collect_ocr_full_items(
        source_dir=str(bundle["pages"]),
        dest_dir=str(bundle["pages"]),
        text_source_dir=str(bundle["pages"]),
        force=False,
        scan_parallelism=1,
    )

    assert items == []


def test_bundle_recheck_preserves_existing_generation_metadata(tmp_path):
    from sneeze.tpn.ocr_vllm import OcrFullItem, write_ocr_page_bundle

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    ocr_path = tmp_path / "page.ocr"
    markdown_path = tmp_path / "page.md"
    json_path = tmp_path / "page.json"
    ocr_path.write_text("grounded evidence", encoding="utf-8")
    markdown_path.write_text("markdown", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {"generation_metadata": {"markdown": {"request_id": "preserved"}}}
        ),
        encoding="utf-8",
    )
    item = OcrFullItem(
        image_path=str(image_path),
        base="page",
        ocr_path=str(ocr_path),
        markdown_path=str(markdown_path),
        json_path=str(json_path),
        text_path=None,
        need_json=True,
    )

    payload = write_ocr_page_bundle(
        item,
        generation_metadata={"ocr": {"request_id": "new"}},
    )

    assert payload["generation_metadata"] == {
        "markdown": {"request_id": "preserved"},
        "ocr": {"request_id": "new"},
    }


def test_bundle_metadata_readers_ignore_non_object_json(tmp_path):
    from sneeze.tpn.ocr_vllm import (
        persisted_markdown_quality_status,
        read_existing_generation_metadata,
    )

    json_path = tmp_path / "page.json"
    json_path.write_text("[]", encoding="utf-8")

    assert persisted_markdown_quality_status(str(json_path)) is None
    assert read_existing_generation_metadata(str(json_path)) is None


def test_ocr_pdf_pages_uses_processor(tmp_path, monkeypatch):
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    captured = {}

    class FakeProcessor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs

        def run(self, tasks):
            captured["tasks"] = list(tasks)

    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    out = StringIO()
    command = OcrPdfPages(None, out, None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.output_format = "md"
    command.prompt_mode = "markdown"
    command._scan_parallelism = 1
    command._max_tokens = 64
    command.temperature = "0.2"

    command.run()

    assert captured["kwargs"]["model_revision"] == (
        "9f30c71f441d010e5429c532364a86705536c53a"
    )
    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["temperature"] == 0.2
    assert captured["kwargs"]["window_size"] == 128
    assert len(captured["tasks"]) == 1
    assert captured["tasks"][0].output_path == str(dest_dir / "page-1.md")


def test_ocr_pdf_pages_can_render_then_ocr_one_destination(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.commands as commands
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    pdf_root = tmp_path / "pdfs"
    dest_dir = tmp_path / "pages"
    pdf_root.mkdir()
    (pdf_root / "paper.pdf").write_bytes(b"synthetic")
    stale_dir = dest_dir / "removed-paper"
    stale_dir.mkdir(parents=True)
    (stale_dir / "page-01.png").write_bytes(VALID_PNG_BYTES)
    captured = {"events": []}

    def fake_render(**kwargs):
        captured["events"].append("render")
        captured["render"] = kwargs
        page_dir = dest_dir / "paper"
        page_dir.mkdir(parents=True)
        (page_dir / "page-01.png").write_bytes(VALID_PNG_BYTES)
        (page_dir / "page-01.txt").write_text(
            "extracted text",
            encoding="utf-8",
        )
        image_path = str(page_dir / "page-01.png")
        return commands.PdfPagesRunResult(
            image_paths=(image_path,),
            rendered_image_paths=frozenset({image_path}),
            rendered_text_image_paths=frozenset({image_path}),
        )

    class FakeProcessor:
        def __init__(self, **kwargs):
            captured["events"].append("processor")

        def run(self, tasks):
            captured["events"].append("ocr")
            captured["tasks"] = list(tasks)

    monkeypatch.setattr(commands, "_run_pdf_to_pages", fake_render)
    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(pdf_root)
    command.dest_dir = str(dest_dir)
    command.pdf_to_pages = True
    command.pdf_recursive = True
    command._pdf_parallelism = 4
    command._pdf_dpi = 350
    command._scan_parallelism = 2
    command.output_format = "md"
    command.prompt_mode = "markdown"
    command.force = True

    command.run()

    assert captured["events"] == ["render", "processor", "ocr"]
    assert captured["render"]["source_dir"] == str(pdf_root)
    assert captured["render"]["dest_dir"] == str(dest_dir)
    assert captured["render"]["recursive"] is True
    assert captured["render"]["parallelism"] == 4
    assert captured["render"]["dpi"] == 350
    assert captured["render"]["force"] is True
    assert captured["render"]["no_text"] is False
    assert captured["render"]["record_encrypted_sentinels"] is False
    assert len(captured["tasks"]) == 1
    task = captured["tasks"][0]
    assert task.image_path == str(dest_dir / "paper/page-01.png")
    assert task.text_source_path == str(dest_dir / "paper/page-01.txt")
    assert task.output_path == str(dest_dir / "paper/page-01.md")


def test_ocr_pdf_pages_reruns_stale_markdown_for_rendered_image(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.commands as commands
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    pdf_root = tmp_path / "pdfs"
    dest_dir = tmp_path / "pages"
    page_dir = dest_dir / "paper"
    pdf_root.mkdir()
    page_dir.mkdir(parents=True)
    image_path = page_dir / "page-01.png"
    markdown_path = page_dir / "page-01.md"
    markdown_path.write_text("stale markdown", encoding="utf-8")
    captured = {}

    def fake_render(**_kwargs):
        image_path.write_bytes(VALID_PNG_BYTES)
        return commands.PdfPagesRunResult(
            image_paths=(str(image_path),),
            rendered_image_paths=frozenset({str(image_path)}),
            rendered_text_image_paths=frozenset(),
        )

    class FakeProcessor:
        def __init__(self, **_kwargs):
            pass

        def run(self, tasks):
            captured["tasks"] = list(tasks)

    monkeypatch.setattr(commands, "_run_pdf_to_pages", fake_render)
    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(pdf_root)
    command.dest_dir = str(dest_dir)
    command.pdf_to_pages = True
    command._scan_parallelism = 1
    command.output_format = "md"
    command.prompt_mode = "markdown"
    command.force = False

    command.run()

    assert len(captured["tasks"]) == 1
    assert captured["tasks"][0].image_path == str(image_path)
    assert captured["tasks"][0].output_path == str(markdown_path)


def test_ocr_pdf_pages_render_ignores_stale_pages_when_no_pdfs(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.commands as commands
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pdfs"
    dest_dir = tmp_path / "pages"
    source_dir.mkdir()
    stale_dir = dest_dir / "removed-paper"
    stale_dir.mkdir(parents=True)
    (stale_dir / "page-01.png").write_bytes(VALID_PNG_BYTES)
    constructed = False

    monkeypatch.setattr(
        commands,
        "_run_pdf_to_pages",
        lambda **_kwargs: commands.PdfPagesRunResult(
            (),
            frozenset(),
            frozenset(),
        ),
    )

    class FakeProcessor:
        def __init__(self, **_kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.pdf_to_pages = True
    command._scan_parallelism = 1

    command.run()

    assert constructed is False


def test_ocr_pdf_pages_render_defaults_destination_to_source(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.commands as commands
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pdfs"
    source_dir.mkdir()
    captured = {}
    processor_constructed = False

    def fake_render(**kwargs):
        captured.update(kwargs)
        return commands.PdfPagesRunResult(
            (),
            frozenset(),
            frozenset(),
        )

    class FakeProcessor:
        def __init__(self, **_kwargs):
            nonlocal processor_constructed
            processor_constructed = True

    monkeypatch.setattr(commands, "_run_pdf_to_pages", fake_render)
    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source_dir)
    command.pdf_to_pages = True
    command._scan_parallelism = 1

    command.run()

    assert captured["source_dir"] == str(source_dir)
    assert captured["dest_dir"] == str(source_dir)
    assert captured["record_encrypted_sentinels"] is False
    assert processor_constructed is False


def test_ocr_pdf_pages_render_failure_stops_before_processor(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    import sneeze.tpn.commands as commands
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pdfs"
    dest_dir = tmp_path / "pages"
    source_dir.mkdir()
    constructed = False

    def fake_render(**_kwargs):
        (dest_dir / "partial.marker").write_text("partial", encoding="utf-8")
        raise RuntimeError("synthetic Poppler failure")

    class FakeProcessor:
        def __init__(self, **_kwargs):
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(commands, "_run_pdf_to_pages", fake_render)
    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.pdf_to_pages = True

    with pytest.raises(
        CommandError,
        match="PDF rendering failed before OCR: synthetic Poppler failure",
    ):
        command.run()

    assert constructed is False
    assert (dest_dir / "partial.marker").read_text() == "partial"


def test_ocr_pdf_pages_retry_refreshes_image_changed_before_failure(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    import sneeze.tpn.commands as commands
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pdfs"
    dest_dir = tmp_path / "pages"
    page_dir = dest_dir / "paper"
    source_dir.mkdir()
    page_dir.mkdir(parents=True)
    image_path = page_dir / "page-01.png"
    markdown_path = page_dir / "page-01.md"
    markdown_path.write_text("stale markdown", encoding="utf-8")
    os.utime(markdown_path, ns=(1_000_000_000, 1_000_000_000))
    render_calls = 0
    captured = {}

    def fake_render(**_kwargs):
        nonlocal render_calls
        render_calls += 1
        if render_calls == 1:
            image_path.write_bytes(VALID_PNG_BYTES)
            os.utime(image_path, ns=(2_000_000_000, 2_000_000_000))
            raise RuntimeError("later page failed")
        return commands.PdfPagesRunResult(
            image_paths=(str(image_path),),
            rendered_image_paths=frozenset(),
            rendered_text_image_paths=frozenset(),
        )

    class FakeProcessor:
        def __init__(self, **_kwargs):
            pass

        def run(self, tasks):
            captured["tasks"] = list(tasks)

    monkeypatch.setattr(commands, "_run_pdf_to_pages", fake_render)
    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.pdf_to_pages = True
    command._scan_parallelism = 1
    command.output_format = "md"
    command.prompt_mode = "markdown"
    command.force = False

    with pytest.raises(CommandError, match="later page failed"):
        command.run()
    command.run()

    assert render_calls == 2
    assert len(captured["tasks"]) == 1
    assert captured["tasks"][0].image_path == str(image_path)
    assert captured["tasks"][0].output_path == str(markdown_path)


def test_ocr_pdf_pages_validates_mode_before_rendering(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    import sneeze.tpn.commands as commands
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pdfs"
    dest_dir = tmp_path / "pages"
    source_dir.mkdir()
    rendered = False

    def fake_render(**_kwargs):
        nonlocal rendered
        rendered = True

    monkeypatch.setattr(commands, "_run_pdf_to_pages", fake_render)
    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.pdf_to_pages = True
    command.output_format = "html"
    command.prompt_mode = "full"

    with pytest.raises(CommandError, match="requires --output-format md"):
        command.run()

    assert rendered is False
    assert not dest_dir.exists()


@pytest.mark.parametrize(
    "option_name",
    ["scheduler_delay_factor", "gpu_memory_utilization", "temperature"],
)
def test_ocr_pdf_pages_validates_float_options_before_rendering(
    tmp_path,
    monkeypatch,
    option_name,
):
    from sneeze.command import CommandError

    import sneeze.tpn.commands as commands
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pdfs"
    dest_dir = tmp_path / "pages"
    source_dir.mkdir()
    rendered = False

    def fake_render(**_kwargs):
        nonlocal rendered
        rendered = True

    monkeypatch.setattr(commands, "_run_pdf_to_pages", fake_render)
    command = OcrPdfPages(None, StringIO(), None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.pdf_to_pages = True
    setattr(command, option_name, "not-a-number")

    with pytest.raises(CommandError, match="Expected a float"):
        command.run()

    assert rendered is False
    assert not dest_dir.exists()


def test_ocr_pdf_pages_full_mode_writes_bundle(tmp_path, monkeypatch):
    import sneeze.tpn.ocr_vllm as ocr_vllm
    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pages"
    dest_dir = tmp_path / "ocr"
    source_dir.mkdir()
    (source_dir / "page-1.png").write_bytes(VALID_PNG_BYTES)
    (source_dir / "page-1.txt").write_text("raw text", encoding="utf-8")
    captured = {"runs": []}

    class FakeProcessor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.output_format = kwargs["output_format"]
            self.prompt_mode = kwargs["prompt_mode"]
            self.use_grounding = kwargs["use_grounding"]
            self.keep_grounding = kwargs["keep_grounding"]
            self.generation_metadata = {}

        def init_progressbar(self, total=None):
            captured["total"] = total

        def close_progressbar(self):
            captured["closed"] = True

        def update_progressbar(self, amount=1):
            captured["updates"] = captured.get("updates", 0) + amount

        def prepare_engine(self):
            captured["prepared"] = True
            return object(), SimpleNamespace(
                extra_args={"ngram_size": 10, "window_size": 50}
            )

        def shutdown_engine(self, llm):
            captured["shutdown"] = llm is not None

        def run(self, tasks, *, llm=None, sampling_params=None):
            tasks = list(tasks)
            captured["runs"].append(
                {
                    "prompt_mode": self.prompt_mode,
                    "use_grounding": self.use_grounding,
                    "keep_grounding": self.keep_grounding,
                    "outputs": [task.output_path for task in tasks],
                    "llm": llm is not None,
                    "sampling_params": sampling_params is not None,
                    "window_size": sampling_params.extra_args["window_size"],
                }
            )
            for task in tasks:
                with open(
                    task.output_path,
                    "w",
                    encoding="utf-8",
                ) as handle:
                    handle.write(f"{self.prompt_mode} output\n")
                self.generation_metadata[task.output_path] = {
                    "status": "completed"
                }
            self.update_progressbar(len(tasks))

    monkeypatch.setattr(
        ocr_vllm,
        "DeepSeekOcrOfflineProcessor",
        FakeProcessor,
    )
    out = StringIO()
    command = OcrPdfPages(None, out, None)
    command._source_dir = str(source_dir)
    command.dest_dir = str(dest_dir)
    command.output_format = "md"
    command.prompt_mode = "full"
    command._scan_parallelism = 1

    command.run()

    assert captured["prepared"] is True
    assert captured["total"] == 3
    assert captured["updates"] == 3
    assert [run["prompt_mode"] for run in captured["runs"]] == [
        "plain_ocr",
        "markdown",
    ]
    assert captured["runs"][0]["use_grounding"] is True
    assert captured["runs"][0]["keep_grounding"] is True
    assert captured["runs"][1]["use_grounding"] is False
    assert captured["runs"][1]["keep_grounding"] is False
    assert captured["runs"][0]["llm"] is True
    assert captured["runs"][1]["sampling_params"] is True
    assert captured["kwargs"]["window_size"] == 50
    assert [run["window_size"] for run in captured["runs"]] == [50, 128]
    payload = json.loads((dest_dir / "page-1.json").read_text())
    assert payload["ocr_text"] == "plain_ocr output\n"
    assert payload["markdown_text"] == "markdown output\n"
    assert payload["raw_text"] == "raw text"
    assert payload["image_dims"] == {"w": 1, "h": 1}


def test_full_mode_local_provider_releases_engine_before_markdown(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_provider as ocr_provider
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_vllm import OcrFullItem

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    text_path = tmp_path / "page.txt"
    text_path.write_text("Table 8 rows Table 9 rows", encoding="utf-8")
    item = OcrFullItem(
        image_path=str(image_path),
        base="paper/page",
        ocr_path=str(tmp_path / "page.ocr"),
        markdown_path=str(tmp_path / "page.md"),
        json_path=str(tmp_path / "page.json"),
        text_path=str(text_path),
        need_ocr=True,
        need_markdown=True,
        need_json=True,
        image_dims={"w": 1, "h": 1},
        image_sha256=(
            "5c9b6e9f825b1ad005cbea8e70758bb8171d43bb66425b430254b9a0ef972d56"
        ),
    )
    events = []

    class FakeProcessor:
        def __init__(self):
            self.generation_metadata = {}

        def init_progressbar(self, total=None):
            events.append(("progress", total))

        def close_progressbar(self):
            events.append(("close", None))

        def update_progressbar(self, amount=1):
            events.append(("update", amount))

        def prepare_engine(self):
            events.append(("prepare", None))
            return object(), object()

        def shutdown_engine(self, llm):
            if llm is not None:
                events.append(("shutdown", None))

        def run(self, tasks, *, llm, sampling_params):
            tasks = list(tasks)
            events.append(("grounded", len(tasks)))
            assert all(task.output_path.endswith(".ocr") for task in tasks)
            for task in tasks:
                with open(task.output_path, "w", encoding="utf-8") as handle:
                    handle.write("Table 8 rows Table 9 rows")
                self.generation_metadata[task.output_path] = {
                    "status": "completed"
                }

    def fake_provider_batch(**kwargs):
        events.append(("provider", len(kwargs["pages"])))
        page = kwargs["pages"][0]
        return {
            "ok": True,
            "protocol_errors": [],
            "results": [
                {
                    "page_id": page["page_id"],
                    "valid": True,
                    "errors": [],
                    "output_text": "Table 8\nrows\n\nTable 9\nrows\n",
                    "normalized_metadata": {
                        "status": "completed",
                        "backend": "local_provider",
                        "termination": {
                            "eos": True,
                            "length": False,
                            "max_tokens": False,
                        },
                    },
                }
            ],
        }

    monkeypatch.setattr(
        ocr_provider,
        "run_local_provider_batch",
        fake_provider_batch,
    )
    provider = {
        "spec_path": str(tmp_path / "provider.json"),
        "spec": {
            "lanes": {
                "best": {
                    "prompt": "native two-stage",
                    "settings": {"backend": "official-two-step-transformers"},
                }
            }
        },
        "lane": "best",
        "run_dir": str(tmp_path / "evidence"),
    }
    command = OcrPdfPages(None, StringIO(), None)
    command._run_full_mode(
        FakeProcessor(),
        [item],
        markdown_provider=provider,
    )

    labels = [event[0] for event in events]
    assert labels.index("shutdown") < labels.index("provider")
    assert labels.count("grounded") == 1
    assert (tmp_path / "page.md").read_text() == (
        "Table 8\nrows\n\nTable 9\nrows\n"
    )
    payload = json.loads((tmp_path / "page.json").read_text())
    assert payload["generation_metadata"]["markdown"]["backend"] == (
        "local_provider"
    )
    promotion = payload["generation_metadata"]["markdown"]["promotion"]
    assert promotion["promoted"] is True
    assert (
        promotion["candidate_sha256"] == promotion["persisted_output_sha256"]
    )
    assert payload["markdown_quality"]["status"] == "pass"


def test_full_mode_local_provider_rejects_candidate_atomically(
    tmp_path,
    monkeypatch,
):
    import sneeze.tpn.ocr_provider as ocr_provider
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_vllm import OcrFullItem

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    ocr_path = tmp_path / "page.ocr"
    ocr_path.write_text("real page prose", encoding="utf-8")
    markdown_path = tmp_path / "page.md"
    markdown_path.write_text(
        "previous validated markdown\n",
        encoding="utf-8",
    )
    item = OcrFullItem(
        image_path=str(image_path),
        base="paper/page",
        ocr_path=str(ocr_path),
        markdown_path=str(markdown_path),
        json_path=str(tmp_path / "page.json"),
        text_path=None,
        need_ocr=False,
        need_markdown=True,
        need_json=True,
        image_dims={"w": 1, "h": 1},
        image_sha256=(
            "5c9b6e9f825b1ad005cbea8e70758bb8171d43bb66425b430254b9a0ef972d56"
        ),
    )

    class FakeProcessor:
        def __init__(self):
            self.generation_metadata = {}

        def init_progressbar(self, total=None):
            pass

        def close_progressbar(self):
            pass

        def update_progressbar(self, amount=1):
            pass

        def prepare_engine(self):
            raise AssertionError("provider-only Markdown must not start vLLM")

        def shutdown_engine(self, llm):
            pass

    def fake_provider_batch(**kwargs):
        page = kwargs["pages"][0]
        axis = " ".join(str(value) for value in range(20, 2226, 2))
        return {
            "ok": True,
            "protocol_errors": [],
            "results": [
                {
                    "page_id": page["page_id"],
                    "valid": True,
                    "errors": [],
                    "output_text": axis,
                    "normalized_metadata": {
                        "status": "completed",
                        "backend": "local_provider",
                        "termination": {
                            "eos": True,
                            "length": False,
                            "max_tokens": False,
                        },
                    },
                }
            ],
        }

    monkeypatch.setattr(
        ocr_provider,
        "run_local_provider_batch",
        fake_provider_batch,
    )
    provider = {
        "spec_path": str(tmp_path / "provider.json"),
        "spec": {
            "lanes": {
                "best": {
                    "prompt": "native two-stage",
                    "settings": {"backend": "official-two-step-transformers"},
                }
            }
        },
        "lane": "best",
        "run_dir": str(tmp_path / "evidence"),
    }
    command = OcrPdfPages(None, StringIO(), None)
    with pytest.raises(
        RuntimeError, match="runaway_monotonic_integer_axis"
    ) as exc_info:
        command._run_full_mode(
            FakeProcessor(),
            [item],
            markdown_provider=provider,
        )

    assert markdown_path.read_text() == "previous validated markdown\n"
    payload = json.loads((tmp_path / "page.json").read_text())
    markdown_metadata = payload["generation_metadata"]["markdown"]
    assert markdown_metadata["status"] == "error"
    assert markdown_metadata["validation"]["flags"] == [
        "runaway_monotonic_integer_axis"
    ]
    promotion = markdown_metadata["promotion"]
    assert promotion["promoted"] is False
    assert (
        promotion["candidate_sha256"] != promotion["persisted_output_sha256"]
    )
    assert (
        promotion["persisted_output_sha256"]
        == hashlib.sha256(b"previous validated markdown\n").hexdigest()
    )
    assert "1 page(s)" in str(exc_info.value)
    assert str(markdown_path) in str(exc_info.value)
    assert str(exc_info.value).count(str(markdown_path)) == 1


def test_markdown_provider_preparation_resolves_spec(
    tmp_path,
    monkeypatch,
):
    from sneeze.tpn.commands import OcrPdfPages

    home = tmp_path / "home"
    home.mkdir()
    spec_path = home / "provider.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provider_id": "fake-local-provider",
                "argv": [sys.executable],
                "egress": "none",
                "model": {
                    "requested": "example/model",
                    "revision": "a" * 40,
                },
                "lanes": {
                    "best": {
                        "prompt": "native pipeline",
                        "settings": {},
                    }
                },
                "secret_env_names": [],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    command = OcrPdfPages(None, StringIO(), None)
    command.markdown_provider_spec = "~/provider.json"
    command.markdown_provider_lane = "best"
    command.markdown_provider_run_dir = str(tmp_path / "evidence")

    provider = command._prepare_markdown_provider(
        prompt_mode="full",
        source_dir=str(tmp_path / "source"),
        dest_dir=str(tmp_path / "dest"),
    )
    assert provider["spec_path"] == str(spec_path.resolve())


def test_provider_evidence_and_profile_runtime_reject_protected_corpora(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    from sneeze.tpn.commands import (
        _create_profile_runtime_root,
        _preflight_markdown_provider_run_dir,
    )

    protected = tmp_path / "canonical" / "papers"
    protected.mkdir(parents=True)
    config_home = tmp_path / "config"
    policy = config_home / "sneeze" / "tpn" / "ocr-protected-roots"
    policy.parent.mkdir(parents=True)
    policy.write_text(f"{protected}\n", encoding="utf-8")
    policy.chmod(0o600)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))

    source_dir = str(tmp_path / "source")
    dest_dir = str(tmp_path / "dest")
    evidence = protected / "evidence"
    with pytest.raises(CommandError, match="protected corpora"):
        _preflight_markdown_provider_run_dir(
            evidence,
            source_dir=source_dir,
            dest_dir=dest_dir,
        )
    assert not evidence.exists()

    with pytest.raises(CommandError, match="protected corpora"):
        _create_profile_runtime_root(
            protected,
            source_dir=source_dir,
            dest_dir=dest_dir,
        )
    assert list(protected.glob(".sneeze-ocr-profile-*")) == []


def test_structured_ocr_rejects_provider_lane_before_runtime_side_effects(
    tmp_path,
    monkeypatch,
):
    from sneeze.command import CommandError

    from sneeze.tpn import ocr_installations
    from sneeze.tpn.commands import OcrPdfPages

    calls = []

    def unexpected_registry_load(*_args, **_kwargs):
        calls.append("registry")
        raise AssertionError("registry must not be read for a CLI conflict")

    monkeypatch.setattr(
        ocr_installations,
        "load_installation_registry",
        unexpected_registry_load,
    )
    evidence_dir = tmp_path / "provider-evidence"
    command = OcrPdfPages(None, StringIO(), None)
    command.pipeline = "local-mineru25"
    command.markdown_provider_lane = "parity"
    command.markdown_provider_run_dir = str(evidence_dir)

    with pytest.raises(CommandError, match="--markdown-provider-lane"):
        command._resolve_structured_execution(
            prompt_mode="markdown",
            output_format="md",
        )

    assert calls == []
    assert not evidence_dir.exists()


def test_atomic_write_text_preserves_or_applies_normal_file_mode(tmp_path):
    from sneeze.tpn.commands import _atomic_write_text

    output_path = tmp_path / "page.md"
    previous_umask = os.umask(0o027)
    try:
        _atomic_write_text(str(output_path), "first\n")
    finally:
        os.umask(previous_umask)
    assert output_path.stat().st_mode & 0o777 == 0o640

    output_path.chmod(0o604)
    _atomic_write_text(str(output_path), "second\n")
    assert output_path.read_text(encoding="utf-8") == "second\n"
    assert output_path.stat().st_mode & 0o777 == 0o604


def test_ocr_pdf_pages_full_rejects_html_output(tmp_path):
    from sneeze.command import CommandError

    from sneeze.tpn.commands import OcrPdfPages

    source_dir = tmp_path / "pages"
    source_dir.mkdir()
    out = StringIO()
    command = OcrPdfPages(None, out, None)
    command._source_dir = str(source_dir)
    command.output_format = "html"
    command.prompt_mode = "full"

    with pytest.raises(CommandError, match="requires --output-format md"):
        command.run()


def test_bootstrap_commands_support_dry_run():
    from sneeze.tpn.commands import (
        DeepseekOcrCreateEnv,
        DeepseekOcrDownloadModel,
    )

    out = StringIO()
    command = DeepseekOcrCreateEnv(None, out, None)
    command.manager = "mamba"
    command.env_name = "ocr-env"
    command.python = "3.12"
    command.torch_backend = "cu130"
    command.dry_run = True
    command.run()
    text = out.getvalue()
    assert "mamba create -n ocr-env" in text
    assert "wheels.vllm.ai/nightly/cu130" in text

    out = StringIO()
    command = DeepseekOcrDownloadModel(None, out, None)
    command.manager = "mamba"
    command.env_name = "ocr-env"
    command.dry_run = True
    command.run()
    text = out.getvalue()
    assert "mamba run -n ocr-env python -c" in text
    assert "deepseek-ai/DeepSeek-OCR" in text


def test_command_docstrings_explain_workflow_prerequisites():
    from sneeze.tpn.commands import (
        DeepseekOcrCreateEnv,
        DeepseekOcrDownloadModel,
        OcrPdfPages,
        PdfToPages,
    )

    assert "pdftocairo" in PdfToPages.__doc__
    assert "pdftotext" in PdfToPages.__doc__
    assert "sne ocr-pdf-pages" in PdfToPages.__doc__
    assert "sne pdf-to-pages" in OcrPdfPages.__doc__
    assert "--pdf-to-pages" in OcrPdfPages.__doc__
    assert "Recommended" in OcrPdfPages.__doc__
    assert "--pipeline local-mineru25" in OcrPdfPages.__doc__
    assert "sne ocr-backend-list" in OcrPdfPages.__doc__
    assert "sne ocr-profile-list" in OcrPdfPages.__doc__
    assert "Legacy manual mode" in OcrPdfPages.__doc__
    assert "sne deepseek-ocr-create-env" in OcrPdfPages.__doc__
    assert "sne deepseek-ocr-download-model" in OcrPdfPages.__doc__
    assert OcrPdfPages.__doc__.index("Recommended") < (
        OcrPdfPages.__doc__.index("Legacy manual mode")
    )
    assert "sne ocr-pdf-pages" in DeepseekOcrCreateEnv.__doc__
    assert "sne ocr-pdf-pages" in DeepseekOcrDownloadModel.__doc__


def test_ocr_pdf_pages_help_leads_with_structured_pipeline(capsys):
    from sneeze import cli as sneeze_cli

    cli = sneeze_cli.CLI(
        program_name="sne",
        module_names=["sneeze", "sneeze.tpn"],
        introspect=True,
        auto_plugins=False,
    )

    with pytest.raises(SystemExit) as exc_info:
        cli._commands_by_name["ocr-pdf-pages"].run(["--help"])

    assert exc_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "Recommended: select `--pipeline local-mineru25`" in help_text
    assert "highest supported tier" in help_text
    assert "Legacy manual mode" in help_text
    assert (
        "Comma-separated absolute profile JSON files or directories."
        in help_text
    )
    assert "Defaults to the highest supported tier." in help_text
    assert (
        "Legacy manual mode only: DeepSeek OCR model name or local path; "
        "structured pipelines pin the model." in help_text
    )
    assert (
        "Legacy manual mode only: immutable Hugging Face model revision; "
        "structured pipelines pin the revision." in help_text
    )
    assert help_text.index("Recommended:") < help_text.index(
        "Legacy manual mode"
    )


def test_markdown_provider_options_belong_only_to_ocr_command():
    from sneeze.tpn.commands import OcrPdfPages, PdfToPages

    for argument in (
        "MarkdownProviderSpecArg",
        "MarkdownProviderLaneArg",
        "MarkdownProviderRunDirArg",
    ):
        assert hasattr(OcrPdfPages, argument)
        assert not hasattr(PdfToPages, argument)


def test_combined_pdf_options_belong_to_ocr_command():
    from sneeze.tpn.commands import OcrPdfPages

    for argument in (
        "PdfToPagesArg",
        "PdfRecursiveArg",
        "PdfParallelismArg",
        "PdfDpiArg",
    ):
        assert hasattr(OcrPdfPages, argument)


def test_combined_pdf_options_preserve_legacy_short_cli_mappings(
    tmp_path,
    monkeypatch,
):
    from sneeze import cli as sneeze_cli
    from sneeze.tpn.commands import OcrPdfPages

    captured = {}

    def fake_run(self):
        captured.update(
            prompt=self.prompt,
            prompt_mode=self.prompt_mode,
            offset=self._offset,
            scan_parallelism=self._scan_parallelism,
            attention_backend=self.attention_backend,
            pdf_to_pages=self.pdf_to_pages,
            pdf_recursive=self.pdf_recursive,
            pdf_parallelism=self._pdf_parallelism,
            pdf_dpi=self._pdf_dpi,
        )

    monkeypatch.setattr(OcrPdfPages, "run", fake_run)
    cli = sneeze_cli.CLI(
        program_name="sne",
        module_names=["sneeze", "sneeze.tpn"],
        introspect=True,
        auto_plugins=False,
    )
    commandline = cli._commands_by_name["ocr-pdf-pages"]
    commandline.run(
        [
            "--source-dir",
            str(tmp_path),
            "-p",
            "legacy prompt",
            "-r",
            "full",
            "-f",
            "2",
            "-a",
            "3",
            "-i",
            "FLASH_ATTN",
            "--pdf-to-pages",
            "--pdf-recursive",
            "--pdf-parallelism",
            "4",
            "--pdf-dpi",
            "350",
        ]
    )

    assert captured == {
        "prompt": "legacy prompt",
        "prompt_mode": "full",
        "offset": 2,
        "scan_parallelism": 3,
        "attention_backend": "FLASH_ATTN",
        "pdf_to_pages": True,
        "pdf_recursive": True,
        "pdf_parallelism": 4,
        "pdf_dpi": 350,
    }


def test_documented_commands_are_visible_to_sneeze_cli():
    from sneeze import cli as sneeze_cli

    cli = sneeze_cli.CLI(
        program_name="sneeze",
        module_names=["sneeze", "sneeze.tpn"],
        introspect=True,
        auto_plugins=False,
    )

    for command_name in (
        "pdf-to-pages",
        "pdfs-to-pages",
        "ocr-pdf-pages",
        "ocr-benchmark",
        "deepseek-ocr-create-env",
        "deepseek-ocr-download-model",
    ):
        assert command_name in cli._commands_by_name


def test_markdown_quality_detects_three_calibrated_failure_classes():
    from sneeze.tpn.ocr_vllm import analyze_markdown_quality

    axis = " ".join(str(value) for value in range(20, 2226, 2))
    quality = analyze_markdown_quality(
        axis,
        ocr_text="real page prose",
        raw_text="real page prose",
    )
    assert quality["flags"] == ["runaway_monotonic_integer_axis"]

    invented = " ".join(f"invented{index}" for index in range(50))
    quality = analyze_markdown_quality(
        "\n\n".join([invented] * 4),
        ocr_text="grounded evidence from the page",
        raw_text="grounded evidence from the page",
    )
    assert quality["flags"] == ["unsupported_long_span_repetition"]

    raw = "Table 8\n" + " ".join(f"row{index}" for index in range(80))
    raw += "\nTable 9\nmore table evidence"
    quality = analyze_markdown_quality(
        "| 1 | 2 |\n| 3 | 4 |",
        ocr_text=raw,
        raw_text=raw,
    )
    assert quality["flags"] == ["premature_table_truncation"]


def test_markdown_quality_accepts_supported_repetition():
    from sneeze.tpn.ocr_vllm import analyze_markdown_quality

    caption = " ".join(f"caption{index}" for index in range(50))
    quality = analyze_markdown_quality(
        "\n\n".join([caption] * 4),
        ocr_text=caption,
        raw_text=caption,
    )
    assert quality["status"] == "pass"
    assert quality["flags"] == []


def test_grounded_recall_ignores_layout_markers_but_keeps_table_numbers():
    from sneeze.tpn.ocr_vllm import table_completion_metrics

    reference = """\
table[[361, 519, 706, 774]]
table_caption[[115, 778, 881, 792]]
Table 9
<table>Dense24 relus Dense16 relu Dense8 relus value 519</table>
"""
    candidate = """\
table[[361, 518, 706, 772]]
table_caption[[115, 777, 881, 791]]
Table 9
<table>Dense24 relus Dense16 relu Dense8 relus value 519</table>
"""

    metrics = table_completion_metrics(candidate, None, reference)

    assert metrics["grounded_ocr_token_multiset_recall"] == 1.0

    missing_layout_block = candidate.replace(
        "table_caption[[115, 777, 881, 791]]\n", ""
    )
    metrics = table_completion_metrics(missing_layout_block, None, reference)

    assert metrics["grounded_ocr_token_multiset_recall"] == 1.0

    missing_table_value = candidate.replace("value 519", "value 518")
    metrics = table_completion_metrics(missing_table_value, None, reference)

    assert metrics["grounded_ocr_token_multiset_recall"] < 1.0


def test_generation_metadata_keeps_vllm_completion_details(tmp_path):
    from sneeze.tpn.ocr_vllm import (
        DeepSeekOcrOfflineProcessor,
        OcrPageTask,
    )

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    processor = DeepSeekOcrOfflineProcessor(
        model="deepseek-ai/DeepSeek-OCR",
        output_format="md",
        prompt_mode="markdown",
    )
    processor._engine_metadata = {
        "model": {"snapshot": "snapshot-sha"},
        "runtime": {"attention_backend_selected": "FLASH_ATTN"},
    }
    completion = SimpleNamespace(
        text="short output",
        token_ids=[10, 11, 12],
        finish_reason="length",
        stop_reason=None,
    )
    request_output = SimpleNamespace(
        request_id="request-1",
        prompt_token_ids=[1, 2],
    )
    sampling_params = SimpleNamespace(
        temperature=0.0,
        max_tokens=3,
        extra_args={"ngram_size": 10, "window_size": 50},
    )

    metadata = processor._generation_metadata(
        task=OcrPageTask(
            image_path=str(image_path),
            output_path=str(tmp_path / "page.md"),
        ),
        prompt="<image>\nConvert the document to markdown.",
        request_output=request_output,
        completion=completion,
        sampling_params=sampling_params,
    )

    assert metadata["image"]["dimensions"] == {"w": 1, "h": 1}
    assert metadata["prompt_mode"] == "markdown"
    assert metadata["generated_token_count"] == 3
    assert metadata["finish_reason"] == "length"
    assert metadata["status"] == "completed"
    assert metadata["termination"] == {
        "eos": False,
        "length": True,
        "max_tokens": True,
    }
    assert metadata["runtime"]["attention_backend_selected"] == "FLASH_ATTN"
    assert metadata["logits_processor_extra_args"] == {
        "ngram_size": 10,
        "window_size": 50,
    }

    completion.token_ids = [10, 11]
    metadata = processor._generation_metadata(
        task=OcrPageTask(
            image_path=str(image_path),
            output_path=str(tmp_path / "page.md"),
        ),
        prompt="<image>\nConvert the document to markdown.",
        request_output=request_output,
        completion=completion,
        sampling_params=sampling_params,
    )
    assert metadata["termination"] == {
        "eos": False,
        "length": True,
        "max_tokens": False,
    }


def test_full_mode_preserves_and_rejects_flagged_markdown(tmp_path):
    from sneeze.tpn.commands import OcrPdfPages
    from sneeze.tpn.ocr_vllm import OcrFullItem

    image_path = tmp_path / "page.png"
    image_path.write_bytes(VALID_PNG_BYTES)
    item = OcrFullItem(
        image_path=str(image_path),
        base="page",
        ocr_path=str(tmp_path / "page.ocr"),
        markdown_path=str(tmp_path / "page.md"),
        json_path=str(tmp_path / "page.json"),
        text_path=str(tmp_path / "page.txt"),
        need_ocr=True,
        need_markdown=True,
        need_json=True,
        image_dims={"w": 1, "h": 1},
    )
    (tmp_path / "page.txt").write_text("real prose", encoding="utf-8")

    class FakeProcessor:
        generation_metadata = {}

        def init_progressbar(self, total=None):
            pass

        def prepare_engine(self):
            return object(), SimpleNamespace(
                extra_args={"ngram_size": 10, "window_size": 50}
            )

        def run(self, tasks, *, llm, sampling_params):
            for task in tasks:
                if task.output_path.endswith(".ocr"):
                    text = "real prose"
                else:
                    text = " ".join(
                        str(value) for value in range(20, 2226, 2)
                    )
                with open(task.output_path, "w", encoding="utf-8") as handle:
                    handle.write(text)

        def update_progressbar(self, amount=1):
            pass

        def close_progressbar(self):
            pass

        def shutdown_engine(self, llm):
            pass

    command = OcrPdfPages(None, StringIO(), None)
    with pytest.raises(
        RuntimeError,
        match="page validation failed",
    ) as exc_info:
        command._run_full_mode(FakeProcessor(), [item])

    assert "grounded_ocr_generation_failed" in str(exc_info.value)
    assert "markdown_generation_failed" in str(exc_info.value)

    payload = json.loads((tmp_path / "page.json").read_text())
    assert payload["markdown_quality"]["status"] == "fail"
    assert payload["markdown_quality"]["flags"] == [
        "runaway_monotonic_integer_axis"
    ]
