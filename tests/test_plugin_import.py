import base64
from io import StringIO
from types import SimpleNamespace

VALID_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+Xk6cAAAAASUVORK5CYII="
)


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

    assert captured["kwargs"]["max_tokens"] == 64
    assert captured["kwargs"]["temperature"] == 0.2
    assert len(captured["tasks"]) == 1
    assert captured["tasks"][0].output_path == str(dest_dir / "page-1.md")


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
        "deepseek-ocr-create-env",
        "deepseek-ocr-download-model",
    ):
        assert command_name in cli._commands_by_name
