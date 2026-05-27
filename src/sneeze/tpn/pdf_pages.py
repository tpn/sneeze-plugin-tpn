from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

ENCRYPTED_SENTINEL_SUFFIX = ".encrypted"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND_CHUNK = b"\x00\x00\x00\x00IEND\xaeB`\x82"
DEFAULT_RENDER_DPI = 400


@dataclass(frozen=True)
class PdfPagesTask:
    source_path: str
    dest_dir: str
    text_dest_dir: str | None = None
    page_count: int | None = None


@dataclass(frozen=True)
class PdfPageTask:
    source_path: str
    dest_dir: str
    page_num: int
    page_count: int
    image_output_path: str
    render_image: bool = True
    render_text: bool = False
    text_output_path: str | None = None
    dpi: int = DEFAULT_RENDER_DPI

    @property
    def image_output_prefix(self) -> str:
        return os.path.splitext(self.image_output_path)[0]

    @property
    def page_stem(self) -> str:
        return os.path.splitext(os.path.basename(self.image_output_path))[0]

    @property
    def text_dest_dir(self) -> str | None:
        if self.text_output_path is None:
            return None
        return os.path.dirname(self.text_output_path)


@dataclass(frozen=True)
class PdfPreparationError:
    source_path: str
    error: str
    kind: str = "unreadable"
    sentinel_path: str | None = None

    @property
    def is_encrypted(self) -> bool:
        return self.kind in {"encrypted", "encrypted_sentinel"}

    @property
    def needs_sentinel(self) -> bool:
        return self.kind == "encrypted" and self.sentinel_path is not None


def list_pdf_paths(source_dir: str, recursive: bool) -> list[str]:
    paths = []
    if recursive:
        for root, _, files in os.walk(source_dir):
            for filename in files:
                if filename.lower().endswith(".pdf"):
                    paths.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(source_dir):
            if not filename.lower().endswith(".pdf"):
                continue
            path = os.path.join(source_dir, filename)
            if os.path.isfile(path):
                paths.append(path)
    return sorted(paths)


def encrypted_sentinel_path(pdf_path: str) -> str:
    return f"{pdf_path}{ENCRYPTED_SENTINEL_SUFFIX}"


def has_encrypted_sentinel(pdf_path: str) -> bool:
    return os.path.exists(encrypted_sentinel_path(pdf_path))


def inspect_png_file(path: str) -> str | None:
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return f"unreadable: {exc}"
    if size == 0:
        return "empty"
    min_size = len(PNG_SIGNATURE) + len(PNG_IEND_CHUNK)
    if size < min_size:
        return "too_small"
    try:
        with open(path, "rb") as handle:
            if handle.read(len(PNG_SIGNATURE)) != PNG_SIGNATURE:
                return "bad_signature"
            handle.seek(-len(PNG_IEND_CHUNK), os.SEEK_END)
            if handle.read(len(PNG_IEND_CHUNK)) != PNG_IEND_CHUNK:
                return "missing_iend"
    except OSError as exc:
        return f"unreadable: {exc}"
    return None


def build_pdf_pages_tasks(
    source_dir: str,
    dest_dir: str,
    *,
    recursive: bool,
    limit: int | None = None,
    text_root: str | None = None,
) -> list[PdfPagesTask]:
    source_root = os.path.abspath(source_dir)
    dest_root = os.path.abspath(dest_dir)
    text_root_abs = os.path.abspath(text_root) if text_root else None
    pdf_paths = list_pdf_paths(source_root, recursive)
    if limit:
        pdf_paths = pdf_paths[:limit]

    tasks = []
    for pdf_path in pdf_paths:
        rel_dir = os.path.relpath(os.path.dirname(pdf_path), source_root)
        if rel_dir == ".":
            rel_dir = ""
        pdf_base = os.path.splitext(os.path.basename(pdf_path))[0]
        target_root = (
            os.path.join(dest_root, rel_dir)
            if rel_dir
            else dest_root
        )
        text_dest_dir = None
        if text_root_abs:
            text_root = (
                os.path.join(text_root_abs, rel_dir)
                if rel_dir
                else text_root_abs
            )
            text_dest_dir = os.path.join(text_root, pdf_base)
        tasks.append(
            PdfPagesTask(
                source_path=pdf_path,
                dest_dir=os.path.join(target_root, pdf_base),
                text_dest_dir=text_dest_dir,
            )
        )
    return tasks


def get_pdf_page_count(pdf_path: str) -> int:
    try:
        import pdf2image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pdf2image is required to inspect PDFs. Install "
            "`sneeze-plugin-tpn[pdf]` and Poppler."
        ) from exc
    try:
        info = pdf2image.pdfinfo_from_path(pdf_path)
    except Exception as exc:
        raise RuntimeError(
            f"Unable to determine page count for {pdf_path}: {exc}"
        ) from exc
    pages = info.get("Pages", info.get("pages"))
    if pages is None:
        raise RuntimeError(f"Unable to determine page count for {pdf_path}")
    try:
        return int(pages)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Unable to determine page count for {pdf_path}: {pages!r}"
        ) from exc


def prepare_pdf_pages_tasks(
    pdf_tasks: list[PdfPagesTask],
    *,
    max_workers: int | None = 1,
    show_progress: bool = True,
) -> tuple[list[PdfPagesTask], list[PdfPreparationError]]:
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - optional dependency
        tqdm = None

    worker_count = int(max_workers or 1)
    if worker_count <= 0:
        worker_count = os.cpu_count() or 1

    def prepare_one(
        task: PdfPagesTask,
    ) -> tuple[PdfPagesTask | None, PdfPreparationError | None]:
        sentinel_path = encrypted_sentinel_path(task.source_path)
        if has_encrypted_sentinel(task.source_path):
            return None, PdfPreparationError(
                source_path=task.source_path,
                error=f"Previously marked encrypted at {sentinel_path}",
                kind="encrypted_sentinel",
                sentinel_path=sentinel_path,
            )
        try:
            page_count = get_pdf_page_count(task.source_path)
        except Exception as exc:
            error = str(exc)
            kind = (
                "encrypted"
                if _is_encrypted_error_message(error)
                else "unreadable"
            )
            return None, PdfPreparationError(
                source_path=task.source_path,
                error=error,
                kind=kind,
                sentinel_path=sentinel_path if kind == "encrypted" else None,
            )
        return replace(task, page_count=page_count), None

    prepared: list[PdfPagesTask] = []
    skipped: list[PdfPreparationError] = []
    if worker_count <= 1:
        iterator = pdf_tasks
        progress = None
        if show_progress and tqdm is not None:
            progress = tqdm(
                iterator,
                total=len(pdf_tasks),
                desc="Preparing PDFs",
            )
            iterator = progress
        try:
            for task in iterator:
                ready, error = prepare_one(task)
                if ready is not None:
                    prepared.append(ready)
                if error is not None:
                    skipped.append(error)
        finally:
            if progress is not None:
                progress.close()
        return prepared, skipped

    progress = None
    if show_progress and tqdm is not None:
        progress = tqdm(total=len(pdf_tasks), desc="Preparing PDFs")
    try:
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="sneeze-pdf",
        ) as executor:
            for ready, error in executor.map(prepare_one, pdf_tasks):
                if ready is not None:
                    prepared.append(ready)
                if error is not None:
                    skipped.append(error)
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()
    return prepared, skipped


def write_encrypted_sentinels(
    skipped: list[PdfPreparationError],
) -> list[str]:
    written = []
    for item in skipped:
        if not item.needs_sentinel or item.sentinel_path is None:
            continue
        with open(item.sentinel_path, "wb"):
            pass
        written.append(item.sentinel_path)
    return written


def build_pdf_page_tasks(
    pdf_tasks: list[PdfPagesTask],
    *,
    extract_text: bool,
    force: bool,
    dpi: int = DEFAULT_RENDER_DPI,
) -> list[PdfPageTask]:
    tasks = []
    for pdf_task in pdf_tasks:
        page_count = (
            pdf_task.page_count
            or get_pdf_page_count(pdf_task.source_path)
        )
        digits = _page_digits(page_count)
        images_complete = False
        text_complete = not extract_text
        if extract_text and pdf_task.text_dest_dir is None:
            raise RuntimeError(
                "text_dest_dir is required for text extraction"
            )
        if not force:
            images_complete = pages_already_rendered(
                pdf_task.dest_dir,
                page_count,
            )
            if extract_text:
                text_complete = text_pages_already_rendered(
                    pdf_task.text_dest_dir or "",
                    page_count,
                    digits,
                )
            if images_complete and text_complete:
                continue
        for page_num in range(1, page_count + 1):
            render_image = force or (
                not images_complete
                and not page_image_exists(pdf_task.dest_dir, page_num, digits)
            )
            text_output_path = None
            render_text = False
            if extract_text:
                text_output_path = page_text_path(
                    pdf_task.text_dest_dir or "",
                    page_num,
                    digits,
                )
                render_text = force or (
                    not text_complete and not os.path.exists(text_output_path)
                )
            if not render_image and not render_text:
                continue
            tasks.append(
                PdfPageTask(
                    source_path=pdf_task.source_path,
                    dest_dir=pdf_task.dest_dir,
                    page_num=page_num,
                    page_count=page_count,
                    image_output_path=preferred_page_image_path(
                        pdf_task.dest_dir,
                        page_num,
                        digits,
                    ),
                    render_image=render_image,
                    render_text=render_text,
                    text_output_path=text_output_path,
                    dpi=dpi,
                )
            )
    return tasks


def extract_pdf_page_and_text(task: PdfPageTask) -> bool:
    did_work = False
    if task.render_image:
        os.makedirs(task.dest_dir, exist_ok=True)
        args = [
            "pdftocairo",
            "-png",
            "-r",
            str(task.dpi),
            "-singlefile",
            "-f",
            str(task.page_num),
            "-l",
            str(task.page_num),
            task.source_path,
            task.image_output_prefix,
        ]
        subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        did_work = True
    if task.render_text:
        if task.text_output_path is None:
            raise RuntimeError(
                "text_output_path is required for text extraction"
            )
        os.makedirs(task.text_dest_dir or "", exist_ok=True)
        args = [
            "pdftotext",
            "-f",
            str(task.page_num),
            "-l",
            str(task.page_num),
            task.source_path,
            task.text_output_path,
        ]
        subprocess.run(
            args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        did_work = True
    return did_work


def run_tasks_in_threads(
    tasks: list,
    fn: Callable,
    *,
    max_workers: int,
    desc: str,
) -> None:
    try:
        from tqdm import tqdm
    except Exception:  # pragma: no cover - optional dependency
        tqdm = None
    workers = max(1, int(max_workers or 1))
    if workers <= 1:
        iterator = tasks
        progress = None
        if tqdm is not None:
            progress = tqdm(iterator, total=len(tasks), desc=desc)
            iterator = progress
        try:
            for task in iterator:
                fn(task)
        finally:
            if progress is not None:
                progress.close()
        return
    progress = None
    if tqdm is not None:
        progress = tqdm(total=len(tasks), desc=desc)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for _ in executor.map(fn, tasks):
                if progress is not None:
                    progress.update(1)
    finally:
        if progress is not None:
            progress.close()


def page_image_exists(dest_dir: str, page_num: int, digits: int) -> bool:
    for path in page_image_path_candidates(dest_dir, page_num, digits):
        if os.path.exists(path) and inspect_png_file(path) is None:
            return True
    return False


def preferred_page_image_path(
    dest_dir: str,
    page_num: int,
    digits: int,
) -> str:
    invalid = None
    for path in page_image_path_candidates(dest_dir, page_num, digits):
        if not os.path.exists(path):
            continue
        if inspect_png_file(path) is None:
            return path
        if invalid is None:
            invalid = path
    return invalid or page_image_path(dest_dir, page_num, digits)


def pages_already_rendered(dest_dir: str, page_count: int | None) -> bool:
    if not page_count or page_count <= 0:
        return True
    if not os.path.isdir(dest_dir):
        return False
    digits = _page_digits(page_count)
    return all(
        page_image_exists(dest_dir, page_num, digits)
        for page_num in range(1, page_count + 1)
    )


def text_pages_already_rendered(
    dest_dir: str,
    page_count: int,
    digits: int,
) -> bool:
    if page_count <= 0:
        return True
    if not os.path.isdir(dest_dir):
        return False
    return all(
        os.path.exists(page_text_path(dest_dir, page_num, digits))
        for page_num in range(1, page_count + 1)
    )


def page_image_path(dest_dir: str, page_num: int, digits: int) -> str:
    return os.path.join(dest_dir, f"{page_stem(page_num, digits)}.png")


def page_image_path_candidates(
    dest_dir: str,
    page_num: int,
    digits: int,
) -> tuple[str, ...]:
    padded = page_image_path(dest_dir, page_num, digits)
    plain = os.path.join(dest_dir, f"page-{page_num}.png")
    if plain == padded:
        return (padded,)
    return (padded, plain)


def page_text_path(dest_dir: str, page_num: int, digits: int) -> str:
    return os.path.join(dest_dir, f"{page_stem(page_num, digits)}.txt")


def page_stem(page_num: int, digits: int) -> str:
    return f"page-{str(page_num).zfill(digits)}"


def _page_digits(page_count: int) -> int:
    return max(1, len(str(page_count)))


def _is_encrypted_error_message(error: str) -> bool:
    text = error.lower()
    markers = (
        "incorrect password",
        "encrypted",
        "requires password",
        "owner password",
    )
    return any(marker in text for marker in markers)
