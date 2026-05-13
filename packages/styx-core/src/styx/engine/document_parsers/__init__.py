"""Document parsers registry (волна 28).

Dispatch по расширению в pure-Python парсеры:

| Расширение | Parser | Mime |
|---|---|---|
| `.pdf` | `parse_pdf` | application/pdf |
| `.docx` | `parse_docx` | docx |
| `.xlsx` | `parse_xlsx` | xlsx |
| `.md`, `.markdown` | `parse_plaintext` | text/markdown |
| `.txt`, `.text` | `parse_plaintext` | text/plain |

Без OCR / vision / PDF-image extraction — image-only PDF возвращает
empty text, caller отвечает 422.
"""

from __future__ import annotations

from pathlib import Path

from styx.engine.document_parsers._mime import (
    is_supported_extension,
    mime_for_extension,
    normalize_extension,
    verify_magic_bytes,
)
from styx.engine.document_parsers._types import ParsedDocument
from styx.engine.document_parsers.docx import parse_docx
from styx.engine.document_parsers.pdf import parse_pdf
from styx.engine.document_parsers.plaintext import parse_plaintext
from styx.engine.document_parsers.xlsx import parse_xlsx


def parse(path: Path) -> ParsedDocument:
    """Распарсить файл по расширению.

    Pipeline:
        1. Normalize расширение (lowercase).
        2. Verify magic bytes для binary форматов.
        3. Dispatch в parser-функцию.
        4. Расширение неизвестно → ValueError.

    Raises:
        ValueError: расширение неподдерживаемо / mime mismatch /
            parser-специфичный fail (encrypted PDF, etc).
    """
    ext = normalize_extension(path)
    if not is_supported_extension(ext):
        raise ValueError(
            f"unsupported extension: {ext or '<none>'} "
            f"(supported: .pdf, .docx, .xlsx, .md, .markdown, .txt, .text)"
        )

    verify_magic_bytes(path, ext)
    mime = mime_for_extension(ext) or ""

    if ext == ".pdf":
        return parse_pdf(path)
    if ext == ".docx":
        return parse_docx(path)
    if ext == ".xlsx":
        return parse_xlsx(path)
    return parse_plaintext(path, mime_type=mime)


__all__ = [
    "ParsedDocument",
    "is_supported_extension",
    "mime_for_extension",
    "normalize_extension",
    "parse",
    "verify_magic_bytes",
]
