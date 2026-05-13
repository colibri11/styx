"""PDF parser через pypdf (волна 28).

Pure-Python, text layer extraction. Image-only PDFs (scan без OCR
layer) → empty text — caller возвращает 422 «empty document».
Encrypted PDFs → ValueError, caller возвращает 422.
"""

from __future__ import annotations

from pathlib import Path

from styx.engine.document_parsers._types import ParsedDocument


def parse_pdf(path: Path) -> ParsedDocument:
    """Извлечь текст из PDF.

    Raises:
        ValueError: PDF encrypted (без поддержки пароля) либо
            malformed (pypdf не распарсил).
    """
    # Lazy import — pypdf тяжёлый, грузим только при реальном вызове.
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(str(path))
    except PdfReadError as exc:
        raise ValueError(f"PDF parse error: {exc}") from exc

    if reader.is_encrypted:
        raise ValueError(
            "encrypted document, password not supported"
        )

    parts: list[str] = []
    for page in reader.pages:
        try:
            txt = page.extract_text() or ""
        except Exception as exc:  # noqa: BLE001
            # Битая страница не должна валить весь parse.
            raise ValueError(f"PDF page extract error: {exc}") from exc
        if txt:
            parts.append(txt)

    text = "\n\n".join(parts).strip()

    return ParsedDocument(
        text=text,
        metadata={
            "page_count": len(reader.pages),
        },
    )
