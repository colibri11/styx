"""Plain text / markdown parser (волна 28).

Builtin ``open`` + utf-8 с ``errors='replace'`` для устойчивости к
смешанным encoding'ам. Для MD — сохраняем raw markdown (не renderим
в plain text, индекс по сырому контенту полезен для search).
"""

from __future__ import annotations

from pathlib import Path

from styx.engine.document_parsers._types import ParsedDocument


def parse_plaintext(path: Path, *, mime_type: str) -> ParsedDocument:
    """Прочитать .txt / .md как utf-8 строку."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()

    line_count = text.count("\n") + 1 if text else 0

    return ParsedDocument(
        text=text,
        metadata={
            "mime_type_input": mime_type,
            "line_count": line_count,
        },
    )
