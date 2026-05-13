"""ParsedDocument — общий return-type парсеров (волна 28)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParsedDocument:
    """Результат парсинга файла.

    ``text`` — единая строка, склеенная из всех страниц/секций; пустая
    при image-only PDF / encrypted / decoy file. Caller сам решает
    как реагировать на empty text (обычно 422 «empty document»).

    ``metadata`` — структурные поля, специфичные для формата
    (page_count для PDF, sheet_names для XLSX, и т.д.). Уходит в
    ``documents.metadata`` JSONB.
    """

    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
