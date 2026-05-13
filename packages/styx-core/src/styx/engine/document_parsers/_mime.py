"""Mime detection — расширение + magic bytes verification (волна 28 D11).

Расширение определяет parser; magic bytes (первые ~8 байт) сверяются как
defense. Без python-magic dependency — ручная table из 5 entries.
"""

from __future__ import annotations

from pathlib import Path


# Canonical mime types по расширению.
_EXT_TO_MIME: dict[str, str] = {
    ".pdf": "application/pdf",
    ".docx": (
        "application/vnd.openxmlformats-officedocument."
        "wordprocessingml.document"
    ),
    ".xlsx": (
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".txt": "text/plain",
    ".text": "text/plain",
}


# Magic bytes (начало файла) для binary форматов. text/* — без проверки.
_MAGIC_BYTES: dict[str, bytes] = {
    ".pdf": b"%PDF-",
    # DOCX и XLSX — zip-контейнеры; signature \x50\x4b\x03\x04.
    ".docx": b"PK\x03\x04",
    ".xlsx": b"PK\x03\x04",
}


def normalize_extension(path: Path) -> str:
    """Lowercase suffix, например ``.PDF`` → ``.pdf``. Пустая строка
    если файл без расширения.
    """
    return path.suffix.lower()


def mime_for_extension(ext: str) -> str | None:
    """Canonical mime по расширению или None если расширение неизвестно."""
    return _EXT_TO_MIME.get(ext)


def verify_magic_bytes(path: Path, ext: str) -> None:
    """Проверка magic bytes (defense): если расширение объявляет
    binary формат, первые байты должны совпасть с signature.

    Raises:
        ValueError: расширение требует magic bytes, но содержимое не
            начинается с ожидаемой signature. Сообщение указывает на
            обнаруженный mismatch.
    """
    expected = _MAGIC_BYTES.get(ext)
    if expected is None:
        # text/* — magic не проверяем (utf-8 не имеет signature).
        return
    with path.open("rb") as fh:
        head = fh.read(len(expected))
    if head != expected:
        raise ValueError(
            f"mime mismatch: extension {ext} expects magic bytes "
            f"{expected!r}, got {head!r}"
        )


def is_supported_extension(ext: str) -> bool:
    """True если расширение известно registry парсеров."""
    return ext in _EXT_TO_MIME
