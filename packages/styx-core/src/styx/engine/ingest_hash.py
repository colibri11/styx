"""Content hash для идемпотентности `/ingest_experience` (волна 23).

Port memorybox `src/ingest-content-hash.ts`. Hash берётся от
``(pipeline_id, pipeline_version, content_ref)`` в канонической
сериализации — ключи отсортированы рекурсивно, чтобы
``{"file_path": "a", "url": "b"}`` и ``{"url": "b", "file_path": "a"}``
давали одинаковый hash.

``agent_id`` в hash не входит — scope дедупа задаётся partial unique
индексом ``memories_agent_content_hash_uniq`` на ``(agent_id,
content_hash)`` (schema 0002, port volna 7).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonicalize(value: Any) -> str:
    """Детерминированная сериализация JSON-совместимого значения.

    Ключи объектов сортируются по unicode codepoint'у (port memorybox
    ``a < b ? -1 : a > b ? 1 : 0``). ``None``-значения внутри объектов
    отфильтровываются — они не должны влиять на hash, иначе
    ``{}`` и ``{"x": None}`` дали бы разные hash'и (memorybox: фильтрует
    ``undefined``).

    Float'ы сериализуются через ``json.dumps`` — это deterministic
    (Python repr фиксирован для конечных чисел). Bool'ы тоже через
    ``json.dumps`` (``true``/``false``).
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        # bool — подкласс int, проверяем раньше
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(canonicalize(item) for item in value) + "]"
    if isinstance(value, dict):
        entries = sorted(
            ((k, v) for k, v in value.items() if v is not None),
            key=lambda kv: kv[0],
        )
        return (
            "{"
            + ",".join(
                json.dumps(str(k), ensure_ascii=False) + ":" + canonicalize(v)
                for k, v in entries
            )
            + "}"
        )
    # Не-JSON-типы: возвращаем "null" как memorybox для unexpected
    # input'ов (вместо raise — hash остаётся детерминированным).
    return "null"


def compute_content_hash(
    *,
    pipeline_id: str,
    pipeline_version: str,
    content_ref: dict[str, Any],
) -> str:
    """sha256 hex от canonicalize'нного payload-tuple.

    ``content_ref`` — словарь с ссылкой на источник (file_path / url /
    inline_text / hash и т.д.). ``agent_id`` намеренно не включён.

    Caller отвечает за валидацию аргументов (non-empty pipeline_id /
    version, не-пустой content_ref). Эта функция предполагает что
    payload уже валиден.
    """
    canonical = canonicalize(
        {
            "pipeline_id": pipeline_id,
            "pipeline_version": pipeline_version,
            "content_ref": content_ref,
        }
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def is_content_ref_empty(content_ref: dict[str, Any] | None) -> bool:
    """Pipeline передал ``content_ref`` но в нём нет ни одного значения.

    Семантически это «нет ссылки на источник», дедуп склеит несвязанные
    payload'ы. Caller должен трактовать как «hash не вычисляем».

    Port memorybox ``hasContentRef`` логики
    (``ingest-experience.ts:297-299``).
    """
    if content_ref is None:
        return True
    return not any(v is not None for v in content_ref.values())
