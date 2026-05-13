"""Relevance-aware eviction — top-K pair-групп по cosine к focus centroid'у.

Волна 12. Поверхность waves-v1 § «Eviction» v2 — оценка релевантности
middle-сообщений к текущему фокусу (centroid из ``focus_tracker`` волны
10) и keep top-K в выживающей части compress'а.

Все функции pure: state живёт в ``eviction_relevance_bridge`` и
``focus_tracker``. Эти модули зовут эти helpers с уже подготовленными
аргументами.

Контракт ``apply_relevance_eviction``:
    in:  messages, head_end, tail_start, handle, centroid
    out: middle_keep — list[dict] из выбранных pair-групп в
         chronological порядке. Кладётся между head и tail.

Fail-open: при любом сбое (handle is None, centroid is None, БД-запрос
упал, нет embed-able групп) → возвращается пустой список (recency-only
eviction в caller'е).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from styx.engine.eviction_relevance_bridge import EvictionRelevanceHandle

log = logging.getLogger(__name__)


def apply_relevance_eviction(
    messages: list[dict[str, Any]],
    head_end: int,
    tail_start: int,
    handle: EvictionRelevanceHandle | None,
    centroid: list[float] | None,
) -> list[dict[str, Any]]:
    """Выбрать top-K relevant pair-групп из middle. Пустой list при skip.

    ``head_end`` / ``tail_start`` — индексы в ``messages``: middle =
    ``messages[head_end:tail_start]``. ``handle`` — bridge handle (None
    если bridge не сконфигурирован). ``centroid`` — текущий focus
    centroid (None если focus_tracker не configured / окно пусто).

    Skip-условия (см. wave-doc D6):
    - ``handle`` is None;
    - ``centroid`` is None;
    - middle пуст;
    - ``handle.keep_k`` == 0 (явно отключённый keep);
    - groups <= keep_k и floor 0 → возвращаем все (no-op eviction для
      relevance, но здесь же caller всё равно будет резать recency'ем
      — лишний кейс, см. ниже);
    - БД-запрос упал;
    - ни одной группы не прошёл floor.
    """
    if handle is None or centroid is None:
        return []
    if handle.keep_k == 0:
        return []
    middle = messages[head_end:tail_start]
    if not middle:
        return []

    groups = _segment_pair_groups(middle)
    if not groups:
        return []

    # Если групп ≤ keep_k и floor=0 — возвращаем все. Но мы всё равно
    # хотим относительный ranking чтобы порядок керопов отражал relevance
    # (хотя в нашем случае мы их вернём в chronological, ranking влияет
    # только при K limit cut'е). Просто пробрасываем через scoring.
    contents = sorted({_message_text(m) for g in groups for m in g if _message_text(m)})
    embeds: dict[str, list[float]] = {}
    if contents:
        try:
            embeds = handle.queries.lookup_embeddings_by_content(contents)
        except Exception as exc:  # noqa: BLE001 — fail-open
            log.warning("eviction_relevance: lookup_embeddings_by_content упал: %s", exc)
            return []

    scored: list[tuple[int, list[dict[str, Any]], float | None]] = []
    for idx, group in enumerate(groups):
        rel = _pair_group_relevance(group, embeds, centroid)
        scored.append((idx, group, rel))

    return _select_top_k(scored, handle.keep_k, handle.threshold)


# ── helpers ──────────────────────────────────────────────────────────


def _segment_pair_groups(
    middle: list[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Сегментировать middle на pair-группы.

    Группа — последовательность messages, начинающаяся с **не-tool**
    message (user / assistant с tool_calls / assistant без tool_calls)
    и продолжающаяся всеми subsequent ``role='tool'`` сообщениями.

    Orphan tool в начале middle (не должно встречаться в нормальном
    flow — head_extension #14 закрывает предыдущую pair) формирует
    свою single-group; он будет дропнут в scoring (нет embed-able
    text — relevance None).
    """
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for msg in middle:
        role = msg.get("role")
        if role == "tool":
            if current:
                current.append(msg)
            else:
                # orphan tool — single-group
                groups.append([msg])
        else:
            if current:
                groups.append(current)
            current = [msg]
    if current:
        groups.append(current)
    return groups


def _pair_group_relevance(
    group: list[dict[str, Any]],
    embeds: dict[str, list[float]],
    centroid: list[float],
) -> float | None:
    """``max cosine(emb(msg), centroid)`` среди embed-able messages группы.

    Embed-able — message, у которого есть text content и для этого
    text'а нашёлся embedding в ``embeds`` map'е. Если ни одного —
    возвращает None (группа выпадает из top-K ranking'а).
    """
    best: float | None = None
    for msg in group:
        text = _message_text(msg)
        if not text:
            continue
        vec = embeds.get(text)
        if vec is None:
            continue
        sim = _cosine(vec, centroid)
        if best is None or sim > best:
            best = sim
    return best


def _select_top_k(
    scored: list[tuple[int, list[dict[str, Any]], float | None]],
    k: int,
    floor: float,
) -> list[dict[str, Any]]:
    """Выбрать top-K групп с relevance >= floor; вернуть в chronological.

    ``scored`` — список ``(chronological_idx, group, relevance | None)``.
    None relevance исключаются. Сортировка по relevance desc, ties
    разрываются исходным chronological порядком (stable sort).
    Затем slice до K. Восстанавливаем chronological через сортировку
    по idx.
    """
    eligible = [(idx, group, rel) for idx, group, rel in scored if rel is not None and rel >= floor]
    eligible.sort(key=lambda t: (-t[2], t[0]))
    chosen = eligible[:k]
    chosen.sort(key=lambda t: t[0])
    return [m for _, group, _ in chosen for m in group]


def _message_text(msg: dict[str, Any]) -> str:
    """Извлечь text-содержимое message'а или пустую строку.

    Только string ``content`` поддерживается (Hermes-формат). Список
    блоков (Anthropic-style) или None → пустая строка (message не
    embed-able, не участвует в ranking).
    """
    content = msg.get("content")
    if isinstance(content, str):
        return content
    return ""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))
