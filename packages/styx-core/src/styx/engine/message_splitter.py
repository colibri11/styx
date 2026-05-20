"""Non-overlapping сплиттер больших реплик дневника (Defect-fix B).

Сообщение (user/assistant реплика) — это речь, дневник-как-разговор.
По концепции (IAmBook §V) оно остаётся в дневнике (``memories``)
целиком, а не уходит в архив (``documents``+``chunks``). Но один ряд
``memories`` ограничен CHECK constraint'ом ``length(content) <= 2400``
(размер семантической единицы — ряд embed'ится в одну вектор-точку).

Поэтому реплику длиннее лимита режем на N рядов того же ``role`` /
session, каждый ≤ ``part_chars``. Все ряды остаются в дневнике;
группа помечается ``msg_group`` / ``part`` / ``parts`` в metadata,
чтобы композиция пересобрала их обратно в один блок.

**Почему отдельный сплиттер, а не ``chunker.chunk_text``:** у chunker'а
из волны 19 есть overlap (~320 chars) — он нужен для retrieval по
chunks архива. Здесь overlap = дублирование текста при пересборке
recent-окна (одни и те же предложения попали бы в окно дважды).
Поэтому split строго non-overlapping, по естественным границам:
абзац (``\\n\\n``) → предложение (``. `` / ``.\\n``) → жёсткий рез
(только если одно предложение само длиннее лимита).
"""

from __future__ import annotations

import logging
import re
from typing import Any

log = logging.getLogger(__name__)

# Реплика ≤ лимита остаётся одним рядом — поведение не меняется.
# Реплика длиннее режется на части ≤ DEFAULT_PART_CHARS. 2000 —
# запас до CHECK constraint'а 2400 (на случай если будущий wrapper
# что-то припишет; и чтобы не сидеть впритык к границе).
DEFAULT_PART_CHARS = 2000

_SENTENCE_BOUNDARY_RE = re.compile(r"\.\s")


def needs_split(content: str, *, part_chars: int = DEFAULT_PART_CHARS) -> bool:
    """True если реплику нужно резать (длиннее ``part_chars``)."""
    return len(content) > part_chars


def split_message(
    content: str,
    *,
    part_chars: int = DEFAULT_PART_CHARS,
) -> list[str]:
    """Разрезать реплику на non-overlapping части ≤ ``part_chars``.

    Реплика ≤ ``part_chars`` → ``[content]`` (один элемент, поведение
    не меняется). Пустая / whitespace-only реплика → ``[content]``
    as-is (caller сам решает писать её или нет).

    Конкатенация результата восстанавливает оригинал byte-for-byte:
    сепараторы (``\\n\\n`` между абзацами, пробел/перевод строки
    после точки) остаются в тексте, не теряются.

    Алгоритм:
      1. Split по ``\\n\\n`` на абзацы (сепаратор сохраняется).
      2. Greedy merge абзацев в части ≤ ``part_chars``.
      3. Абзац сам длиннее лимита → sentence-split.
      4. Предложение само длиннее лимита → жёсткий рез по символам.
    """
    if part_chars <= 0:
        raise ValueError(f"part_chars должен быть > 0, получено {part_chars}")
    if len(content) <= part_chars:
        return [content]

    paragraphs = _split_keep_separator(content, "\n\n")
    parts: list[str] = []
    current = ""

    for para in paragraphs:
        if len(para) > part_chars:
            # Абзац не влезает целиком — сначала flush current,
            # потом дробим абзац на предложения.
            if current:
                parts.append(current)
                current = ""
            for piece in _split_long_paragraph(para, part_chars):
                if len(piece) > part_chars:
                    parts.extend(_hard_split(piece, part_chars))
                else:
                    parts.append(piece)
            continue
        if not current:
            current = para
            continue
        if len(current) + len(para) <= part_chars:
            current += para
        else:
            parts.append(current)
            current = para

    if current:
        parts.append(current)

    return parts


def _split_keep_separator(text: str, sep: str) -> list[str]:
    """Split по ``sep``, оставляя сепаратор в конце каждого куска
    (кроме последнего). Конкатенация восстанавливает оригинал."""
    raw = text.split(sep)
    out: list[str] = []
    for i, piece in enumerate(raw):
        if i < len(raw) - 1:
            out.append(piece + sep)
        elif piece:
            out.append(piece)
    return out


def _split_long_paragraph(text: str, part_chars: int) -> list[str]:
    """Sentence-split абзаца, который сам длиннее ``part_chars``.

    Greedy merge предложений в части ≤ part_chars. Предложение само
    длиннее лимита возвращается as-is (caller дёрнет ``_hard_split``).
    """
    sentences = _split_sentences(text)
    out: list[str] = []
    current = ""
    for sent in sentences:
        if len(sent) > part_chars:
            if current:
                out.append(current)
                current = ""
            out.append(sent)
            continue
        if not current:
            current = sent
            continue
        if len(current) + len(sent) <= part_chars:
            current += sent
        else:
            out.append(current)
            current = sent
    if current:
        out.append(current)
    return out


def _split_sentences(text: str) -> list[str]:
    """Split на границах предложений ``. `` / ``.\\n``.

    Trailing-сепаратор остаётся в предыдущем предложении —
    конкатенация воссоздаёт оригинал byte-for-byte.
    """
    parts: list[str] = []
    remaining = text
    while remaining:
        match = _SENTENCE_BOUNDARY_RE.search(remaining)
        if match is None:
            parts.append(remaining)
            break
        end = match.end()
        parts.append(remaining[:end])
        remaining = remaining[end:]
    return parts


def _hard_split(text: str, part_chars: int) -> list[str]:
    """Грубый рез по фиксированному char-limit'у.

    Используется только когда одно предложение длиннее лимита
    (длинная строка без пробелов и точек).
    """
    return [text[i : i + part_chars] for i in range(0, len(text), part_chars)]


def _group_key(msg: dict[str, Any]) -> tuple[str, int, int] | None:
    """Извлечь (msg_group, part, parts) из metadata сообщения.

    Возвращает ``None`` если сообщение не часть группы (нет
    ``msg_group`` в metadata) или metadata невалидна. msg_group хранит
    либо ``metadata`` (формат StoredMessage / dict сообщения), либо
    ничего — в обоих случаях ключ берётся из ``msg["metadata"]``.
    """
    meta = msg.get("metadata")
    if not isinstance(meta, dict):
        return None
    group = meta.get("msg_group")
    part = meta.get("part")
    parts = meta.get("parts")
    if not isinstance(group, str) or not group:
        return None
    if not isinstance(part, int) or not isinstance(parts, int):
        return None
    if parts < 1 or part < 0 or part >= parts:
        return None
    return (group, part, parts)


def reassemble_message_groups(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Пересобрать ряды одной группы (Defect-fix B) обратно в один блок.

    Группа — последовательность смежных сообщений с одинаковым
    ``metadata.msg_group``; их ``content`` склеивается по возрастанию
    ``metadata.part`` в один блок. В результирующем сообщении остаётся
    ``role`` / прочие поля первого ряда группы, ``content`` —
    конкатенация всех частей, а group-маркеры (``msg_group`` / ``part``
    / ``parts``) из metadata убираются (блок снова целостный).

    Сообщения без ``msg_group`` проходят без изменений. Порядок
    остальных сообщений сохраняется. Если части группы идут не
    подряд (перемешаны другими сообщениями) — каждый смежный
    под-сегмент группы пересобирается отдельно; это defensive-ветка,
    в норме части одной группы всегда смежны.

    Идемпотентна: повторный вызов на уже пересобранном списке —
    no-op (нет msg_group → ничего не меняется).
    """
    out: list[dict[str, Any]] = []
    i = 0
    n = len(messages)
    while i < n:
        key = _group_key(messages[i])
        if key is None:
            out.append(messages[i])
            i += 1
            continue
        group_id = key[0]
        # Собираем смежный run сообщений той же группы.
        run: list[dict[str, Any]] = [messages[i]]
        j = i + 1
        while j < n:
            kj = _group_key(messages[j])
            if kj is None or kj[0] != group_id:
                break
            run.append(messages[j])
            j += 1
        out.append(_merge_group_run(run))
        i = j
    return out


def _merge_group_run(run: list[dict[str, Any]]) -> dict[str, Any]:
    """Склеить смежный run рядов одной группы в одно сообщение."""
    # Сортируем по part — на случай если ряды пришли не по порядку.
    ordered = sorted(run, key=lambda m: m["metadata"]["part"])
    # Q4: диагностика неполной группы. Если собрано частей меньше чем
    # заявлено в metadata.parts — часть потеряна (eviction, миграция,
    # обрыв LIMIT-boundary'ем выше recent_messages). Поведение не
    # меняем — пересобираем что есть; только сигналим в лог.
    declared = ordered[0]["metadata"].get("parts")
    if isinstance(declared, int) and len(ordered) != declared:
        log.warning(
            "reassemble: группа %s неполная — собрано %d частей из "
            "заявленных %d (parts=%r)",
            ordered[0]["metadata"].get("msg_group"),
            len(ordered),
            declared,
            sorted(m["metadata"].get("part") for m in ordered),
        )
    merged = dict(ordered[0])
    merged["content"] = "".join(m.get("content") or "" for m in ordered)
    # Снимаем group-маркеры — блок снова целостный.
    meta = dict(merged.get("metadata") or {})
    for marker in ("msg_group", "part", "parts"):
        meta.pop(marker, None)
    merged["metadata"] = meta
    return merged
