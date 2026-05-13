"""Pure формат строки transcript'а для dialogue_prepare_summary (волна 24).

Формат: ``[YYYY-MM-DD HH:MM:SS] Speaker: content``.

Speaker mapping (D3 в waves/24): ``user → Human``, ``assistant → Agent``.
Любой иной role — ValueError; SQL фильтр в `dialogue_prepare_summary`
не пропускает их, но pure-функция честно проверяет на случай прямого
вызова.
"""

from __future__ import annotations

import datetime as _dt


_SPEAKER_MAP: dict[str, str] = {
    "user": "Human",
    "assistant": "Agent",
}


def _format_timestamp(ts: _dt.datetime) -> str:
    """ISO без 'T' и без миллисекунд (port memorybox формата)."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return ts.replace(microsecond=0).isoformat(sep=" ")


def format_transcript_line(role: str, content: str, ts: _dt.datetime) -> str:
    speaker = _SPEAKER_MAP.get(role)
    if speaker is None:
        raise ValueError(
            f"format_transcript_line: unsupported role={role!r} "
            "(только 'user'/'assistant')"
        )
    return f"[{_format_timestamp(ts)}] {speaker}: {content}"
