"""Channel B (волна 15) — peer VAD note.

Короткая отметка о тоне последней peer-реплики из ``emotional_state``.
``sync_turn`` пишет hot-path VAD под ``source='hot_sentiment'`` с
``metadata={"hot_vad": [v, a, d]}`` (raw VAD, не scaled-delta — чтобы
канал видел «острый» сигнал без амортизации с базой).

8-октантный словарь phrase'ов по знакам трёх осей VAD (без
intensity gradation в волне 15). Семантика — descriptive («peer
прозвучал X»), не директива.

Skip-условия:
- ``handle.peer_vad_enabled = False``;
- нет hot_sentiment записи за последние ``peer_vad_ttl_s`` секунд;
- ``metadata.hot_vad`` отсутствует / некорректен (legacy запись);
- ``norm < peer_vad_min_norm`` (peer слишком нейтрален).
"""

from __future__ import annotations

import logging
import math
from typing import Any

from styx.engine.pre_llm_inject import ChannelHandle

log = logging.getLogger(__name__)


# 8 октантов × 1 phrase. Ключ — `${sign(v)}${sign(a)}${sign(d)}`,
# где sign возвращает "pos" если значение >= 0, "neg" если < 0.
# Стартовая редакция; intensity gradation (как 24-таблица memorybox
# mood'а) — следующая волна если потребуется.
OCTANTS: dict[str, str] = {
    "pospospos": "оживлённо и уверенно",
    "posposneg": "взволнованно и радостно",
    "posnegpos": "спокойно и удовлетворённо",
    "posnegneg": "мягко и расслабленно",
    "negpospos": "напряжённо и собранно",
    "negposneg": "тревожно и взволнованно",
    "negnegpos": "сдержанно и тяжело",
    "negnegneg": "устало и подавленно",
}


def _sign(v: float) -> str:
    return "neg" if v < 0 else "pos"


def channel_peer_vad(
    handle: ChannelHandle, hermes_kwargs: dict[str, Any]
) -> str | None:
    """Сформировать «Peer прозвучал: <phrase>.» или None."""
    del hermes_kwargs  # волна 15: канал не зависит от Hermes-полей

    if not handle.peer_vad_enabled:
        return None

    try:
        entry = handle.queries.get_latest_hot_sentiment(
            within_seconds=handle.peer_vad_ttl_s,
        )
    except Exception as exc:  # noqa: BLE001 — fail-open на DB-ошибках
        log.warning("peer_vad: get_latest_hot_sentiment failed: %s", exc)
        return None

    if entry is None:
        return None
    vad, _at = entry

    norm = math.sqrt(vad[0] ** 2 + vad[1] ** 2 + vad[2] ** 2)
    if norm < handle.peer_vad_min_norm:
        return None

    octant = _sign(vad[0]) + _sign(vad[1]) + _sign(vad[2])
    phrase = OCTANTS.get(octant)
    if phrase is None:  # paranoia
        return None
    return f"Peer прозвучал: {phrase}."
