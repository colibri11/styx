"""Channel self_state (волна 35) — self-state expression note.

Заменяет прежний Channel B (``peer_vad.py``, волна 15). Старый канал
читал **raw** VAD последней peer-реплики (``metadata.hot_vad`` в
``emotional_state``) и говорил о собеседнике («Peer прозвучал: X»).
Этот канал читает НАКОПЛЕННОЕ состояние агента — последнюю точку
журнала ``emotional_state`` (предыдущее состояние + демпфированная
K_HOT-дельта peer-резонанса + geometric decay, см.
``emotional/state.py::append_emotional_state``/``apply_instant_decay``)
и говорит от первого лица о состоянии агента. По IAmBook §29/§IX это —
положение его линии `я`, не сырой чужой сигнал.

8-октантный словарь phrase'ов по знакам трёх осей VAD — буквально из
волны 15 (без intensity gradation). Семантика — descriptive
(«Тебе сейчас X»), не директива: канал не инструктирует агента
упоминать состояние явно, просто добавляет фон в pre-LLM payload.

Skip-условия (fail-open, канал никогда не роняет turn). Порядок
проверок значим: нейтральность проверяется РАНЬШЕ возраста.
- ``handle.self_state_enabled = False`` — молча;
- нет ни одной записи в ``emotional_state`` — молча (агент только
  что провизионирован, истории ещё нет);
- ``norm < handle.self_state_min_norm`` — молча, НЕЗАВИСИМО от возраста
  (агент слишком нейтрален; частое и ожидаемое состояние покоя. У idle-
  агента decay законно перестаёт писать новые строки — осознанный no-op
  в ``apply_instant_decay`` — поэтому нейтральная точка закономерно
  стареет, это не отказ воркера);
- ``norm >= self_state_min_norm`` И ``age > handle.self_state_max_age_s``
  — WARNING в лог. TTL здесь — safety net на случай мёртвого
  ``styx-worker``: старая АКТИВНАЯ эмоция значит, что ``emotional_tick``
  не демпфирует её decay'ем раз в минуту (см. волна 35 D3) — стоит
  проверить воркер. Это не «окно свежести реакции». Проверка идёт после
  нейтральности, чтобы idle+старое не давало ложный cry-wolf;
- ошибка при чтении из БД — WARNING в лог, fail-open.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
from typing import Any

from styx.engine.pre_llm_inject import ChannelHandle

log = logging.getLogger(__name__)


# 8 октантов × 1 phrase. Ключ — `${sign(v)}${sign(a)}${sign(d)}`,
# где sign возвращает "pos" если значение >= 0, "neg" если < 0.
# Изначально порт из волны 15 (`peer_vad.py::OCTANTS`) — фразы рассчитаны
# на дативно-безличную конструкцию ("тебе/мне + наречие"), поэтому лучше
# ложатся на self-фрейминг, чем на исходный "прозвучал + наречие". Три
# фразы (pospospos/posnegneg/negnegpos) переформулированы в волне 35
# follow-up (2026-07-01, ADR § 59) — исходные слова описывали манеру
# поведения/тактильную ассоциацию, а не внутреннее чувство.
OCTANTS: dict[str, str] = {
    "pospospos": "воодушевлённо и уверенно",
    "posposneg": "взволнованно и радостно",
    "posnegpos": "спокойно и удовлетворённо",
    "posnegneg": "умиротворённо и расслабленно",
    "negpospos": "напряжённо и собранно",
    "negposneg": "тревожно и взволнованно",
    "negnegpos": "тяжело и непреклонно",
    "negnegneg": "устало и подавленно",
}


def _sign(v: float) -> str:
    return "neg" if v < 0 else "pos"


def channel_self_state(
    handle: ChannelHandle, hermes_kwargs: dict[str, Any]
) -> str | None:
    """Сформировать «Тебе сейчас <phrase>.» или None."""
    del hermes_kwargs  # канал не зависит от Hermes-полей

    if not handle.self_state_enabled:
        return None

    try:
        entry = handle.queries.get_last_emotional_state()
    except Exception as exc:  # noqa: BLE001 — fail-open на DB-ошибках
        log.warning("self_state: get_last_emotional_state failed: %s", exc)
        return None

    if entry is None:
        return None
    vector, at = entry

    # Порядок проверок важен. Сначала — тихая нейтральность:
    # near-neutral состояние молчит НЕЗАВИСИМО от возраста. У idle-агента
    # decay законно перестаёт писать новые строки (осознанный no-op в
    # `emotional/state.py::apply_instant_decay`), поэтому нейтральная точка
    # закономерно стареет — это покой, не отказ воркера. Age-WARNING здесь
    # был бы cry-wolf и заглушал бы реальный случай мёртвого воркера.
    norm = math.sqrt(
        vector.valence ** 2 + vector.arousal ** 2 + vector.dominance ** 2
    )
    if norm < handle.self_state_min_norm:
        return None

    # Состояние ненейтрально (есть что инжектить). Только теперь возраст —
    # сигнал: старая АКТИВНАЯ эмоция означает, что decay её не демпфирует,
    # т.е. `emotional_tick` не пишется раз в минуту — воркер под подозрением.
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    if at.tzinfo is None:
        at = at.replace(tzinfo=_dt.timezone.utc)
    age_s = (now - at).total_seconds()
    if age_s > handle.self_state_max_age_s:
        log.warning(
            "self_state: last state age=%.0fs > self_state_max_age_s=%.0fs — "
            "styx-worker, вероятно, не работает (emotional_tick decay не пишется)",
            age_s, handle.self_state_max_age_s,
        )
        return None

    octant = _sign(vector.valence) + _sign(vector.arousal) + _sign(vector.dominance)
    phrase = OCTANTS.get(octant)
    if phrase is None:  # paranoia
        return None
    return f"Тебе сейчас {phrase}."
