"""Builder для salient memory block — recall'нутые memories для compose'а.

Волна 9 — главная инверсия: вместо «LLM спросит recall, если надо»
Styx сам встраивает релевантные memories в каждый turn. Builder
изолирует логику recall'а от ``StyxContextEngine.compress()``: skip-
условия, timeout, форматирование.

Волна 10 — drift detection + cached salient. На стабильной теме
salient block кэшируется в ``focus_tracker`` и переиспользуется между
compress'ами; при обнаруженном drift'е (cosine последнего user-embed'а
с centroid'ом окна < threshold) — кэш инвалидируется, идёт fresh
recall. Включается через ``focus_tracker.configure(...)`` в
``MemoryProvider.initialize()``; если tracker не configured —
fallback на волну-9 поведение (fresh каждый turn).

Семантика fail-open аналогична hot-path sentiment'у (§ 21.2):
любая ошибка / timeout / отсутствие данных → ``None``, compose
возвращает старый head+tail без salient. Latency hot-path'а
критичнее точности recall'а на единичном turn'е.
"""

from __future__ import annotations

import concurrent.futures
import logging
from typing import Any

from styx import turn_state
from styx.embedding import EmbeddingError
from styx.engine import focus_tracker
from styx.engine.salient_bridge import SalientHandle
from styx.storage.recall import format_recall_text, recall_full

log = logging.getLogger(__name__)

SALIENT_MARKER = "[Styx — релевантные memories из долгой памяти]"


def build_salient_block(
    messages: list[dict[str, Any]],
    handle: SalientHandle | None,
) -> dict[str, Any] | None:
    """Собрать один user-message с recall'нутыми memories или None.

    Skip (вернёт None) если:
    - bridge не сконфигурирован (handle is None);
    - в messages нет содержательного user-сообщения;
    - последний user короче ``handle.min_query_len`` (короткие запросы
      шумят embedding-пространство);
    - embed last_user'а упал (Ollama outage, embed-fail);
    - recall превысил ``handle.timeout_s`` или бросил исключение;
    - после фильтра min_score память пуста.

    Все skip-ы тихие (WARNING лог, без ошибок) — fail-open.

    Если ``focus_tracker`` configured — на стабильной теме (no drift)
    возвращает кэшированный salient. На drift'е / первом call'е —
    fresh recall, кэшируется.
    """
    if handle is None:
        return None

    last_user = _find_last_user_text(messages)
    if last_user is None:
        return None
    if len(last_user) < handle.min_query_len:
        return None

    state = focus_tracker.get_state(handle.agent_id)
    if state is None:
        # Drift detection отключён → волна-9 поведение (fresh каждый turn).
        return _fresh_salient(handle, last_user, query_vector=None)

    # Embed заранее — нужен и для drift detection, и для recall'а.
    try:
        last_user_embed = handle.embed_client.embed(last_user)
    except EmbeddingError as exc:
        log.warning("salient skip: embed last_user упал: %s", exc)
        return None

    drift = focus_tracker.observe(handle.agent_id, last_user_embed)
    if drift or state.cached_salient is None:
        salient = _fresh_salient(handle, last_user, query_vector=last_user_embed)
        focus_tracker.set_cached(handle.agent_id, salient)
        return salient
    return state.cached_salient


def _fresh_salient(
    handle: SalientHandle,
    last_user: str,
    *,
    query_vector: list[float] | None,
) -> dict[str, Any] | None:
    """Сделать новый recall и сформатировать salient block. None если пусто.

    Волна 14: при наличии agent_id'а в handle (configured) — observe()
    turn_state и передаём snapshot в recall_full. Без agent_id'а
    (unit-тесты с stub handle) — snapshot=None, fence не применяется.
    """
    snapshot = turn_state.observe(handle.agent_id) if handle.agent_id else None
    try:
        result = _call_with_timeout(
            handle=handle,
            query=last_user,
            query_vector=query_vector,
            snapshot=snapshot,
        )
    except concurrent.futures.TimeoutError:
        log.warning("salient skip: recall timeout (%.2fs)", handle.timeout_s)
        return None
    except Exception as exc:  # noqa: BLE001 — fail-open
        log.warning("salient skip: recall failed: %s", exc)
        return None

    if not result.memories:
        return None

    text = format_recall_text(result)
    return {"role": "user", "content": f"{SALIENT_MARKER}\n{text}"}


def _find_last_user_text(messages: list[dict[str, Any]]) -> str | None:
    """Последний user-message с извлечённым текстом.

    Поддерживает string content и multi-modal list content
    (``[{type:"text", text:"..."}, {type:"image", ...}]``) — pi-
    embedded runner (OpenClaw 2026.5.7) передаёт user messages
    именно в multi-modal shape. До мини-волны 26.8 round 6 функция
    игнорировала не-string content → boевые агенты получали
    salient=None.

    Image / audio / другие non-text parts пропускаются — embed
    работает на plain text.
    """
    from styx.engine.content_parts import extract_text_from_content

    for msg in reversed(messages):
        if msg.get("role") != "user":
            continue
        text = extract_text_from_content(msg.get("content"))
        if text:
            return text
    return None


def _call_with_timeout(
    *,
    handle: SalientHandle,
    query: str,
    query_vector: list[float] | None,
    snapshot=None,
):
    """recall_full в отдельном thread'е с timeout.

    ThreadPoolExecutor выбран вместо ``signal.SIGALRM`` (упомянутого как
    альтернатива в design'е) потому что:
    - не имеет проблем с signal-mask'ами Hermes runloop'а;
    - работает из non-main threads (тесты, в т.ч. pytest worker'ы);
    - cancel невозможен в любом случае — Python thread API не даёт
      прерывать blocking C-call'ы (urllib socket в embed). Если recall
      повис — наш thread продолжит ждать, но caller вернётся через
      timeout и compose продолжится без salient. Ленивый thread
      завершится сам когда сокет ответит.
    """
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="styx-salient"
    ) as pool:
        fut = pool.submit(
            recall_full,
            queries=handle.queries,
            embed_client=handle.embed_client,
            query=query,
            query_vector=query_vector,
            full_config=handle.recall_config.full,
            snapshot=snapshot,
        )
        return fut.result(timeout=handle.timeout_s)
