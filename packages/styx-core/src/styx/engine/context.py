"""StyxContextEngine — head+tail композиция с tool-pair sanitization.

V1 минимум семи поверхностей waves-v1, доставляемых волной 3:

- Бюджет окна: threshold_tokens = context_length * threshold_percent;
  reserve под user message и ответ заложен в (1 - threshold_percent).
- Eviction recency: при current_tokens > threshold отбрасываем середину,
  оставляя protect_first_n головы + protect_last_n хвоста.
- Tool-pair integrity: orphan tool_result → drop, orphan tool_call →
  стаб-result. Sanitize прогоняется при каждом compress().
- Working set runtime: in-process поля (last_prompt_tokens и др.) —
  без persistence (это v2 поверхности).
- Suffix composition stable+recent: head стабилен пока первые
  protect_first_n message не меняются — это и даёт байт-стабильность
  prefix'а для prefix-кэша провайдера.
- Cache placement (часть): сохраняем стабильные первые N байт history
  между turn'ами; OpenAI auto-prefix-cache срабатывает.
- Tier связи (active часть): active suffix = head + tail; long и hot
  не задействованы в v1 (retrieval — поздняя волна).

Волна 26.5 — cache-friendly salient placement + defensive markers:

- Salient вставляется ПЕРЕД последним message (insert_at = len(body)-1),
  не после head'а. Это делает префикс окна (head + middle + tail-без-
  последнего) стабильным между turn'ами; провайдеры (Anthropic,
  OpenAI, z.ai) кешируют этот префикс, recompute только salient
  + last user. Production economics: $2250/мес → $225/мес на Opus 4.7.
- Salient content оборачивается в XML-style теги (recommended
  Anthropic prompt-engineering). Defensive sanitize в начале
  ``compress()`` вырезает любые такие блоки из входящих messages —
  защита от leak'а если runtime когда-то начнёт persist'ить
  assembled view (salient не должен попадать в memories или в
  next-turn history).

Волна 30 — taxonomy маркеров (различение источника):

- Generic ``<styx>`` тег волны 26.5 заменён семейством
  ``<styx-salient>...</styx-salient>``,
  ``<styx-recall>...</styx-recall>``, ``<styx-archive>...``,
  ``<styx-dialogue>...``, ``<styx-relations>...``,
  ``<styx-explain>...``, ``<styx-working-set>...``. Salient inject
  здесь использует ``<styx-salient>``; tool-result wrapping для
  остальных каналов — в ``http/_wrap.py`` (Phase B/C).
- Sanitize regex принимает family ``<styx-[a-z-]+>...</styx-[a-z-]+>``
  и параллельно legacy ``<styx>...</styx>`` (без суффикса) — для
  историческо persist'нутых данных эпохи 26.5.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any

from styx.engine import (
    eviction_relevance_bridge,
    focus_tracker,
    salient_bridge,
)
from styx.engine.eviction_relevance import apply_relevance_eviction
from styx.engine.salient import build_salient_block
from styx.observability.logging import log_event

log = logging.getLogger(__name__)

STUB_TOOL_RESULT = "[Result from earlier conversation — see context summary above]"

# Регексы для defensive sanitize. Family покрывает любой тег вида
# `<styx-foo>...</styx-foo>` с парным суффиксом (см. STYX_TAG_* ниже);
# legacy — голый `<styx>...</styx>` эпохи 26.5 (для historical
# persist'нутых данных, до волны 30). Оба применяются в
# `_sanitize_styx_blocks`. Парность суффикса гарантирована regex'ом
# через backreference — `<styx-recall>...<styx-archive>` НЕ матчится
# как один блок.
_STYX_FAMILY_BLOCK_RE = re.compile(
    r"<styx-([a-z-]+)>.*?</styx-\1>", re.DOTALL
)
_STYX_LEGACY_BLOCK_RE = re.compile(r"<styx>.*?</styx>", re.DOTALL)

# Канонические имена тегов под каждый канал инжекта/wrap'а (волна 30).
# Используются wrap-helper'ами здесь и в `http/_wrap.py`. Менять
# существующие значения — breaking change для скиллов 26.6 и LLM-
# поведения; добавлять новые имена — расширение taxonomy.
STYX_TAG_SALIENT = "styx-salient"
STYX_TAG_WORKING_SET = "styx-working-set"
STYX_TAG_RECALL = "styx-recall"
STYX_TAG_ARCHIVE = "styx-archive"
STYX_TAG_DIALOGUE = "styx-dialogue"
STYX_TAG_RELATIONS = "styx-relations"
STYX_TAG_EXPLAIN = "styx-explain"

# -- production observability counter (Fix 6, волна 26.5) -------------------
#
# Считает суммарно сколько <styx>...</styx> блоков было вырезано из input
# messages с момента старта daemon'а. Production-критично для волны 27 deploy:
# оператор видит ненулевое значение → значит runtime где-то persist'ит
# assembled view (transcript echo, snapshot replay) и нужно искать утечку.
# Counter — module-global, защищён lock'ом, чтобы worker pool / multiple
# StyxComposer instances корректно агрегировали.
_styx_sanitized_blocks_total = 0
_styx_sanitized_lock = threading.Lock()


def _increment_sanitized_counter(n: int) -> None:
    """Атомарно прибавляет n к агрегатному счётчику вырезанных блоков."""
    if n <= 0:
        return
    global _styx_sanitized_blocks_total
    with _styx_sanitized_lock:
        _styx_sanitized_blocks_total += n


def get_styx_sanitized_blocks_total() -> int:
    """Геттер для analytics endpoint'а / production метрик."""
    with _styx_sanitized_lock:
        return _styx_sanitized_blocks_total


def reset_styx_sanitized_blocks_total() -> None:
    """Сбросить агрегат + breakdown. Для тестов; production не вызывает."""
    global _styx_sanitized_blocks_total
    with _styx_sanitized_lock:
        _styx_sanitized_blocks_total = 0
        _styx_sanitized_blocks_by_tag.clear()


# Per-tag breakdown (волна 30 D2 / Phase F): отдельные счётчики для
# каждого family-тега + legacy. ``legacy`` — псевдо-имя для голого
# ``<styx>...</styx>``; family-теги хранятся как они приходят
# (``salient``, ``recall``, ...). Анализ leak-источника по breakdown'у:
# например, рост ``recall`` без роста ``salient`` указывает на то что
# tool result где-то persist'ится, а не assembled view.
_styx_sanitized_blocks_by_tag: dict[str, int] = {}


def _increment_sanitized_by_tag(tag_suffix: str, n: int) -> None:
    """Атомарно прибавляет n к per-tag счётчику. ``tag_suffix`` — без
    префикса ``styx-`` (``salient``, ``recall``, ...) или строка
    ``legacy`` для голого ``<styx>...</styx>``."""
    if n <= 0:
        return
    with _styx_sanitized_lock:
        _styx_sanitized_blocks_by_tag[tag_suffix] = (
            _styx_sanitized_blocks_by_tag.get(tag_suffix, 0) + n
        )


def get_styx_sanitized_blocks_by_tag() -> dict[str, int]:
    """Snapshot per-tag breakdown'а для analytics endpoint'а."""
    with _styx_sanitized_lock:
        return dict(_styx_sanitized_blocks_by_tag)


class StyxComposer:
    """Head+tail композиция с tool-pair sanitization (host-agnostic core).

    Hermes-обёртка ``StyxContextEngine(ContextEngine)`` живёт в
    styx-hermes и проксирует ``compress`` по HTTP в core daemon, который
    держит state и вызывает этот класс.
    """

    def __init__(
        self,
        agent_id: str = "",
        *,
        context_length: int = 0,
        threshold_percent: float = 0.75,
        protect_first_n: int = 3,
        protect_last_n: int = 6,
    ) -> None:
        self.agent_id = agent_id

        # state поля (Hermes run_agent читает напрямую)
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0
        self.context_length = context_length
        self.threshold_percent = threshold_percent
        self.threshold_tokens = int(context_length * threshold_percent) if context_length else 0

        # composition параметры
        self.protect_first_n = protect_first_n
        self.protect_last_n = protect_last_n

    @property
    def name(self) -> str:
        return "styx"

    # -- token tracking --------------------------------------------------

    def update_from_response(self, usage: dict[str, Any]) -> None:
        if not usage:
            return
        self.last_prompt_tokens = int(usage.get("prompt_tokens") or 0)
        self.last_completion_tokens = int(usage.get("completion_tokens") or 0)
        self.last_total_tokens = int(
            usage.get("total_tokens")
            or self.last_prompt_tokens + self.last_completion_tokens
        )

    # -- compaction gating -----------------------------------------------

    def should_compress(self, prompt_tokens: int | None = None) -> bool:
        # Styx хочет владеть каждым turn'ом — compress() пропускает no-op
        # внутри себя. См. .design/integrations/hermes-v1.md § StyxContextEngine.
        return True

    # -- main compose entry-point ----------------------------------------

    def compress(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> list[dict[str, Any]]:
        del focus_topic  # v1: focus_topic игнорируется (см. waves-v1)

        # Defensive: вырезаем любые <styx-*>...</styx-*> и legacy
        # <styx>...</styx> блоки из входящих messages. Эти теги —
        # exclusive маркеры Styx (см. STYX_TAG_*). Если они появились
        # во входе — значит кто-то persist'нул assembled view или
        # tool result (LLM-провайдер вернул transcript? runtime сохранил
        # compressed window? native memory дамп assembled?). Утечка
        # ломает кэш-стабильность префикса и засоряет память. Удаляем
        # до того, как salient считается заново. Волны 26.5 + 30.
        messages = _sanitize_styx_blocks(messages)

        handle = salient_bridge.get_handle(self.agent_id)

        # build_salient_block внутри observe'ит focus_tracker по last
        # user embed'у — это единственная точка обновления focus state'а
        # за turn. Делаем заранее, чтобы apply_relevance_eviction (волна
        # 12) видел уже свежий centroid. На no-op / no-eviction путях
        # salient вставляется через _inject_salient_block; на eviction-
        # пути сначала relevance eviction поверх уже наблюдённого
        # centroid'а, затем salient.
        salient = build_salient_block(messages, handle)

        # Волна 26.5: salient вставляется ПЕРЕД последним сообщением
        # окна, а не после head'а. Это делает префикс (head + middle +
        # tail-без-последнего) стабильным между turn'ами и позволяет
        # provider'ам (Anthropic/OpenAI/z.ai) кешировать его.
        # Edge cases: len(body)==0 → salient не вставляется (insert_at=0,
        # _inject_salient_block обрабатывает); len(body)==1 → insert_at=0,
        # salient идёт перед единственной репликой («вот память — теперь
        # твой ход»).

        # No-op путь: бюджет не превышен → только sanitize.
        if (
            current_tokens is None
            or self.threshold_tokens <= 0
            or current_tokens <= self.threshold_tokens
        ):
            body = list(messages)
            insert_at = max(0, len(body) - 1)
            return self._inject_salient_block(salient, body, insert_at)

        # Eviction-минимум недостижим — нечего эвиктить.
        protected = self.protect_first_n + self.protect_last_n
        if len(messages) <= protected:
            body = list(messages)
            insert_at = max(0, len(body) - 1)
            return self._inject_salient_block(salient, body, insert_at)

        # #6: tail_start гарантирует непересечение head и tail
        tail_start = max(self.protect_first_n, len(messages) - self.protect_last_n)

        # #14: расширяем head вперёд пока последнее сообщение — незакрытый tool-pair
        head_end = self.protect_first_n
        while head_end < tail_start and _is_open_tool_pair_end(messages, head_end):
            head_end += 1
        # Если расширение поглотило всё — fallback без eviction
        if head_end >= tail_start:
            body = list(messages)
            insert_at = max(0, len(body) - 1)
            return self._inject_salient_block(salient, body, insert_at)

        head = list(messages[:head_end])
        tail = list(messages[tail_start:])

        # Волна 12: relevance-aware keep. Top-K pair-групп из middle с
        # cosine ≥ floor к focus centroid'у. Fail-open → пустой list при
        # любом skip-условии (см. wave-doc D6).
        ev_handle = eviction_relevance_bridge.get_handle(self.agent_id)
        centroid = focus_tracker.get_centroid(self.agent_id)
        middle_keep = apply_relevance_eviction(
            messages, head_end, tail_start, ev_handle, centroid
        )

        self.compression_count += 1
        log_event(
            log,
            "compress",
            compression_count=self.compression_count,
            messages_in=len(messages),
            head=len(head),
            middle_keep=len(middle_keep),
            tail=len(tail),
            evicted=len(messages) - len(head) - len(middle_keep) - len(tail),
            salient_injected=salient is not None,
        )
        # Волна 26.5: salient вставляется ПЕРЕД последним сообщением
        # финального body (= head + middle_keep + tail). Префикс
        # (head + middle_keep + tail-без-последнего) — стабилен пока
        # первые компоненты не меняются → provider кеширует.
        body = head + middle_keep + tail
        return self._inject_salient_block(
            salient, body, max(0, len(body) - 1)
        )

    def _inject_salient_block(
        self,
        salient: dict[str, Any] | None,
        body: list[dict[str, Any]],
        insert_at: int,
    ) -> list[dict[str, Any]]:
        """Вставить уже посчитанный salient на ``insert_at`` и санитайзить.

        ``salient`` — результат ``build_salient_block`` посчитанный в
        начале ``compress`` (там же выполнен ``focus_tracker.observe``).
        Sanitize tool-pair'ов прогоняется в самом конце над итоговым
        списком.

        Волна 26.5/30: salient.content оборачивается в
        ``<styx-salient>...</styx-salient>`` теги — defensive marker
        для последующего sanitize'а на входе compress'а (если runtime
        когда-то persist'нет assembled view). Обёртка применяется один
        раз, тут (а не в build_salient_block), чтобы кэшированный
        salient в focus_tracker оставался без тегов и
        `_inject_salient_block` оставался единственной точкой wrap'а —
        симметрично с `_sanitize_styx_blocks` на входе.
        """
        if salient is not None:
            wrapped = _wrap_salient(salient)
            body = body[:insert_at] + [wrapped] + body[insert_at:]
        return _sanitize_tool_pairs(body)

    # -- assemble entry-point (волна 26.7 — channel split) ---------------

    def assemble_for_runtime(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> dict[str, Any]:
        """Compose для runtime'а через **систему промптов**, не messages.

        Волна 26.7. Используется ``/context/assemble`` (OpenClaw embedded
        path). Возвращает ``{messages, salient_text}`` где:

        - ``messages`` — eviction-нормализованный window БЕЗ inject'а
          salient. Alternation user/assistant сохраняется как было в
          input'е (за вычетом эвикции середины).
        - ``salient_text`` — salient block уже обёрнутый в
          ``<styx-salient>...</styx-salient>`` (или ``None`` если
          salient пустой). Runtime ставит его в system prompt через
          ``systemPromptAddition`` ContextEngine return field; LLM
          увидит его как часть system prompt'а, минуя schema-валидацию
          conversation alternation.

        Зачем расщеплено (см. ADR § 41.10):
        - Salient как user-role message между existing user и assistant
          ломал strict alternation OpenAI Responses API (codex backend).
          API silently отбрасывал двух подряд user → boevые агенты не
          видели salient в input'е.
        - ``systemPromptAddition`` — задокументированный channel в
          OpenClaw runtime (cf. `selection-*.js: if (assembled.
          systemPromptAddition) systemPromptText = prependSystemPrompt
          Addition(...)`), без alternation requirements.

        ``focus_topic`` игнорируется (v1 не использует — см. waves-v1).
        ``compress()`` (для Hermes path через ``/context/build``)
        остаётся неизменным — там alternation issue не наблюдалось и
        cache invariant 26.5 на messages-inject держится.
        """
        del focus_topic

        # Sanitize input (волна 26.5) — вырезаем любые исторические
        # <styx>...</styx> блоки. Семантика идентична compress.
        messages = _sanitize_styx_blocks(messages)

        # Salient — посчитан и обёрнут, но не инжектится в messages.
        handle = salient_bridge.get_handle(self.agent_id)
        salient = build_salient_block(messages, handle)
        salient_text: str | None = None
        if salient is not None:
            wrapped = _wrap_salient(salient)
            wrapped_content = wrapped.get("content")
            if isinstance(wrapped_content, str):
                salient_text = wrapped_content

        # Eviction-normalize messages — те же ветки что compress(), но
        # БЕЗ вызова _inject_salient_block в конце.
        body = self._compose_window(messages, current_tokens)
        body = _sanitize_tool_pairs(body)

        return {"messages": body, "salient_text": salient_text}

    def _compose_window(
        self,
        messages: list[dict[str, Any]],
        current_tokens: int | None,
    ) -> list[dict[str, Any]]:
        """Eviction-normalize window без salient inject'а.

        Поведенчески идентичен compress() в части head/middle/tail
        формирования, но не вставляет salient block и не делает
        финальный sanitize_tool_pairs (вызывающий ответственен).
        Используется assemble_for_runtime (волна 26.7).
        """
        # No-op путь: бюджет не превышен.
        if (
            current_tokens is None
            or self.threshold_tokens <= 0
            or current_tokens <= self.threshold_tokens
        ):
            return list(messages)

        # Eviction-минимум недостижим.
        protected = self.protect_first_n + self.protect_last_n
        if len(messages) <= protected:
            return list(messages)

        tail_start = max(self.protect_first_n, len(messages) - self.protect_last_n)
        head_end = self.protect_first_n
        while head_end < tail_start and _is_open_tool_pair_end(messages, head_end):
            head_end += 1
        if head_end >= tail_start:
            return list(messages)

        head = list(messages[:head_end])
        tail = list(messages[tail_start:])

        ev_handle = eviction_relevance_bridge.get_handle(self.agent_id)
        centroid = focus_tracker.get_centroid(self.agent_id)
        middle_keep = apply_relevance_eviction(
            messages, head_end, tail_start, ev_handle, centroid
        )

        self.compression_count += 1
        log_event(
            log,
            "compose_window",
            compression_count=self.compression_count,
            messages_in=len(messages),
            head=len(head),
            middle_keep=len(middle_keep),
            tail=len(tail),
            evicted=len(messages) - len(head) - len(middle_keep) - len(tail),
            channel="assemble",
        )

        return head + middle_keep + tail

    # -- model switch ----------------------------------------------------

    def update_model(
        self,
        model: str,
        context_length: int,
        base_url: str = "",
        api_key: str = "",
        provider: str = "",
    ) -> None:
        del model, base_url, api_key, provider
        self.context_length = context_length
        self.threshold_tokens = int(context_length * self.threshold_percent)

    # -- session lifecycle -----------------------------------------------

    def on_session_reset(self) -> None:
        """Сбросить per-session счётчики. Не трогает focus/hot/salient state."""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0


def wrap_text_with_styx_tag(text: str, tag: str) -> str:
    """Обернуть строку в ``<{tag}>\\n...\\n</{tag}>``. Идемпотентно:
    если уже обёрнута тем же тегом — возвращает как есть.

    Используется wrap-helper'ами в ``http/_wrap.py`` для tool result
    каналов (recall/archive/dialogue/relations/explain) — единая точка
    форматирования (CRLF/whitespace), чтобы LLM видел один и тот же
    shape для всех tag'ов. Tag должен быть из STYX_TAG_* констант.
    """
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    if text.startswith(open_tag) and text.endswith(close_tag):
        return text
    return f"{open_tag}\n{text}\n{close_tag}"


def _wrap_salient(salient: dict[str, Any]) -> dict[str, Any]:
    """Обернуть salient.content в ``<styx-salient>\\n...\\n</styx-salient>``.

    Возвращает копию salient'а — caller'ский dict (или кэшированный в
    focus_tracker) не мутируется. Если content уже обёрнут (повторный
    inject? not expected, но defensive) — оставляем как есть.
    """
    content = salient.get("content")
    if not isinstance(content, str):
        # multi-modal salient не делается build_salient_block'ом, но
        # на всякий случай — pass-through.
        return salient
    wrapped = dict(salient)
    wrapped["content"] = wrap_text_with_styx_tag(content, STYX_TAG_SALIENT)
    return wrapped


def _strip_styx_text(text: str) -> tuple[str, dict[str, int]]:
    """Вырезает family + legacy блоки из строки.

    Возвращает кортеж ``(cleaned_text, per_tag_counts)`` где
    per_tag_counts — суффикс тега → количество вырезанных блоков.
    Legacy ``<styx>...</styx>`` агрегируется под ключом ``"legacy"``.

    Сначала вырезаются family-блоки (regex с backreference на
    суффикс), потом legacy. Порядок важен: legacy regex (``<styx>``
    без суффикса) при naive применении мог бы матчить открывающий
    family-тег ``<styx-recall>``? Нет — ``<styx>`` это literal
    закрытый ``>``, а ``<styx-recall>`` имеет дефис перед ``>``,
    поэтому legacy не пересекается с family. Но раздельные subn
    проще для счётчиков.
    """
    counts: dict[str, int] = {}
    if "<styx-" in text:
        def _family_sub(m: "re.Match[str]") -> str:
            counts[m.group(1)] = counts.get(m.group(1), 0) + 1
            return ""
        text = _STYX_FAMILY_BLOCK_RE.sub(_family_sub, text)
    if "<styx>" in text:
        text, legacy_removed = _STYX_LEGACY_BLOCK_RE.subn("", text)
        if legacy_removed:
            counts["legacy"] = counts.get("legacy", 0) + legacy_removed
    return text, counts


def _has_styx_marker(text: str) -> bool:
    """Substring fast-path: совпадает ли строка хоть с одним styx-тегом."""
    return "<styx-" in text or "<styx>" in text


def _sanitize_styx_blocks(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Вырезает все ``<styx-*>...</styx-*>`` и ``<styx>...</styx>``
    блоки из text-content.

    Волна 26.5/30 (defensive markers). Salient block + tool result
    wrapping оборачиваются в семейство ``<styx-{salient,recall,
    archive,dialogue,relations,explain,working-set}>...``. Эти теги —
    exclusive маркер Styx: ничего другого в pipeline такого не пишет.
    Если они приехали в input — значит кто-то случайно persist'нул
    assembled view или tool result (LLM transcript echo, runtime
    snapshot, log replay, native memory dump). Вырезаем до того, как
    считается новый salient.

    Legacy ``<styx>...</styx>`` (без суффикса, эпоха 26.5) тоже
    sanitize'ится — backwards-compat для historical persist'нутых
    данных (ADR § 45 D2).

    Спецификация:
    - Применяется к ``content`` типа ``str`` И к multi-modal list
      content (``[{type:"text", text:"..."}, {type:"image", ...}]``).
      Pi-embedded-runner (OpenClaw 2026.5.7) передаёт messages в
      multi-modal shape — до мини-волны 26.8 round 6 sanitize был
      pass-through для list-content (давнее TODO).
    - Sanitize применяется только к text-частям. Image / audio /
      tool_use parts остаются без изменений.
    - Если после очистки text-content становится пустой/whitespace-only:
      * string content → message целиком удаляется из array.
      * multi-modal с другими parts → text-part'ы удаляются, остальные
        parts сохраняются (message с image-only остаётся).
      * multi-modal только из text-частей → message удаляется.
    - Counter pass'ит точное количество вырезанных блоков (через
      ``re.subn`` / ``re.sub`` с callback'ом), не количество
      затронутых messages — два ``<styx-recall>...</styx-recall>``
      в одном message считаются как 2.
    - Module-global ``_styx_sanitized_blocks_total`` инкрементируется
      агрегированно (волна 26.5 Fix 6); per-tag breakdown — в
      ``_styx_sanitized_blocks_by_tag`` (волна 30, Phase F → analytics).
    - WARNING лог per-call pass'ит сумму блоков для observability.
    """
    sanitized_blocks = 0
    per_tag_total: dict[str, int] = {}

    def _accumulate(per_tag: dict[str, int]) -> None:
        nonlocal sanitized_blocks
        for tag, n in per_tag.items():
            sanitized_blocks += n
            per_tag_total[tag] = per_tag_total.get(tag, 0) + n

    out: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")

        # String content — простой случай.
        if isinstance(content, str):
            if not _has_styx_marker(content):
                out.append(msg)
                continue
            new_content, per_tag = _strip_styx_text(content)
            _accumulate(per_tag)
            if not new_content.strip():
                # Message стал пустым → удаляем целиком.
                continue
            new_msg = dict(msg)
            new_msg["content"] = new_content
            out.append(new_msg)
            continue

        # Multi-modal list content — sanitize text-parts inline.
        if isinstance(content, list):
            # Fast path: ни одна text-part не содержит styx-тег → no-op.
            has_marker = any(
                isinstance(p, dict)
                and p.get("type") in ("text", "input_text")
                and isinstance(p.get("text"), str)
                and _has_styx_marker(p["text"])
                for p in content
            )
            if not has_marker:
                out.append(msg)
                continue

            # Sanitize text-parts, считая удалённые блоки.
            new_parts: list[Any] = []
            for part in content:
                if not isinstance(part, dict):
                    new_parts.append(part)
                    continue
                ptype = part.get("type")
                if ptype not in ("text", "input_text"):
                    new_parts.append(part)
                    continue
                txt = part.get("text")
                if not isinstance(txt, str):
                    new_parts.append(part)
                    continue
                if not _has_styx_marker(txt):
                    new_parts.append(part)
                    continue
                new_txt, per_tag = _strip_styx_text(txt)
                _accumulate(per_tag)
                if new_txt.strip():
                    new_part = dict(part)
                    new_part["text"] = new_txt
                    new_parts.append(new_part)
                # Иначе — text-part стал пустой, drop'аем его.

            # Если ни одной части не осталось — message целиком drop.
            if not new_parts:
                continue
            new_msg = dict(msg)
            new_msg["content"] = new_parts
            out.append(new_msg)
            continue

        # Unknown content shape (None, etc.) — pass-through.
        out.append(msg)

    if sanitized_blocks > 0:
        # Production observability: per-call WARNING + counters.
        log.warning(
            "compress sanitized %d styx block(s) from input messages: %s",
            sanitized_blocks,
            ",".join(f"{tag}={n}" for tag, n in sorted(per_tag_total.items())),
        )
        _increment_sanitized_counter(sanitized_blocks)
        for tag, n in per_tag_total.items():
            _increment_sanitized_by_tag(tag, n)
    return out


def _is_open_tool_pair_end(messages: list[dict[str, Any]], head_end: int) -> bool:
    """Возвращает True если messages[head_end - 1] — незакрытый конец tool-pair.

    «Незакрытый» означает:
    - assistant с tool_calls (ожидает следующий tool result), или
    - tool с tool_call_id (сам является tool result — пара уже открыта выше).

    Используется для расширения head чтобы не обрывать посередине tool-pair.
    """
    if head_end <= 0 or head_end > len(messages):
        return False
    last = messages[head_end - 1]
    if last.get("role") == "assistant" and last.get("tool_calls"):
        return True
    if last.get("role") == "tool" and last.get("tool_call_id"):
        return True
    return False


def _get_tool_call_id(tc: Any) -> str:
    if isinstance(tc, dict):
        return tc.get("id", "") or ""
    return getattr(tc, "id", "") or ""


def _sanitize_tool_pairs(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Чинит orphan tool_call / tool_result после eviction.

    Симметрично дефолтному ContextCompressor.\\_sanitize\\_tool\\_pairs:
    - tool result без живого assistant.tool_call → удаляем.
    - assistant.tool_call без следующего за ним tool result → вставляем
      стаб-result, чтобы парность не сломалась.
    """
    surviving_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in msg.get("tool_calls") or ():
                cid = _get_tool_call_id(tc)
                if cid:
                    surviving_call_ids.add(cid)

    result_call_ids: set[str] = set()
    for msg in messages:
        if msg.get("role") == "tool":
            cid = msg.get("tool_call_id")
            if cid:
                result_call_ids.add(cid)

    orphaned_results = result_call_ids - surviving_call_ids
    if orphaned_results:
        messages = [
            m for m in messages
            if not (m.get("role") == "tool" and m.get("tool_call_id") in orphaned_results)
        ]
        log.debug("sanitize: removed %d orphan tool result(s)", len(orphaned_results))

    missing_results = surviving_call_ids - result_call_ids
    if not missing_results:
        return messages

    patched: list[dict[str, Any]] = []
    for msg in messages:
        patched.append(msg)
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or ():
            cid = _get_tool_call_id(tc)
            if cid in missing_results:
                patched.append(
                    {
                        "role": "tool",
                        "content": STUB_TOOL_RESULT,
                        "tool_call_id": cid,
                    }
                )
    log.debug("sanitize: stubbed %d orphan tool call(s)", len(missing_results))
    return patched
