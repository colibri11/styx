"""Юнит-тесты для dialogue batch consolidation handler'а."""

from __future__ import annotations

import datetime as _dt
import json
import logging
import uuid
from typing import Any

import psycopg
import pytest

from styx.emotional.sentiment_batch import K_BATCH, SentimentBatchMetrics
from styx.emotional.state import EmotionalVector
from styx.llm import LLMRateLimiter, OllamaChatClient, OllamaTerminalError
from styx.workers.handlers.dialogue_batch_consolidation import (
    DIALOGUE_BATCH_TASK_TYPE,
    L_BATCH_CHARS,
    OVERLAP_CHARS,
    SYSTEM_PROMPT,
    USER_ONLY_VAD_EMPTY_NOTE,
    USER_ONLY_VAD_FOOTER,
    USER_ONLY_VAD_HEADER,
    USER_ONLY_VAD_MAX_CHARS,
    VAD_INSTRUCTION_REMINDER,
    VAD_INSTRUCTION_REMINDER_EMPTY,
    _validate_batch_response,
    build_archive_ref,
    build_user_only_vad_section,
    build_user_prompt,
    chunk_window_text,
    create_dialogue_batch_handler,
    format_memory_context,
    format_user_only_window,
    format_window,
    truncate,
)
from styx.workers.runtime import HandlerContext, LlmTask


# ── Validator ──────────────────────────────────────────────────────────


def test_validator_skip_path() -> None:
    raw = {
        "skip": True,
        "skip_reason": "короткий служебный обмен",
        "summary": None,
        "archive_hints": None,
        "vad": None,
    }
    parsed = _validate_batch_response(raw)
    assert parsed.skip is True
    assert parsed.skip_reason == "короткий служебный обмен"
    assert parsed.summary is None
    assert parsed.archive_hints is None
    assert parsed.vad is None


def test_validator_scored_path() -> None:
    raw = {
        "skip": False,
        "skip_reason": None,
        "summary": "Мы пришли к решению X.",
        "archive_hints": [{"snippet": "ключевой момент"}],
        "vad": {"valence": 0.5, "arousal": 0.3, "dominance": 0.2},
    }
    parsed = _validate_batch_response(raw)
    assert parsed.skip is False
    assert parsed.summary == "Мы пришли к решению X."
    assert parsed.archive_hints is not None
    assert len(parsed.archive_hints) == 1
    assert parsed.archive_hints[0].snippet == "ключевой момент"
    assert parsed.vad == EmotionalVector(0.5, 0.3, 0.2)


def test_validator_skip_must_have_reason() -> None:
    raw = {"skip": True, "skip_reason": "", "summary": None,
           "archive_hints": None, "vad": None}
    with pytest.raises(ValueError, match="skip_reason"):
        _validate_batch_response(raw)


def test_validator_scored_summary_required() -> None:
    raw = {"skip": False, "skip_reason": None, "summary": None,
           "archive_hints": [], "vad": None}
    with pytest.raises(ValueError, match="summary"):
        _validate_batch_response(raw)


def test_validator_archive_hints_capped_at_two() -> None:
    raw = {
        "skip": False,
        "skip_reason": None,
        "summary": "Сводка",
        "archive_hints": [
            {"snippet": "a"}, {"snippet": "b"}, {"snippet": "c"},
        ],
        "vad": None,
    }
    parsed = _validate_batch_response(raw)
    assert parsed.archive_hints is not None
    assert len(parsed.archive_hints) == 2  # truncated


def test_validator_vad_optional() -> None:
    """vad=null допустим в обеих ветках (skip и scored)."""
    raw = {
        "skip": False, "skip_reason": None, "summary": "summary",
        "archive_hints": [], "vad": None,
    }
    parsed = _validate_batch_response(raw)
    assert parsed.vad is None


def test_validator_vad_invalid_axis() -> None:
    raw = {
        "skip": False, "skip_reason": None, "summary": "s",
        "archive_hints": [],
        "vad": {"valence": 2.0, "arousal": 0, "dominance": 0},  # > 1.0
    }
    with pytest.raises(ValueError, match="valence"):
        _validate_batch_response(raw)


def test_validator_top_level_not_object() -> None:
    with pytest.raises(ValueError, match="object"):
        _validate_batch_response("not a dict")


# ── Chunker ───────────────────────────────────────────────────────────


def test_chunker_short_text_one_chunk() -> None:
    text = "x" * 1000
    chunks = chunk_window_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunker_exactly_l_batch_one_chunk() -> None:
    text = "x" * L_BATCH_CHARS
    chunks = chunk_window_text(text)
    assert len(chunks) == 1


def test_chunker_overlapping_chunks() -> None:
    # 2 × L_BATCH = должно быть несколько chunk'ов с overlap'ом
    text = "x" * (L_BATCH_CHARS * 2)
    chunks = chunk_window_text(text)
    assert len(chunks) >= 2
    # Каждый chunk не больше L_BATCH_CHARS
    assert all(len(c) <= L_BATCH_CHARS for c in chunks)
    # Overlap: первые OVERLAP_CHARS chunk'а N+1 совпадают с последними
    # OVERLAP_CHARS chunk'а N (на 'x'-сплошном тексте это выполняется
    # тривиально).
    step = L_BATCH_CHARS - OVERLAP_CHARS
    # Длина последнего chunk'а: text[step*(N-1) : step*(N-1)+L_BATCH]
    # Для текста 2*L_BATCH chars: chunks = [0:85k], [68k:153k], [136k:170k] = 3.


# ── Formatting ────────────────────────────────────────────────────────


def test_format_window() -> None:
    rows = [
        {"role": "user", "content": "привет"},
        {"role": "assistant", "content": "здравствуй"},
    ]
    text = format_window(rows)
    assert "[user]: привет" in text
    assert "[assistant]: здравствуй" in text


def test_format_memory_context_empty() -> None:
    assert format_memory_context([]) == "(пусто)"


def test_format_memory_context_lines() -> None:
    rows = [{"kind_src": "subjective", "content": "факт"}]
    out = format_memory_context(rows)
    assert "- subjective: факт" in out


def test_build_user_prompt_shape() -> None:
    out = build_user_prompt("[user]: hi", "(пусто)", "user-only-section")
    assert "window:" in out
    assert "memory_context:" in out
    assert "[user]: hi" in out
    assert "user-only-section" in out


# ── User-only VAD section (волна 35 D7 — batch-path role separation) ──


def test_format_user_only_window_filters_to_user_role() -> None:
    rows = [
        {"role": "user", "content": "мне тревожно"},
        {"role": "assistant", "content": "я тебя понимаю"},
        {"role": "user", "content": "спасибо"},
    ]
    text = format_user_only_window(rows)
    assert "[user]: мне тревожно" in text
    assert "[user]: спасибо" in text
    assert "[assistant]" not in text
    assert "я тебя понимаю" not in text


def test_format_user_only_window_empty_when_no_user_rows() -> None:
    rows = [{"role": "assistant", "content": "монолог агента"}]
    assert format_user_only_window(rows) == ""


def test_build_user_only_vad_section_contains_only_user_lines() -> None:
    """(б) секция содержит ТОЛЬКО user-строки данного окна — никаких
    assistant-строк внутри демаркированного блока."""
    rows = [
        {"role": "user", "content": "мне тревожно"},
        {"role": "assistant", "content": "я тебя понимаю"},
        {"role": "user", "content": "спасибо"},
    ]
    section = build_user_only_vad_section(rows)
    assert USER_ONLY_VAD_HEADER in section
    assert USER_ONLY_VAD_FOOTER in section

    body = section.split(USER_ONLY_VAD_HEADER, 1)[1].split(USER_ONLY_VAD_FOOTER, 1)[0]
    assert "[user]: мне тревожно" in body
    assert "[user]: спасибо" in body
    assert "[assistant]" not in body
    assert "я тебя понимаю" not in body
    # Инструкция-напоминание идёт ПОСЛЕ футера, не смешана с данными.
    assert VAD_INSTRUCTION_REMINDER in section
    assert section.index(USER_ONLY_VAD_FOOTER) < section.index(VAD_INSTRUCTION_REMINDER)


def test_build_user_only_vad_section_no_user_rows_is_explicit_not_empty() -> None:
    """Edge case: окно без единой user-реплики — секция не пустая
    (header/footer без содержимого), а явно помечена; инструкция
    говорит вернуть vad: null, а не молчит про отсутствие блока."""
    rows = [{"role": "assistant", "content": "монолог агента, без ответа"}]
    section = build_user_only_vad_section(rows)
    assert USER_ONLY_VAD_HEADER in section
    assert USER_ONLY_VAD_FOOTER in section
    assert USER_ONLY_VAD_EMPTY_NOTE in section
    assert "монолог агента" not in section
    assert VAD_INSTRUCTION_REMINDER_EMPTY in section
    assert "null" in section


def test_build_user_prompt_places_user_only_section_after_window_before_reminder() -> None:
    """(в) порядок в итоговом user-промпте: общий текст окна → ... →
    user-only блок → VAD-инструкция (волна 35 D7)."""
    rows = [
        {"role": "user", "content": "вопрос про qwen3"},
        {"role": "assistant", "content": "qwen3 50k окно"},
    ]
    window_text = format_window(rows)
    section = build_user_only_vad_section(rows)
    prompt = build_user_prompt(window_text, "(пусто)", section)

    idx_general_window = prompt.index("[assistant]: qwen3 50k окно")
    idx_header = prompt.index(USER_ONLY_VAD_HEADER)
    idx_footer = prompt.index(USER_ONLY_VAD_FOOTER)
    idx_reminder = prompt.index(VAD_INSTRUCTION_REMINDER)

    assert idx_general_window < idx_header, (
        "user-only блок должен идти после общего текста окна"
    )
    assert idx_header < idx_footer < idx_reminder, (
        "VAD-инструкция должна идти сразу после user-only блока"
    )
    # Не в самом начале промпта.
    assert prompt.index("window:") < idx_header


def test_truncate_no_op() -> None:
    assert truncate("short", 100) == "short"


def test_truncate_with_ellipsis() -> None:
    out = truncate("x" * 200, 50)
    assert len(out) == 50
    assert out.endswith("…")


def test_build_archive_ref_single_chunk() -> None:
    rows = [
        {"created_at": _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc),
         "role": "user", "content": "hello"},
        {"created_at": _dt.datetime(2026, 5, 2, 12, 5, tzinfo=_dt.timezone.utc),
         "role": "assistant", "content": "hi"},
    ]
    from styx.workers.handlers.dialogue_batch_consolidation import ArchiveHint
    ref = build_archive_ref(rows, "[user]: hello", [ArchiveHint("hello")], None)
    assert ref["kind"] == "dialogue_message"
    assert ref["snippet"] == "hello"
    assert "12:00" in ref["locator"]
    assert "12:05" in ref["locator"]
    assert "#" not in ref["locator"]  # без chunk index


def test_build_archive_ref_chunked_index() -> None:
    rows = [
        {"created_at": _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc),
         "role": "user", "content": "x"},
    ]
    ref = build_archive_ref(rows, "x" * 1000, [], chunk_index=2)
    assert "#2" in ref["locator"]
    assert ref["snippet"] == "x" * 300


def test_system_prompt_mentions_vad() -> None:
    """26b: prompt должен упоминать VAD оценку peer-части."""
    assert "valence" in SYSTEM_PROMPT
    assert "arousal" in SYSTEM_PROMPT
    assert "dominance" in SYSTEM_PROMPT
    assert "role=assistant" in SYSTEM_PROMPT


# ── Handler with migrated_db ───────────────────────────────────────────


class _ScriptedChat(OllamaChatClient):
    """OllamaChatClient, но chat_json возвращает заранее заданный JSON.

    Поддерживает sequence ответов (по одному на chunk) или single response.
    """

    def __init__(self, responses: list[Any] | dict | Exception) -> None:
        super().__init__(base_url="http://x", model="m", max_attempts=1)
        if isinstance(responses, list):
            self._responses = responses
        else:
            self._responses = [responses]
        self.calls = 0
        self.captured_messages: list[list[dict]] = []

    def chat_json(self, messages, *, timeout_s=None, max_attempts=None):  # type: ignore[override]
        self.captured_messages.append(list(messages))
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        resp = self._responses[idx]
        if isinstance(resp, Exception):
            raise resp
        return resp


@pytest.fixture
def db(migrated_db: str):
    conn = psycopg.connect(migrated_db)
    yield conn
    conn.close()


@pytest.fixture
def rate_limit() -> LLMRateLimiter:
    return LLMRateLimiter(capacity=4, refill_per_second=10.0)


def _ctx(conn, llm, rate_limit) -> HandlerContext:
    return HandlerContext(
        conn=conn, llm=llm, rate_limit=rate_limit, logger=logging.getLogger("test"),
    )


def _insert_dialogue(
    conn: psycopg.Connection, agent_id: str,
    rows: list[tuple[str, str, _dt.datetime]],
) -> None:
    with conn.cursor() as cur:
        for role, content, at in rows:
            cur.execute(
                "INSERT INTO memories (agent_id, role, content, kind, created_at) "
                "VALUES (%s, %s, %s, 'episode', %s)",
                (agent_id, role, content, at),
            )
    conn.commit()


def _make_task(payload: dict) -> LlmTask:
    return LlmTask(
        id=uuid.uuid4(), task_type=DIALOGUE_BATCH_TASK_TYPE,
        memory_id=None, payload=payload, retry_count=0,
    )


def test_handler_empty_window_advances_state(db, rate_limit) -> None:
    """Пустое окно → не дёргаем LLM, продвигаем state до window_to."""
    handler = create_dialogue_batch_handler()
    llm = _ScriptedChat({"never": "called"})
    window_to = _dt.datetime.now(tz=_dt.timezone.utc)
    task = _make_task({
        "agent_id": "alpha-empty",
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": window_to.isoformat(),
        "with_overlap": False,
    })
    out = handler(task, _ctx(db, llm, rate_limit))
    assert out.result == {"skipped": "empty_window"}
    assert llm.calls == 0
    db.commit()
    # state продвинут
    from styx.storage.queries import get_batch_state
    state = get_batch_state(db, "alpha-empty")
    assert state is not None
    assert state["last_window_end_at"] == window_to.isoformat()


def test_handler_happy_path_creates_memory(db, rate_limit) -> None:
    agent = "alpha-happy"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    _insert_dialogue(db, agent, [
        ("user", "вопрос про qwen3", base),
        ("assistant", "qwen3 50k окно", base + _dt.timedelta(seconds=10)),
    ])
    handler = create_dialogue_batch_handler()
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None,
        "summary": "Обсудили qwen3:4b-local — 50k окно подходит.",
        "archive_hints": [{"snippet": "qwen3 50k окно"}],
        "vad": None,
    })
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=30)).isoformat(),
        "with_overlap": False,
    })
    out = handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    assert out.skipped_by_llm is False
    assert out.result["chunks"] == 1
    assert len(out.result["memories_created"]) == 1
    assert llm.calls == 1

    with db.cursor() as cur:
        cur.execute(
            "SELECT content, kind, kind_src, archive_ref FROM memories "
            "WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        row = cur.fetchone()
    assert row is not None
    assert "qwen3" in row[0]
    assert row[1] == "episode"
    assert row[2] == "dialogue_batch_consolidation"
    assert row[3]["kind"] == "dialogue_message"


def test_handler_llm_skip_no_memory(db, rate_limit) -> None:
    agent = "alpha-skip"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    _insert_dialogue(db, agent, [
        ("user", "ок", base),
        ("assistant", "ок", base + _dt.timedelta(seconds=5)),
    ])
    handler = create_dialogue_batch_handler()
    llm = _ScriptedChat({
        "skip": True, "skip_reason": "слишком короткое",
        "summary": None, "archive_hints": None, "vad": None,
    })
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=30)).isoformat(),
        "with_overlap": False,
    })
    out = handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    assert out.skipped_by_llm is True
    assert out.result["memories_created"] == []
    assert "слишком короткое" in out.result["skipped_reasons"]

    # Memory не создана
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories "
            "WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        assert cur.fetchone()[0] == 0


def test_handler_schema_mismatch_terminal(db, rate_limit) -> None:
    agent = "alpha-schema"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    _insert_dialogue(db, agent, [
        ("user", "test", base),
    ])
    handler = create_dialogue_batch_handler()
    llm = _ScriptedChat({"skip": "not a bool"})  # invalid
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=10)).isoformat(),
        "with_overlap": False,
    })
    with pytest.raises(OllamaTerminalError, match="schema_mismatch"):
        handler(task, _ctx(db, llm, rate_limit))


def test_handler_vad_apply_first_in_transaction(db, rate_limit) -> None:
    """VAD apply ПЕРВЫМ — emotional_state записан до INSERT memory."""
    agent = "alpha-vad"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    _insert_dialogue(db, agent, [
        ("user", "тревожно", base),
    ])
    metrics = SentimentBatchMetrics()
    handler = create_dialogue_batch_handler(
        batch_sentiment_enabled=True, batch_sentiment_metrics=metrics,
    )
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None,
        "summary": "Диалог был тревожным, обсудили.",
        "archive_hints": [],
        "vad": {"valence": -0.6, "arousal": 0.4, "dominance": -0.3},
    })
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=30)).isoformat(),
        "with_overlap": False,
    })
    handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    # emotional_state has sentiment:batch entry
    with db.cursor() as cur:
        cur.execute(
            "SELECT valence, arousal, dominance, source, metadata "
            "FROM emotional_state WHERE agent_id = %s "
            "AND source = 'sentiment:batch'",
            (agent,),
        )
        row = cur.fetchone()
    assert row is not None
    # K_BATCH=0.4 scale
    assert row[0] == pytest.approx(-0.6 * K_BATCH, abs=0.01)
    assert row[3] == "sentiment:batch"
    assert row[4]["chunks"] == 1
    assert row[4]["vad_samples"] == 1
    assert metrics.snapshot()["applied"] == 1


def test_handler_vad_disabled_no_apply(db, rate_limit) -> None:
    """batch_sentiment_enabled=False → memory создаётся, VAD не apply'ится."""
    agent = "alpha-vad-off"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    _insert_dialogue(db, agent, [
        ("user", "test", base),
    ])
    handler = create_dialogue_batch_handler(batch_sentiment_enabled=False)
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None,
        "summary": "Сводка.",
        "archive_hints": [],
        "vad": {"valence": 0.5, "arousal": 0.5, "dominance": 0.5},
    })
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=30)).isoformat(),
        "with_overlap": False,
    })
    handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    # Memory создана
    with db.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM memories "
            "WHERE agent_id = %s AND kind_src='dialogue_batch_consolidation'",
            (agent,),
        )
        assert cur.fetchone()[0] == 1
        # emotional_state НЕ имеет sentiment:batch записи
        cur.execute(
            "SELECT count(*) FROM emotional_state "
            "WHERE agent_id = %s AND source = 'sentiment:batch'",
            (agent,),
        )
        assert cur.fetchone()[0] == 0


def test_handler_vad_null_no_apply(db, rate_limit) -> None:
    """LLM вернул vad=null → нет apply, метрика skips_no_vad++."""
    agent = "alpha-vad-null"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    _insert_dialogue(db, agent, [
        ("user", "техническое", base),
    ])
    metrics = SentimentBatchMetrics()
    handler = create_dialogue_batch_handler(batch_sentiment_metrics=metrics)
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None,
        "summary": "Технический обмен.",
        "archive_hints": [], "vad": None,
    })
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=30)).isoformat(),
        "with_overlap": False,
    })
    handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    snap = metrics.snapshot()
    assert snap["calls"] == 1
    assert snap["skips_no_vad"] == 1
    assert snap["applied"] == 0


def test_handler_chunked_multiple_calls(db, rate_limit) -> None:
    """Окно > L_BATCH → несколько LLM-вызовов → несколько memories."""
    agent = "alpha-chunked"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    # CHECK ограничение memories.content = 2400 символов. Создаём 50
    # реплик по ~2000 chars каждая → window text ~100k > L_BATCH (85k).
    big_chunk = "x" * 2000
    rows = [
        (
            "user" if i % 2 == 0 else "assistant",
            f"{big_chunk} реплика {i}",
            base + _dt.timedelta(seconds=i),
        )
        for i in range(50)
    ]
    _insert_dialogue(db, agent, rows)
    handler = create_dialogue_batch_handler()
    llm = _ScriptedChat([
        {"skip": False, "skip_reason": None, "summary": "Сводка chunk 0.",
         "archive_hints": [], "vad": None},
        {"skip": False, "skip_reason": None, "summary": "Сводка chunk 1.",
         "archive_hints": [], "vad": None},
        {"skip": False, "skip_reason": None, "summary": "Сводка chunk 2.",
         "archive_hints": [], "vad": None},
    ])
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=120)).isoformat(),
        "with_overlap": False,
    })
    out = handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    assert out.result["chunks"] >= 2
    assert llm.calls >= 2
    # Каждый chunk → отдельная memory
    assert len(out.result["memories_created"]) >= 2


# ── User-only VAD section — end-to-end wiring (волна 35 D7) ────────────


def test_handler_sends_user_only_vad_section_to_llm(db, rate_limit) -> None:
    """(a)(б)(в) сквозная проверка: то, что реально уходит в
    ``ctx.llm.chat_json`` (не только изолированные функции), содержит
    демаркированный user-only блок после общего окна и перед
    VAD-инструкцией, и блок содержит только user-строки этого окна."""
    agent = "alpha-user-only-vad"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    _insert_dialogue(db, agent, [
        ("user", "мне сегодня тревожно", base),
        ("assistant", "я рядом, расскажи подробнее", base + _dt.timedelta(seconds=5)),
        ("user", "спасибо что выслушал", base + _dt.timedelta(seconds=20)),
    ])
    handler = create_dialogue_batch_handler()
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None,
        "summary": "Обсудили тревогу пользователя, стало легче.",
        "archive_hints": [],
        "vad": {"valence": -0.3, "arousal": 0.4, "dominance": -0.1},
    })
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=30)).isoformat(),
        "with_overlap": False,
    })
    handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    assert llm.calls == 1
    messages = llm.captured_messages[0]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    content = messages[1]["content"]

    # (a) блок присутствует в итоговом тексте, отправленном LLM.
    assert USER_ONLY_VAD_HEADER in content
    assert USER_ONLY_VAD_FOOTER in content

    # (б) блок содержит ТОЛЬКО user-строки этого окна.
    body = content.split(USER_ONLY_VAD_HEADER, 1)[1].split(USER_ONLY_VAD_FOOTER, 1)[0]
    assert "[user]: мне сегодня тревожно" in body
    assert "[user]: спасибо что выслушал" in body
    assert "[assistant]" not in body
    assert "я рядом" not in body

    # (в) порядок: общий текст окна → ... → блок → VAD-инструкция.
    idx_general_window = content.index("я рядом, расскажи подробнее")
    idx_header = content.index(USER_ONLY_VAD_HEADER)
    idx_reminder = content.index(VAD_INSTRUCTION_REMINDER)
    assert idx_general_window < idx_header < idx_reminder


def test_handler_all_assistant_window_marks_no_user_replies(db, rate_limit) -> None:
    """Edge case сквозь весь handler: окно без единой user-реплики — LLM
    получает явную пометку "нет реплик пользователя", а не пустой блок."""
    agent = "alpha-no-user-rows"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    _insert_dialogue(db, agent, [
        ("assistant", "агент рассуждает сам с собой", base),
    ])
    handler = create_dialogue_batch_handler()
    llm = _ScriptedChat({
        "skip": False, "skip_reason": None,
        "summary": "Агент зафиксировал собственную мысль.",
        "archive_hints": [], "vad": None,
    })
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=30)).isoformat(),
        "with_overlap": False,
    })
    handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    content = llm.captured_messages[0][1]["content"]
    assert USER_ONLY_VAD_HEADER in content
    assert USER_ONLY_VAD_EMPTY_NOTE in content
    assert VAD_INSTRUCTION_REMINDER_EMPTY in content
    assert "агент рассуждает сам с собой" not in content.split(
        USER_ONLY_VAD_HEADER, 1
    )[1].split(USER_ONLY_VAD_FOOTER, 1)[0]


def test_handler_user_only_vad_section_bounded_on_large_window(db, rate_limit) -> None:
    """Регрессионный фикс (волна 35): раньше ``user_only_vad_section``
    строился из ПОЛНОГО ``rows`` без ограничения размера и
    приклеивался неизменным к промпту КАЖДОГО chunk'а в цикле. При
    большом backlog'е (окно > L_BATCH_CHARS, несколько chunk'ов) это
    могло дать HTTP 400 "exceeds available context size" на каждом
    chunk-вызове — причём неудачный tick не продвигает ``window_from``
    (``set_batch_state`` двигает состояние только при успехе), так что
    окно продолжает расти и отказ повторяется. Теперь секция ограничена
    ``USER_ONLY_VAD_MAX_CHARS`` независимо от размера ``rows``."""
    agent = "alpha-user-only-vad-bounded"
    base = _dt.datetime(2026, 5, 2, 12, 0, tzinfo=_dt.timezone.utc)
    # CHECK constraint memories.content ≤ 2400 (MAX_CONTENT_CHARS) —
    # держим построчный content под лимитом, как в
    # test_handler_chunked_multiple_calls. Преимущественно user-реплики
    # (единственная assistant-реплика — i == 0).
    big_chunk = "x" * 2000
    entries = [
        {
            "role": "assistant" if i == 0 else "user",
            "content": f"{big_chunk} реплика {i}",
        }
        for i in range(150)
    ]
    # sanity: окно действительно большое (≥200k) и user-only текст (до
    # обрезки) значительно больше USER_ONLY_VAD_MAX_CHARS — иначе тест
    # не нагружает обрезку и не проверяет фикс.
    assert len(format_window(entries)) >= 200_000
    assert len(format_user_only_window(entries)) > USER_ONLY_VAD_MAX_CHARS

    rows = [
        (e["role"], e["content"], base + _dt.timedelta(seconds=i))
        for i, e in enumerate(entries)
    ]
    _insert_dialogue(db, agent, rows)
    handler = create_dialogue_batch_handler()
    llm = _ScriptedChat([
        {"skip": False, "skip_reason": None, "summary": f"Сводка {i}.",
         "archive_hints": [], "vad": None}
        for i in range(10)
    ])
    task = _make_task({
        "agent_id": agent,
        "window_from": _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc).isoformat(),
        "window_to": (base + _dt.timedelta(seconds=200)).isoformat(),
        "with_overlap": False,
    })
    handler(task, _ctx(db, llm, rate_limit))
    db.commit()

    assert llm.calls >= 2  # окно > L_BATCH_CHARS → несколько chunk'ов,
    # секция приклеивается к промпту каждого из них.
    for messages in llm.captured_messages:
        content = messages[1]["content"]
        body = content.split(USER_ONLY_VAD_HEADER, 1)[1].split(
            USER_ONLY_VAD_FOOTER, 1
        )[0].strip("\n")
        assert len(body) <= USER_ONLY_VAD_MAX_CHARS
