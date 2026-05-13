"""Юнит-тесты StyxComposer — composition, eviction, sanitization."""

from __future__ import annotations

import pytest

from styx.engine import salient_bridge
from styx.engine.context import (
    STUB_TOOL_RESULT,
    STYX_TAG_ARCHIVE,
    STYX_TAG_DIALOGUE,
    STYX_TAG_RECALL,
    STYX_TAG_SALIENT,
    StyxComposer,
    _sanitize_tool_pairs,
    get_styx_sanitized_blocks_by_tag,
    get_styx_sanitized_blocks_total,
    reset_styx_sanitized_blocks_total,
    wrap_text_with_styx_tag,
)
from styx.engine.salient import SALIENT_MARKER
from styx.storage.recall import RecallResult
from styx.storage.recall_config import DEFAULT_RECALL_CONFIG


def _msg(role: str, content: str = "", **extra) -> dict:
    return {"role": role, "content": content, **extra}


@pytest.fixture(autouse=True)
def _reset_salient_bridge() -> None:
    from styx.engine import focus_tracker
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    reset_styx_sanitized_blocks_total()
    yield
    salient_bridge.reset_all()
    focus_tracker.reset_all()
    reset_styx_sanitized_blocks_total()


# -- identity --------------------------------------------------------------
# (ABC inheritance проверяется в hermes/tests на StyxContextEngine — он
# в styx-hermes наследуется от Hermes ContextEngine. Core StyxComposer
# host-agnostic, без ABC.)


def test_name() -> None:
    e = StyxComposer("test-agent", context_length=128_000)
    assert e.name == "styx"


def test_state_fields_initialized() -> None:
    e = StyxComposer("test-agent", context_length=100_000)
    assert e.last_prompt_tokens == 0
    assert e.last_completion_tokens == 0
    assert e.last_total_tokens == 0
    assert e.compression_count == 0
    assert e.context_length == 100_000
    assert e.threshold_tokens == 75_000  # 0.75 * 100_000
    assert e.protect_first_n == 3
    assert e.protect_last_n == 6


# -- update_from_response --------------------------------------------------


def test_update_from_response_basic() -> None:
    e = StyxComposer("test-agent", context_length=100_000)
    e.update_from_response({
        "prompt_tokens": 1500,
        "completion_tokens": 300,
        "total_tokens": 1800,
    })
    assert e.last_prompt_tokens == 1500
    assert e.last_completion_tokens == 300
    assert e.last_total_tokens == 1800


def test_update_from_response_derives_total() -> None:
    e = StyxComposer("test-agent", context_length=100_000)
    e.update_from_response({"prompt_tokens": 100, "completion_tokens": 50})
    assert e.last_total_tokens == 150


def test_update_from_response_handles_empty() -> None:
    e = StyxComposer("test-agent", context_length=100_000)
    e.update_from_response({})
    e.update_from_response(None)  # type: ignore[arg-type]
    assert e.last_prompt_tokens == 0


# -- should_compress / model switch ---------------------------------------


def test_should_compress_always_true() -> None:
    e = StyxComposer("test-agent", context_length=100_000)
    assert e.should_compress() is True
    assert e.should_compress(prompt_tokens=1) is True
    assert e.should_compress(prompt_tokens=10**9) is True


def test_update_model_recalculates_threshold() -> None:
    e = StyxComposer("test-agent", context_length=10_000)
    assert e.threshold_tokens == 7_500
    e.update_model(model="gpt-x", context_length=200_000)
    assert e.context_length == 200_000
    assert e.threshold_tokens == 150_000


# -- compress: no-op paths --------------------------------------------------


def test_compress_noop_when_tokens_unknown() -> None:
    e = StyxComposer("test-agent", context_length=100_000)
    msgs = [_msg("user", "u"), _msg("assistant", "a")]
    out = e.compress(msgs, current_tokens=None)
    assert out == msgs
    assert e.compression_count == 0


def test_compress_noop_when_under_threshold() -> None:
    e = StyxComposer("test-agent", context_length=100_000)
    msgs = [_msg("user", f"u{i}") for i in range(20)]
    out = e.compress(msgs, current_tokens=10_000)  # < 75_000
    assert out == msgs
    assert e.compression_count == 0


def test_compress_noop_when_too_few_messages() -> None:
    e = StyxComposer("test-agent", context_length=100_000)
    # 9 = protect_first_n (3) + protect_last_n (6)
    msgs = [_msg("user", f"u{i}") for i in range(9)]
    out = e.compress(msgs, current_tokens=99_000)
    assert out == msgs
    assert e.compression_count == 0


# -- compress: eviction ----------------------------------------------------


def test_compress_evicts_middle_when_overflow() -> None:
    e = StyxComposer("test-agent", 
        context_length=100_000,
        protect_first_n=2,
        protect_last_n=3,
    )
    msgs = [_msg("user", f"m{i}") for i in range(10)]
    out = e.compress(msgs, current_tokens=99_000)

    assert len(out) == 5
    assert [m["content"] for m in out] == ["m0", "m1", "m7", "m8", "m9"]
    assert e.compression_count == 1


def test_compress_head_is_byte_stable_across_calls() -> None:
    """Голова стабильна между turn'ами при неменяющемся messages prefix.

    Это предусловие auto-prefix-cache на провайдере.
    """
    e = StyxComposer("test-agent", 
        context_length=100_000,
        protect_first_n=2,
        protect_last_n=3,
    )

    base = [_msg("user", f"m{i}") for i in range(10)]
    out1 = e.compress(list(base), current_tokens=99_000)

    # Симулируем следующий turn — добавили новый assistant + user в хвост
    base.append(_msg("assistant", "new-assistant"))
    base.append(_msg("user", "new-user"))
    out2 = e.compress(list(base), current_tokens=99_000)

    head1 = out1[: e.protect_first_n]
    head2 = out2[: e.protect_first_n]
    assert head1 == head2  # байт-в-байт


# -- sanitize tool pairs ---------------------------------------------------


def test_sanitize_drops_orphan_tool_result() -> None:
    msgs = [
        _msg("user", "q"),
        _msg("tool", "result", tool_call_id="call-x"),  # orphan
        _msg("assistant", "a"),
    ]
    out = _sanitize_tool_pairs(msgs)
    assert all(m.get("tool_call_id") != "call-x" for m in out)
    assert len(out) == 2


def test_sanitize_stubs_orphan_tool_call() -> None:
    msgs = [
        _msg(
            "assistant",
            "",
            tool_calls=[{"id": "call-y", "type": "function"}],
        ),
        _msg("user", "next"),
    ]
    out = _sanitize_tool_pairs(msgs)
    # стаб-result должен быть прямо после assistant
    assert out[0]["role"] == "assistant"
    assert out[1]["role"] == "tool"
    assert out[1]["tool_call_id"] == "call-y"
    assert out[1]["content"] == STUB_TOOL_RESULT
    assert out[2]["role"] == "user"


def test_sanitize_handles_simplenamespace_tool_call() -> None:
    """Hermes хранит tool_calls и как dict, и как SimpleNamespace."""
    from types import SimpleNamespace

    msgs = [
        _msg(
            "assistant",
            "",
            tool_calls=[SimpleNamespace(id="call-z", type="function")],
        ),
    ]
    out = _sanitize_tool_pairs(msgs)
    assert any(m.get("tool_call_id") == "call-z" for m in out)


def test_sanitize_keeps_matched_pairs() -> None:
    msgs = [
        _msg("user", "q"),
        _msg(
            "assistant",
            "",
            tool_calls=[{"id": "c1"}, {"id": "c2"}],
        ),
        _msg("tool", "r1", tool_call_id="c1"),
        _msg("tool", "r2", tool_call_id="c2"),
    ]
    out = _sanitize_tool_pairs(msgs)
    assert out == msgs


def test_sanitize_empty_list() -> None:
    assert _sanitize_tool_pairs([]) == []


# -- compress + sanitize integration --------------------------------------


def test_compress_stubs_orphan_call_when_result_evicted() -> None:
    """assistant с tool_call в protected head, его result в evicted middle.

    Fix #14: head расширяется чтобы включить tool result → пара закрыта,
    стаб НЕ нужен. Если расширение поглощает всь history → fallback no-op.

    Для демонстрации stub используем сценарий где tool result находится
    достаточно далеко — за tail_start, т.е. в tail, а не в middle.
    """
    e = StyxComposer("test-agent", 
        context_length=10_000,
        protect_first_n=3,
        protect_last_n=2,
    )

    # В этом сценарии assistant с tool_calls на индексе 2.
    # tool result на индексе 3 — между head_end(3) и tail_start(5).
    # Fix #14: head расширяется до 5 (включая tool result),
    # head_end(5) == tail_start(5) → fallback no-op (eviction не происходит).
    msgs = [
        _msg("user", "h0"),
        _msg("user", "h1"),
        _msg("assistant", "", tool_calls=[{"id": "kept-call"}]),  # index 2
        _msg("tool", "result", tool_call_id="kept-call"),          # index 3
        _msg("user", "filler"),                                     # index 4
        _msg("user", "t-1"),                                        # index 5
        _msg("user", "t-0"),                                        # index 6
    ]
    out = e.compress(msgs, current_tokens=9_000)

    # Расширение head до index 5 == tail_start → fallback, все сообщения сохранены
    assert len(out) == len(msgs)
    # tool result сохранён как настоящий, не стаб
    tool_results = [m for m in out if m.get("role") == "tool"]
    assert len(tool_results) == 1
    assert tool_results[0]["content"] == "result"
    assert not any(m.get("content") == STUB_TOOL_RESULT for m in out)


def test_compress_stubs_orphan_call_when_result_in_middle() -> None:
    """Stub вставляется когда tool result в evicted middle, не сразу за assistant.

    Fix #14: расширение head останавливается на non-tool-pair сообщении.
    Если между assistant[tool_calls] и tool result стоит user — расширение
    остановится, tool result уйдёт в evicted middle → orphan → stub.
    """
    e = StyxComposer("test-agent", 
        context_length=10_000,
        protect_first_n=2,
        protect_last_n=3,
    )

    # len=8, tail_start = max(2, 8-3) = 5
    # messages[1]=assistant[call] → head_end=3
    # messages[2]=user_filler → стоп (False)
    # head=[0,1,2], tool_result на index 3 → evicted middle → orphan
    msgs = [
        _msg("user", "h0"),
        _msg("assistant", "", tool_calls=[{"id": "call-mid"}]),  # index 1
        _msg("user", "filler-2"),                                 # index 2 — прерывает расширение
        _msg("tool", "mid-result", tool_call_id="call-mid"),      # index 3 — в middle
        _msg("user", "filler-4"),                                 # index 4 — в middle
        _msg("user", "t-5"),                                      # index 5 — tail
        _msg("user", "t-6"),                                      # index 6 — tail
        _msg("user", "t-7"),                                      # index 7 — tail
    ]
    out = e.compress(msgs, current_tokens=9_000)

    # assistant остался без result → стаб
    stubs = [m for m in out if m.get("content") == STUB_TOOL_RESULT]
    assert len(stubs) == 1, "стаб должен быть вставлен для orphan tool_call"
    assert stubs[0]["tool_call_id"] == "call-mid"
    # mid-result в evicted middle — его нет в output
    assert not any(m.get("content") == "mid-result" for m in out)
    assert e.compression_count == 1


def test_compress_drops_orphan_tool_result_when_call_evicted() -> None:
    """assistant с tool_call попадает в evicted middle, result в tail.

    После eviction'а result остался без call'а — sanitize выкидывает result.
    """
    e = StyxComposer("test-agent", 
        context_length=10_000,
        protect_first_n=2,
        protect_last_n=2,
    )

    msgs = [
        _msg("user", "h0"),
        _msg("user", "h1"),
        _msg("assistant", "", tool_calls=[{"id": "evicted"}]),  # уйдёт
        _msg("user", "filler"),
        _msg("tool", "result", tool_call_id="evicted"),  # tail[0] — orphan
        _msg("user", "t-0"),
    ]
    out = e.compress(msgs, current_tokens=9_000)

    # tool result остался один в tail — должен быть удалён.
    assert all(m.get("tool_call_id") != "evicted" for m in out)
    assert [m["content"] for m in out if m["role"] != "tool"] == ["h0", "h1", "t-0"]


def test_compress_pure_eviction_no_orphans() -> None:
    """Eviction без tool_calls в head/tail — sanitize no-op."""
    e = StyxComposer("test-agent", 
        context_length=10_000,
        protect_first_n=2,
        protect_last_n=2,
    )
    msgs = [
        _msg("user", f"m{i}") for i in range(10)
    ]
    out = e.compress(msgs, current_tokens=9_000)
    assert [m["content"] for m in out] == ["m0", "m1", "m8", "m9"]


# -- #6: no overlap at boundary -------------------------------------------


def test_compress_no_overlap_at_boundary() -> None:
    """protect_first_n=6, protect_last_n=6, len=10: сообщения[4],[5] не дублируются.

    Bug #6: tail = messages[-6:] = messages[4:] → [4],[5] попадали и в head и в tail.
    Фикс: tail_start = max(6, 10-6) = 6, tail = messages[6:] — нет пересечения.
    """
    e = StyxComposer("test-agent", 
        context_length=100_000,
        protect_first_n=6,
        protect_last_n=6,
    )
    msgs = [_msg("user", f"m{i}") for i in range(10)]
    out = e.compress(msgs, current_tokens=99_000)

    # Нет дублей — каждый content встречается ровно раз
    contents = [m["content"] for m in out]
    assert len(contents) == len(set(contents)), f"duplicates found: {contents}"
    # head = m0..m5, tail = m6..m9
    assert contents == ["m0", "m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]
    # При len == protect_first_n + protect_last_n eviction no-op (ничего не эвиктировано)
    assert e.compression_count == 0


def test_compress_no_overlap_exact_protection_boundary() -> None:
    """protect_first_n=4, protect_last_n=4, len=9: mid=[4] эвиктируется, нет дублей."""
    e = StyxComposer("test-agent", 
        context_length=100_000,
        protect_first_n=4,
        protect_last_n=4,
    )
    msgs = [_msg("user", f"m{i}") for i in range(9)]
    out = e.compress(msgs, current_tokens=99_000)

    contents = [m["content"] for m in out]
    assert len(contents) == len(set(contents)), f"duplicates found: {contents}"
    assert contents == ["m0", "m1", "m2", "m3", "m5", "m6", "m7", "m8"]
    assert e.compression_count == 1


# -- #14: head extension for tool-pair closure ----------------------------


def test_compress_extends_head_to_close_tool_pair() -> None:
    """assistant с tool_calls на границе head → head расширяется до tool result.

    Bug #14: head обрывался на assistant[tool_calls] → sanitize вставлял стаб
    прямо в head, меняя его между turn'ами.
    Фикс: head расширяется пока последнее сообщение — незакрытый tool-pair.
    """
    e = StyxComposer("test-agent", 
        context_length=100_000,
        protect_first_n=3,
        protect_last_n=2,
    )
    # messages[2] = assistant с tool_calls (граница protect_first_n=3)
    # messages[3] = tool result (пара к [2])
    # messages[4,5,6] = filler (уйдут в evicted middle)
    # messages[7,8] = tail
    msgs = [
        _msg("user", "h0"),
        _msg("user", "h1"),
        _msg("assistant", "", tool_calls=[{"id": "tc-1"}]),  # index 2
        _msg("tool", "result-1", tool_call_id="tc-1"),       # index 3 — пара
        _msg("user", "filler-4"),
        _msg("user", "filler-5"),
        _msg("user", "filler-6"),
        _msg("user", "t-7"),
        _msg("user", "t-8"),
    ]
    out = e.compress(msgs, current_tokens=99_000)

    # head должен включать [0,1,2,3] — tool-pair закрыта
    head_contents = []
    for m in out:
        if m["content"] in ("t-7", "t-8"):
            break
        head_contents.append(m["content"])

    assert "result-1" in head_contents, (
        "tool result должен быть в head после расширения, а не стаб"
    )
    # Стаба быть не должно
    assert not any(m["content"] == STUB_TOOL_RESULT for m in out), (
        "стаб не должен появляться когда пара закрыта расширением head"
    )
    assert e.compression_count == 1


def test_compress_fallback_when_extension_absorbs_all() -> None:
    """Если расширение head поглощает весь history — fallback без eviction."""
    e = StyxComposer("test-agent", 
        context_length=100_000,
        protect_first_n=2,
        protect_last_n=2,
    )
    # assistant с tool_calls стоит так что расширение поглощает весь список
    msgs = [
        _msg("user", "h0"),
        _msg("assistant", "", tool_calls=[{"id": "tc-x"}]),  # index 1
        _msg("tool", "res-x", tool_call_id="tc-x"),           # index 2
        _msg("user", "t0"),
    ]
    # protect_first_n=2 → head_end=2 → last=assistant[tool_calls] → расширяем
    # head_end=3 → last=tool → расширяем → head_end=4 == tail_start=2 → fallback
    out = e.compress(msgs, current_tokens=99_000)

    assert e.compression_count == 0
    assert len(out) == 4


# -- on_session_reset ------------------------------------------------------


# -- Волна 9: salient memories injection ---------------------------------


class _StubQueries:
    pass


class _StubEmbed:
    @property
    def dim(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        return [0.0] * 768


def _make_hit(content: str = "remembered fact"):
    import uuid

    from styx.storage.queries import MemoryHit

    return MemoryHit(
        id=uuid.uuid4(),
        agent_id="test",
        kind="episode",
        role="user",
        content=content,
        metadata={},
        created_at=None,
        score=0.7,
        match_score=0.7,
    )


def _enable_salient(
    monkeypatch: pytest.MonkeyPatch,
    *,
    content: str = "remembered fact",
    min_query_len: int = 10,
) -> None:
    salient_bridge.configure("test-agent", 
        queries=_StubQueries(),
        embed_client=_StubEmbed(),
        recall_config=DEFAULT_RECALL_CONFIG,
        timeout_s=1.0,
        min_query_len=min_query_len,
    )

    def fake_recall(**_kw):
        return RecallResult(
            memories=[_make_hit(content)],
            queried_count=1,
            internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)


def test_compress_inserts_salient_in_no_op_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Волна 26.5: salient вставляется ПЕРЕД последним сообщением окна.

    Это даёт стабильный префикс (head + middle + tail-без-последнего)
    между turn'ами для prompt-cache провайдеров. До волны 26.5 salient
    шёл сразу за head'ом — это ломало кэш.
    """
    _enable_salient(monkeypatch, content="apples-fact")

    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("system", "you are styx assistant"),
        _msg("user", "first turn user content here please"),
        _msg("assistant", "ok understood"),
        _msg("user", "second turn user content here please"),
    ]
    out = e.compress(msgs, current_tokens=None)

    # Префикс окна стабилен — system + первый user + assistant.
    assert out[0]["role"] == "system"
    assert out[1]["content"] == "first turn user content here please"
    assert out[2]["role"] == "assistant"
    # Salient — перед последним user (= len(body)-1 после insert'а это
    # индекс len-2). На месте было 4 сообщения → после insert 5,
    # salient на индексе 3.
    assert out[3]["role"] == "user"
    assert SALIENT_MARKER in out[3]["content"]
    assert "apples-fact" in out[3]["content"]
    # Salient обёрнут в <styx-salient>...</styx-salient> (волны 26.5 + 30).
    assert out[3]["content"].startswith("<styx-salient>\n")
    assert out[3]["content"].endswith("\n</styx-salient>")
    # Последнее — оригинальный last user.
    assert out[4]["content"] == "second turn user content here please"


def test_compress_no_salient_when_bridge_unset() -> None:
    """Без configure() — поведение как до волны 9 (regression guard)."""
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", "first user content here please"),
        _msg("assistant", "ok"),
        _msg("user", "second user content here please"),
    ]
    out = e.compress(msgs, current_tokens=None)
    assert out == msgs


def test_compress_no_salient_when_no_user_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_salient(monkeypatch)
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("system", "boot"),
        _msg("assistant", "hi"),
    ]
    out = e.compress(msgs, current_tokens=None)
    # build_salient_block вернёт None потому что нет user'а
    assert out == msgs


def test_compress_inserts_salient_in_eviction_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Волна 26.5: salient вставляется перед последним сообщением body.

    body = head(2) + middle_keep(0) + tail(2) = 4 messages; salient на
    индексе 3 (= len-1) → после insert'а out имеет 5 messages, salient
    на индексе 3, последнее — оригинальный last message.
    """
    _enable_salient(monkeypatch, content="evict-fact")

    e = StyxComposer("test-agent",
        context_length=10_000, protect_first_n=2, protect_last_n=2,
    )
    # 10 messages — eviction случится
    msgs = [_msg("user", f"long-user-content-{i:02d}-here") for i in range(10)]
    out = e.compress(msgs, current_tokens=9_000)

    # head + tail = 4 (m0, m1, m8, m9); salient вставляется перед m9.
    assert out[0]["content"] == "long-user-content-00-here"
    assert out[1]["content"] == "long-user-content-01-here"
    assert out[2]["content"] == "long-user-content-08-here"
    # Salient на индексе 3 (= перед последним)
    assert out[3]["role"] == "user"
    assert SALIENT_MARKER in out[3]["content"]
    assert "evict-fact" in out[3]["content"]
    assert out[3]["content"].startswith("<styx-salient>\n")
    assert out[3]["content"].endswith("\n</styx-salient>")
    # Последнее — оригинальное последнее сообщение
    assert out[4]["content"] == "long-user-content-09-here"
    assert e.compression_count == 1


def test_compress_first_n_byte_stable_across_turns_with_salient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Волна 26.5: префикс окна стабилен между turn'ами.

    Salient теперь вставляется ПЕРЕД последним message — значит вся
    history до last message остаётся стабильной (включая первые
    protect_first_n). Это и есть инвариант prompt-cache: provider
    кеширует префикс до salient'а, recompute только salient + last
    user.
    """
    contents = iter(["recall-turn-1", "recall-turn-2"])

    salient_bridge.configure("test-agent",
        queries=_StubQueries(),
        embed_client=_StubEmbed(),
        recall_config=DEFAULT_RECALL_CONFIG,
        timeout_s=1.0,
        min_query_len=10,
    )

    def fake_recall(**_kw):
        return RecallResult(
            memories=[_make_hit(next(contents))],
            queried_count=1,
            internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)

    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    base = [
        _msg("system", "boot styx"),
        _msg("user", "first user content turn 1 here"),
        _msg("assistant", "ok-1"),
        _msg("user", "second user content turn 1 here"),
    ]
    out1 = e.compress(list(base), current_tokens=None)

    base.append(_msg("assistant", "ok-2"))
    base.append(_msg("user", "next-turn user content fully different topic"))
    out2 = e.compress(list(base), current_tokens=None)

    # Первые 3 — байт-стабильны (= digest stable; см. compute_prefix_digest).
    assert out1[:3] == out2[:3]
    # Турн 1: out1 = [base 4] + [salient] = 5 messages, salient на 3.
    # Турн 2: out2 = [base 6] + [salient] = 7 messages, salient на 5.
    salient1 = next(m for m in out1 if SALIENT_MARKER in m.get("content", ""))
    salient2 = next(m for m in out2 if SALIENT_MARKER in m.get("content", ""))
    assert salient1["content"] != salient2["content"]
    # Salient обёрнут в styx-salient теги (волна 30).
    assert salient1["content"].startswith("<styx-salient>\n")
    assert salient2["content"].endswith("\n</styx-salient>")
    # Salient — на позиции len-2 (перед last user).
    assert out1[-2] is salient1 or SALIENT_MARKER in out1[-2]["content"]
    assert out2[-2] is salient2 or SALIENT_MARKER in out2[-2]["content"]


def test_compress_salient_byte_stable_within_epoch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Волна 10: salient block байт-идентичен между двумя compress'ами одной
    эпохи (стабильная тема). Закрывает условие переоткрытия из § 23.2.

    Волна 26.5: позиция salient'а — len-2 (перед last user). Тут
    проверяем что content идентичен независимо от позиции.
    """
    from styx.engine import focus_tracker

    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)

    calls = {"n": 0}

    def fake_recall(**_kw):
        calls["n"] += 1
        return RecallResult(
            memories=[_make_hit(f"epoch-fact-{calls['n']}")],
            queried_count=1, internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)

    same_embed = [1.0] + [0.0] * 767

    class _DetEmbed:
        @property
        def dim(self): return 768
        def embed(self, t): return list(same_embed)

    salient_bridge.configure("test-agent",
        queries=_StubQueries(),
        embed_client=_DetEmbed(),
        recall_config=DEFAULT_RECALL_CONFIG,
        timeout_s=1.0,
        min_query_len=10,
    )

    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    base = [
        _msg("system", "you are styx assistant"),
        _msg("user", "stable topic — first version of the question"),
        _msg("assistant", "ok"),
        _msg("user", "stable topic — second version of the same question"),
    ]
    out1 = e.compress(list(base), current_tokens=None)

    base.append(_msg("assistant", "got it"))
    base.append(_msg("user", "stable topic — third version still about the same"))
    out2 = e.compress(list(base), current_tokens=None)

    # Salient block — байт-идентичен между turn'ами (cache hit).
    salient1 = next(m for m in out1 if SALIENT_MARKER in m.get("content", ""))
    salient2 = next(m for m in out2 if SALIENT_MARKER in m.get("content", ""))
    assert salient1 == salient2
    assert calls["n"] == 1  # recall сделан только один раз


def test_compress_salient_does_not_break_tool_pair_integrity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Salient = role=user без tool_call_id — sanitize не trip'ается на нём.

    Волна 26.5: salient вставлен перед последним сообщением — это
    user, и tool-pair tc-1 остаётся закрытой выше salient'а.
    """
    _enable_salient(monkeypatch)

    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", "first long user content here please"),
        _msg("assistant", "", tool_calls=[{"id": "tc-1"}]),
        _msg("tool", "result-1", tool_call_id="tc-1"),
        _msg("assistant", "ok"),
        _msg("user", "follow up user content here please"),
    ]
    out = e.compress(msgs, current_tokens=None)

    # Salient на индексе len-2 (= перед last user).
    salient_idx = next(
        i for i, m in enumerate(out) if SALIENT_MARKER in m.get("content", "")
    )
    assert salient_idx == len(out) - 2
    # все tool_call/tool_result пары сохранены
    assert any(m.get("role") == "assistant" and m.get("tool_calls") for m in out)
    assert any(
        m.get("role") == "tool" and m.get("tool_call_id") == "tc-1" for m in out
    )
    # Стаба нет
    assert not any(m.get("content") == STUB_TOOL_RESULT for m in out)


# -- Волна 26.5: cache-friendly placement + defensive markers ------------


def test_compress_salient_wrapped_in_styx_salient_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Salient.content обёрнут в <styx-salient>...</styx-salient> (волна 30)."""
    _enable_salient(monkeypatch, content="wrapped-fact")
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", "first long user content here please"),
        _msg("user", "second long user content here please"),
    ]
    out = e.compress(msgs, current_tokens=None)
    salient = next(m for m in out if SALIENT_MARKER in m.get("content", ""))
    assert salient["content"].startswith("<styx-salient>\n")
    assert salient["content"].endswith("\n</styx-salient>")
    # Marker и фактический content внутри обёртки.
    assert SALIENT_MARKER in salient["content"]
    assert "wrapped-fact" in salient["content"]
    # Generic <styx> тег больше НЕ используется для inject'а (D5).
    assert "<styx>" not in salient["content"]
    assert "</styx>" not in salient["content"]


def test_compress_sanitizes_styx_blocks_from_input(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Defensive: legacy <styx>...</styx> в input messages — вырезаем
    (backwards-compat волны 30 D2 для historical persist'а эпохи 26.5)."""
    import logging as _logging

    _enable_salient(monkeypatch, content="fresh-salient")
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    polluted = (
        "real user content "
        "<styx>\nold leaked salient block\n</styx> "
        "more real content"
    )
    msgs = [
        _msg("user", polluted),
        _msg("user", "second long user content here please"),
    ]
    with caplog.at_level(_logging.WARNING):
        out = e.compress(msgs, current_tokens=None)

    # Strong assert — все non-salient messages не должны содержать ни
    # legacy `<styx>`, ни family `<styx-…>` после sanitize. Salient сам
    # обёрнут в `<styx-salient>` и должен остаться — пропускаем через
    # marker-check ("fresh-salient" — content из _enable_salient).
    for i, msg in enumerate(out):
        content = msg.get("content", "")
        if isinstance(content, str) and "fresh-salient" not in content:
            assert "<styx>" not in content and "<styx-" not in content, (
                f"out[{i}] содержит styx-тег после sanitize: {content!r}"
            )
            assert "</styx>" not in content and "</styx-" not in content, (
                f"out[{i}] содержит styx закрывающий тег после sanitize: {content!r}"
            )
    # В первом message не должно быть остатков "old leaked".
    user_blob = "".join(m["content"] for m in out if m["role"] == "user" and SALIENT_MARKER not in m["content"])
    assert "old leaked salient block" not in user_blob
    assert "real user content" in user_blob
    # Warning вылетел.
    assert any("sanitized" in rec.message for rec in caplog.records)


def test_compress_drops_message_when_only_styx_block(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Если message целиком был <styx>...</styx> → удаляется.

    Fix 3 (волна 26.5): WARNING лог обязан появиться — production
    observability видит когда runtime вырезает leaked salient.
    """
    import logging as _logging

    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    # salient_bridge не configured — salient None, проверяем чистый sanitize-эффект.
    msgs = [
        _msg("user", "real content here"),
        _msg("user", "<styx>\nfully leaked salient\n</styx>"),
        _msg("user", "another real content"),
    ]
    with caplog.at_level(_logging.WARNING, logger="styx.engine.context"):
        out = e.compress(msgs, current_tokens=None)
    contents = [m["content"] for m in out]
    assert "real content here" in contents
    assert "another real content" in contents
    # Полностью-styx message удалён.
    assert not any("leaked" in c for c in contents)
    assert len(out) == 2
    # WARNING лог про sanitize — обязателен.
    assert "sanitized" in caplog.text.lower(), (
        f"ожидался WARNING с 'sanitized', получено: {caplog.text!r}"
    )


def test_compress_sanitizes_multiple_styx_blocks_one_message() -> None:
    """Внутри одного content — несколько <styx>...</styx> → все вырезаются."""
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg(
            "user",
            "before <styx>\nblock1\n</styx> middle <styx>\nblock2\n</styx> after",
        ),
    ]
    out = e.compress(msgs, current_tokens=None)
    assert len(out) == 1
    assert "block1" not in out[0]["content"]
    assert "block2" not in out[0]["content"]
    assert "before " in out[0]["content"]
    assert "after" in out[0]["content"]


def test_compress_passthrough_when_no_styx_blocks() -> None:
    """Без <styx> в input — sanitize no-op, никаких WARNING."""
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [_msg("user", "plain content"), _msg("user", "more plain")]
    out = e.compress(msgs, current_tokens=None)
    # Сообщения дошли как были (salient bridge не configured).
    assert out == msgs


def test_compress_sanitizes_multimodal_text_parts() -> None:
    """Multi-modal list-content sanitize'ится для text-частей.

    До мини-волны 26.8 round 6 list-content был pass-through (TODO волны
    27+). Pi-embedded-runner (OpenClaw 2026.5.7) передаёт messages в
    multi-modal shape, и без sanitize multi-part'ов `<styx>` leak
    переживал бы persist-цикл.
    """
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", [{"type": "text", "text": "<styx>\nleaked\n</styx>"}]),
        _msg("user", "trailing text"),
    ]
    out = e.compress(msgs, current_tokens=None)
    # Multi-modal: text-part был полностью inside <styx>...</styx>, после
    # sanitize text стал пустой → text-part drop'нут. Других parts в
    # message не было → message целиком удалён.
    # Финальный output содержит только trailing user-message.
    assert len(out) == 1
    assert out[0]["content"] == "trailing text"


def test_compress_sanitizes_multimodal_partial_text() -> None:
    """Multi-modal с частичным <styx>: text-part остаётся с очищенным
    text'ом, остальные parts (image) сохраняются."""
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg(
            "user",
            [
                {"type": "text", "text": "Вопрос: <styx>\nsalient\n</styx> что это?"},
                {"type": "image", "source": "..."},
            ],
        ),
    ]
    out = e.compress(msgs, current_tokens=None)
    assert len(out) == 1
    content = out[0]["content"]
    assert isinstance(content, list)
    # Text-part: <styx>...</styx> вырезан, остался prefix + suffix.
    text_parts = [p for p in content if p.get("type") == "text"]
    assert len(text_parts) == 1
    assert "<styx>" not in text_parts[0]["text"]
    assert "salient" not in text_parts[0]["text"]
    assert "Вопрос:" in text_parts[0]["text"]
    assert "что это?" in text_parts[0]["text"]
    # Image part сохранён.
    image_parts = [p for p in content if p.get("type") == "image"]
    assert len(image_parts) == 1


def test_find_last_user_text_handles_multimodal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """salient.build_salient_block для pi-embedded messages с
    multi-modal content должен находить последний user-message.

    Round 6 fix. До этого `_find_last_user_text` искал только
    `isinstance(content, str)` → multi-modal pass-through → salient
    skip → boевые агенты не получали salient в production.
    """
    _enable_salient(monkeypatch, content="multimodal-fact")
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("system", "sys"),
        _msg("user", [{"type": "text", "text": "u1 text part содержательный вопрос пожалуйста"}]),
        _msg("assistant", "a1"),
        _msg(
            "user",
            [{"type": "text", "text": "u2 content with multi-modal text part here"}],
        ),
    ]
    out = e.compress(msgs, current_tokens=None)
    # Salient должен быть вставлен (last_user найден через multi-modal extractor).
    styx_msgs = [
        m for m in out
        if isinstance(m.get("content"), str) and "<styx-salient>" in m["content"]
    ]
    assert len(styx_msgs) == 1, (
        f"salient должен быть найден для multi-modal user-message: out={out}"
    )
    assert "multimodal-fact" in styx_msgs[0]["content"]


def test_compress_salient_position_single_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Edge case: len(body)==1 → salient вставлен перед единственным
    сообщением (insert_at=0)."""
    _enable_salient(monkeypatch, content="single-fact")
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [_msg("user", "the only long user message here please")]
    out = e.compress(msgs, current_tokens=None)
    assert len(out) == 2
    # Salient идёт первым.
    assert SALIENT_MARKER in out[0]["content"]
    assert out[1]["content"] == "the only long user message here please"


def test_compress_salient_position_with_eviction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """В eviction-пути salient тоже идёт перед последним message."""
    _enable_salient(monkeypatch, content="evict-fact")
    e = StyxComposer("test-agent",
        context_length=10_000, protect_first_n=2, protect_last_n=2,
    )
    msgs = [_msg("user", f"long-user-content-{i:02d}-here") for i in range(10)]
    out = e.compress(msgs, current_tokens=9_000)
    # Последнее сообщение — оригинальное (m9), salient — перед ним.
    assert out[-1]["content"] == "long-user-content-09-here"
    assert SALIENT_MARKER in out[-2]["content"]


def test_compress_prefix_stable_across_turns_for_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix 1 (волна 26.5): байтовый prefix окна стабилен между turn'ами.

    Это ключевой инвариант волны: provider-cache prefix (head + middle +
    tail-без-последнего) не меняется при добавлении нового turn'а, так
    как salient вставлен ПЕРЕД последним message (insert_at = len-1).
    Без этого assert claim про cache hit unverified.

    Salient содержит deterministic-фиксированный content (через _DetEmbed
    + один и тот же fact в recall) → focus_tracker одна эпоха → salient
    блок байт-идентичен между turn'ами тоже. Но тест assert'ит ТОЛЬКО
    prefix до salient'а — этого достаточно для prefix-cache.
    """
    import hashlib

    from styx.engine import focus_tracker

    focus_tracker.configure("cache-agent", window_size=3, drift_threshold=0.4)

    def fake_recall(**_kw):
        return RecallResult(
            memories=[_make_hit("stable-fact")],
            queried_count=1,
            internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)

    same_embed = [1.0] + [0.0] * 767

    class _DetEmbed:
        @property
        def dim(self): return 768
        def embed(self, t): return list(same_embed)

    salient_bridge.configure(
        "cache-agent",
        queries=_StubQueries(),
        embed_client=_DetEmbed(),
        recall_config=DEFAULT_RECALL_CONFIG,
        timeout_s=1.0,
        min_query_len=10,
    )

    e = StyxComposer(
        "cache-agent", context_length=100_000, protect_first_n=3, protect_last_n=6
    )

    base = [
        _msg("system", "you are styx assistant"),
        _msg("user", "first user content turn one here please"),
        _msg("assistant", "ok-1"),
        _msg("user", "follow up question on the same stable topic here"),
    ]
    out1 = e.compress(list(base), current_tokens=None)

    base.append(_msg("assistant", "ok-2"))
    base.append(_msg("user", "third turn user content on the same stable topic"))
    out2 = e.compress(list(base), current_tokens=None)

    # Salient вставлен на индекс len-1 ПЕРЕД последним message.
    # Найдём salient в обоих out'ах, prefix = всё ДО salient'а.
    salient_idx_1 = next(
        i for i, m in enumerate(out1) if SALIENT_MARKER in m.get("content", "")
    )
    salient_idx_2 = next(
        i for i, m in enumerate(out2) if SALIENT_MARKER in m.get("content", "")
    )
    prefix_1 = out1[:salient_idx_1]
    prefix_2 = out2[:salient_idx_2]

    # Prefix у out2 длиннее (новый assistant + чужой user в середину
    # попали ДО salient'а — это и есть accreting history). Старая часть
    # должна совпадать байт-в-байт.
    assert len(prefix_2) >= len(prefix_1), (
        f"prefix turn2 ({len(prefix_2)}) должен быть ≥ turn1 ({len(prefix_1)})"
    )

    # Точная проверка: префикс turn1 == первые len(prefix_1) элементов turn2.
    digest_1 = hashlib.sha256(repr(prefix_1).encode("utf-8")).hexdigest()
    digest_2_truncated = hashlib.sha256(
        repr(prefix_2[: len(prefix_1)]).encode("utf-8")
    ).hexdigest()
    assert digest_1 == digest_2_truncated, (
        "Prefix-байты до salient'а изменились между turn'ами — "
        "provider prefix-cache invariant нарушен.\n"
        f"  out1 prefix: {prefix_1}\n"
        f"  out2 prefix[:{len(prefix_1)}]: {prefix_2[: len(prefix_1)]}"
    )


def test_compress_does_not_break_tool_pair_when_inserting_salient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fix 2 (волна 26.5): tool-pair safety при insert salient.

    Сценарий: messages = [user, assistant[tool_calls=abc], tool[id=abc]].
    salient вставляется на len-1 = ПЕРЕД tool message → разорвал бы pair.
    После compress'а expect: либо salient переставлен ПЕРЕД assistant
    (закрытый pair выше), либо `_sanitize_tool_pairs` удалил orphan tool
    result. В любом случае output должен быть consistent —
    каждый assistant.tool_calls[id] имеет соответствующий tool result, и
    наоборот.

    Текущая реализация _sanitize_tool_pairs срабатывает после insert'а:
    salient (role=user) разрывает pair → tool result становится orphan'ом
    относительно immediately preceding assistant — но _sanitize находит
    pair по id'шке независимо от позиции, так что pair остаётся
    valid (call_id присутствует с обеих сторон).

    Тест assert'ит финальный invariant: id'шки tool_use и tool_result
    в out совпадают. Если это сломается — нужен fix в позиции
    salient'а (смещать insert_at вверх через open tool-pair).
    """
    _enable_salient(monkeypatch, content="tool-pair-fact")

    e = StyxComposer(
        "tp-agent", context_length=100_000, protect_first_n=3, protect_last_n=6,
    )
    msgs = [
        _msg("user", "long user content here please for embed length check"),
        _msg("assistant", "", tool_calls=[{"id": "abc", "type": "function"}]),
        _msg("tool", "result-abc", tool_call_id="abc"),
    ]
    out = e.compress(msgs, current_tokens=None)

    # Все surviving assistant.tool_calls должны иметь matching tool_result.
    surviving_call_ids: set[str] = set()
    for m in out:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or ():
                cid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if cid:
                    surviving_call_ids.add(cid)

    result_call_ids: set[str] = set()
    for m in out:
        if m.get("role") == "tool":
            cid = m.get("tool_call_id")
            if cid:
                result_call_ids.add(cid)

    # Pair "abc" должен быть закрыт в обе стороны.
    assert "abc" in surviving_call_ids, (
        f"assistant.tool_calls[abc] потерян в out: {out}"
    )
    assert "abc" in result_call_ids, (
        f"tool_result[tool_call_id=abc] потерян в out (orphan call): {out}"
    )

    # Никаких разрывов: surviving tool_calls == tool_results.
    assert surviving_call_ids == result_call_ids, (
        f"orphans found: surviving={surviving_call_ids}, "
        f"results={result_call_ids}"
    )
    # И никаких stub'ов (т.е. реальный tool result сохранён, не replaced
    # на STUB_TOOL_RESULT).
    assert not any(
        m.get("role") == "tool" and m.get("content") == STUB_TOOL_RESULT
        for m in out
    ), f"stub'ы не должны появляться когда оба конца pair'а в input: {out}"


def test_session_reset_clears_counters() -> None:
    e = StyxComposer("test-agent", context_length=10_000, protect_first_n=2, protect_last_n=2)
    e.update_from_response({"prompt_tokens": 100, "completion_tokens": 50})
    e.compress([_msg("user", str(i)) for i in range(10)], current_tokens=9_000)
    assert e.last_prompt_tokens == 100
    assert e.compression_count == 1

    e.on_session_reset()
    assert e.last_prompt_tokens == 0
    assert e.last_completion_tokens == 0
    assert e.last_total_tokens == 0
    assert e.compression_count == 0


# -- Волна 26.7: assemble_for_runtime (system_prompt_addition channel) ---


def test_assemble_for_runtime_returns_salient_text_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Волна 26.7: assemble_for_runtime отдаёт salient через salient_text,
    а не инжектит в messages.

    Зачем: OpenAI Responses API (openai-codex backend) требует strict
    user/assistant alternation; inject salient как role=user между
    existing user и assistant — silent отбрасывание. OpenClaw runtime
    canonical channel — systemPromptAddition.
    """
    _enable_salient(monkeypatch, content="apples-fact")

    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("system", "you are styx assistant"),
        _msg("user", "first turn user content here please"),
        _msg("assistant", "ok understood"),
        _msg("user", "second turn user content here please"),
    ]
    result = e.assemble_for_runtime(msgs, current_tokens=None)

    # messages — без inject'а: оригинальные сообщения без обёртки.
    assert isinstance(result["messages"], list)
    assert len(result["messages"]) == len(msgs)
    for orig, out in zip(msgs, result["messages"]):
        assert out["role"] == orig["role"]
        assert out["content"] == orig["content"]
        # Никакой обёртки <styx-*>/<styx> в messages.
        assert "<styx" not in out["content"]

    # salient_text — отдельным полем, обёрнут в <styx-salient>...</styx-salient>.
    assert isinstance(result["salient_text"], str)
    assert result["salient_text"].startswith("<styx-salient>\n")
    assert result["salient_text"].endswith("\n</styx-salient>")
    assert SALIENT_MARKER in result["salient_text"]
    assert "apples-fact" in result["salient_text"]


def test_assemble_for_runtime_preserves_alternation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Волна 26.7 root cause regression guard.

    После assemble_for_runtime в messages не должно быть consecutive
    user/user или assistant/assistant пар. Это инвариант OpenAI
    Responses API (strict alternation). До 26.7 inject salient как
    role=user генерировал нарушение — алгоритм работал, провайдер
    silently дропал второй consecutive user.
    """
    _enable_salient(monkeypatch, content="some-memory")

    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("system", "you are styx assistant"),
        _msg("user", "u1 turn content please long enough"),
        _msg("assistant", "a1 reply"),
        _msg("user", "u2 turn content please long enough"),
        _msg("assistant", "a2 reply"),
        _msg("user", "u3 turn content please long enough"),
    ]
    result = e.assemble_for_runtime(msgs, current_tokens=None)

    out = result["messages"]
    # Salient присутствует — значит salient_bridge сработал.
    assert isinstance(result["salient_text"], str), (
        "salient_text должен быть установлен — иначе тест не проверяет regression"
    )
    # Alternation: после фильтра system, не должно быть подряд role=user
    # или подряд role=assistant.
    non_system = [m for m in out if m["role"] != "system"]
    for i in range(1, len(non_system)):
        assert non_system[i]["role"] != non_system[i - 1]["role"], (
            f"alternation нарушена в позиции {i}: "
            f"{non_system[i-1]['role']} → {non_system[i]['role']}"
        )


def test_assemble_for_runtime_no_salient_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Если salient_bridge не configure'нут (или recall не вернул hits) —
    salient_text=None, messages — passthrough (с sanitize_styx_blocks
    если в input были маркеры)."""
    # Без _enable_salient — salient_bridge.get_handle вернёт None →
    # build_salient_block тоже вернёт None.

    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", "вопрос"),
        _msg("assistant", "ответ"),
    ]
    result = e.assemble_for_runtime(msgs, current_tokens=None)

    assert result["salient_text"] is None
    assert result["messages"] == msgs


def test_assemble_for_runtime_sanitizes_legacy_styx_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanitize защита волн 26.5 + 30 — если runtime persist'нул прошлый
    assembled view в historical messages, и legacy `<styx>...</styx>`
    (эпоха 26.5) и family `<styx-…>` блоки вырезаются на входе.

    assemble_for_runtime использует тот же _sanitize_styx_blocks что и
    compress().
    """
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", "клин <styx>\nстарый salient\n</styx> вопрос"),
        _msg("assistant", "ответ"),
    ]
    result = e.assemble_for_runtime(msgs, current_tokens=None)

    # Salient block из historical content вырезан.
    assert "<styx>" not in result["messages"][0]["content"]
    assert "старый salient" not in result["messages"][0]["content"]
    assert "клин" in result["messages"][0]["content"]
    assert "вопрос" in result["messages"][0]["content"]


def test_assemble_for_runtime_multimodal_user_produces_salient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production-критичный кейс: pi-embedded передаёт user-message в
    multi-modal shape ({type:"text", text:"..."}). До round 6
    `_find_last_user_text` пропускал это → salient_text=None → hook
    возвращал undefined → боевая Алёна не видела блок.

    После round 6 multi-modal text-parts извлекаются через
    `extract_text_from_content` и идут в `build_salient_block`.
    """
    _enable_salient(monkeypatch, content="round6-fact")
    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("system", "sys"),
        _msg("user", [{"type": "text", "text": "u1 content в multi-modal формате"}]),
        _msg("assistant", "a1"),
        _msg(
            "user",
            [
                {"type": "text", "text": "Что ты помнишь про migration переезд?"},
            ],
        ),
    ]
    result = e.assemble_for_runtime(msgs, current_tokens=None)

    # salient_text должен быть produced — обёрнут в <styx-salient>...</styx-salient>.
    assert isinstance(result["salient_text"], str), (
        f"salient_text должен быть produced для multi-modal user-message; "
        f"got {result['salient_text']!r}"
    )
    assert "<styx-salient>" in result["salient_text"]
    assert "round6-fact" in result["salient_text"]


def test_compress_still_injects_salient_in_messages_for_hermes_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard: compress() (Hermes path через /context/build)
    остаётся unchanged — salient инжектится в messages как было.

    Это критично потому что Hermes pipeline (волна 29 parity recheck
    — open) полагается на messages-inject; cache invariant 26.5 тоже
    привязан к messages-position. Менять compress() в волне 26.7
    нельзя — только новый channel через assemble_for_runtime.
    """
    _enable_salient(monkeypatch, content="hermes-still-injected")

    e = StyxComposer("test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("system", "sys"),
        _msg("user", "u1 content for compress test"),
        _msg("assistant", "a1"),
        _msg("user", "u2 content for compress test"),
    ]
    out = e.compress(msgs, current_tokens=None)

    # Salient присутствует внутри messages, обёрнут в <styx-salient>...</styx-salient>.
    styx_msgs = [m for m in out if "<styx-salient>" in m.get("content", "")]
    assert len(styx_msgs) == 1, f"ожидаем ровно один salient в messages, нашли {len(styx_msgs)}"
    assert "hermes-still-injected" in styx_msgs[0]["content"]


# -- Волна 30: tag taxonomy (family + legacy) ----------------------------


def test_wrap_text_with_styx_tag_basic() -> None:
    """Generic helper форматирует строку как `<tag>\\n...\\n</tag>`."""
    out = wrap_text_with_styx_tag("hello", STYX_TAG_RECALL)
    assert out == "<styx-recall>\nhello\n</styx-recall>"


def test_wrap_text_with_styx_tag_idempotent() -> None:
    """Повторный wrap тем же тегом — no-op (защита от двойной обёртки)."""
    once = wrap_text_with_styx_tag("hello", STYX_TAG_RECALL)
    twice = wrap_text_with_styx_tag(once, STYX_TAG_RECALL)
    assert twice == once


def test_compress_sanitizes_family_styx_recall_block() -> None:
    """Family-тег `<styx-recall>...</styx-recall>` вырезается из input.

    Вол. 30: tool result от styx_recall может протекать в historical
    через native memory dump / transcript echo. Sanitize ловит его.
    """
    e = StyxComposer("a", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", "вопрос <styx-recall>\nhit-1: foo\n</styx-recall> хвост"),
        _msg("assistant", "ok"),
    ]
    out = e.compress(msgs, current_tokens=None)
    content = out[0]["content"]
    assert "<styx-recall>" not in content
    assert "hit-1: foo" not in content
    assert "вопрос" in content and "хвост" in content


def test_compress_sanitizes_mixed_family_and_legacy_blocks() -> None:
    """Один message содержит и family и legacy теги — оба удаляются,
    оба учитываются в per-tag breakdown'е."""
    e = StyxComposer("a", context_length=100_000, protect_first_n=3, protect_last_n=6)
    polluted = (
        "head "
        "<styx-recall>\nrec1\n</styx-recall> mid1 "
        "<styx-recall>\nrec2\n</styx-recall> mid2 "
        "<styx-archive>\narch1\n</styx-archive> mid3 "
        "<styx>\nlegacy1\n</styx> tail"
    )
    msgs = [_msg("user", polluted)]
    out = e.compress(msgs, current_tokens=None)
    content = out[0]["content"]
    for substr in (
        "<styx-recall>", "rec1", "rec2",
        "<styx-archive>", "arch1",
        "<styx>", "legacy1",
    ):
        assert substr not in content, f"{substr!r} осталось: {content!r}"
    assert "head" in content and "tail" in content
    # Breakdown: 2 recall + 1 archive + 1 legacy = 4 total.
    assert get_styx_sanitized_blocks_total() == 4
    by_tag = get_styx_sanitized_blocks_by_tag()
    assert by_tag.get("recall") == 2
    assert by_tag.get("archive") == 1
    assert by_tag.get("legacy") == 1


def test_compress_family_regex_requires_matching_suffix() -> None:
    """Backreference в regex'е: открывающий и закрывающий теги
    должны иметь одинаковый суффикс. `<styx-recall>...</styx-archive>`
    НЕ матчится как один блок (защита от ошибочного склеивания).

    Если суффиксы разные и закрывающего парного нет — блок остаётся
    как есть; если есть валидная пара внутри — она вырезается.
    """
    e = StyxComposer("a", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", "x <styx-recall>data</styx-archive> y"),  # mismatched
    ]
    out = e.compress(msgs, current_tokens=None)
    # Mismatched теги остаются (нет валидной пары) — sanitize не triggers.
    assert "<styx-recall>" in out[0]["content"]
    assert "</styx-archive>" in out[0]["content"]
    assert get_styx_sanitized_blocks_total() == 0


def test_compress_per_tag_counter_split_for_dialogue() -> None:
    """Dialogue tool result wrapping тоже учитывается отдельно."""
    e = StyxComposer("a", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", "до <styx-dialogue>\nrec\n</styx-dialogue> после"),
    ]
    e.compress(msgs, current_tokens=None)
    by_tag = get_styx_sanitized_blocks_by_tag()
    assert by_tag == {"dialogue": 1}


def test_compress_legacy_styx_block_increments_legacy_counter() -> None:
    """Legacy `<styx>...</styx>` (без суффикса) идёт в bucket "legacy"."""
    e = StyxComposer("a", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", "x <styx>\nold\n</styx> y"),
    ]
    e.compress(msgs, current_tokens=None)
    assert get_styx_sanitized_blocks_total() == 1
    assert get_styx_sanitized_blocks_by_tag() == {"legacy": 1}


def test_compress_prefix_includes_styx_salient_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache invariant волны 26.5 + 30: salient-message в финальном
    окне обёрнут именно `<styx-salient>` (а не legacy `<styx>`).
    Это гарантирует что cached префикс ASCII-стабилен между turn'ами
    с новой taxonomy."""
    _enable_salient(monkeypatch, content="taxonomy-fact")
    e = StyxComposer(
        "test-agent", context_length=100_000, protect_first_n=3, protect_last_n=6,
    )
    msgs = [
        _msg("system", "sys"),
        _msg("user", "u1 long enough content please for embed"),
        _msg("assistant", "a1"),
        _msg("user", "u2 long enough content please for embed"),
    ]
    out = e.compress(msgs, current_tokens=None)
    salient = next(m for m in out if SALIENT_MARKER in m.get("content", ""))
    assert salient["content"].startswith("<styx-salient>\n")
    assert salient["content"].endswith("\n</styx-salient>")
    # Новые таги — НЕ равны legacy `<styx>...</styx>` (defensive).
    assert "<styx>\n" not in salient["content"]


def test_compress_multimodal_sanitizes_family_tag() -> None:
    """Multi-modal text-part с family `<styx-recall>` — sanitize'ится
    как и legacy. Регрессия не должна случиться при волне 30."""
    e = StyxComposer("a", context_length=100_000, protect_first_n=3, protect_last_n=6)
    msgs = [
        _msg("user", [{"type": "text", "text": "<styx-recall>\nleak\n</styx-recall>"}]),
        _msg("user", "trailing"),
    ]
    out = e.compress(msgs, current_tokens=None)
    # Text-part внутри multimodal был полностью inside family-блока →
    # text-part пуст после sanitize → message целиком удалён (нет других parts).
    assert len(out) == 1
    assert out[0]["content"] == "trailing"
    assert get_styx_sanitized_blocks_by_tag() == {"recall": 1}
