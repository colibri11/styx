"""Юнит-тесты build_salient_block — все skip-условия + happy path.

recall_full патчится через monkeypatch на ``styx.engine.salient.recall_full``;
БД не нужна. Unit-тесты для D5 покрывают каждый из 5 skip-кейсов из
wave-doc'а.
"""

from __future__ import annotations

import time

import pytest

from styx.engine import salient_bridge
from styx.engine.salient import SALIENT_MARKER, build_salient_block
from styx.storage.queries import MemoryHit
from styx.storage.recall import RecallResult
from styx.storage.recall_config import DEFAULT_RECALL_CONFIG


class _StubQueries:
    pass


class _StubEmbed:
    @property
    def dim(self) -> int:
        return 768

    def embed(self, text: str) -> list[float]:
        return [0.0] * 768


def _handle(timeout_s: float = 1.0, min_query_len: int = 20):
    return salient_bridge.SalientHandle(
        queries=_StubQueries(),
        embed_client=_StubEmbed(),
        recall_config=DEFAULT_RECALL_CONFIG,
        timeout_s=timeout_s,
        min_query_len=min_query_len,
        agent_id="test-agent",
    )


def _hit(content: str = "remembered fact", score: float = 0.7) -> MemoryHit:
    """Минимальный MemoryHit для тестов — поля что использует format_recall_text."""
    import uuid

    return MemoryHit(
        id=uuid.uuid4(),
        agent_id="test",
        kind="episode",
        role="user",
        content=content,
        metadata={},
        created_at=None,
        score=score,
        match_score=score,
    )


# -- D5 skip conditions ---------------------------------------------------


def test_skip_when_handle_is_none() -> None:
    msgs = [{"role": "user", "content": "hello world this is a long enough query"}]
    assert build_salient_block(msgs, None) is None


def test_skip_when_no_user_message() -> None:
    msgs = [
        {"role": "system", "content": "..."},
        {"role": "assistant", "content": "..."},
    ]
    assert build_salient_block(msgs, _handle()) is None


def test_skip_when_user_content_too_short() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    assert build_salient_block(msgs, _handle()) is None


def test_skip_when_user_content_is_not_string() -> None:
    msgs = [{"role": "user", "content": [{"type": "text", "text": "..."}]}]
    assert build_salient_block(msgs, _handle()) is None


def test_skip_when_recall_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**_kw):
        raise RuntimeError("ollama down")

    monkeypatch.setattr("styx.engine.salient.recall_full", boom)
    msgs = [{"role": "user", "content": "what did we decide about apples in march"}]
    assert build_salient_block(msgs, _handle()) is None


def test_skip_when_memories_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def empty(**_kw):
        return RecallResult(
            memories=[], queried_count=0, internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", empty)
    msgs = [{"role": "user", "content": "what did we decide about apples in march"}]
    assert build_salient_block(msgs, _handle()) is None


def test_skip_when_recall_times_out(monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(**_kw):
        time.sleep(0.5)
        return RecallResult(memories=[_hit()], queried_count=1, internal_duplicates_removed=0)

    monkeypatch.setattr("styx.engine.salient.recall_full", slow)
    msgs = [{"role": "user", "content": "what did we decide about apples in march"}]
    out = build_salient_block(msgs, _handle(timeout_s=0.05))
    assert out is None


# -- happy path -----------------------------------------------------------


def test_happy_path_returns_user_message_with_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict = {}

    def fake_recall(*, queries, embed_client, query, full_config, **_kw):
        captured["query"] = query
        captured["limit"] = full_config.memory_limit
        return RecallResult(
            memories=[_hit("apples and pears in march", score=0.71)],
            queried_count=1,
            internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)
    msgs = [
        {"role": "system", "content": "you are styx"},
        {"role": "user", "content": "first message that is not the focus"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "what did we decide about apples in march"},
    ]
    out = build_salient_block(msgs, _handle())

    assert out is not None
    assert out["role"] == "user"
    assert SALIENT_MARKER in out["content"]
    assert "apples and pears in march" in out["content"]
    # last user — фокус-query, а не первый
    assert captured["query"] == "what did we decide about apples in march"


def test_picks_last_user_when_multiple(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def fake_recall(*, query, **_kw):
        seen.append(query)
        return RecallResult(
            memories=[_hit()], queried_count=1, internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)
    msgs = [
        {"role": "user", "content": "first user query that is plenty long here"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second user query that is plenty long here"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "third and final user query plenty long"},
    ]
    out = build_salient_block(msgs, _handle())
    assert out is not None
    assert seen == ["third and final user query plenty long"]


def test_min_query_len_boundary_inclusive(monkeypatch: pytest.MonkeyPatch) -> None:
    """len == min_query_len → пропускает (>= оператор внутри)."""

    def fake_recall(**_kw):
        return RecallResult(
            memories=[_hit()], queried_count=1, internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)
    text = "x" * 20  # ровно min_query_len по дефолту
    msgs = [{"role": "user", "content": text}]
    out = build_salient_block(msgs, _handle(min_query_len=20))
    assert out is not None


# -- Волна 10: drift detection + cached salient -------------------------


@pytest.fixture
def _focus_tracker_off():
    """focus_tracker не configured → fallback на волна-9 (fresh каждый turn)."""
    from styx.engine import focus_tracker

    focus_tracker.reset_all()
    yield
    focus_tracker.reset_all()


@pytest.fixture
def _focus_tracker_on():
    """focus_tracker configured → drift detection активен."""
    from styx.engine import focus_tracker

    focus_tracker.reset_all()
    focus_tracker.configure("test-agent", window_size=3, drift_threshold=0.4)
    yield
    focus_tracker.reset_all()


class _DeterministicEmbed:
    """Embed-клиент с заданным маппингом text → vector. Для тестов drift'а."""

    def __init__(self, mapping: dict[str, list[float]], dim: int = 768) -> None:
        self._mapping = mapping
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if text in self._mapping:
            return list(self._mapping[text])
        # Fallback: синтетический ортогональный embed для не-зарегистрированного текста.
        h = hash(text) % self._dim
        v = [0.0] * self._dim
        v[h] = 1.0
        return v


def _handle_with_embed(embed_client) -> "salient_bridge.SalientHandle":
    return salient_bridge.SalientHandle(
        queries=_StubQueries(),
        embed_client=embed_client,
        recall_config=DEFAULT_RECALL_CONFIG,
        timeout_s=1.0,
        min_query_len=10,
        agent_id="test-agent",
    )


def test_drift_disabled_falls_back_to_fresh_each_turn(
    monkeypatch: pytest.MonkeyPatch, _focus_tracker_off,
) -> None:
    """Без focus_tracker.configure("test-agent") — каждый build вызывает recall (волна-9 поведение)."""
    calls = {"n": 0}

    def fake_recall(**_kw):
        calls["n"] += 1
        return RecallResult(
            memories=[_hit(f"hit-{calls['n']}")],
            queried_count=1,
            internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)
    msgs = [{"role": "user", "content": "long enough query for the recall pipeline"}]

    out1 = build_salient_block(msgs, _handle())
    out2 = build_salient_block(msgs, _handle())
    assert out1 is not None and out2 is not None
    assert calls["n"] == 2  # каждый раз новый recall


def test_first_call_caches_salient(
    monkeypatch: pytest.MonkeyPatch, _focus_tracker_on,
) -> None:
    from styx.engine import focus_tracker

    def fake_recall(**_kw):
        return RecallResult(
            memories=[_hit("cached-fact")], queried_count=1,
            internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)
    e_a = [1.0] + [0.0] * 767
    embed = _DeterministicEmbed({"первый запрос про embedding модели styx": e_a})
    handle = _handle_with_embed(embed)

    msgs = [{"role": "user", "content": "первый запрос про embedding модели styx"}]
    out = build_salient_block(msgs, handle)
    assert out is not None
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.cached_salient is out  # тот же dict в кэше


def test_stable_topic_returns_cached_salient(
    monkeypatch: pytest.MonkeyPatch, _focus_tracker_on,
) -> None:
    """Идентичный embed на втором turn'е → no drift → reuse cached."""
    calls = {"n": 0}

    def fake_recall(**_kw):
        calls["n"] += 1
        return RecallResult(
            memories=[_hit(f"hit-{calls['n']}")],
            queried_count=1,
            internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)
    same_embed = [1.0] + [0.0] * 767
    embed = _DeterministicEmbed({
        "stable topic message version one here": same_embed,
        "stable topic message version two here": same_embed,
    })
    handle = _handle_with_embed(embed)

    msgs1 = [{"role": "user", "content": "stable topic message version one here"}]
    msgs2 = [{"role": "user", "content": "stable topic message version two here"}]

    out1 = build_salient_block(msgs1, handle)
    out2 = build_salient_block(msgs2, handle)

    assert out1 is not None and out2 is not None
    assert out1 is out2  # тот же объект из кэша
    assert calls["n"] == 1  # recall сделан только один раз


def test_drift_invalidates_and_recomputes(
    monkeypatch: pytest.MonkeyPatch, _focus_tracker_on,
) -> None:
    """Ортогональный embed → drift → fresh recall, новый salient."""
    calls = {"n": 0}

    def fake_recall(**_kw):
        calls["n"] += 1
        return RecallResult(
            memories=[_hit(f"hit-turn-{calls['n']}")],
            queried_count=1,
            internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", fake_recall)
    e_a = [1.0] + [0.0] * 767
    e_b = [0.0, 1.0] + [0.0] * 766  # ортогональный e_a
    embed = _DeterministicEmbed({
        "topic A — message one is long enough": e_a,
        "topic B — completely unrelated query for sure": e_b,
    })
    handle = _handle_with_embed(embed)

    out1 = build_salient_block(
        [{"role": "user", "content": "topic A — message one is long enough"}], handle,
    )
    out2 = build_salient_block(
        [{"role": "user", "content": "topic B — completely unrelated query for sure"}],
        handle,
    )

    assert out1 is not None and out2 is not None
    assert out1 is not out2
    assert calls["n"] == 2  # оба turn'а сделали recall
    assert "hit-turn-1" in out1["content"]
    assert "hit-turn-2" in out2["content"]


def test_drift_skip_when_embed_fails(
    monkeypatch: pytest.MonkeyPatch, _focus_tracker_on,
) -> None:
    """EmbeddingError на last_user → return None (fail-open), кэш не трогаем."""
    from styx.embedding import EmbeddingError

    class _BoomEmbed:
        @property
        def dim(self) -> int:
            return 768

        def embed(self, text: str) -> list[float]:
            raise EmbeddingError("ollama down")

    handle = _handle_with_embed(_BoomEmbed())
    out = build_salient_block(
        [{"role": "user", "content": "long enough query but embed will fail"}],
        handle,
    )
    assert out is None


def test_drift_invalidates_cache_when_recall_returns_empty(
    monkeypatch: pytest.MonkeyPatch, _focus_tracker_on,
) -> None:
    """На drift'е recall вернул empty → cached_salient становится None.

    Следующий compress (даже если drift не повторится) должен попытаться
    fresh recall ещё раз — потому что cached_salient is None.
    """
    from styx.engine import focus_tracker

    def empty_recall(**_kw):
        return RecallResult(
            memories=[], queried_count=0, internal_duplicates_removed=0,
        )

    monkeypatch.setattr("styx.engine.salient.recall_full", empty_recall)
    e_a = [1.0] + [0.0] * 767
    embed = _DeterministicEmbed({"first message about a topic — long": e_a})
    handle = _handle_with_embed(embed)

    out = build_salient_block(
        [{"role": "user", "content": "first message about a topic — long"}], handle,
    )
    assert out is None
    state = focus_tracker.get_state("test-agent")
    assert state is not None
    assert state.cached_salient is None
