"""Тесты StyxMemoryCore — lifecycle, sync_turn, изоляция scope."""

from __future__ import annotations

import os
import uuid

import pytest

from styx.providers.memory import StyxMemoryCore


@pytest.fixture
def styx_env(monkeypatch: pytest.MonkeyPatch, migrated_db: str):
    monkeypatch.setenv("STYX_DATABASE_URL", migrated_db)


def test_initialize_rejects_empty_agent_identity() -> None:
    """RuntimeError выбрасывается до psycopg.connect — DB не нужна.

    После Phase B сообщение об ошибке сменилось — agent_id может задаваться
    в __init__ или через kwargs.agent_identity; ошибка возникает только если
    оба источника пусты.
    """
    sid = str(uuid.uuid4())

    # Пустая строка
    p = StyxMemoryCore()
    with pytest.raises(RuntimeError, match="non-empty agent_id"):
        p.initialize(session_id=sid, agent_identity="")

    # Только пробелы
    p2 = StyxMemoryCore()
    with pytest.raises(RuntimeError, match="non-empty agent_id"):
        p2.initialize(session_id=sid, agent_identity="   ")

    # Kwarg вовсе не передан
    p3 = StyxMemoryCore()
    with pytest.raises(RuntimeError, match="non-empty agent_id"):
        p3.initialize(session_id=sid)


def test_provider_metadata() -> None:
    p = StyxMemoryCore()
    assert p.name == "styx-memory"
    # Волна 7 — styx_recall; волна 20 — styx_search_archive (pull-канал);
    # волна 22 — styx_reinterpret (explicit reinterpret tool); волна 24
    # follow-up — три read-only dialogue tools (search/recent/prepare_summary);
    # волна 28 — styx_ingest_document (file-ingest, symmetric с OpenClaw).
    schemas = p.get_tool_schemas()
    names = {s["name"] for s in schemas}
    assert names == {
        "styx_recall",
        "styx_search_archive",
        "styx_reinterpret",
        "styx_dialogue_search",
        "styx_dialogue_recent",
        "styx_dialogue_prepare_summary",
        "styx_ingest_document",
    }
    recall = next(s for s in schemas if s["name"] == "styx_recall")
    assert "query" in recall["parameters"]["properties"]
    reinterpret = next(s for s in schemas if s["name"] == "styx_reinterpret")
    assert "memory_id" in reinterpret["parameters"]["properties"]
    assert "new_understanding_text" in reinterpret["parameters"]["properties"]
    dialogue_search = next(
        s for s in schemas if s["name"] == "styx_dialogue_search"
    )
    assert "query" in dialogue_search["parameters"]["properties"]
    assert "semantic_only" in dialogue_search["parameters"]["properties"]
    prepare = next(
        s for s in schemas if s["name"] == "styx_dialogue_prepare_summary"
    )
    assert "session_id" in prepare["parameters"]["properties"]
    assert prepare["parameters"]["required"] == ["session_id"]
    schema = p.get_config_schema()
    assert any(field["key"] == "database_url" for field in schema)


def test_is_available_via_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STYX_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    p = StyxMemoryCore()
    assert p.is_available() is False
    monkeypatch.setenv("STYX_DATABASE_URL", "postgresql://stub")
    assert p.is_available() is True


def test_initialize_registers_session(styx_env, migrated_db: str) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        assert p.queries.agent_id == "alpha"
        assert p.queries.count_messages() == 0
    finally:
        p.shutdown()


def test_sync_turn_writes_user_and_assistant(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        p.sync_turn("привет", "здравствуй", session_id=sid)
        recent = p.queries.recent_messages(limit=10, session_id=uuid.UUID(sid))
        roles = [m.role for m in recent]
        contents = [m.content for m in recent]
        # recent_messages — DESC по created_at: assistant первым
        assert roles == ["assistant", "user"]
        assert contents == ["здравствуй", "привет"]
    finally:
        p.shutdown()


def test_sync_turn_default_session_id(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        # session_id="" — провайдер должен использовать сессию из initialize
        p.sync_turn("u", "a")
        assert p.queries.count_messages() == 2
    finally:
        p.shutdown()


def test_sync_turn_skips_empty_content(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        p.sync_turn("", "только assistant")
        p.sync_turn("только user", "")
        assert p.queries.count_messages() == 2  # одно user + одно assistant
    finally:
        p.shutdown()


def test_sync_turn_splits_oversized_message(styx_env) -> None:
    """Defect-fix B + регрессия бага: реплика 15090 символов
    (документ-Markdown через turn-канал) больше НЕ роняет
    sync_turn по CHECK constraint memories_content_length_check.

    Реплика режется на N рядов дневника ≤ лимита; пересборка через
    recent_messages даёт обратно целостный блок.
    """
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        # 15090 символов — точный размер из боевого инцидента.
        big_user = "Параграф документа с текстом.\n\n" * 503  # ~15090 chars
        assert len(big_user) > 14000
        # Раньше это бросало CheckViolation; теперь — успешный split.
        p.sync_turn(big_user, "краткий ответ", session_id=sid)

        # Сырые ряды: user разрезан на несколько частей, assistant — один.
        raw = p.queries.recent_messages(
            limit=50, session_id=uuid.UUID(sid), reassemble_groups=False,
        )
        user_parts = [m for m in raw if m.role == "user"]
        assert len(user_parts) > 1
        # Каждая часть ≤ CHECK constraint лимита.
        assert all(len(m.content) <= 2400 for m in raw)
        # Все части — одна группа.
        groups = {m.metadata["msg_group"] for m in user_parts}
        assert len(groups) == 1

        # Пересборка: группа склеена обратно в один блок.
        reassembled = p.queries.recent_messages(
            limit=50, session_id=uuid.UUID(sid),
        )
        users = [m for m in reassembled if m.role == "user"]
        assert len(users) == 1
        assert users[0].content == big_user
    finally:
        p.shutdown()


def test_sync_turn_rollback_guard_keeps_connection_usable(
    styx_env, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Defect-fix 2: при ошибке записи в sync_turn соединение
    остаётся рабочим (rollback), а не aborted-state до рестарта."""
    from styx.storage import queries as queries_mod

    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        # Эмулируем сбой записи: insert_message бросает исключение.
        orig = queries_mod.AgentScopedQueries.insert_message

        def _boom(self, **kwargs):  # noqa: ANN001
            raise RuntimeError("эмулированный сбой записи")

        monkeypatch.setattr(
            queries_mod.AgentScopedQueries, "insert_message", _boom,
        )
        with pytest.raises(RuntimeError, match="эмулированн"):
            p.sync_turn("реплика", "ответ", session_id=sid)

        # Восстанавливаем insert_message — соединение должно быть рабочим.
        monkeypatch.setattr(
            queries_mod.AgentScopedQueries, "insert_message", orig,
        )
        # Следующий sync_turn проходит — connection не в aborted-state.
        p.sync_turn("после сбоя", "и ответ", session_id=sid)
        assert p.queries.count_messages(session_id=uuid.UUID(sid)) == 2
    finally:
        p.shutdown()


def test_prefetch_and_system_prompt_block_empty_in_v1(styx_env) -> None:
    p = StyxMemoryCore()
    sid = str(uuid.uuid4())
    p.initialize(session_id=sid, agent_identity="alpha")
    try:
        assert p.prefetch("anything") == ""
        assert p.system_prompt_block() == ""
    finally:
        p.shutdown()


def test_two_providers_isolated_by_agent_id(styx_env) -> None:
    a = StyxMemoryCore()
    b = StyxMemoryCore()
    sid_a = str(uuid.uuid4())
    sid_b = str(uuid.uuid4())
    a.initialize(session_id=sid_a, agent_identity="alpha")
    b.initialize(session_id=sid_b, agent_identity="beta")
    try:
        a.sync_turn("a-user", "a-assistant")
        b.sync_turn("b-user", "b-assistant")

        assert a.queries.count_messages() == 2
        assert b.queries.count_messages() == 2

        a_contents = {m.content for m in a.queries.recent_messages(limit=10)}
        b_contents = {m.content for m in b.queries.recent_messages(limit=10)}
        assert a_contents == {"a-user", "a-assistant"}
        assert b_contents == {"b-user", "b-assistant"}
        assert a_contents.isdisjoint(b_contents)
    finally:
        a.shutdown()
        b.shutdown()


def test_save_config_round_trip(tmp_path) -> None:
    p = StyxMemoryCore()
    p.save_config(
        {"database_url": "postgresql://u:p@h/d", "ollama_url": "http://x:11434"},
        str(tmp_path),
    )
    config_file = tmp_path / "styx.json"
    assert config_file.exists()
    import json
    data = json.loads(config_file.read_text())
    assert data["database_url"] == "postgresql://u:p@h/d"

    # повторный save — merge, а не overwrite
    p.save_config({"embedding_model": "qwen3-embedding"}, str(tmp_path))
    data = json.loads(config_file.read_text())
    assert data["database_url"] == "postgresql://u:p@h/d"
    assert data["embedding_model"] == "qwen3-embedding"


def test_does_not_inherit_hermes_abc() -> None:
    """После split (v1.0.0) StyxMemoryCore — host-agnostic класс без ABC.
    Hermes-обёртка StyxMemoryProvider(MemoryProvider) живёт в styx_hermes."""
    p = StyxMemoryCore("alpha")
    # Не должно быть наследования от Hermes ABC (модуль может быть не
    # доступен — core импортируется без HERMES_PATH).
    cls_names = {c.__name__ for c in type(p).__mro__}
    assert "MemoryProvider" not in cls_names
