"""POST /memory_store: rollback-guard + FK→NULL деградация (волна 34).

Defect-fix боевого инцидента в продакшене 2026-06-09: `memory_store` с `session_id`,
которого нет в `sessions`, давал FK-violation → HTTP 500, память
потеряна, постоянный per-agent `self._conn` оставался в aborted-state →
следующие запросы агента падали `InFailedSqlTransaction`.

Тесты — **black-box по приёмке** (`.design/waves/34-write-path-rollback-guard.md`
§ «Приёмка»), на наблюдаемое поведение (HTTP-статус / outcome.action +
факт записи в `memories` + `session_id IS NULL` + здоровье соединения на
следующем вызове), НЕ на внутренности реализации (имя/сигнатуру хелпера
`_guarded_write` не предполагаем — его пишет другой разработчик
параллельно).

Требует ``STYX_TEST_DATABASE_URL`` + Ollama (как остальные integration
в этой папке). Гоняются в Phase C (Docker).

Покрытие приёмки:
- Приёмка №2 (FK→NULL) → ``test_memory_store_missing_session_degrades_to_null``.
- Приёмка №1 (rollback-guard, отравление→восстановление) →
  ``test_memory_store_db_failure_leaves_connection_usable`` +
  ``..._recall_after_failure`` + ``..._sync_turn_after_failure``.
- Приёмка №3 (routed regress) →
  ``test_memory_store_routed_missing_session_degrades_to_null`` +
  ``..._routed_db_failure_leaves_connection_usable``.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import replace

import psycopg
import pytest
from fastapi.testclient import TestClient

from styx.config import StyxConfig, load as load_config
from styx.http import registry
from styx.http.app import create_app
from styx.providers.memory import StyxMemoryCore
from styx.storage import migrate


pytestmark = pytest.mark.skipif(
    not os.environ.get("STYX_TEST_DATABASE_URL"),
    reason="STYX_TEST_DATABASE_URL не задан — integration tests skip",
)


# Маркер-строка для спровоцированного DB-сбоя в сценарии «отравление».
# На `memories` навешивается временный CHECK, который этот маркер
# нарушает → INSERT отлетает на уровне Postgres (CheckViolation, НЕ
# ValueError) → постоянный self._conn остаётся в aborted-state, ровно
# как FK-violation в боевом инциденте. Длинный content / неверный kind
# для этого НЕ годятся: первый ловит app-guard ContentTooLongError
# (ValueError → 422, БД не тронута), второй — _role_for_kind (тоже до
# INSERT'а). Нужен сбой, реально долетающий до Postgres.
_POISON_MARKER = "STYX_W34_POISON_MARKER"
_POISON_CONSTRAINT = "styx_w34_poison_guard"


def _make_stack(clean_db: str):
    migrate.run(clean_db)
    cfg: StyxConfig = load_config()
    # Override DSN на test DB, убираем токен (TestClient ходит без
    # Authorization header'а).
    cfg = replace(cfg, database_url=clean_db, http_token=None)

    agent = "alpha"
    core = StyxMemoryCore(agent_id=agent)
    core._config = cfg
    core.initialize(session_id=str(uuid.uuid4()), agent_identity=agent)
    registry.reset_all()
    registry.register(agent_id=agent, core=core, write_lock=core._write_lock)

    app = create_app(cfg)
    # raise_server_exceptions=False → серверный 500 возвращается как HTTP
    # response, а не re-raise'ится в тест (нам нужен именно код 500, как
    # его увидит реальный HTTP-клиент плагина).
    client = TestClient(app, raise_server_exceptions=False)
    return client, agent, clean_db, core


@pytest.fixture
def stack(clean_db: str):
    client, agent, dsn, core = _make_stack(clean_db)
    yield client, agent, dsn, core
    core.shutdown()
    registry.reset_all()


# ── helpers (read-only наблюдение в БД из отдельного соединения) ──────


def _count_subjective(dsn: str, agent: str) -> int:
    """Живые subjective-memory (не tail) агента."""
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM memories "
                " WHERE agent_id = %s AND superseded_by IS NULL "
                "   AND kind_src = 'subjective'",
                (agent,),
            )
            return int(cur.fetchone()[0])


def _fetch_one_memory(dsn: str, memory_id: str):
    """(content, session_id) записанной памяти по id, или None."""
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT content, session_id FROM memories WHERE id = %s",
                (memory_id,),
            )
            return cur.fetchone()


def _session_id_of(dsn: str, memory_id: str):
    row = _fetch_one_memory(dsn, memory_id)
    assert row is not None, f"память {memory_id} не найдена в БД"
    return row[1]


def _install_poison_constraint(dsn: str) -> None:
    """Навесить CHECK, нарушаемый маркером, через отдельное соединение.

    Любой INSERT в `memories` с content, содержащим `_POISON_MARKER`,
    отлетит CheckViolation'ом на уровне Postgres — детерминированный
    DB-сбой, не зависящий от деталей реализации фикса.
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"ALTER TABLE memories ADD CONSTRAINT {_POISON_CONSTRAINT} "
                f"CHECK (content NOT LIKE '%%{_POISON_MARKER}%%')"
            )
        conn.commit()


def _drop_poison_constraint(dsn: str) -> None:
    """Снять poison-constraint через отдельное соединение.

    ``DROP CONSTRAINT`` требует AccessExclusiveLock на `memories`. Если
    rollback-guard ещё НЕ откатил aborted-транзакцию постоянного
    self._conn (т.е. фикс не на месте / сломан), тот держит
    RowExclusiveLock от провалившегося INSERT'а — DROP завис бы. Ставим
    `lock_timeout`, чтобы в этом случае получить внятный LockNotAvailable
    (тест зафейлится с диагностикой), а не вечный hang.
    """
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SET lock_timeout = '15s'")
            cur.execute(
                f"ALTER TABLE memories DROP CONSTRAINT IF EXISTS "
                f"{_POISON_CONSTRAINT}"
            )
        conn.commit()


# ── Приёмка №2: FK-missing-session → 200 + session_id=NULL ───────────


def test_memory_store_missing_session_degrades_to_null(stack) -> None:
    """session_id, которого нет в sessions → 200, память сохранена,
    session_id IS NULL, не потеряна, без FK-violation."""
    client, agent, dsn, _ = stack
    ghost_session = str(uuid.uuid4())  # точно нет в sessions

    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": "мысль про отсутствующую сессию деградирует в NULL",
            "kind": "note",
            "kind_src": "subjective",
            "session_id": ghost_session,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "store"
    mid = body["memory_id"]
    assert mid is not None

    # Память реально записана.
    assert _count_subjective(dsn, agent) == 1
    row = _fetch_one_memory(dsn, mid)
    assert row is not None, "память не записана в БД"
    # И session_id деградировал в NULL (FK не нарушен, ряд не потерян).
    assert row[1] is None, f"ожидался session_id IS NULL, получили {row[1]!r}"


def test_memory_store_existing_session_keeps_id(stack) -> None:
    """Контроль: при существующей сессии session_id сохраняется (не
    деградирует огульно)."""
    client, agent, dsn, core = stack
    # Сессия из initialize существует — берём свежую и upsert'им явно
    # через отдельное соединение, чтобы быть независимыми от core-state.
    real_session = uuid.uuid4()
    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sessions (id, agent_id) VALUES (%s, %s) "
                "ON CONFLICT (id) DO NOTHING",
                (real_session, agent),
            )
        conn.commit()

    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": "мысль на реально существующей сессии сохраняет id",
            "kind": "note",
            "kind_src": "subjective",
            "session_id": str(real_session),
        },
    )
    assert resp.status_code == 200, resp.text
    mid = resp.json()["memory_id"]
    assert mid is not None
    assert _session_id_of(dsn, mid) == real_session


# ── Приёмка №1: отравление → восстановление соединения ───────────────


def _poison_once(client, agent: str) -> None:
    """Спровоцировать DB-сбой на memory_store → HTTP 500, self._conn
    остаётся aborted."""
    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": f"отравляющая запись {_POISON_MARKER} роняет insert",
            "kind": "note",
            "kind_src": "subjective",
        },
    )
    # Настоящий DB-сбой (CheckViolation, не ValueError) → 500.
    assert resp.status_code == 500, (
        f"ожидали 500 от спровоцированного DB-сбоя, получили "
        f"{resp.status_code}: {resp.text}"
    )


def test_memory_store_db_failure_leaves_connection_usable(stack) -> None:
    """ЯДРО приёмки №1: одиночный сбой записи НЕ оставляет self._conn в
    aborted — следующий memory_store того же агента проходит без
    InFailedSqlTransaction."""
    client, agent, dsn, _ = stack
    _install_poison_constraint(dsn)
    try:
        _poison_once(client, agent)
    finally:
        # Снимаем constraint ДО восстановительного вызова — иначе
        # следующий store упал бы по той же причине, а не по
        # InFailedSqlTransaction (мы проверяем здоровье соединения, не
        # повторный сбой).
        _drop_poison_constraint(dsn)

    # Немедленно следующий memory_store того же агента — должен пройти
    # чисто (соединение живо, не InFailedSqlTransaction).
    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": "восстановительная запись после спровоцированного сбоя",
            "kind": "note",
            "kind_src": "subjective",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["action"] == "store"
    mid = body["memory_id"]
    assert mid is not None
    # Восстановительная запись реально в БД; отравляющая — нет.
    assert _count_subjective(dsn, agent) == 1
    assert _fetch_one_memory(dsn, mid) is not None


def test_memory_store_db_failure_recall_after_failure(stack) -> None:
    """После спровоцированного 500 немедленный /recall того же агента
    проходит без InFailedSqlTransaction (соединение read-path тоже
    здорово)."""
    client, agent, dsn, _ = stack
    # Сначала положим одну валидную память, чтобы recall имел что искать.
    seed = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": "опорная память для последующего recall после сбоя",
            "kind": "note",
            "kind_src": "subjective",
        },
    )
    assert seed.status_code == 200, seed.text

    _install_poison_constraint(dsn)
    try:
        _poison_once(client, agent)
    finally:
        _drop_poison_constraint(dsn)

    resp = client.post(
        "/recall",
        json={
            "agent_id": agent,
            "query": "опорная память для recall",
        },
    )
    # Главное — НЕ 500 из-за InFailedSqlTransaction; recall отрабатывает.
    assert resp.status_code == 200, resp.text


def test_memory_store_db_failure_sync_turn_after_failure(stack) -> None:
    """После спровоцированного 500 немедленный /sync_turn того же агента
    проходит без InFailedSqlTransaction (приёмка №1 явно называет
    sync_turn в числе восстановительных вызовов)."""
    client, agent, dsn, _ = stack
    _install_poison_constraint(dsn)
    try:
        _poison_once(client, agent)
    finally:
        _drop_poison_constraint(dsn)

    resp = client.post(
        "/sync_turn",
        json={
            "agent_id": agent,
            "user_content": "проверка живости соединения после сбоя",
            "assistant_content": "соединение восстановлено, отвечаю штатно",
        },
    )
    assert resp.status_code == 200, resp.text


# ── Приёмка №3: routed regress (длинный content + отсутствующая сессия)


def _long_content() -> str:
    # > store_routing_limit (default 2400) → routed-путь.
    return "Длинный исследовательский фрагмент про геометрию входа. " * 80


def test_memory_store_routed_missing_session_degrades_to_null(stack) -> None:
    """Длинный content (> routing.limit) на ОТСУТСТВУЮЩЕЙ сессии:
    routed-путь не теряет память и применяет FK→NULL (tail-memory
    с session_id IS NULL)."""
    client, agent, dsn, _ = stack
    content = _long_content()
    assert len(content) > 2400, "контент должен превышать store_routing_limit"
    ghost_session = str(uuid.uuid4())

    resp = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": content,
            "kind": "note",
            "kind_src": "subjective",
            "session_id": ghost_session,
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["routed"] is True, body
    mid = body["memory_id"]  # tail-memory
    assert mid is not None
    # Tail-memory записана и её session_id деградировал в NULL.
    row = _fetch_one_memory(dsn, mid)
    assert row is not None, "tail-memory не записана"
    assert row[1] is None, (
        f"routed-путь: ожидался session_id IS NULL, получили {row[1]!r}"
    )


def test_memory_store_routed_db_failure_leaves_connection_usable(stack) -> None:
    """Routed-путь: спровоцированный DB-сбой на длинном content не травит
    соединение — следующий memory_store того же агента проходит без
    InFailedSqlTransaction."""
    client, agent, dsn, _ = stack
    # Маркер ПРЕФИКСОМ: routed-путь пишет в memories только tail-summary
    # (make_tail_summary — обрезок ГОЛОВЫ content'а ≤ summary_chars).
    # Маркер в конце ушёл бы только в chunks (CHECK на них не висит) →
    # tail-memory insert прошёл бы. Префикс гарантирует, что маркер
    # попадёт в tail-summary → CHECK на memories отлетит на вставке
    # tail-memory именно на routed-пути.
    content = f"{_POISON_MARKER} " + _long_content()
    assert len(content) > 2400

    _install_poison_constraint(dsn)
    try:
        resp = client.post(
            "/memory_store",
            json={
                "agent_id": agent,
                "content": content,
                "kind": "note",
                "kind_src": "subjective",
            },
        )
        # Routed insert tail-memory отлетает CheckViolation'ом → 500.
        assert resp.status_code == 500, (
            f"ожидали 500 от спровоцированного DB-сбоя на routed-пути, "
            f"получили {resp.status_code}: {resp.text}"
        )
    finally:
        _drop_poison_constraint(dsn)

    # Следующий обычный memory_store того же агента — соединение живо.
    resp2 = client.post(
        "/memory_store",
        json={
            "agent_id": agent,
            "content": "короткая запись после сбоя на routed-пути",
            "kind": "note",
            "kind_src": "subjective",
        },
    )
    assert resp2.status_code == 200, resp2.text
    assert resp2.json()["action"] == "store"
