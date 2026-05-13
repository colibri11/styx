"""OpenClaw plugin integration tests (волна 26 Phase E).

Запускаются host-side (или внутри hermes-styx как обычные integration
tests). Обращаются к docker-стику `docker/docker-compose.test.yml`:

- `openclaw-cli` (sidecar для CLI-команд) — через `docker compose exec`
  (subprocess) — `plugins inspect styx --runtime --json` для lifecycle
  проверок и `agent --message ... --local` для real chat round-trip.
- `styx-daemon` HTTP API — через `httpx` на `http://127.0.0.1:8788`
  (порт expose'нут в compose) — для верификации side-effects (записи
  в memories через ContextEngine.ingest).

Skip-паттерн идентичен другим integration tests: при отсутствии
`STYX_TEST_DATABASE_URL` тесты пропускаются (значит стик не поднят, нет
смысла дёргать docker compose).

Архитектурный смысл (Phase E из `.design/openclaw-plugin-v1.md`):
проверяем, что plugin не просто **регистрируется** (это тестировалось
вручную в Phase A-D через `plugins inspect`), а **участвует в реальном
turn'е** OpenClaw агента — `ContextEngine.ingest` пишет в Styx PG, на
следующем turn assemble() поднимает то что записано.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path

import psycopg
import pytest


def _has_compose_root() -> bool:
    """True если запущены host-side (видим docker-compose.test.yml).

    В hermes-styx / styx-daemon контейнерах compose-файла нет в
    parent path'ах — эти тесты host-only по дизайну, нужны для
    `docker compose exec` против поднятого стика.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docker" / "docker-compose.test.yml").is_file():
            return True
    return False


# Все тесты файла — integration. Skip если стик не поднят либо если
# мы внутри Docker контейнера (host-only — дёргают `docker compose exec`).
pytestmark = [
    pytest.mark.skipif(
        not os.environ.get("STYX_TEST_DATABASE_URL"),
        reason="STYX_TEST_DATABASE_URL не задан — docker compose стик не поднят",
    ),
    pytest.mark.skipif(
        not _has_compose_root(),
        reason="host-only: docker-compose.test.yml не виден (тесты дёргают docker compose exec host-side)",
    ),
]


# ─── helpers ──────────────────────────────────────────────────────────


def _compose_root() -> Path:
    """Корень репо styx (где лежит docker/docker-compose.test.yml)."""
    here = Path(__file__).resolve()
    # tests/integration/openclaw → packages/styx-core → packages → repo root
    for parent in here.parents:
        if (parent / "docker" / "docker-compose.test.yml").is_file():
            return parent
    raise RuntimeError("docker-compose.test.yml not found above " + str(here))


def _docker_exec(
    service: str,
    *cmd: str,
    timeout: float = 60.0,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """`docker compose exec -T <service> <cmd...>` с timeout'ом + capture."""
    root = _compose_root()
    full = [
        "docker",
        "compose",
        "-f",
        "docker/docker-compose.test.yml",
        "exec",
        "-T",
        service,
        *cmd,
    ]
    return subprocess.run(
        full,
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=check,
    )


def _inspect_plugin_runtime() -> dict:
    """`openclaw plugins inspect styx --runtime --json` → dict.

    Реальный shape (openclaw 2026.5.7):
        {
          "workspaceDir": "...",
          "plugin": {
            "id": "styx",
            "activated": true,
            "activationReason": "selected context engine slot",
            "toolNames": [...16],
            "contextEngineIds": ["styx"],
            ...
          }
        }
    """
    res = _docker_exec(
        "openclaw-cli",
        "node",
        "/app/dist/index.js",
        "plugins",
        "inspect",
        "styx",
        "--runtime",
        "--json",
        timeout=30.0,
    )
    # OpenClaw CLI печатает JSON в stdout, прогресс-сообщения в stderr.
    return json.loads(res.stdout)


# ─── core lifecycle (всегда работают, не зависят от LLM) ──────────────


def test_compose_stack_up() -> None:
    """5+ сервисов запущены и healthy.

    Sanity check: убедиться что стик поднят прежде чем дёргать CLI.
    """
    root = _compose_root()
    res = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            "docker/docker-compose.test.yml",
            "ps",
            "--format",
            "json",
        ],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=15.0,
        check=True,
    )
    # `compose ps --format json` печатает по одной JSON-строке на сервис.
    services = [json.loads(line) for line in res.stdout.splitlines() if line.strip()]
    names = {svc["Service"] for svc in services}
    # S1: openclaw-cli включён в обязательный набор — без него `agent
    # --local` chat-тесты не могут отработать.
    expected = {"postgres", "styx-daemon", "hermes-styx", "openclaw-gateway", "openclaw-cli"}
    missing = expected - names
    assert not missing, f"Сервисы не подняты: {missing}. Запустить: docker compose ... up -d --wait"

    # Минимум gateway+daemon+cli healthy для следующих тестов.
    for svc in services:
        if svc["Service"] in ("openclaw-gateway", "styx-daemon", "openclaw-cli"):
            assert svc.get("State") == "running", (
                f"{svc['Service']} не running: state={svc.get('State')}, status={svc.get('Status')}"
            )


def test_plugin_capability_registered() -> None:
    """ContextEngine capability зарегистрирована (Phase A-D smoke parity)."""
    info = _inspect_plugin_runtime()
    plugin = info.get("plugin", {})
    assert plugin.get("id") == "styx", f"plugin id mismatch: {plugin!r}"
    assert plugin.get("activated") is True, (
        f"plugin not activated: activated={plugin.get('activated')}, "
        f"activationReason={plugin.get('activationReason')!r}, "
        f"status={plugin.get('status')!r}"
    )
    ce_ids = plugin.get("contextEngineIds", [])
    assert ce_ids == ["styx"], (
        f"contextEngineIds={ce_ids!r}, ожидали ['styx']. "
        f"plugin.activationReason={plugin.get('activationReason')!r}"
    )


def test_17_tools_registered() -> None:
    """Все 17 styx_* LLM tools зарегистрированы (D3 в plugin-v1 + D14
    в waves/28 — styx_ingest_document добавлен волной 28)."""
    info = _inspect_plugin_runtime()
    plugin = info.get("plugin", {})
    names = set(plugin.get("toolNames", []))
    expected = {
        "styx_store",
        "styx_recall",
        "styx_search_archive",
        "styx_reinterpret",
        "styx_ingest_experience",
        "styx_ingest_document",
        "styx_dialogue_save",
        "styx_dialogue_search",
        "styx_dialogue_recent",
        "styx_dialogue_sessions",
        "styx_dialogue_prepare_summary",
        "styx_relations_query",
        "styx_graph_traverse",
        "styx_analytics",
        "styx_explain",
        "styx_confirm_usage",
        "styx_link",
    }
    assert names == expected, (
        f"Tool set mismatch.\n  Missing: {expected - names}\n  Unexpected: {names - expected}"
    )
    assert len(plugin.get("toolNames", [])) == 17, (
        f"toolNames={plugin.get('toolNames')!r}, ожидали 17 уникальных"
    )


def test_context_engine_slot_active() -> None:
    """`plugins.slots.contextEngine` указывает на styx (config применён)."""
    res = _docker_exec(
        "openclaw-cli",
        "node",
        "/app/dist/index.js",
        "config",
        "get",
        "plugins.slots.contextEngine",
        "--json",
        timeout=15.0,
    )
    # `config get --json` возвращает чистый JSON value, а не объект-обёртку:
    # для leaf-string получаем `"styx"` как одиночный JSON-литерал.
    value = json.loads(res.stdout.strip())
    assert value == "styx", (
        f"contextEngine slot ≠ 'styx': {value!r}. Config bootstrap (openclaw-init.sh) не применил."
    )


# ─── e2e chat round-trip через `agent --local` ────────────────────────


@pytest.fixture
def database_url() -> str:
    url = os.environ.get("STYX_TEST_DATABASE_URL")
    if not url:
        pytest.skip("STYX_TEST_DATABASE_URL не задан")
    return url


def test_chat_turn1_writes_to_styx_memories(database_url: str) -> None:
    """Real chat round-trip: `openclaw agent --local` отрабатывает,
    embedded agent ходит в z.ai/glm-5.1 и получает ответ, ContextEngine
    записывает user+assistant turn в Styx memories.

    M3 (strict assert на write): после fix'а в plugin (Phase E) memories
    действительно растут — assert'им delta строго.

    M6 (idempotency): session_id и target — uuid'ы per invocation, тест
    зелёный при повторном запуске без `docker compose down -v`.

    M5: count'ы фильтрованы по session_id — параллельные writes от
    Hermes-плеч (тот же стик) не интерферируют.
    """
    # M6 + S4: per-test uuid-маркер. OpenClaw "phone-as-actor"
    # convention — `target` используется как derived part session key;
    # uuid делает прогон idempotent.
    #
    # Phase E observation: OpenClaw 2026.5.x в `agent --local` НЕ
    # пробрасывает `--session-id` в lifecycle sessionId. CLI-флаг
    # используется для assemble-pre-LLM, но runtime генерирует свой
    # session_id для ingest. Фильтр по session_id нерелевантен —
    # используем content_marker (per-test UUID, шансы collision
    # астрономически малы).
    session_id = str(uuid.uuid4())
    marker = session_id[:8]
    target = f"+1555555{uuid.uuid4().int % 10000:04d}"
    message = f"Phase E e2e test {marker}. Просто скажи hi одним предложением."

    before_memories = _count_table(database_url, "memories", content_marker=marker)

    res = _docker_exec(
        "openclaw-cli",
        "node",
        "/app/dist/index.js",
        "agent",
        "--local",
        "--to",
        target,
        "--message",
        message,
        "--session-id",
        session_id,
        "--json",
        "--timeout",
        "300",
        timeout=400.0,
        check=False,
    )

    assert res.returncode == 0, (
        f"`openclaw agent --local` failed (exit={res.returncode}).\n"
        f"  stdout: {res.stdout[:2000]}\n"
        f"  stderr: {res.stderr[:2000]}"
    )

    # Парсим JSON-output agent --local; должен содержать финальный
    # assistant message (модель ответила).
    try:
        payload = json.loads(res.stdout)
    except json.JSONDecodeError as err:
        pytest.fail(
            f"`agent --local --json` дал невалидный JSON: {err}.\n"
            f"  stdout (first 500 chars): {res.stdout[:500]}"
        )

    # Schema (openclaw 2026.5.7 с --to/--session-id):
    #   { meta: {agentMeta: {model, ...}, ...}, payloads: [{text, mediaUrl}] }
    payloads = payload.get("payloads", [])
    assert payloads, (
        f"`agent --local` payloads пуст → embedded agent не получил "
        f"ответ. payload keys: {list(payload.keys())}"
    )
    assistant_text = payloads[0].get("text") or ""
    # S2: модель не assert'им (могут быть fallback'и/A-B провайдеры).
    # Достаточно что текст ответа непустой.
    assert assistant_text.strip(), (
        f"payloads[0].text пустой → LLM не вернул content.\n"
        f"  meta: {payload.get('meta')}"
    )

    # M3 strict: ContextEngine.afterTurn вызвал /context/ingest_batch с
    # tail последнего turn'а (см. plugin-side fix Phase E). Должен быть
    # минимум +1 memory с marker'ом — это user-message, который шёл с
    # текстом from this test. Проверка через ILIKE %marker% работает
    # независимо от того под какой session_id runtime пишет.
    after_memories = _count_table(database_url, "memories", content_marker=marker)
    assert after_memories >= before_memories + 1, (
        f"memories с marker={marker!r}: "
        f"{before_memories} → {after_memories}; ожидали ≥+1 "
        f"(user-message с текстом теста). "
        f"ContextEngine.afterTurn не вызвал /context/ingest_batch?"
    )


def _count_table(
    database_url: str,
    table: str,
    session_id: str | None = None,
    content_marker: str | None = None,
) -> int:
    """COUNT(*) с опциональным фильтром по session_id и/или
    подстроке content (для memories).

    S5: whitelist таблиц — table-name приходит в f-string SQL, поэтому
    защищаемся от ошибок (опечаток / accidental SQL injection если в
    тесте проедет user-input в этот аргумент). Допустимые таблицы
    зафиксированы в schema (см. schema/0002_*.sql).

    M5: фильтр для chat-тестов нужен — иначе параллельные writes от
    Hermes pytest battery (тот же стик) дадут flake.

    Phase E observation: OpenClaw 2026.5.x в `agent --local` сам
    генерирует runtime sessionId (DEFAULT_AGENT_ID=main → собственный
    UUID), игнорируя `--session-id` CLI flag для actual lifecycle
    sessionId. Поэтому первый assemble идёт с CLI session_id (где user
    появляется), а ingest_batch для assistant идёт с runtime session_id.
    Фильтрация по `content_marker` (UUID-фрагмент в сообщении) даёт
    надёжный способ найти оба event'а одного chat-вызова.
    """
    assert table in {"sessions", "memories"}, (
        f"_count_table: неразрешённое имя таблицы {table!r}; "
        f"допустимы только sessions/memories"
    )
    clauses: list[str] = []
    args: list[object] = []
    if session_id is not None:
        clauses.append("session_id = %s")
        args.append(session_id)
    if content_marker is not None:
        assert table == "memories", (
            "content_marker применим только к memories"
        )
        clauses.append("content ILIKE %s")
        args.append(f"%{content_marker}%")
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*)::int FROM {table}{where}", args)
            (n,) = cur.fetchone()
            return int(n)


def _count_table_styx_leak(database_url: str) -> int:
    """COUNT(*) memories с любым `<styx*>` маркером в content.

    Волны 26.5 + 30 invariant: Styx wrap'ит блоки тегами семейства
    `<styx-{salient,recall,archive,dialogue,relations,explain,
    working-set}>...` при assembled view inject'е и tool-result
    обёртке. Legacy `<styx>...</styx>` (эпоха 26.5) тоже учитывается
    — backwards-compat для historical persist'нутых данных. Эти теги
    — exclusive marker, ничего другого их не пишет; присутствие в
    memories означает что assembled view или tool result утёк в
    afterTurn / ingest_batch.
    """
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*)::int FROM memories "
                "WHERE content LIKE %s OR content LIKE %s OR content LIKE %s",
                ("%<styx>%", "%</styx>%", "%<styx-%"),
            )
            (n,) = cur.fetchone()
            return int(n)


def test_chat_two_turns_same_session(database_url: str) -> None:
    """Два последовательных embedded turn'а в одной session.

    Проверяет что:
      - openclaw-cli может прогонять несколько turn'ов подряд
        (state shared volume сохраняет sessions/workspace);
      - ContextEngine.assemble не падает на turn 2 (если turn 1 что-то
        записал, assemble попробует восстановить — должен корректно
        обрабатывать любое состояние Styx PG, включая пустое);
      - после двух turn'ов в session_id есть строго больше memories
        чем после одного (M3 strict — afterTurn вызывается на каждом
        turn'е, не один раз глобально).

    M6: session_id и target — uuid per invocation.
    """
    # S4 + M6: см. test_chat_turn1_writes_to_styx_memories.
    # OpenClaw "phone-as-actor" convention — target используется для
    # derived session key; uuid делает прогон idempotent.
    #
    # Используем 2 отдельных marker'а — по одному на turn, чтобы
    # отличать writes turn 1 от turn 2 без зависимости от session_id
    # routing'а в runtime (см. test_chat_turn1_writes_to_styx_memories).
    session_id = str(uuid.uuid4())
    target = f"+1555555{uuid.uuid4().int % 10000:04d}"
    marker1 = uuid.uuid4().hex[:10]
    marker2 = uuid.uuid4().hex[:10]

    msg1 = f"Phase E turn1-marker-{marker1} — поздоровайся коротко."
    msg2 = f"Phase E turn2-marker-{marker2} — скажи 'OK turn2 received'."

    before1 = _count_table(database_url, "memories", content_marker=marker1)
    before2 = _count_table(database_url, "memories", content_marker=marker2)

    res1 = _docker_exec(
        "openclaw-cli",
        "node",
        "/app/dist/index.js",
        "agent",
        "--local",
        "--to",
        target,
        "--message",
        msg1,
        "--session-id",
        session_id,
        "--json",
        "--timeout",
        "300",
        timeout=400.0,
        check=False,
    )
    assert res1.returncode == 0, (
        f"Turn 1 failed: stderr={res1.stderr[:1000]}"
    )

    after_turn1_m1 = _count_table(database_url, "memories", content_marker=marker1)

    res2 = _docker_exec(
        "openclaw-cli",
        "node",
        "/app/dist/index.js",
        "agent",
        "--local",
        "--to",
        target,
        "--message",
        msg2,
        "--session-id",
        session_id,
        "--json",
        "--timeout",
        "300",
        timeout=400.0,
        check=False,
    )
    assert res2.returncode == 0, (
        f"Turn 2 failed: stderr={res2.stderr[:1000]}"
    )

    after_turn2_m2 = _count_table(database_url, "memories", content_marker=marker2)

    # Оба turn'а получили ответ от LLM (см. shape в test_chat_turn1_*).
    p1 = json.loads(res1.stdout)
    p2 = json.loads(res2.stdout)
    text1 = (p1.get("payloads", [{}])[0].get("text") or "").strip()
    text2 = (p2.get("payloads", [{}])[0].get("text") or "").strip()
    assert text1, f"Turn 1: payloads[0].text пустой; meta={p1.get('meta')}"
    assert text2, f"Turn 2: payloads[0].text пустой; meta={p2.get('meta')}"

    # M3 strict: каждый turn пишет user-message со своим marker'ом.
    # marker1 появляется только после turn 1, marker2 — только после
    # turn 2 (если afterTurn действительно вызвал /context/ingest_batch
    # на каждом turn'е).
    assert after_turn1_m1 >= before1 + 1, (
        f"После turn 1: memories с marker1={marker1!r} "
        f"{before1} → {after_turn1_m1}; ожидали ≥+1. "
        f"ContextEngine.afterTurn не вызвал ingest_batch на turn 1?"
    )
    assert after_turn2_m2 >= before2 + 1, (
        f"После turn 2: memories с marker2={marker2!r} "
        f"{before2} → {after_turn2_m2}; ожидали ≥+1. "
        f"ContextEngine.afterTurn не вызвал ingest_batch на turn 2 "
        f"(возможный regression — afterTurn fired только on first turn)."
    )

    # Волна 26.5 (defensive markers): salient block оборачивается в
    # <styx>...</styx>. Эти теги ВСТАВЛЯЮТСЯ только в assembled view
    # для LLM, и НЕ должны попадать в memories. ContextEngine.afterTurn
    # пишет original messages (rawMessages slice от last user) — там
    # никогда не было salient'а. Защитный assert: проверяем что в
    # memories нет ни одной content-строки с <styx> или </styx>.
    leaked = _count_table_styx_leak(database_url)
    assert leaked == 0, (
        f"Найдено {leaked} memories с <styx> тегами в content. "
        f"Salient block утёк в memories → bug в afterTurn / ingest_batch "
        f"(должен брать только original user/assistant messages, не "
        f"assembled view)."
    )


def test_anonymous_session_no_writes(database_url: str) -> None:
    """Negative test (M4): non-`agent:` sessionKey даёт passthrough,
    Styx ничего не пишет.

    План был передать sessionKey в legacy/alias формате через
    `--session-id <bare-string>`, но OpenClaw 2026.5.x в `agent --local`
    встраивает default agent context (DEFAULT_AGENT_ID="main") и через
    factoryCtx.agentDir = `${stateDir}/agents/main/agent` — даже если
    sessionKey передан без `agent:`-prefix'а, `deriveOpenclawAgentId`
    парсит agentDir и возвращает "main". То есть anonymous поток
    через CLI недостижим.

    Логику anonymous пути валидируют unit-тесты на
    `extensions/styx/src/context-engine.ts::deriveOpenclawAgentId`
    (TS-сторона) — там можно подать пустой ctx / sessionKey и проверить
    что engine возвращает passthrough. Здесь оставляем skip с явным
    reason, чтобы Phase F+ не забыл переключить на runtime-bypass когда
    OpenClaw добавит "anonymous" agent CLI flag.
    """
    pytest.skip(
        "OpenClaw 2026.5.x всегда добавляет default 'main' agent context в "
        "`agent --local` (через factoryCtx.agentDir); anonymous поток через "
        "CLI недостижим без runtime-flag. Логика anonymous-passthrough "
        "покрыта unit-тестами на TS-стороне (deriveOpenclawAgentId)."
    )
