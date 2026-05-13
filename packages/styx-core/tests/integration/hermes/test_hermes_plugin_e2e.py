"""Hermes plugin integration tests (волна 29 Phase F).

Mirror к ``tests/integration/openclaw/test_openclaw_plugin.py``.

Запускаются host-side; обращаются к docker-стику
``docker/docker-compose.test.yml``:

- ``hermes-styx`` — через ``docker compose exec`` (subprocess) запускают
  настоящий ``hermes -z PROMPT`` через z.ai/glm-5 backend (ZAI_API_KEY /
  GLM_API_KEY уже в test stack env).
- ``styx-daemon`` HTTP API — через ``psycopg`` напрямую к postgres :15432
  для верификации side-effects (memories writes, recall_events, etc.).

Архитектурный смысл (Phase F волны 29): проверить что после реализации
``StyxMemoryProvider.prefetch()`` (волна 29 Phase B) Hermes-агент
**действительно использует Styx Locus** для inject через
``/context/assemble``, а не только write-side через ``/sync_turn``.

Skip-паттерн: при отсутствии ``STYX_TEST_DATABASE_URL`` пропускаем —
docker стик не поднят.
"""

from __future__ import annotations

import os
import subprocess
import time
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
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "docker" / "docker-compose.test.yml").is_file():
            return parent
    raise RuntimeError(
        "docker-compose.test.yml not found above " + str(here)
    )


def _glm_api_key() -> str:
    """GLM_API_KEY из .env репо (test stack env уже его пробрасывает в
    container, но subprocess .env не видит — читаем явно для exec'а)."""
    env_file = _compose_root() / ".env"
    if not env_file.is_file():
        return ""
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line.startswith("GLM_API_KEY="):
            return line.split("=", 1)[1]
    return ""


def _hermes_prompt(prompt: str, *, timeout: float = 180.0) -> str:
    """Выполнить hermes -z PROMPT через GLM/zai в hermes-styx контейнере.

    Возвращает stdout (текстовый ответ агента). Поднимает CalledProcessError
    при non-zero exit code.
    """
    glm_key = _glm_api_key()
    if not glm_key:
        pytest.skip("GLM_API_KEY не найден в .env — z.ai backend недоступен")
    res = subprocess.run(
        [
            "docker", "compose",
            "-f", "docker/docker-compose.test.yml",
            "exec", "-T",
            "-e", f"GLM_API_KEY={glm_key}",
            "hermes-styx",
            "/opt/hermes/.venv/bin/hermes",
            "-z", prompt,
            "--provider", "zai",
            "--model", "zai/glm-5",
        ],
        cwd=str(_compose_root()),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return res.stdout.strip()


def _count_styx_marker_leaks(database_url: str) -> int:
    """Anti-leak invariant (волны 26.5 + 30): styx-маркеры не должны
    попадать обратно в memories."""
    with psycopg.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*)::int FROM memories WHERE "
                "content LIKE %s OR content LIKE %s OR "
                "content LIKE %s OR content LIKE %s",
                ("%<styx-%", "%</styx-%", "%<styx>%", "%</styx>%"),
            )
            (n,) = cur.fetchone()
            return int(n)


def _count_assemble_calls() -> int:
    """Считает число POST /context/assemble в логах styx-daemon.

    `prefetch()` каждый turn делает один assemble call. Регрессия — если
    Hermes перестанет звать prefetch (т.е. Hermes upstream API изменится
    или MemoryProvider stub'нется обратно).
    """
    res = subprocess.run(
        [
            "docker", "compose",
            "-f", "docker/docker-compose.test.yml",
            "logs", "styx-daemon",
        ],
        cwd=str(_compose_root()),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return res.stdout.count("POST /context/assemble")


# ─── fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def database_url() -> str:
    url = os.environ.get("STYX_TEST_DATABASE_URL")
    if not url:
        pytest.skip("STYX_TEST_DATABASE_URL не задан")
    return url


# ─── tests ────────────────────────────────────────────────────────────


def test_hermes_plugin_loads_and_responds(database_url: str) -> None:
    """Smoke: Hermes стартует, plugin загружен, GLM отвечает."""
    out = _hermes_prompt(
        f"echo-{uuid.uuid4().hex[:8]} ответь одним словом 'ok'"
    )
    assert out, "Hermes вернул пустой ответ"


def test_prefetch_calls_context_assemble(database_url: str) -> None:
    """Phase B invariant: каждый turn вызывает /context/assemble через
    `StyxMemoryProvider.prefetch()`. Это главное доказательство что
    Hermes реально подключён к Styx Locus, а не только write-only.
    """
    before = _count_assemble_calls()
    _hermes_prompt(f"проба связи {uuid.uuid4().hex[:8]}")
    # Hermes делает несколько assemble calls per turn (prefetch + on_pre_compress
    # на boundary, обычно ≥1).
    after = _count_assemble_calls()
    assert after > before, (
        f"prefetch() не вызвался: до={before}, после={after}. "
        "Phase B (StyxMemoryProvider.prefetch) или Hermes runtime интеграция сломана."
    )


def test_recall_through_prefetch_two_turns(database_url: str) -> None:
    """End-to-end recall через Phase B prefetch path.

    Turn 1: записываем уникальный marker → sync_turn пишет в memories.
    Turn 2: вопрос про marker → prefetch() вызывает /context/assemble →
    salient_text возвращается с marker'ом → Hermes-агент видит и
    цитирует.
    """
    marker = f"кодовое-слово-{uuid.uuid4().hex[:6]}"
    # Turn 1: запомнить
    _hermes_prompt(
        f"запомни на будущее: моё кодовое слово сегодня — {marker}"
    )
    # Дать sync_turn время записать в БД (background flush ~1s).
    time.sleep(2)

    # Turn 2: recall через prefetch.
    out = _hermes_prompt("какое моё кодовое слово сегодня?")
    assert marker in out, (
        f"Hermes не вспомнил marker {marker!r} через prefetch path: {out!r}"
    )


def test_no_styx_marker_leak_into_memories(database_url: str) -> None:
    """Anti-leak invariant: после нескольких turn'ов с prefetch
    (`<styx-salient>` инжект'ы) — НИ ОДНОЙ memory с styx-маркером в content.

    Если leak — sync_turn где-то persist'ит assembled view вместо clean
    user/assistant content (волны 26.5 + 30 invariant).
    """
    # Несколько turn'ов чтобы prefetch гарантированно сработал
    for _ in range(2):
        _hermes_prompt(
            f"короткая реплика {uuid.uuid4().hex[:6]}, ответь одним словом"
        )
    time.sleep(2)
    leaks = _count_styx_marker_leaks(database_url)
    assert leaks == 0, (
        f"найдено {leaks} memories со styx-маркерами в content — "
        "leak assembled view в sync_turn (волны 26.5/30 invariant)."
    )
