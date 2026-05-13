"""Юнит-тесты `http/_wrap.py` — opt-in LLM wrap helper (волна 30 Phase B)."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from styx.http._wrap import (
    WRAP_CHANNELS,
    should_wrap_for_llm,
    wrap_for_llm,
)


def test_wrap_channels_match_taxonomy() -> None:
    """Совпадает с taxonomy волны 30 (D1) — без неожиданных каналов."""
    assert WRAP_CHANNELS == frozenset(
        {
            "salient",
            "recall",
            "archive",
            "dialogue",
            "relations",
            "explain",
            "working-set",
        }
    )


def test_wrap_for_llm_dict_payload() -> None:
    """Dict сериализуется в pretty-JSON и оборачивается в `<styx-recall>`."""
    out = wrap_for_llm({"hits": [{"id": 1, "text": "foo"}]}, "recall")
    assert out.startswith("<styx-recall>\n")
    assert out.endswith("\n</styx-recall>")
    inner = out[len("<styx-recall>\n") : -len("\n</styx-recall>")]
    parsed = json.loads(inner)
    assert parsed == {"hits": [{"id": 1, "text": "foo"}]}


def test_wrap_for_llm_pydantic_payload() -> None:
    """Pydantic BaseModel сериализуется через `model_dump(by_alias=True)`."""

    class Hit(BaseModel):
        id_: int
        text: str

        model_config = {"populate_by_name": True}

    out = wrap_for_llm(Hit(id_=1, text="привет"), "dialogue")
    assert out.startswith("<styx-dialogue>\n")
    inner = out[len("<styx-dialogue>\n") : -len("\n</styx-dialogue>")]
    parsed = json.loads(inner)
    assert parsed == {"id_": 1, "text": "привет"}


def test_wrap_for_llm_unicode_preserved() -> None:
    """ensure_ascii=False — русский остаётся читаемым."""
    out = wrap_for_llm({"q": "вопрос"}, "archive")
    assert "вопрос" in out
    assert "\\u" not in out


def test_wrap_for_llm_unknown_channel_raises() -> None:
    with pytest.raises(ValueError, match="unknown wrap channel"):
        wrap_for_llm({}, "memory")


def test_wrap_for_llm_string_payload_gets_json_encoded() -> None:
    """Если payload — уже строка (не dict), она JSON-кодируется внутри
    обёртки. Это не идемпотентность, а defensive-shape: caller не
    должен вызывать `wrap_for_llm` дважды (только один раз в HTTP
    route после получения raw response). Тест документирует
    наблюдаемое поведение, чтобы случайный двойной вызов не приводил
    к неинтерпретируемому output'у — просто экранированной строке."""
    out = wrap_for_llm("plain string", "explain")
    assert out.startswith("<styx-explain>\n")
    assert out.endswith("\n</styx-explain>")
    inner = out[len("<styx-explain>\n") : -len("\n</styx-explain>")]
    # JSON-encoded строка — она quoted внутри.
    assert json.loads(inner) == "plain string"


def test_wrap_for_llm_preserves_complex_types() -> None:
    """Datetime/UUID через `default=str` — не падает."""
    import uuid
    from datetime import datetime, timezone

    payload = {
        "ts": datetime(2026, 5, 11, 12, 0, 0, tzinfo=timezone.utc),
        "id": uuid.UUID("12345678-1234-5678-1234-567812345678"),
    }
    out = wrap_for_llm(payload, "explain")
    inner = out[len("<styx-explain>\n") : -len("\n</styx-explain>")]
    parsed = json.loads(inner)
    assert "2026-05-11" in parsed["ts"]
    assert parsed["id"] == "12345678-1234-5678-1234-567812345678"


def _make_app() -> FastAPI:
    """FastAPI-приложение для тестирования dependency."""
    from fastapi import Depends

    app = FastAPI()

    @app.get("/probe")
    def probe(wrap: bool = Depends(should_wrap_for_llm)) -> dict[str, bool]:
        return {"wrap": wrap}

    return app


def test_should_wrap_default_false() -> None:
    """Без флагов — wrap False."""
    client = TestClient(_make_app())
    r = client.get("/probe")
    assert r.status_code == 200
    assert r.json() == {"wrap": False}


def test_should_wrap_query_param_one() -> None:
    """`?wrap_for_llm=1` → True."""
    client = TestClient(_make_app())
    r = client.get("/probe", params={"wrap_for_llm": 1})
    assert r.json() == {"wrap": True}


def test_should_wrap_query_param_zero() -> None:
    """`?wrap_for_llm=0` → False (явный opt-out)."""
    client = TestClient(_make_app())
    r = client.get("/probe", params={"wrap_for_llm": 0})
    assert r.json() == {"wrap": False}


def test_should_wrap_query_param_invalid_rejected() -> None:
    """`?wrap_for_llm=2` — 422 (range validation)."""
    client = TestClient(_make_app())
    r = client.get("/probe", params={"wrap_for_llm": 2})
    assert r.status_code == 422


def test_should_wrap_header_truthy_values() -> None:
    """Header принимает 1 / true / yes / on (case-insensitive)."""
    client = TestClient(_make_app())
    for v in ("1", "true", "yes", "on", "TRUE", "Yes", "  1  "):
        r = client.get("/probe", headers={"X-Wrap-For-LLM": v})
        assert r.json() == {"wrap": True}, f"value {v!r} should enable wrap"


def test_should_wrap_header_falsy_values() -> None:
    """Header с другими значениями → wrap остаётся False."""
    client = TestClient(_make_app())
    for v in ("0", "false", "no", "", "off", "garbage"):
        r = client.get("/probe", headers={"X-Wrap-For-LLM": v})
        assert r.json() == {"wrap": False}, f"value {v!r} should NOT enable wrap"


def test_should_wrap_header_takes_priority_when_query_absent() -> None:
    """Header работает независимо от query-param."""
    client = TestClient(_make_app())
    r = client.get("/probe", headers={"X-Wrap-For-LLM": "1"})
    assert r.json() == {"wrap": True}


def test_should_wrap_either_flag_enables() -> None:
    """OR-логика: либо query, либо header — достаточно одного."""
    client = TestClient(_make_app())
    # Query + header off → True (через query).
    r1 = client.get(
        "/probe",
        params={"wrap_for_llm": 1},
        headers={"X-Wrap-For-LLM": "0"},
    )
    assert r1.json() == {"wrap": True}
    # Query off + header on → True (через header).
    r2 = client.get(
        "/probe",
        params={"wrap_for_llm": 0},
        headers={"X-Wrap-For-LLM": "1"},
    )
    assert r2.json() == {"wrap": True}
