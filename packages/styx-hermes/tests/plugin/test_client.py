"""Unit-тесты StyxCoreClient — без живого daemon, через mock requests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from styx_hermes.client import StyxCoreClient


@pytest.fixture
def client_no_token(monkeypatch: pytest.MonkeyPatch) -> StyxCoreClient:
    # В Docker окружении STYX_HTTP_TOKEN может быть в env — тест должен
    # явно сбросить чтобы проверить путь "ни token, ни env".
    monkeypatch.delenv("STYX_HTTP_TOKEN", raising=False)
    return StyxCoreClient(base_url="http://daemon.local:8788", token=None)


@pytest.fixture
def client_with_token(monkeypatch: pytest.MonkeyPatch) -> StyxCoreClient:
    monkeypatch.delenv("STYX_HTTP_TOKEN", raising=False)
    return StyxCoreClient(
        base_url="http://daemon.local:8788", token="test-token-12345"
    )


def _mock_response(status: int = 200, json_payload: dict | None = None):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status
    resp.json.return_value = json_payload or {}
    resp.text = ""
    if 200 <= status < 300:
        resp.raise_for_status = MagicMock()
    else:
        resp.raise_for_status = MagicMock(side_effect=requests.HTTPError())
    return resp


def test_base_url_strips_trailing_slash() -> None:
    c = StyxCoreClient(base_url="http://a:1234/", token=None)
    assert c.base_url == "http://a:1234"


def test_no_token_no_auth_header(client_no_token: StyxCoreClient) -> None:
    assert "Authorization" not in client_no_token._session.headers


def test_token_sets_auth_header(client_with_token: StyxCoreClient) -> None:
    assert (
        client_with_token._session.headers.get("Authorization")
        == "Bearer test-token-12345"
    )


def test_initialize_agent_payload(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"agent_id": "alyona", "tools": []}
        )
        resp = client_no_token.initialize_agent(
            "alyona",
            session_id="sid-1",
            agent_identity="alyona",
            platform="cli",
        )
        assert resp == {"agent_id": "alyona", "tools": []}
        args, kwargs = mock_post.call_args
        assert args[0] == "http://daemon.local:8788/agent/initialize"
        assert kwargs["json"]["agent_id"] == "alyona"
        assert kwargs["json"]["session_id"] == "sid-1"
        assert kwargs["json"]["agent_identity"] == "alyona"


def test_shutdown_204_returns_empty(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(204, {})
        client_no_token.shutdown_agent("alyona")
        mock_post.assert_called_once()


def test_sync_turn_payload(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"memory_ids": []})
        client_no_token.sync_turn(
            "alyona",
            user_content="hi",
            assistant_content="hello",
            session_id="sid-x",
        )
        args, kwargs = mock_post.call_args
        body = kwargs["json"]
        assert body["agent_id"] == "alyona"
        assert body["user_content"] == "hi"
        assert body["assistant_content"] == "hello"
        assert body["session_id"] == "sid-x"


def test_recall_long_timeout(client_no_token: StyxCoreClient) -> None:
    """recall использует long_timeout (по умолчанию 30s)."""
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"memories": []})
        client_no_token.recall("alyona", "query", limit=5)
        kwargs = mock_post.call_args.kwargs
        assert kwargs["timeout"] == client_no_token._long_timeout


def test_build_context_uses_long_timeout(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(
            200, {"messages": [], "compression_count": 0, "salient_injected": False}
        )
        client_no_token.build_context("alyona", [{"role": "user", "content": "x"}])
        kwargs = mock_post.call_args.kwargs
        assert kwargs["timeout"] == client_no_token._long_timeout


def test_pre_llm_inject_returns_context_or_none(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(200, {"context": "Peer прозвучал: ..."})
        out = client_no_token.pre_llm_inject(
            "alyona", session_id="sid", user_message="hi"
        )
        assert out["context"] == "Peer прозвучал: ..."

        mock_post.return_value = _mock_response(200, {"context": None})
        out2 = client_no_token.pre_llm_inject("alyona")
        assert out2["context"] is None


def test_5xx_raises(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(503, {"detail": "down"})
        with pytest.raises(requests.HTTPError):
            client_no_token.sync_turn("alyona")


def test_401_raises(client_no_token: StyxCoreClient) -> None:
    with patch.object(client_no_token._session, "post") as mock_post:
        mock_post.return_value = _mock_response(401, {"detail": "missing bearer"})
        with pytest.raises(requests.HTTPError):
            client_no_token.sync_turn("alyona")


def test_close_releases_session(client_no_token: StyxCoreClient) -> None:
    sess = client_no_token._session
    with patch.object(sess, "close") as mock_close:
        client_no_token.close()
        mock_close.assert_called_once()


def test_default_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STYX_DAEMON_URL", "http://envurl:9999")
    c = StyxCoreClient()
    assert c.base_url == "http://envurl:9999"


def test_default_token_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("STYX_HTTP_TOKEN", "env-token")
    c = StyxCoreClient()
    assert c._session.headers.get("Authorization") == "Bearer env-token"
