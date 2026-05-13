"""POST /pre_llm_inject — multi-channel framework для injection в user msg."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from styx.engine import pre_llm_inject
from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import PreLlmInjectRequest, PreLlmInjectResponse

router = APIRouter()


@router.post(
    "/pre_llm_inject",
    response_model=PreLlmInjectResponse,
    dependencies=[Depends(require_auth)],
)
def pre_llm(req: PreLlmInjectRequest) -> PreLlmInjectResponse:
    # registry.get валидирует agent_id зарегистрирован, но если
    # pre_llm_inject не configure'нут — возвращаем None (fail-open).
    registry.get(req.agent_id)

    kwargs = dict(req.extra)
    if req.session_id is not None:
        kwargs.setdefault("session_id", req.session_id)
    if req.user_message is not None:
        kwargs.setdefault("user_message", req.user_message)
    kwargs.setdefault("is_first_turn", req.is_first_turn)
    if req.model is not None:
        kwargs.setdefault("model", req.model)
    if req.platform is not None:
        kwargs.setdefault("platform", req.platform)

    result = pre_llm_inject.on_pre_llm_call(req.agent_id, **kwargs)
    if result is None:
        return PreLlmInjectResponse(context=None)
    return PreLlmInjectResponse(context=result.get("context"))
