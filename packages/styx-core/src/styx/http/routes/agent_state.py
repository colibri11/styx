"""GET /agent_state — snapshot эмоционального состояния агента."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from styx.emotional.baseline import read_baseline_for_scoring
from styx.emotional.state import read_last_state
from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import VAD, AgentStateResponse

router = APIRouter()


@router.get(
    "/agent_state",
    response_model=AgentStateResponse,
    dependencies=[Depends(require_auth)],
)
def agent_state(agent_id: str) -> AgentStateResponse:
    session = registry.get(agent_id)
    core = session.core
    if core._conn is None:
        return AgentStateResponse(agent_id=agent_id)

    last = read_last_state(core._conn, agent_id)
    baseline = read_baseline_for_scoring(core._conn, agent_id)

    instant_vad = None
    if last is not None:
        vec, _ts = last
        instant_vad = VAD(
            valence=vec.valence,
            arousal=vec.arousal,
            dominance=vec.dominance,
        )
    baseline_vad = (
        VAD(
            valence=baseline.valence,
            arousal=baseline.arousal,
            dominance=baseline.dominance,
        )
        if baseline is not None
        else None
    )
    return AgentStateResponse(
        agent_id=agent_id,
        instant=instant_vad,
        baseline=baseline_vad,
        mood=None,
    )
