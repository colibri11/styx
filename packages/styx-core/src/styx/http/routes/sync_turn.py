"""POST /sync_turn — записать turn (user/assistant пара) в memory."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import SyncTurnRequest, SyncTurnResponse

router = APIRouter()


@router.post(
    "/sync_turn",
    response_model=SyncTurnResponse,
    dependencies=[Depends(require_auth)],
)
def sync_turn(req: SyncTurnRequest) -> SyncTurnResponse:
    session = registry.get(req.agent_id)
    session.core.sync_turn(
        user_content=req.user_content,
        assistant_content=req.assistant_content,
        session_id=req.session_id or "",
    )
    # Текущая sync_turn возвращает None — memory_ids недоступны через
    # этот вызов. Возвращаем пустые списки; контракт расширится когда
    # MemoryCore.sync_turn научится возвращать вставленные ids.
    return SyncTurnResponse(memory_ids=[], recall_event_ids=[])
