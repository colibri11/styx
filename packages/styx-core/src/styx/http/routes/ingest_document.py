"""POST /ingest_document — file-ingest pipeline (волна 28).

Plugin шлёт абсолютный path; core читает диск, парсит (pypdf /
python-docx / openpyxl / builtin), режет на chunks через
``route_long_content`` (волна 19), embed'ит, INSERT'ит document +
chunks. tail-memory НЕ создаётся — pull-only архив (D5 в waves/28).

Идемпотентен через SHA256 от file bytes: повторный ingest того же
файла возвращает existing ``document_id`` с ``deduplicated=True``,
без побочных эффектов.

Pipeline-канал — без gatekeeper'а / auto-link'а / classifier'а
(symmetric с /ingest_experience, D13 в waves/28). Path validation
под whitelist ``STYX_INGEST_DOC_ROOTS`` + size guard.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from styx.http import registry
from styx.http.auth import require_auth
from styx.http.models import IngestDocumentRequest, IngestDocumentResponse

router = APIRouter()


@router.post(
    "/ingest_document",
    response_model=IngestDocumentResponse,
    dependencies=[Depends(require_auth)],
)
def ingest_document(req: IngestDocumentRequest) -> IngestDocumentResponse:
    session = registry.get(req.agent_id)
    try:
        outcome = session.core.ingest_document(
            path=req.path,
            source_ref=req.source_ref,
            visibility=req.visibility,
            metadata=req.metadata,
            content_hash=req.content_hash,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        # Provider не инициализирован или ingest_document disabled.
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IngestDocumentResponse(
        document_id=outcome.document_id,
        deduplicated=outcome.deduplicated,
        chunks_count=outcome.chunks_count,
        mime_type=outcome.mime_type,
        original_name=outcome.original_name,
        size_bytes=outcome.size_bytes,
        char_count=outcome.char_count,
        content_hash=outcome.content_hash,
    )
