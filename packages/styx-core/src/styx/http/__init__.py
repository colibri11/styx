"""Styx HTTP API daemon — FastAPI app + uvicorn entry.

Поверхность ``packages/styx-core/src/styx/http/``:

- ``app.py``       — FastAPI app factory
- ``auth.py``      — bearer token dependency
- ``models.py``    — Pydantic request/response models
- ``registry.py``  — agent registry (agent_id → AgentSession)
- ``routes/*``     — route modules (healthz, agent, sync_turn, recall,
                     context, pre_llm, agent_state)
- ``server.py``    — uvicorn entry для ``styx daemon run``

Контракт API — ``.design/host-agnostic-split-v1.md`` § 6.
"""
