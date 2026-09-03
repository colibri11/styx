# styx-core

Host-agnostic ядро Styx — дирижёра динамической части контекстного окна
LLM-агента. Содержит storage layer (Postgres + pgvector), recall pipeline,
focus tracker, hot-tier, eviction relevance, salient inject, working set
persistence, emotional baseline, workers и HTTP API daemon.

Не зависит от Hermes Agent. Используется как Python-библиотека или как
standalone HTTP daemon (`styx daemon run`).

Scoped social evidence — отдельный opt-in ledger/API. Он закрыт без
operator-managed hash-only principal registry, не читает dialogue/model output
для автоматических attestations и доставляет разрешённое evidence в cognition
только через явный observation bridge.

См. корневой `README.md` репо для общего описания архитектуры и
`docs/HTTP_API.md` для контракта API.
