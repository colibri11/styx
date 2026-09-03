# styx-core

Текущая версия пакета: **1.7.0** (репозиторный релиз
[`v1.0.16`](https://github.com/colibri11/styx/releases/tag/v1.0.16)).

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

См. корневой [`README.md`](../../README.md) для общего описания архитектуры,
[`docs/HTTP_API.md`](../../docs/HTTP_API.md) для контракта API и
[`docs/DEPLOYMENT.md`](../../docs/DEPLOYMENT.md) для порядка обновления.
