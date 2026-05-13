# styx-core

Host-agnostic ядро Styx — дирижёра динамической части контекстного окна
LLM-агента. Содержит storage layer (Postgres + pgvector), recall pipeline,
focus tracker, hot-tier, eviction relevance, salient inject, working set
persistence, emotional baseline, workers и HTTP API daemon.

Не зависит от Hermes Agent. Используется как Python-библиотека или как
standalone HTTP daemon (`styx daemon run`).

См. корневой `README.md` репо для общего описания архитектуры и
`docs/HTTP_API.md` для контракта API.
