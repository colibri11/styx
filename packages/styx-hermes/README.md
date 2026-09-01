# styx-hermes

Тонкая Hermes-обёртка для styx-core. Регистрирует слоты Hermes
(`MemoryProvider`, два `Transport`, `pre_llm_call` и `post_llm_call` hooks)
и проксирует все вызовы по HTTP в `styx-core` daemon.

`post_llm_call` передаёт finalized turn (включая bounded tool trajectory)
в `/affect/observe_turn` до фонового `sync_turn`. Core сохраняет причинный
переход, а следующие memory rows получают его provenance-снимок. Hook
fail-open и idempotent: сбой наблюдения не меняет уже завершённый ответ.

Никакого state'а на стороне Hermes-процесса. Daemon — отдельный процесс
(`styx daemon run`).

См. корневой `README.md` репо для установочного пути и
`docs/HTTP_API.md` для контракта API.
