# styx-hermes

Тонкая Hermes-обёртка для styx-core. Регистрирует слоты Hermes
(`MemoryProvider`, `ContextEngine`, два `Transport`, `pre_llm_call` hook)
и проксирует все вызовы по HTTP в `styx-core` daemon.

Никакого state'а на стороне Hermes-процесса. Daemon — отдельный процесс
(`styx daemon run`).

См. корневой `README.md` репо для установочного пути и
`docs/HTTP_API.md` для контракта API.
