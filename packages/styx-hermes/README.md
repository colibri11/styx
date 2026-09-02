# styx-hermes

Тонкая Hermes-обёртка для styx-core. Регистрирует слоты Hermes
(`MemoryProvider`, два `Transport`, `pre_llm_call` и `post_llm_call` hooks)
и проксирует все вызовы по HTTP в `styx-core` daemon.

`pre_llm_call` получает единый fenced envelope через `/cognition/preturn`,
передаёт физический `host_key` когда Hermes его предоставляет и сохраняет
`snapshot_token`. Same-act preturn retry идемпотентен; abandoned snapshot не
теряет feedback навсегда — observation lease делает доставку recoverable
at-least-once. `post_llm_call` после завершения tool loop
передаёт finalized channel projection, ordered bounded tool trajectory,
`host_key`, declared parent и snapshot в `/cognition/commit`. Retry одного
физического turn идемпотентен по `host_key`; declared ancestry не выводится из
времени доставки. Ack идемпотентен; exactly-once при падении host не
обещается. Terminal hook fail-open: сбой Styx не меняет уже завершённый ответ
Hermes.

Adapter передаёт bounded finalized projection до 64 ordered
`call|result|error` events суммарно. Явный client method
`cognition_observe` предназначен только для независимо пришедших после act,
предварительно редуцированных различий. `result`/`error`, уже увиденные внутри
этого tool loop, остаются same-act journal events и не переиздаются как
future observation. После commit core создаёт durable reduction outcome;
canonical reducer асинхронно выводит 0..4 evidence-bound residues и обновляет
causal carrier. Повтор с тем же `host_key`, но иным bounded request получает
`409`, а не молча принимается как прежний act.

Legacy `/pre_llm_inject` и `/affect/observe_turn` + provider `sync_turn`
используются только в mixed-version deployment, когда cognition endpoint
ответил именно `404`. Другие HTTP-ошибки не запускают двойную запись.

На стороне Hermes-процесса нет durable или authoritative state Styx.
Адаптер держит только bounded transient coordination caches для
snapshot/ancestry и idempotent retry; после рестарта источником истины остаётся
daemon/PostgreSQL (`styx daemon run`).

См. корневой `README.md` репо для установочного пути и
`docs/HTTP_API.md` для контракта API.
