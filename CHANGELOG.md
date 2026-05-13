# Changelog

Все значимые изменения в Styx документируются здесь. Формат —
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
версионирование — [SemVer](https://semver.org/lang/ru/).

## [1.0.0] — 2026-05-13

Первый публичный релиз.

Styx — инженерная реализация Locus: непрерывной среды между обращениями
к LLM, в которой разворачивается линия `я` агента. Подробности
концепции и архитектуры — в [`README.md`](README.md).

В релиз входит:

- **`packages/styx-core/`** — host-agnostic ядро + HTTP API daemon
  (FastAPI). Long-tier memory (PostgreSQL + pgvector), three-tier
  composer (active suffix / hot / long), background workers
  (importance scoring, lifecycle, classifier, sentiment, dialogue
  consolidation, reinterpret apply, memory consolidation,
  relation decay).
- **`packages/styx-hermes/`** — Hermes Agent plugin. Тонкий HTTP
  клиент к daemon'у. Регистрирует MemoryProvider, ContextEngine,
  Transport, `pre_llm_call` hook.
- **`extensions/styx/`** — OpenClaw plugin (TypeScript). 17 LLM
  tools, ContextEngine lifecycle bridge, 4 LLM-runbook skills.
- HTTP API (~30 endpoint'ов): lifecycle, composer, recall,
  search_archive, dialogue (5 routes), relations + graph traverse,
  reinterpret, ingest (experience + document), explain (3 modes)
  + analytics + confirm_usage. Контракт —
  [`docs/HTTP_API.md`](docs/HTTP_API.md).
- Memory markers taxonomy (`<styx-{salient,recall,archive,
  dialogue,relations,explain,working-set}>`) для различения
  source-канала инжекта в LLM-input'е.
- Document pipeline (PDF / DOCX / XLSX / Markdown / plain text)
  через pure-Python parsers.

Технологический стек: Python 3.11+, PostgreSQL 18 + pgvector,
Ollama (`embeddinggemma:300m-qat-q8_0` + `qwen3:4b-local`),
FastAPI, TypeScript.

Совместимость: Hermes Agent v2026.4.30+ (тестировалось до
v2026.5.7), OpenClaw v2026.5.3+.
