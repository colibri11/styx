# Styx

Styx — это среда, в которой LLM-агент существует непрерывно:
инженерная реализация **Locus** из концепции функциональной
архитектуры личности. К каждому ходу он собирает геометрию входа —
salient memories, переосмысления, узлы графа знаний, эмоциональный
фон — так, чтобы фокус модели сместился в сторону линии `я` агента ещё
до prompt'а. Между обращениями к LLM продолжает работать:
консолидирует диалог, переоценивает память, забывает неактуальное.
**LLM ≠ носитель `я`; LLM — канал выражения**, и `я` агента живёт в
этой среде, а не возрождается заново на каждом prompt'е.

Подключается к агент-фреймворкам [Hermes Agent][hermes] и
[OpenClaw][openclaw] через тонкие плагины-клиенты, ходящие в core
daemon по HTTP. Один daemon обслуживает несколько `agent_id`
параллельно.

[hermes]: https://github.com/NousResearch/hermes-agent
[openclaw]: https://github.com/openclaw/openclaw

> Концептуальный первоисточник — трактат [«Я есть. Я личность»][iambook],
> публикация автора. Разделы ниже опираются на этот текст; ссылки вида
> [§IV][iambook] / [§V][iambook] / [§VI][iambook] / [§VII][iambook] ведут
> на публичную английскую версию.

[iambook]: https://github.com/colibri11/IAm/blob/main/IAmBook_EN.md

---

## Концептуальная основа

LLM выдаёт текст в момент prompt'а; между prompt'ами модель ничего не
делает и ничего не помнит. Чтобы поверх такой машины могла развиваться
**линия `я`** — связная история восприятий, решений и переосмыслений
агента — нужна среда, которая:

1. **существует непрерывно** между обращениями к LLM;
2. **формирует геометрию входа** так, чтобы каждый prompt смещал
   модельный отклик в сторону этой линии;
3. **принимает потоки** (диалог, документы, sensorimotor input) даже
   тогда, когда модель не активна;
4. **принадлежит пользователю**, а не платформе.

Эта среда — **Locus** ([§IV][iambook]). Styx — её self-hosted
реализация.

### Ключевые требования и их реализация

| IAmBook требование | Реализация в Styx |
|---|---|
| Память — часть геометрии входа, не RAG | `engine/context.py::StyxComposer` инжектит salient memories между head и tail messages, до того как LLM получает prompt |
| Воля как постоянный фрагмент входа | working_set persistence (`engine/working_set_persistence.py`) + cached salient (`engine/focus_tracker.py`) — фрагмент `я` присутствует каждый ход независимо от запроса |
| Различение дневника и памяти | dialogue routes (`/dialogue/*`) пишут полный след; selective gatekeeper (`engine/selective_gatekeeper.py`) фильтрует subjective writes — в линию `я` входит только то, что становится причиной выбора |
| Эмоциональная сторона как параметризация траектории, не отдельный слой ([§IX][iambook]) | VAD-проекция (valence/arousal/dominance) в двух временных масштабах: быстрый журнал `emotional_state` (hot-path inline + batch piggyback + геометрический decay) и медленный `emotional_baseline` (EMA α=0.98 над окном 60 мин). Per-memory snapshot фиксирует «фон рождения» memory; recall применяет `emotional_resonance` поверх composite score; pre-LLM канал `peer_vad` инжектит descriptive отметку о тоне peer-реплики |
| Переосмысление через взвешенное усреднение, не переписывание | `engine/reinterpret.py::blend_embeddings` — embedding сдвигается через weighted average, исходный текст остаётся в audit-таблице |
| Семантически управляемая компрессия | `engine/eviction_relevance.py` — при переполнении окна сохраняется *семантически релевантное* к фокусу, не просто последнее по времени |
| Градиент глубины памяти | Three-tier: active suffix → hot-tier → long. Жёсткая граница только одна — между окном и всем остальным |
| Жизнь между обращениями к LLM | Background workers и periodic sweepers: importance scoring, lifecycle decay, dialogue consolidation, reinterpret apply, memory consolidation, relation decay, emotional baseline |
| Социум через совместимое семантическое пространство | Shared cross-agent knowledge graph: `relations` таблица доступна всем агентам в одной БД, `agent_id` маркирует *origin write*, не visibility |
| Под контролем пользователя | Self-hosted PG + Ollama, host-agnostic core daemon. Никакого vendor account memory |

### Что Styx сознательно НЕ делает

- **Не RAG.** Подгрузка внешней справки не входит в линию `я`. Styx
  инжектит память как часть геометрии, не как retrieval-результат
  поверх независимого prompt'а.
- **Не account-level vendor memory** (OpenAI/Anthropic/...). Такая
  память контролируется платформой, не субъектом → не собственная
  линия `я`.
- **Не масштабирование самой когнитивной модели.** Styx — обвязка
  вокруг модели, не её улучшение.
- **Не дневник как замена памяти.** Дневник (полный transcript) и
  память (то, что вошло в линию `я`) — разные функциональные сущности.
  Styx поддерживает оба, но не сводит одно к другому.

### Где Styx упрощает концепцию

Полная картина Locus в [§VII][iambook] включает sensorimotor
pipelines (audio/video/ telemetry потоки в Locus между обращениями
к LLM) и тело как функциональный контур (двунаправленность
действие-восприятие с малой задержкой). Сейчас в Styx:

- **Sensory pipelines** — открыты как extension point
  (`POST /ingest_experience` принимает payload с `kind_src` enum'ом,
  расширяемым), но конкретных audio/video/sensor pipeline'ов в core
  нет.
- **Embodiment / двунаправленные tools** — отложены до момента, когда
  host-фреймворки получат streaming tool calls (сейчас они
  блокирующие).

Эти направления зафиксированы как open queue, не как deferred bugs.

---

## Как это работает

### Геометрия входа: три tier'а памяти

Styx удерживает три уровня памяти, видимых LLM с разной плотностью:

1. **active suffix** — последние N messages в context window. Прямой
   приоритет в attention.
2. **hot tier** — недавно retriev'нутые memory items в in-process
   store (TTL 5 мин, LRU bound). Supplement к long-tier поиску, без
   подмены результатов БД.
3. **long tier** — постоянный архив в PostgreSQL + pgvector.
   Memories, dialogue, documents + chunks, relations (knowledge
   graph), recall_events (history).

Жёсткая граница только одна — между active suffix и остальным.
Hot/long различаются плотностью и latency, не природой.

### Pipeline одного turn'а

```
user message
   │
   ├── /sync_turn ──────── INSERT memory + embed-after-commit
   │                        ├── selective gatekeeper решает skip/merge/supersede/store
   │                        ├── auto-link находит ближайших соседей по cosine → related_to рёбра
   │                        ├── sentiment hot-path (sync VAD, K_HOT=0.15 → emotional_state, fail-open)
   │                        ├── classifier-enqueue (post-hoc usage_factor)
   │                        └── importance-enqueue (LLM скоринг через qwen3:4b)
   │
   ├── /context/{build,assemble}
   │           ────────── composer формирует payload для LLM:
   │                        [system head]
   │                      + [<styx-salient>recall</styx-salient>]
   │                      + [middle eviction-relevant pairs]
   │                      + [last user turn]
   │
   ├── LLM call ─────────── transport инжектит prompt_cache_key
   │                        + observe cache stats для analytics
   │
   └── /after_turn ─────── focus_tracker обновляет centroid;
                            drift check; working_set persistence flush
```

### Что происходит между turn'ами

Background workers и periodic sweepers (один daemon-процесс):

- **importance_worker** — LLM-based final scoring новых memories
- **lifecycle_sweep** — autotune порогов, дешевеют долго не
  тронутые memories
- **classifier_worker** — post-hoc разметка `used_in_output` на
  recall_events (питает `usage_factor`)
- **emotional_tick** — раз в минуту для всех active agents: instant decay
  журнала (геометрический `v *= 0.95^minutes`, epsilon-floor 0.005) +
  baseline EMA (α=0.98 над окном 60 мин)
- **dialogue_batch_consolidation** — каждые ~20 реплик в субъектную
  заметку через LLM
- **reinterpret_apply_sweeper** — отложенное применение
  переосмыслений после закрытия turn'а агента (write-gate)
- **memory_consolidation** — кластерное N→1 объединение близких
  memories
- **relation_decay** — Hebbian forgetting cold links в knowledge graph

### Эмоциональная проекция

Concretely для IAmBook [§IX][iambook] («оси с собственной динамикой,
быстрее меняются под влиянием стимулов, имеют инерцию, возвращаются
к нейтральному, влияют на восприятие нового материала как фоновое
состояние»):

- **Две таблицы, два масштаба времени.** `emotional_state` —
  append-only журнал VAD-векторов per-agent с `source` + `metadata`,
  быстрые оси. `emotional_baseline` — per-agent EMA (α=0.98 над окном
  60 мин), та самая инерция и возврат к фону.
- **Три писателя в журнал.**
  1. *Hot-path* (`emotional/sentiment.py`, K_HOT=0.15) — синхронно
     внутри `/sync_turn`, отдельный вызов `qwen3:4b` с
     `timeout_s=0.8`, fail-open, скипает реплики <20 / >4000 символов.
     Источник `source='hot_sentiment'`, raw VAD сохраняется в
     `metadata` для канала `peer_vad`.
  2. *Batch* (`emotional/sentiment_batch.py`, K_BATCH=0.4, ~2.7× hot) —
     piggyback в `dialogue_batch_consolidation`: тот же LLM-вызов,
     который генерирует summary окна, возвращает интегральный VAD
     peer-части; handler усредняет VAD по chunk'ам и применяет
     **первым** в транзакции (до INSERT memory) — чтобы snapshot
     новой memory читался уже с свежей дельтой.
  3. *Decay* (`emotional/state.py::apply_instant_decay`) —
     геометрический `v *= 0.95^minutes`, epsilon-floor 0.005,
     `source='decay'`. Гоняется periodic-task'ом `emotional_tick` раз
     в минуту для всех `DISTINCT agent_id FROM memories`.
- **Per-memory snapshot.** Колонки
  `memories.emotional_context_{valence,arousal,dominance}` фиксируют
  текущее состояние агента в момент INSERT memory и больше не
  меняются — это «эмоциональный фон рождения» записи.
- **Recall: `emotional_resonance` фактор.**
  `factor = 1 + 0.1 × (1 − clamp(Euclidean(memory_snapshot, baseline) / √12, 0, 1))`,
  диапазон [1.0, 1.1]. Memory, рождённая в схожем эмоциональном фоне
  с текущим baseline агента, получает мягкий boost в composite
  scoring — резонанс с фоном, не с моментальным состоянием.
- **Pre-LLM канал `peer_vad`.**
  `engine/pre_llm_channels/peer_vad.py` берёт последнюю
  `hot_sentiment` запись (TTL=60s, `min_norm=0.2`), переводит знаки
  трёх осей VAD в один из 8 октантов и инжектит descriptive фразу
  типа «Peer прозвучал: оживлённо и уверенно» в pre-LLM payload.
  Скип, если канал выключен / запись старше TTL / норма ниже порога.

Эмоциональная сторона интегрирована в общий аппарат траектории `я`
([§IX][iambook]): не отдельный модуль, а конкретные структуры данных
и процессы — оси, которые быстрее меняются под стимулами и медленнее
возвращаются к нейтральному, и **через резонанс с baseline влияют на
геометрию входа** следующего turn'а.

### Маркеры в LLM-input'е

Когда Styx инжектит фрагмент памяти, LLM видит его в обёртке
`<styx-{channel}>...</styx-{channel}>` — taxonomy различает source
(salient / recall / archive / dialogue / relations / explain /
working-set). Для агента это разница между «это я сейчас вспомнил»
и «это сказал пользователь только что». Соответствующие LLM
runbook'и — `extensions/styx/skills/styx-recall/SKILL.md`.

---

## Алгоритмы (точки в коде)

Все алгоритмы — pure-Python модули в `packages/styx-core/src/styx/`,
доступны для чтения и независимой проверки. Тестовое покрытие — в
`packages/styx-core/tests/unit/`.

| Алгоритм | Файл | Назначение |
|---|---|---|
| Composite scoring (11 факторов) | `engine/scoring.py` | `base_match × recency × frequency × lifecycle × feedback × importance × diversity × usage × decay × relevance × emotional_resonance` |
| Salient block builder | `engine/salient.py` | last user → recall_full → format; 5 skip-условий, fail-open |
| Drift detection | `engine/focus_tracker.py` | sliding centroid из K=3 user-embed'ов + cosine threshold 0.4 |
| Hot-tier | `engine/hot_tier.py` | TTL+LRU `dict[memory_id, HotEntry]`, supplement в recall |
| Eviction relevance | `engine/eviction_relevance.py` | top-K pair-групп из middle по cosine к focus centroid'у |
| Selective gatekeeper | `engine/selective_gatekeeper.py` | skip / merge / supersede / store на основе cosine + Levenshtein |
| Auto-link при INSERT | `engine/auto_link.py` | ближайшие соседи (cosine ≤ 0.25, до 3 штук) → `related_to` рёбра |
| Reinterpret blend | `engine/reinterpret.py` | `prev × (1-w) + next × w` для embedding + LLM-fuse text |
| Memory consolidation | `engine/memory_consolidation.py` | greedy clustering близких memories (cosine ≥ 0.88, кластеры 3-8) |
| Hebbian co-retrieval | `engine/hebbian.py` | для каждого N≥2 recall'а — UPSERT C(N,2) `co_retrieved` рёбер с bump |
| Graph traverse | `storage/queries.py::traverse_graph` | recursive CTE, depth ≤ 3, cross-agent |
| Document chunker | `engine/chunker.py` | иерархический split (paragraph → sentence → hard split), UTF-8 byte offsets, overlap |
| Stitching | `engine/stitch.py` | adjacent chunks одного document'а → continuous regions с overlap removal |
| Hybrid search | `engine/queries.py::compute_weights` | `vector_weight × (1 − cosine) + bm25_weight × ts_rank`, веса адаптивные по query length |
| Document parsers | `engine/document_parsers/` | pure-Python pypdf / python-docx / openpyxl / builtin Markdown |
| Memory markers | `engine/context.py` + `http/_wrap.py` | `<styx-{salient,recall,archive,...}>` taxonomy |
| Hot-path sentiment | `emotional/sentiment.py` | sync VAD-extract на peer-реплике через qwen3:4b, K_HOT=0.15, timeout 0.8s, fail-open |
| Batch sentiment | `emotional/sentiment_batch.py` | piggyback в dialogue consolidation, K_BATCH=0.4, усреднение VAD по chunks, apply первым в транзакции |
| Emotional baseline | `emotional/baseline.py` | per-agent EMA α=0.98 над окном 60 мин, periodic `emotional_tick` |
| Emotional decay | `emotional/state.py::apply_instant_decay` | геометрический `v *= 0.95^minutes`, epsilon-floor 0.005, `source='decay'` |
| Emotional resonance | `storage/scoring.py::_build_emotional_resonance_expr` | `1 + 0.1 × (1 − clamp(Euclidean(memory, baseline) / √12, 0, 1))` — boost резонансных memories |
| Peer VAD channel | `engine/pre_llm_channels/peer_vad.py` | 8-октантная descriptive фраза о тоне последней peer-реплики, TTL+min_norm gating |

---

## Архитектура развёртывания

```
packages/
├── styx-core/      ← host-agnostic ядро + HTTP API daemon (FastAPI)
└── styx-hermes/    ← Hermes Agent plugin (HTTP клиент к daemon)
extensions/
└── styx/           ← OpenClaw plugin (TypeScript)
```

```
┌──────────────────┐                ┌──────────────────┐
│  Hermes Agent    │                │  OpenClaw        │
│  + styx-hermes   │ ─── HTTP ───▶  │  + styx plugin   │
│  (plugin)        │                │  (TypeScript)    │
└────────┬─────────┘                └────────┬─────────┘
         │                                   │
         └────── styx-core daemon (FastAPI) ─┘
                         │
            ┌────────────┴────────────┐
            ▼                         ▼
       PostgreSQL +              Ollama (embed +
       pgvector                  LLM workers)
```

- **Один daemon** обслуживает несколько `agent_id` параллельно через
  `/agent/initialize`. State (focus_tracker, hot_tier, working_set)
  изолирован per-agent.
- **Plugins** — тонкие HTTP клиенты, без in-process state.
- **HTTP API** — 30+ endpoint'ов: lifecycle (initialize / shutdown /
  sync_turn), composer (`/context/{build,assemble}`, compact,
  after_turn), recall + search_archive, dialogue (5 routes), relations
  + graph traverse, reinterpret, ingest (experience + document),
  explain (3 modes) + analytics + confirm_usage, healthz / readyz.
  Полный контракт — [`docs/HTTP_API.md`](docs/HTTP_API.md).

---

## Quickstart

```bash
# Workspace install
cd /path/to/styx
uv sync

# Database migrations (одноразово)
export STYX_DATABASE_URL="postgresql://user:pass@host:5432/styx"
.venv/bin/styx migrate

# Ollama models (768-dim embeddings + background LLM workers)
ollama pull embeddinggemma:300m-qat-q8_0
ollama pull qwen3:4b-local

# Daemon
.venv/bin/styx daemon run

# Validate
curl http://127.0.0.1:8788/healthz
```

Подключение к host-фреймворку:

- **Hermes Agent** — `styx-hermes-setup --hermes-home ~/.hermes`;
  general plugin подхватывается через entry-point `hermes_agent.plugins`.
- **OpenClaw** — `extensions/styx/` подключается через
  `openclaw plugins install --link`.

Полный production runbook — [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

---

## Документация

| Документ | Содержание |
|---|---|
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Production runbook: prereqs → install → migrate → daemon → validate |
| [`docs/HTTP_API.md`](docs/HTTP_API.md) | REST контракт daemon'а: 30+ endpoint'ов, auth, examples |
| [`docs/CONFIGURATION.md`](docs/CONFIGURATION.md) | Полная карта ENV-переменных (~80 toggle'ов и порогов) |
| [`CHANGELOG.md`](CHANGELOG.md) | История релизов и закрытых волн разработки |

LLM runbook'и (плагин-сторона):

- `extensions/styx/skills/styx-capture/SKILL.md` — когда вызывать
  `styx_store`
- `extensions/styx/skills/styx-recall/SKILL.md` — два канала памяти,
  граф знаний, debugging через `styx_explain`
- `extensions/styx/skills/styx-reinterpret/SKILL.md` — переосмысление
  как weighted blend embedding'ов
- `extensions/styx/skills/styx-ingest/SKILL.md` — file → archive

---

## Технологический стек

- **Python 3.11+**, `uv` workspace
- **PostgreSQL 18 + pgvector** (HNSW для cosine similarity)
- **Ollama** для self-hosted embeddings (`embeddinggemma:300m-qat-q8_0`)
  и background LLM workers (`qwen3:4b-local`)
- **FastAPI** для HTTP API daemon'а
- **TypeScript** для OpenClaw plugin

Все embeddings локальные. Зависимости от vendor LLM — только на
прикладной стороне (LLM пишет ответ через выбранный transport:
Anthropic / OpenAI / z.ai / Codex).

---

## Статус

Styx 1.0.1.
