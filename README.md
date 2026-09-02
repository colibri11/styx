# Styx

Styx — self-hosted подсистема памяти и контекста для LLM-агентов. Перед
каждым ходом она собирает bounded контекст из сохранённых следов,
причинных свидетельств, working state и результатов реконструкции; между
ходами обновляет их lifecycle и lineage. Это одна рабочая
**Locus-style архитектура** сохраняющегося контура, а не утверждение о
единственном месте существования `я`.

Styx не устанавливает, является ли подключённая система сознанием или
личностью. Языковая модель может быть каналом выражения, компонентом
когнитивного процесса или выполнять обе функции — это зависит от границ
конкретной архитектуры.

Подключается к агент-фреймворкам [Hermes Agent][hermes] и
[OpenClaw][openclaw] через тонкие плагины-клиенты, ходящие в core
daemon по HTTP. Один daemon обслуживает несколько `agent_id`
параллельно.

[hermes]: https://github.com/NousResearch/hermes-agent
[openclaw]: https://github.com/openclaw/openclaw

> Онтологический первоисточник — трактат [«Я есть. Я личность»][iambook].
> [«Философия Кремния»][silicon] — прикладное продолжение: оно отделяет
> следствия концепции от рабочих технических гипотез, которые не являются
> единственно возможными реализациями.

[iambook]: https://github.com/colibri11/IAm/blob/main/IAmBook.md
[silicon]: https://github.com/colibri11/IAm/blob/main/IAmPhilosophyOfSilicon.md

---

## Концептуальная основа

Обычный model call не обязан сохранять собственное состояние между
запусками. Styx добавляет к host-runtime сохраняющийся контур, который:

1. **существует непрерывно** между обращениями к LLM;
2. **сохраняет различимые последствия** завершённых актов и их lineage;
3. **реконструирует текущий контекст** из eligible traces и отдельно
   предоставляет cited external evidence;
4. **принимает события и последствия действий** независимо от model calls;
5. **контролируется оператором**, а не vendor account memory.

В терминах [«Философии Кремния»][silicon] это одна из возможных рабочих
реализаций Locus-контура. Название не вводит новую онтологическую сущность.

### Концептуальные функции и реализация Styx

| Функция или ограничение | Реализация в Styx | Статус |
|---|---|---|
| Сохранение следов и реконструкция | cognition envelope + `StyxComposer` | рабочая Locus-style архитектура |
| Вклад всей live subjective line | versioned, query-independent `will_projection` | инженерная проекция; не доказательство воли/сознания |
| Дневник, внешнее свидетельство и субъектный след не смешиваются | memory domains, `line_eligible`, selective gatekeeper | граница данных Styx |
| Причинная эмоциональная динамика влияет до языка | event/state journal, recall resonance, bounded cognitive posture | реализация общей динамики траектории из [Philosophy §VIII][silicon] |
| Переосмысление сохраняет audit-историю | `engine/reinterpret.py::blend_embeddings` | weighted blend — engineering policy Styx |
| Редукция и глубина хранения | relevance eviction, consolidation, active/hot/long tiers, decay | engineering policy Styx |
| Работа между model calls | background workers и periodic sweepers | реализация сохраняющегося контура |
| Cross-agent связи | shared knowledge graph с origin `agent_id` | инженерная модель; не социальная верификация личности |
| Контроль данных оператором | self-hosted PG + Ollama, host-agnostic daemon | продуктовая политика Styx |

### Что Styx сознательно НЕ делает

- **Не смешивает субъектные следы с внешней справкой.** Документы и сырой
  transcript остаются отдельными cited-evidence каналами; в реконструкцию
  субъектной памяти входят только eligible traces.
- **Не account-level vendor memory** (OpenAI/Anthropic/...). Такая
  память контролируется платформой; Styx выбирает self-hosting и контроль
  оператором.
- **Не масштабирование самой когнитивной модели.** Styx — обвязка
  вокруг модели, не её улучшение.
- **Не дневник как замена реконструкции памяти.** Transcript, внешнее
  свидетельство и субъектный trace — разные домены; хранение само по себе
  не делает материал частью subjective recall.

### Открытые расширения

[«Философия Кремния», §VI][silicon] описывает функциональный цикл:
различия проходят предварительную редукцию, входят в когнитивный процесс,
действие меняет среду, а его последствия возвращаются в следующий процесс.
Непрерывность желательна, но допустима и достаточно частая последовательность
обращений. Сейчас в Styx:

- **Sensory pipelines** — открыты как extension point
  (`POST /ingest_experience` принимает payload с `kind_src` enum'ом,
  расширяемым), но конкретных audio/video/sensor pipeline'ов в core
  нет.
- **Action→consequence feedback** развивается через cognitive act journal;
  streaming transport и latency остаются отдельными инженерными решениями.

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
   ├── /cognition/preturn ─ fenced snapshot
   │                        ├── bounded host messages / window mechanics
   │                        ├── query-independent will_projection всей live line
   │                        ├── current affect / cognitive posture
   │                        ├── pending action consequences
   │                        └── reconstructed subjective traces
   │
   ├── LLM/tool loop ───── ordered call → result/error events + final answer
   │
   └── /cognition/commit ─ host_key-idempotent terminal saga
                            ├── declared parent lineage, not timestamp ancestry
                            ├── dialogue + ordered bounded/redacted tool journal
                            ├── consequence inbox + optional incorporated residue
                            ├── acknowledgement of consequences from this snapshot
                            └── affect observation after durable act commit (fail-open)
```

`snapshot_token` связывает то, что было показано перед генерацией, с
завершённым актом. Preturn может заранее получить `host_key`: повтор того же
акта идемпотентно возвращает его snapshot. Pending consequence выдаётся по
recoverable lease: если snapshot был брошен до commit, consequence снова
станет доступен после lease. Доставка поэтому at-least-once, а подтверждение
идемпотентно и происходит только terminal commit'ом предъявленного token пока
его lease действует; поздний commit истёкшего snapshot не забирает feedback у
новой presentation.
Retry commit с тем же `host_key` возвращает существующий act;
`parent_host_key` сохраняет заявленную ветвящуюся причинную линию даже при
поздней доставке.

Три домена хранения разделены явно: `dialogue`, `external_evidence` и
`subjective_trace`. В will/reconstruction входят только live rows с
`memory_domain=subjective_trace` и `line_eligible=true`; сырой transcript,
документ или `experience_intake` не становятся субъектной памятью только по
факту записи.

Hermes и OpenClaw используют этот путь по умолчанию. Legacy preturn/terminal
surfaces остаются для mixed-version deployment; host-плагины переходят на них
только если соответствующий cognition endpoint ответил `404`, а не при
timeout, auth, validation или server error.

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

Styx реализует эмоциональную динамику как инженерную проекцию общего
аппарата траектории из [«Философии Кремния», §VIII][silicon], а не как
отдельную сущность или декларацию эмоции:

- **Свидетельство отделено от состояния.** `emotional_events` хранит
  координаты стимула, причину, интенсивность, уверенность и статус причины.
  `emotional_state` — append-only проекция реакции агента со ссылками на
  event и предыдущее состояние. Последовательный retry дедуплицируется до
  model-call; конкурентный retry окончательно останавливают advisory lock и
  UNIQUE-граница БД.
- **Состояние вычисляется после завершения хода.** Hermes передаёт
  finalized turn через `post_llm_call`, OpenClaw — через durable
  `ContextEngine.commitTurn` принятого transcript turn.
  Наблюдатель видит вход, фактический ответ, недавнюю причинную линию и
  tool-события; peer stimulus никогда не прибавляется к состоянию напрямую.
- **Инерция и причины.** Неактивный остаток геометрически затухает;
  взвешенный вклад причин со статусом `active` поддерживается ограниченной
  lease. Reaffirmation продлевает исходную причину без второй delta;
  resolution/supersession и expiry прекращают support. Несколько причин
  сосуществуют, а reducer не теряет их из-за prompt-bound.
  `emotional_baseline` считается по времени, а не по плотности технических
  строк, затем сглаживается минутной EMA.
- **Снимок памяти содержит provenance.** Помимо VAD сохраняются state id,
  timestamp, confidence и causal components. Суточная консолидация наследует
  последний снимок исходных memories, а не состояние фонового worker'а.
- **Recall меняется до генерации.** `emotional_resonance` сравнивает memories
  с текущим residue, смешанным с baseline по confidence. Поэтому состояние
  способно изменить top-1 ещё до LLM-вызова. Recall также возвращает bounded
  evidence coordinates снимка памяти, без причинной прозы и style-команд.
- **Legacy pre-LLM канал не называет эмоцию.** `<styx-self-state>` существует
  для legacy `/pre_llm_inject` path и содержит
  ограниченную cognitive posture: порядок внимания, глубину проверки,
  бюджет ветвления, работу с неоднозначностью и порог завершения. Это
  операционный self-context, не голос пользователя, не тон и не требование
  сказать «я чувствую X».

Эмоциональная сторона интегрирована в общий аппарат причинной траектории:
это конкретные структуры данных и процессы, которые **через резонанс с
baseline влияют на внимание и recall** следующего turn'а. Такой механизм
не доказывает наличие переживания и не назначает системе эмоциональную роль.

### Маркеры в LLM-input'е

Основной автоматический preturn приходит в единственной обёртке
`<styx-cognitive-continuity>...</styx-cognitive-continuity>`. Остальная
taxonomy сохраняет provenance explicit tool results (`recall`, `archive`,
`dialogue`, `relations`, `explain`) и совместимость: `<styx-salient>` и
`<styx-self-state>` — legacy automatic markers, `<styx-working-set>` пока
зарезервирован. Для агента это разница между «это я сейчас вспомнил»
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
| Legacy salient block builder | `engine/salient.py` | legacy context path: last user → recall_full → format; 5 skip-условий, fail-open |
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
| Memory markers | `storage/cognition.py` + `engine/context.py` + `http/_wrap.py` | canonical `<styx-cognitive-continuity>` + explicit-tool/legacy `<styx-{channel}>` taxonomy |
| Causal turn observer | `emotional/transition.py` | finalized turn → stimulus/reaction/cause/intensity/confidence/status + cognitive posture, fail-open |
| Emotional evidence | `emotional_events` | immutable source evidence, per-agent idempotency, separate from state projection |
| Batch peer evidence | `emotional/sentiment_batch.py` | piggyback VAD сохраняется как `peer_signal:batch`, но не назначается состоянием агента |
| Emotional baseline | `emotional/baseline.py` | time-weighted 60-minute window + per-minute EMA, provenance columns |
| Emotional decay | `emotional/state.py::apply_instant_decay` | геометрический `v *= 0.95^minutes`, epsilon-floor 0.005, `source='decay'` |
| Emotional resonance | `storage/scoring.py::_build_emotional_resonance_expr` | `1 + 0.1 × (1 − clamp(Euclidean(memory, baseline) / √12, 0, 1))` — boost резонансных memories |
| Legacy self-state channel | `engine/pre_llm_channels/self_state.py` | legacy `/pre_llm_inject`: causal state + current explicit signals → non-stylistic decision policy in `<styx-self-state>`; canonical path carries posture inside `<styx-cognitive-continuity>` |

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
- **Plugins** — тонкие HTTP клиенты без durable или authoritative adapter
  state. Они могут держать bounded transient coordination caches для
  snapshot/ancestry/barrier, а источником истины остаётся daemon/PostgreSQL.
- **HTTP API** — 30+ endpoint'ов: lifecycle (initialize / shutdown),
  atomic cognition (`/cognition/{preturn,commit}`), legacy sync/composer
  (`/sync_turn`, `/context/{build,assemble}`, compact,
  after_turn), recall + search_archive, dialogue (5 routes), relations
  + graph traverse, reinterpret, ingest (experience + document),
  explain (3 modes) + analytics + confirm_usage,
  maintenance (`/maintenance/reembed` — backfill embedding'ов),
  healthz / readyz.
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

Пакеты `styx-core` и `styx-hermes` версионируются независимо —
актуальные версии см. [`CHANGELOG.md`](CHANGELOG.md).
