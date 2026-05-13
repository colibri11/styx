#!/bin/sh
# OpenClaw gateway init wrapper для интеграционного стика Styx.
#
# Идемпотентный bootstrap → запуск gateway. На каждом старте безопасно
# крутит `config set` (merge) для восстановления критичных ключей:
#
# - gateway.{mode,bind,controlUi}: gateway не стартует без mode=local +
#   allowed origins. Restore при любых правках через Control UI.
# - agents.defaults.model.primary: ollama/qwen3:4b-local (LAN Ollama).
#   В Phase A-D было zai/glm-5.1 через z.ai, но (1) z.ai
#   unreachable из контейнера в текущем стике, (2) embedded
#   `agent --local` потом использует этот же config. Ollama проверена.
# - models.providers.ollama.baseUrl: переопределение default
#   http://127.0.0.1:11434 на http://ollama:11434 (docker extra_hosts).
# - plugins.slots.contextEngine = "styx": exclusive слот, отключает
#   Pi legacy ContextEngine.
# - plugins.entries.styx.{enabled,config.*}: daemonUrl, httpToken,
#   agentMapping="auto", ownsCompaction=true. Plugin регистрируется
#   автоматически через bind-mount `/home/node/.openclaw/plugins/styx`
#   (manifest activation.onStartup=true) — `plugins install --link`
#   не требуется, OpenClaw сканирует $CONFIG_DIR/plugins/* при старте.
#
# Headless-сетап описан в https://docs.openclaw.ai/install/docker
# секция "Manual flow". Phase E reference: docker/config/openclaw-styx-test.json5.

set -eu

CONFIG_DIR="${OPENCLAW_CONFIG_DIR:-/home/node/.openclaw}"
mkdir -p "$CONFIG_DIR"

STYX_DAEMON_URL_BOOT="${STYX_DAEMON_URL:-http://styx-daemon:8788}"
STYX_HTTP_TOKEN_BOOT="${STYX_HTTP_TOKEN:-test-token-do-not-use-in-prod}"

echo "[openclaw-init] applying bootstrap config…"
# Phase E choice: primary LLM = zai/glm-5.1.
#
# Изначально пробовали ollama/qwen3:4b-local (LAN Ollama гарантированно
# доступна), но все локальные модели в нашей Ollama (qwen3:4b-local,
# bge-m3, embeddinggemma) имеют capabilities=["completion"] или
# ["embedding"] — НЕТ tool-calling capability hint в metadata. OpenClaw
# embedded agent посылает 16 styx_* tool schemas, Ollama отклоняет
# запрос: 400 "registry.ollama.ai/library/qwen3:4b-local does not
# support tools".
#
# z.ai (GLM) — tools-capable из коробки + reachable из контейнера
# (404/401 от api.z.ai/v1/models = network OK, gen Phase A-D гипотеза
# unreachable z.ai была неверной — мы имели configuration issue, не
# network). ZAI_API_KEY мостится из GLM_API_KEY в compose (см. comment
# в openclaw-gateway service).
#
# Чтобы переключиться обратно на Ollama для caching/offline сценариев:
# нужна tools-capable модель в Ollama (qwen3:14b/llama3.3 etc).
node /app/dist/index.js config set --batch-json "[
  {\"path\":\"gateway.mode\",\"value\":\"local\"},
  {\"path\":\"gateway.bind\",\"value\":\"lan\"},
  {\"path\":\"gateway.controlUi.allowedOrigins\",\"value\":[\"http://localhost:18789\",\"http://127.0.0.1:18789\"]},
  {\"path\":\"agents.defaults.model.primary\",\"value\":\"zai/glm-5.1\"},
  {\"path\":\"plugins.slots.contextEngine\",\"value\":\"styx\"},
  {\"path\":\"plugins.entries.styx.enabled\",\"value\":true},
  {\"path\":\"plugins.entries.styx.config.daemonUrl\",\"value\":\"${STYX_DAEMON_URL_BOOT}\"},
  {\"path\":\"plugins.entries.styx.config.httpToken\",\"value\":\"${STYX_HTTP_TOKEN_BOOT}\"},
  {\"path\":\"plugins.entries.styx.config.agentMapping\",\"value\":{\"*\":\"auto\"}},
  {\"path\":\"plugins.entries.styx.config.requestTimeoutMs\",\"value\":30000},
  {\"path\":\"plugins.entries.styx.config.logging\",\"value\":true},
  {\"path\":\"plugins.entries.styx.config.ownsCompaction\",\"value\":true}
]"

echo "[openclaw-init] starting gateway on :18789"
exec node /app/dist/index.js gateway --bind lan --port 18789
