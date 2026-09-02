#!/bin/sh
# OpenClaw gateway init wrapper для интеграционного стика Styx.
#
# Идемпотентный bootstrap → запуск gateway. На каждом старте безопасно
# крутит `config set` (merge) для восстановления критичных ключей:
#
# - gateway.{mode,bind,controlUi}: gateway не стартует без mode=local +
#   allowed origins. Restore при любых правках через Control UI.
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

CONFIG_DIR="${OPENCLAW_STATE_DIR:-${OPENCLAW_CONFIG_DIR:-/home/node/.openclaw}}"
mkdir -p "$CONFIG_DIR"

STYX_DAEMON_URL_BOOT="${STYX_DAEMON_URL:-http://styx-daemon:8788}"
STYX_HTTP_TOKEN_BOOT="${STYX_HTTP_TOKEN:-test-token-do-not-use-in-prod}"

echo "[openclaw-init] applying bootstrap config…"
# Базовый compatibility lane намеренно не выбирает provider/model. В
# OpenClaw 2026.8.2 ZAI поставляется отдельным provider package; принудительный
# zai/glm здесь делал чистый официальный image неработоспособным.
# Штатный config writer сохраняет JSON5/$include semantics и выполняет schema
# validation. Эти unsets мигрируют только test-stack state старой линии.
unset_if_present() {
  if node /app/dist/index.js config get "$1" >/dev/null 2>&1; then
    node /app/dist/index.js config unset "$1"
  fi
}
unset_if_present plugins.entries.zai
unset_if_present agents.defaults.model.primary
unset_if_present plugins.entries.styx.hooks
node /app/dist/index.js config set --batch-json "[
  {\"path\":\"gateway.mode\",\"value\":\"local\"},
  {\"path\":\"gateway.bind\",\"value\":\"lan\"},
  {\"path\":\"gateway.controlUi.allowedOrigins\",\"value\":[\"http://localhost:18789\",\"http://127.0.0.1:18789\"]},
  {\"path\":\"plugins.slots.contextEngine\",\"value\":\"styx\"},
  {\"path\":\"plugins.entries.styx.enabled\",\"value\":true},
  {\"path\":\"plugins.entries.styx.config.daemonUrl\",\"value\":\"${STYX_DAEMON_URL_BOOT}\"},
  {\"path\":\"plugins.entries.styx.config.httpToken\",\"value\":\"${STYX_HTTP_TOKEN_BOOT}\"},
  {\"path\":\"plugins.entries.styx.config.agentMapping\",\"value\":{\"*\":\"auto\"}},
  {\"path\":\"plugins.entries.styx.config.requestTimeoutMs\",\"value\":30000},
  {\"path\":\"plugins.entries.styx.config.logging\",\"value\":true},
  {\"path\":\"plugins.entries.styx.config.ownsCompaction\",\"value\":true}
]"

# Context-engine capability requires explicit consent in current OpenClaw.
node /app/dist/index.js plugins enable styx --accept-capabilities

# v2026.8.2 migrates legacy workspace/session setup markers through doctor.
# Record completion in the persistent test state so ordinary restarts stay fast.
MIGRATION_MARKER="$CONFIG_DIR/.styx-openclaw-2026.8.2-migrated"
if [ ! -f "$MIGRATION_MARKER" ]; then
  node /app/dist/index.js doctor --fix --non-interactive --yes
  touch "$MIGRATION_MARKER"
fi

echo "[openclaw-init] starting gateway on :18789"
exec node /app/dist/index.js gateway --bind lan --port 18789
