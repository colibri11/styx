#!/bin/bash
# Styx-Hermes bootstrap внутри hermes-styx контейнера.
#
# Вызывается как $1 в CMD (см. ENTRYPOINT родительского образа —
# `if command -v "$1" >/dev/null 2>&1; then exec "$@"; fi`).
#
# Hermes-родитель к этому моменту уже:
# - применил UID/GID remap через usermod+gosu (мы под hermes user)
# - bootstrap'нул HERMES_HOME (.env, config.yaml, SOUL.md, skills)
#
# Наша работа:
# 1. styx-hermes-setup — копирует styx-memory shim в HERMES_HOME/plugins/.
#    General plugin `styx` подхватывается через entry-point — shim не нужен.
# 2. (миграции БД делает styx-daemon контейнер на старте; этот скрипт
#    их НЕ применяет — иначе race с daemon'ом).
# 3. патч config.yaml: memory.provider: styx-memory + plugins.enabled+=styx
# 4. exec оставшихся аргументов (hermes gateway run / sleep infinity / etc.)

set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
PYTHON="/opt/hermes/.venv/bin/python"
SETUP="/opt/hermes/.venv/bin/styx-hermes-setup"

echo "[styx-bootstrap] HERMES_HOME=$HERMES_HOME"

# 1. Установить styx-memory shim (--force чтобы перезаписать при rebuild)
echo "[styx-bootstrap] styx-hermes-setup..."
STYX_ALLOW_HERMES_HOME=1 "$SETUP" --hermes-home "$HERMES_HOME" --force

# 2. Патч config.yaml:
#    - memory.provider: styx-memory (активирует memory-shim через memory discovery)
#    - plugins.enabled включает styx (общий PluginManager opt-in:
#      без включения plugin не загрузится, ContextEngine не зарегистрируется)
CONFIG="$HERMES_HOME/config.yaml"
if [ -f "$CONFIG" ]; then
    echo "[styx-bootstrap] patching $CONFIG..."
    "$PYTHON" - <<PYEOF
import yaml
from pathlib import Path

p = Path("$CONFIG")
data = yaml.safe_load(p.read_text()) or {}
data.setdefault("memory", {})["provider"] = "styx-memory"

plugins = data.setdefault("plugins", {})
enabled = plugins.setdefault("enabled", [])
if not isinstance(enabled, list):
    enabled = []
if "styx" not in enabled:
    enabled.append("styx")
plugins["enabled"] = enabled

p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
print(f"[styx-bootstrap] memory.provider=styx-memory; plugins.enabled+=styx in {p}")
PYEOF
fi

echo "[styx-bootstrap] complete; exec ${*}"
exec "$@"
