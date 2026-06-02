#!/command/with-contenv sh
# shellcheck shell=sh
# Styx-Hermes bootstrap как s6-overlay cont-init.d hook.
#
# Официальный образ nousresearch/hermes-agent — s6-overlay: gateway НЕ
# самостоятельный бинарь на PATH, он поднимается s6-супервизором, а
# docker CMD `["gateway","run"]` маршрутизируется main-wrapper.sh в
# `s6-setuidgid hermes hermes gateway run`. Поэтому НЕЛЬЗЯ перехватывать
# запуск через `exec gateway` (прежний entrypoint-wrapper давал
# `exec: gateway: not found` → exit 127, crash-loop). Наша работа —
# одноразовый setup ДО старта сервисов; gateway запускает сам s6.
#
# Dockerfile.styx-hermes кладёт этот файл в /etc/cont-init.d/30-styx-bootstrap.
# s6 выполняет /etc/cont-init.d/* в лексикографическом порядке как root,
# ДО старта user-сервисов. К моменту `30-` уже отработали:
#   01-hermes-setup       — stage2-hook: chown тома + seed HERMES_HOME
#                           (config.yaml уже создан)
#   015-supervise-perms   — права на supervise-деревья
#   02-reconcile-profiles — per-profile gateway-сервисы
# `30-` гарантированно после них, поэтому config.yaml уже на месте.
#
# Shebang `#!/command/with-contenv sh`: /init скрабит env перед cont-init.d;
# with-contenv восстанавливает контейнерное окружение (HERMES_HOME и т.п.)
# из /run/s6/container_environment — как делают штатные хуки образа.
#
# Наша работа (идемпотентна, выполняется при каждом старте контейнера):
#   1. styx-hermes-setup — копирует styx-memory shim в HERMES_HOME/plugins/.
#      General plugin `styx` подхватывается через entry-point — shim не нужен.
#   2. патч config.yaml: memory.provider: styx-memory + plugins.enabled+=styx
#      + context.engine: styx.
# Миграции БД делает styx-daemon контейнер на старте — здесь их НЕ трогаем
# (иначе race с daemon'ом).
#
# Работаем под `s6-setuidgid hermes` (как 02-reconcile-profiles), чтобы
# созданные файлы shim'а принадлежали hermes-пользователю (UID 10000),
# под которым потом крутится gateway.
set -e

HERMES_HOME="${HERMES_HOME:-/opt/data}"
PYTHON="/opt/hermes/.venv/bin/python"
SETUP="/opt/hermes/.venv/bin/styx-hermes-setup"

# styx-hermes-setup отказывается писать в HERMES_HOME без явного opt-in;
# выставляем inline, чтобы не зависеть от compose env.
export STYX_ALLOW_HERMES_HOME=1

echo "[styx-bootstrap] cont-init.d hook; HERMES_HOME=$HERMES_HOME"

# 1. Установить styx-memory shim (--force чтобы перезаписать при rebuild).
#    Перед этим (мы под root) снимаем возможный stale cross-uid мусор в нашем
#    shim-каталоге: персистентный HERMES_HOME-volume мог накопить root-owned
#    __pycache__ от прежних прогонов, а `styx-hermes-setup --force` под hermes
#    делает rmtree(dst) и упал бы с EACCES → cont-init exit 1. s6-overlay по
#    дефолту (S6_BEHAVIOUR_IF_STAGE2_FAILS=0) НЕ халтит контейнер на падении
#    cont-init — значит без этого получили бы тихо-сломанный bootstrap (shim
#    не доустановлен, memory-провайдер молча неактивен). chown под root делает
#    rmtree под hermes всегда возможным.
SHIM_DIR="$HERMES_HOME/plugins/styx-memory"
if [ -d "$SHIM_DIR" ]; then
    chown -R hermes:hermes "$SHIM_DIR" 2>/dev/null || true
fi

echo "[styx-bootstrap] styx-hermes-setup..."
s6-setuidgid hermes "$SETUP" --hermes-home "$HERMES_HOME" --force

# 2. Патч config.yaml:
#    - memory.provider: styx-memory (активирует memory-shim через memory discovery)
#    - plugins.enabled включает styx (общий PluginManager opt-in:
#      без включения plugin не загрузится, ContextEngine не зарегистрируется)
#    - context.engine: styx (без этого Hermes берёт built-in compressor;
#      styx-движок зарегистрирован, но НЕ выбран)
CONFIG="$HERMES_HOME/config.yaml"
if [ -f "$CONFIG" ]; then
    echo "[styx-bootstrap] patching $CONFIG..."
    s6-setuidgid hermes "$PYTHON" - "$CONFIG" <<'PYEOF'
import sys
import yaml
from pathlib import Path

p = Path(sys.argv[1])
data = yaml.safe_load(p.read_text()) or {}
data.setdefault("memory", {})["provider"] = "styx-memory"

plugins = data.setdefault("plugins", {})
enabled = plugins.setdefault("enabled", [])
if not isinstance(enabled, list):
    enabled = []
if "styx" not in enabled:
    enabled.append("styx")
plugins["enabled"] = enabled

data.setdefault("context", {})["engine"] = "styx"

p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
print(f"[styx-bootstrap] memory.provider=styx-memory; plugins.enabled+=styx; context.engine=styx in {p}")
PYEOF
else
    echo "[styx-bootstrap] WARNING: $CONFIG отсутствует — 01-hermes-setup не отработал?" >&2
fi

echo "[styx-bootstrap] complete; gateway запускает s6 (cont-init.d hook, без exec)"
