#!/bin/bash
# A separate ComfyUI instance for YuE2 work, so batches of takes never sit in (or block) a shared queue.
# It runs the ComfyUI install in COMFY_DIR on its own port with the yue2_continue node allowed, and stops only the process it
# started, and only when its queue is empty. Nothing else is touched.
#   COMFY_DIR   ComfyUI folder (scripts/yue2_continue copied into its custom_nodes/)
#   COMFY_PY    the python that runs it (default: python3)
#   PORT        default 8190
#   COMFY_ARGS  extra main.py arguments, passed through as they are. A YuE2 install kept apart from a shared ComfyUI usually
#               needs its own folders, e.g.
#               COMFY_ARGS="--disable-all-custom-nodes --offline --models-directory /x/models --input-directory /x/in
#                           --output-directory /x/out --temp-directory /x/temp --user-directory /x/user"
#   PYTHONPATH, CPATH and the rest of the environment are inherited: export them first if the install uses a package
#   overlay or its own C headers.
# usage: yue2_service.sh start|status|stop
set -u
PORT=${PORT:-8190}
PY=${COMFY_PY:-python3}
EXTRA=${COMFY_ARGS:-}
PIDFILE="${COMFY_DIR:?set COMFY_DIR}/yue2_${PORT}.pid"
LOG="$COMFY_DIR/yue2_${PORT}.log"
queue() { curl -s -m 5 "http://127.0.0.1:$PORT/queue"; }
case "${1:-status}" in
  start)
    if queue > /dev/null; then echo "already running on $PORT"; exit 0; fi
    cd "$COMFY_DIR" || exit 1
    # shellcheck disable=SC2086
    setsid nohup "$PY" main.py --listen 127.0.0.1 --port "$PORT" $EXTRA --whitelist-custom-nodes yue2_continue >> "$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDFILE"
    for i in $(seq 1 60); do queue > /dev/null && break; sleep 3; done
    queue | head -c 120; echo
    curl -s -m 5 "http://127.0.0.1:$PORT/object_info/YuE2ContinueMusic" | head -c 60; echo ;;
  status)
    queue | head -c 200; echo ;;
  stop)
    q=$(queue)
    if echo "$q" | "$PY" -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if not d['queue_running'] and not d['queue_pending'] else 1)"; then
      [ -f "$PIDFILE" ] && kill -TERM "$(cat "$PIDFILE")" && rm -f "$PIDFILE" && echo stopped
    else
      echo "queue not empty, not stopping: $q" | head -c 200; echo
    fi ;;
esac
