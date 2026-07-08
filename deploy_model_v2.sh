#!/usr/bin/env bash
# Waits for a train.py retrain to finish, deploys the new checkpoint as the
# live model.pt, and restarts the API server -- fully offline/unattended.
#
# Designed to survive Claude Code / the network / this SSH session going
# away: it only depends on the local filesystem (watches for the output
# checkpoint file to appear and stop growing) and local processes (pgrep/
# pkill), never the network, and should itself be launched fully detached:
#
#   cd /mnt/data/nblong-t04/VMMRdb.explain-more-plainly-moonlit-sundae
#   nohup setsid ./deploy_model_v2.sh </dev/null >deploy_model_v2.out 2>&1 &
#   disown
#
# Usage: ./deploy_model_v2.sh [new_model.pt] [new_classes.json]
#   Defaults to model_v2.pt / model_v2.json (this run's --out).
set -uo pipefail

WORKTREE="/mnt/data/nblong-t04/VMMRdb.explain-more-plainly-moonlit-sundae"
NEW_MODEL="$WORKTREE/${1:-model_v2.pt}"
NEW_CLASSES="$WORKTREE/${2:-model_v2.json}"
LIVE_MODEL="$WORKTREE/model.pt"
LIVE_CLASSES="$WORKTREE/classes.json"
LOG="$WORKTREE/deploy_model_v2.log"
API_LOG="$WORKTREE/api_server.log"
POLL_SEC=60
STABLE_CHECK_SEC=5     # file size must be unchanged across this gap before treating a write as "done"
MAX_WAIT_SEC=$((5 * 24 * 3600))  # 5-day safety valve so a runaway process can't wedge this forever

log() { printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" | tee -a "$LOG"; }

log "watcher started, waiting on $NEW_MODEL"

# -o = oldest matching pid -- train.py's DataLoader workers (persistent_workers
# is off) get torn down and respawned every epoch and would otherwise be
# indistinguishable by cmdline from the long-lived parent process.
TRAIN_PID=$(pgrep -o -f "train.py.*--out $(basename "$NEW_MODEL")" || true)
if [ -z "$TRAIN_PID" ]; then
  log "WARNING: no running train.py process found for $(basename "$NEW_MODEL") at watcher startup -- will still wait for the output file, but can't detect a mid-run crash"
else
  log "tracking training pid $TRAIN_PID"
fi

elapsed=0
while [ "$elapsed" -lt "$MAX_WAIT_SEC" ]; do
  if [ -f "$NEW_MODEL" ]; then
    size1=$(stat -c%s "$NEW_MODEL" 2>/dev/null || echo 0)
    sleep "$STABLE_CHECK_SEC"
    size2=$(stat -c%s "$NEW_MODEL" 2>/dev/null || echo 0)
    if [ "$size1" = "$size2" ] && [ "$size1" -gt 0 ]; then
      log "$(basename "$NEW_MODEL") present and stable ($size2 bytes) -- training finished"
      break
    fi
    log "$(basename "$NEW_MODEL") exists but still being written ($size1 -> $size2 bytes), waiting"
  fi
  if [ -n "$TRAIN_PID" ] && ! kill -0 "$TRAIN_PID" 2>/dev/null && [ ! -f "$NEW_MODEL" ]; then
    log "ERROR: training pid $TRAIN_PID is no longer running and $(basename "$NEW_MODEL") was never created -- training crashed or was killed. NOT deploying. Check the training output for details."
    exit 1
  fi
  sleep "$POLL_SEC"
  elapsed=$((elapsed + POLL_SEC))
done

if [ ! -f "$NEW_MODEL" ]; then
  log "ERROR: gave up after $((MAX_WAIT_SEC / 3600))h -- $(basename "$NEW_MODEL") never appeared. NOT deploying."
  exit 1
fi

log "=== deploying $(basename "$NEW_MODEL") as the live model ==="
ts=$(date -u +%Y%m%dT%H%M%SZ)
if [ -f "$LIVE_MODEL" ]; then
  cp -p "$LIVE_MODEL" "$WORKTREE/model.pt.bak-$ts"
  log "backed up old model.pt -> model.pt.bak-$ts"
fi
if [ -f "$LIVE_CLASSES" ]; then
  cp -p "$LIVE_CLASSES" "$WORKTREE/classes.json.bak-$ts"
fi

cp -p "$NEW_MODEL" "$LIVE_MODEL"
if [ -f "$NEW_CLASSES" ]; then
  cp -p "$NEW_CLASSES" "$LIVE_CLASSES"
fi
log "copied $(basename "$NEW_MODEL") -> model.pt (original left in place too)"

log "=== stopping any running API server ==="
OLD_API_PID=$(pgrep -f "uvicorn api:app" || true)
if [ -n "$OLD_API_PID" ]; then
  log "found running API server pid(s): $OLD_API_PID -- sending SIGTERM"
  kill $OLD_API_PID 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    sleep 2
    pgrep -f "uvicorn api:app" >/dev/null || break
  done
  if pgrep -f "uvicorn api:app" >/dev/null; then
    log "still alive after grace period, sending SIGKILL"
    pkill -9 -f "uvicorn api:app" 2>/dev/null || true
  fi
  log "old API server stopped"
else
  log "no running API server found"
fi

log "=== starting API server with the new model ==="
cd "$WORKTREE" || { log "ERROR: cannot cd to $WORKTREE"; exit 1; }
nohup uv run uvicorn api:app --host 0.0.0.0 --port 8100 </dev/null >>"$API_LOG" 2>&1 &
NEW_API_PID=$!
disown
log "started new API server, pid $NEW_API_PID, logging to $API_LOG"

sleep 15
if curl -sf "http://0.0.0.0:8100/health" >"$WORKTREE/deploy_health_check.json" 2>/dev/null; then
  log "health check OK: $(cat "$WORKTREE/deploy_health_check.json")"
else
  log "WARNING: health check failed or curl unavailable -- check $API_LOG manually"
fi

log "=== deploy complete ==="
