#!/bin/zsh
set -u

ROOT=/Users/hmax/qlibAssistant
PYTHON=/Users/hmax/miniconda3/envs/qlibAssistant/bin/python
OUTPUT="$ROOT/.qlibAssistant/analysis/alpha360_e0_e6_260828"
PLIST="$HOME/Library/LaunchAgents/com.hmax.qlib.alpha360-e0-e6-finalize.plist"
LOG="$ROOT/.qlibAssistant/analysis/alpha360_e0_e6_260828_launchd.log"

mkdir -p "$ROOT/.qlibAssistant/analysis"
if [[ -f "$OUTPUT/local_completion.json" ]]; then
  launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
  exit 0
fi

cd "$ROOT" || exit 1
"$PYTHON" script/finalize_alpha360_e0_e6_local.py >>"$LOG" 2>&1
status=$?

if [[ $status -eq 0 && -f "$OUTPUT/local_completion.json" ]]; then
  launchctl bootout "gui/$(id -u)" "$PLIST" >/dev/null 2>&1 || true
  exit 0
fi

# Exit code 3 means the remote lockbox is still running.  This is expected,
# so do not leave a failed launchd job or a resident polling process.
if [[ $status -eq 3 ]]; then
  exit 0
fi
exit $status
