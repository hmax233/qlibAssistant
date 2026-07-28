#!/usr/bin/env bash
set -euo pipefail

ROOT=/Users/hmax/qlibAssistant
PY=/Users/hmax/miniconda3/envs/qlibAssistant/bin/python

cd "$ROOT"
exec "$PY" script/run_daily_decision_pipeline.py --update "$@"
