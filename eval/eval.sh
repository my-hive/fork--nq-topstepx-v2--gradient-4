#!/bin/bash
# TopstepX 50K combine eval v2 on real NQ 1s data. Do not modify.
# Data: $NQ_DATA or ~/hive-data/nq/train.parquet. A locked holdout is scored separately by the task owner.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf __pycache__ eval/__pycache__          # stale bytecode once scored code that never ran
PYTHONDONTWRITEBYTECODE=1 python3 eval/sim.py "$@" 2>&1 | tee run.log
