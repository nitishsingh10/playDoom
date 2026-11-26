#!/usr/bin/env bash
# Run inside tmux on the CUDA VM. Starts six fresh, bounded training runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test -x .venv/bin/python
test -s submission/weights/checkpoint.pt
test -s submission/weights/config.json
mkdir -p training/runs
exec 9>training/runs/.doom-training.lock
if ! flock -n 9; then
    echo "Another training launcher is running. Attach its tmux session."
    exit 1
fi
if pgrep -f 'python[^ ]* .*training\.(train|combat_experiment|directional_experiment)( |$)' >/dev/null; then
    echo "A training process is already running; comparison was not started."
    exit 1
fi
.venv/bin/python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable in .venv/bin/python; comparison was not started.")
print("GPU:", torch.cuda.get_device_name(0))
PY
task_log=$(mktemp training/runs/directional_comparison_console_XXXXXX.log)
echo "Console log: $task_log"
echo "Fresh v1/v2 x seeds 17,23,41: 64,000 decisions each; 384,000 total."
echo "Then 480 matched evaluation episodes. Existing rewards; no checkpoint promotion."
.venv/bin/python -u -m training.directional_experiment \
    --seeds 17 23 41 --updates 500 --eval-episodes 10 --device cuda \
    2>&1 | tee "$task_log"
