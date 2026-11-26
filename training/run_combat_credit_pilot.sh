#!/usr/bin/env bash
# Run on the CUDA VM, inside tmux. No weights are promoted.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

task_python=.venv/bin/python
task_parent=training/runs/structured_l4_seed17_iXoMX8/checkpoints/checkpoint_500.pt
test -x "$task_python"
test -s "$task_parent"
test -s submission/weights/checkpoint.pt
test -s submission/weights/config.json
mkdir -p training/runs

# Prevent concurrent pilots and refuse to overlap an existing training process.
exec 9>training/runs/.doom-training.lock
if ! flock -n 9; then
    echo "Another combat pilot is running. Attach its tmux session."
    exit 1
fi
if pgrep -f 'python[^ ]* .*training\.(train|combat_experiment|directional_experiment)( |$)' >/dev/null; then
    echo "A training process is already running. Wait for it to finish."
    exit 1
fi

"$task_python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in .venv/bin/python; training was not started.")
print("GPU:", torch.cuda.get_device_name(0))
PY

task_log=$(mktemp training/runs/combat_credit_seed17_XXXXXX.log)
echo "Console log: $task_log"
echo "Two arms, 64,000 additional decisions each; then 240 evaluation episodes."
"$task_python" -u -m training.combat_experiment \
    --parent "$task_parent" \
    --seed 17 --device cuda --additional-updates 500 --eval-episodes 10 \
    2>&1 | tee "$task_log"
