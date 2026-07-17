#!/usr/bin/env bash
# HRSSM smoke test -- run this on the cluster BEFORE launching the full 9-run batch.
# Validates: (1) osmesa headless rendering works, (2) the --steps override
# actually yields ~1 gradient update per 2 env-steps (so a full run lands at
# the intended gradient-update count, matching the paper's reported x-axis),
# (3) per-run GPU memory footprint (used to suggest RUNS_PER_GPU), and
# (4) an ETA for a single run and for the full batch, from the logged `fps`
# metric.
#
# STEPS / NUM_GPUS / TOTAL_RUNS below must match launch_experiments.sh's
# config (STEPS, NUM_GPUS, and TASKS*SEEDS respectively) for the ETA to be
# meaningful.
#
# Usage:
#   HRSSM_DIR=/path/to/HRSSM DURATION=600 bash smoke_test.sh
set -euo pipefail

HRSSM_DIR="${HRSSM_DIR:-$HOME/HRSSM}"
DURATION="${DURATION:-600}"          # bumped from 180 -> 600s: torch.compile warmup
                                      # skews the first few fps readings, and a longer
                                      # window gives a steadier average for the ETA
GPU_CHECK_DELAY="${GPU_CHECK_DELAY:-60}"  # seconds to wait before sampling nvidia-smi
LOGDIR="${LOGDIR:-/tmp/hrssm_smoketest}"
STEPS="${STEPS:-2e6}"                # must match launch_experiments.sh's STEPS
NUM_GPUS="${NUM_GPUS:-2}"            # must match launch_experiments.sh's NUM_GPUS
TOTAL_RUNS="${TOTAL_RUNS:-9}"        # 3 tasks x 3 seeds, must match launch_experiments.sh

cd "$HRSSM_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hrssm
export MUJOCO_GL=osmesa

rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

echo "==> Launching smoke-test run for up to ${DURATION}s ..."
python -u dreamer.py \
  --configs dmc_vision \
  --task dmc_walker_stand \
  --seed 0 \
  --steps "$STEPS" \
  --log_every 2000 \
  --simple_log \
  --logdir "$LOGDIR" \
  --device cuda:0 \
  > "$LOGDIR/run.out" 2>&1 &
PID=$!

sleep "$GPU_CHECK_DELAY"

if ! kill -0 "$PID" 2>/dev/null; then
  echo "!! Process exited early -- check $LOGDIR/run.out for an error"
  echo "   (common cause: osmesa rendering failure -- try MUJOCO_GL=egl instead)"
  tail -n 40 "$LOGDIR/run.out"
  exit 1
fi

# Defaults if nvidia-smi is unavailable; overwritten below if it is.
USED_MIB=0
TOTAL_MIB=16384

if command -v nvidia-smi &> /dev/null; then
  echo
  echo "==> GPU memory while one run is active (GPU 0, the one this run is on):"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
  read -r USED_MIB TOTAL_MIB <<< "$(nvidia-smi --query-gpu=memory.used,memory.total \
      --format=csv,noheader,nounits -i 0 | tr -d ',')"
  echo
fi

REMAINING=$(( DURATION - GPU_CHECK_DELAY ))
if [ "$REMAINING" -gt 0 ]; then
  sleep "$REMAINING"
fi

kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true

echo
echo "==> Checking $LOGDIR/metrics.jsonl for update_count/step ratio, fps, and ETA ..."
python3 - "$LOGDIR/metrics.jsonl" "$STEPS" "$NUM_GPUS" "$TOTAL_RUNS" "$USED_MIB" "$TOTAL_MIB" <<'PYEOF'
import json
import math
import sys

path, steps_raw, num_gpus, total_runs, used_mib, total_mib = sys.argv[1:7]
steps_raw = float(steps_raw)
num_gpus = int(num_gpus)
total_runs = int(total_runs)
used_mib = float(used_mib)
total_mib = float(total_mib)

rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

train_rows = [r for r in rows if "update_count" in r and r.get("step", 0) > 0]
if not train_rows:
    print("No training-metric lines with update_count found yet -- let it run longer,")
    print("or check run.out for errors (rendering, missing deps, etc.).")
    sys.exit(0)

print(f"{'step':>10}  {'update_count':>12}  {'ratio':>6}  (expect ratio ~0.5)")
for r in train_rows[-5:]:
    ratio = r["update_count"] / r["step"]
    print(f"{r['step']:>10}  {r['update_count']:>12}  {ratio:>6.3f}")

print()
print("If the ratio is consistently ~0.5, this --steps value will land at the")
print("intended gradient-update count by the end of a full run.")

# ---- fps / ETA ----
fps_rows = [r["fps"] for r in rows if r.get("fps", 0) > 0]
print()
if len(fps_rows) < 2:
    print("Not enough 'fps' samples yet to estimate ETA -- rerun with a longer DURATION.")
    sys.exit(0)

# Drop the first nonzero sample too: it typically still includes torch.compile
# warmup and skews low.
stable_fps = fps_rows[1:] if len(fps_rows) > 2 else fps_rows
avg_fps = sum(stable_fps) / len(stable_fps)
print(f"fps samples used for estimate: {[round(f, 2) for f in stable_fps]}")
print(f"average fps (post-warmup): {avg_fps:.2f}")

per_run_seconds = steps_raw / avg_fps
per_run_hours = per_run_seconds / 3600
print()
print(f"--> Estimated single-run time: {per_run_hours:.1f} hours "
      f"({per_run_hours / 24:.2f} days) to reach --steps {steps_raw:.0f}")

# ---- suggest RUNS_PER_GPU from the observed VRAM footprint ----
safety_margin_mib = 1024
usable_mib = max(total_mib - safety_margin_mib, 0)
if used_mib > 0:
    suggested_runs_per_gpu = max(1, int(usable_mib // used_mib))
else:
    suggested_runs_per_gpu = 1

print(f"\nObserved single-run GPU memory: {used_mib:.0f} MiB / {total_mib:.0f} MiB total")
print(f"Suggested RUNS_PER_GPU (leaving ~{safety_margin_mib} MiB headroom): "
      f"{suggested_runs_per_gpu}")

max_concurrent = num_gpus * suggested_runs_per_gpu
waves = math.ceil(total_runs / max_concurrent)
batch_hours = waves * per_run_hours
print(f"\nWith NUM_GPUS={num_gpus} and RUNS_PER_GPU={suggested_runs_per_gpu} "
      f"({max_concurrent} concurrent slots), {total_runs} runs = {waves} wave(s).")
print(f"--> Rough full-batch estimate: {batch_hours:.1f} hours "
      f"({batch_hours / 24:.2f} days)")
print("    (this ignores GPU/CPU contention slowdown from packing >1 run/GPU --")
print("     treat it as a lower bound, not a guarantee.)")
PYEOF

echo
echo "==> Smoke test done. Inspect $LOGDIR/run.out for any warnings, and pass"
echo "    RUNS_PER_GPU from above to launch_experiments.sh."
