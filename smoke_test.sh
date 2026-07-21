#!/usr/bin/env bash
# HRSSM smoke test -- run this on the cluster BEFORE launching the full 9-run batch.
# Validates: (1) osmesa headless rendering works, (2) the --steps override
# yields the intended gradient-update cadence (~0.25 update_count/step delta,
# see comment near the ratio check below for why it's 0.25 and not 0.5), (3)
# per-run GPU memory footprint (max observed over the run, used to suggest
# RUNS_PER_GPU), and (4) an ETA for a single run and for the full batch, from
# the logged `fps` metric.
#
# STEPS / NUM_GPUS / TOTAL_RUNS below must match launch_experiments.sh's
# config (STEPS, NUM_GPUS, and TASKS*SEEDS respectively) for the ETA to be
# meaningful.
#
# Usage:
#   HRSSM_DIR=/path/to/HRSSM DURATION=1800 bash smoke_test.sh
set -euo pipefail

HRSSM_DIR="${HRSSM_DIR:-$HOME/HRSSM}"
DURATION="${DURATION:-1800}"   # 30 min: startup (env creation + prefill + first eval
                                # batch + one-time pretrain burst) can take several
                                # minutes on CPU-rendered (osmesa) clusters -- this
                                # leaves room for at least one post-startup training
                                # log line, which the ratio/fps checks below need.
POLL_INTERVAL="${POLL_INTERVAL:-15}"   # seconds between GPU-memory samples
LOGDIR="${LOGDIR:-/tmp/hrssm_smoketest}"
STEPS="${STEPS:-2e6}"                # must match launch_experiments.sh's STEPS
NUM_GPUS="${NUM_GPUS:-2}"            # must match launch_experiments.sh's NUM_GPUS
TOTAL_RUNS="${TOTAL_RUNS:-9}"        # 3 tasks x 3 seeds, must match launch_experiments.sh

cd "$HRSSM_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hrssm
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"   # override e.g. MUJOCO_GL=egl bash smoke_test.sh
echo "==> Rendering backend: MUJOCO_GL=${MUJOCO_GL}  PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-<unset>}"

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

# Poll GPU memory continuously in the background rather than sampling once at a
# fixed delay -- startup (env creation, first eval batch, pretrain burst) can
# take several minutes, so a single early sample risks reading near-idle memory
# before the model has even finished constructing.
GPU_MEM_LOG="$LOGDIR/gpu_mem_samples.csv"
: > "$GPU_MEM_LOG"
TOTAL_MIB=16384
POLL_PID=""
if command -v nvidia-smi &> /dev/null; then
  TOTAL_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 | tr -d ' ')"
  (
    while kill -0 "$PID" 2>/dev/null; do
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 \
        >> "$GPU_MEM_LOG" 2>/dev/null
      sleep "$POLL_INTERVAL"
    done
  ) &
  POLL_PID=$!
fi

sleep 30
if ! kill -0 "$PID" 2>/dev/null; then
  echo "!! Process exited early -- check $LOGDIR/run.out for an error"
  echo "   (common cause: osmesa rendering failure -- try MUJOCO_GL=egl instead)"
  tail -n 40 "$LOGDIR/run.out"
  [ -n "$POLL_PID" ] && kill "$POLL_PID" 2>/dev/null || true
  exit 1
fi

REMAINING=$(( DURATION - 30 ))
if [ "$REMAINING" -gt 0 ]; then
  sleep "$REMAINING"
fi

kill "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
[ -n "$POLL_PID" ] && kill "$POLL_PID" 2>/dev/null || true

USED_MIB=0
if [ -s "$GPU_MEM_LOG" ]; then
  USED_MIB="$(sort -n "$GPU_MEM_LOG" | tail -1)"
fi

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

print(f"logged training-metric lines: {len(train_rows)}")
for r in train_rows[-5:]:
    print(f"  step={r['step']:<10} update_count={r['update_count']:<10} fps={r.get('fps', 0):.2f}")

print()
if len(train_rows) < 2:
    print("Only one training-metric line so far -- that line includes the one-time")
    print("`config.pretrain` (100) burst, so its raw update_count/step ratio is NOT")
    print("a steady-state measurement. Increase DURATION and rerun to get a second")
    print("line (needed to compute a clean delta-based ratio).")
    sys.exit(0)

# Use the last two lines' deltas -- this skips whatever one-time pretrain offset
# is baked into any single line's absolute update_count.
r_prev, r_last = train_rows[-2], train_rows[-1]
d_step = r_last["step"] - r_prev["step"]
d_updates = r_last["update_count"] - r_prev["update_count"]
ratio = d_updates / d_step if d_step else float("nan")
print(f"steady-state ratio (delta update_count / delta step): {ratio:.3f}")
print("(expect ~0.25 -- NOT ~0.5: the logged `step` field is action_repeat *")
print(" agent._step, but the gradient-update cadence is gated on agent._step")
print(" directly, so (batch_size*batch_length/train_ratio) / (envs*action_repeat)")
print(" = 2 / (4*2) = 0.25)")

# ---- fps / ETA ----
fps_rows = [r["fps"] for r in rows if r.get("fps", 0) > 0]
print()
if len(fps_rows) < 2:
    print("Not enough 'fps' samples yet to estimate ETA -- rerun with a longer DURATION.")
    sys.exit(0)

# Drop the first nonzero sample: it typically still includes startup/torch.compile
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

# ---- suggest RUNS_PER_GPU from the observed (max) VRAM footprint ----
safety_margin_mib = 1024
usable_mib = max(total_mib - safety_margin_mib, 0)
if used_mib > 0:
    suggested_runs_per_gpu = max(1, int(usable_mib // used_mib))
else:
    suggested_runs_per_gpu = 1

print(f"\nObserved single-run GPU memory (max over the run so far): "
      f"{used_mib:.0f} MiB / {total_mib:.0f} MiB total")
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
