"""
Rebuild a SINGLE continuous wandb run from a logdir's metrics.jsonl.

Use this when a seed's wandb curve is broken/partial and can NOT be fixed in
place -- e.g. finger_spin_dcs_seed0_v5: an untracked "zombie" process trained
0 -> ~167.5k with no --use_wandb, then a checkpoint-resumed process logged
~167.5k -> 500k as run `hjww4q4g`. Backfilling the missing 0 -> 167.5k into
`hjww4q4g` was tried and is IMPOSSIBLE: wandb enforces step ordering server-side
per run, so once a run's step has advanced past a point nothing can log behind
it, ever (see memory: hrssm-checkpoint-resume-feature, 2026-08-28 entry).

The fix that works: create a brand-new run and replay the COMPLETE history from
metrics.jsonl (which tools.py writes unconditionally on every Logger.write(),
independent of --use_wandb, so it always has the full 0 -> 500k history in one
file, appended across every crash/resume).

Usage (inside the container, conda env active, from the worktree root):

    # 1. inspect the file first -- confirm it really spans ~0 -> full training:
    python rebuild_wandb_run.py \
        --metrics_path log/finger_spin_dcs_seed0_v5/metrics.jsonl \
        --action_repeat 2 --dry_run

    # 2. then create the new run:
    python rebuild_wandb_run.py \
        --metrics_path log/finger_spin_dcs_seed0_v5/metrics.jsonl \
        --action_repeat 2 \
        --project hrssm-dcs \
        --entity vivaan-singh21-carnegie-mellon-university \
        --run_name finger_spin_dcs_seed0_v5_full \
        --config_json log/finger_spin_dcs_seed0_v5/config.json

Notes:
  * Rows sharing a wandb step are merged (last occurrence wins) -- handles the
    small overlap left by the failed intermediate resume (run `x81ot62l`).
  * Only numeric/bool scalars are replayed.
  * This creates a NEW run id every time you run it without --dry_run. Delete the
    old partial run (`hjww4q4g`) and the dead OOM run (`x81ot62l`) by hand in the
    wandb UI once the new one looks right.
"""
import argparse
import json


def load_rows(metrics_path, action_repeat):
    merged = {}
    raw_min = raw_max = None
    n_lines = 0
    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            row = json.loads(line)
            if "step" not in row:
                continue
            raw_step = row["step"]
            raw_min = raw_step if raw_min is None else min(raw_min, raw_step)
            raw_max = raw_step if raw_max is None else max(raw_max, raw_step)
            wandb_step = raw_step // action_repeat
            scalars = {
                k: v
                for k, v in row.items()
                if k != "step" and isinstance(v, (int, float, bool))
            }
            merged.setdefault(wandb_step, {}).update(scalars)
    ordered = sorted(merged.items())
    return ordered, n_lines, raw_min, raw_max


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics_path", required=True)
    p.add_argument("--action_repeat", type=int, required=True)
    p.add_argument("--project", default="hrssm-dcs")
    p.add_argument("--entity", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--config_json", default=None,
                   help="path to the run's config.json, copied into the new run's config panel")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()

    ordered, n_lines, raw_min, raw_max = load_rows(args.metrics_path, args.action_repeat)
    if not ordered:
        raise SystemExit("No rows with a 'step' field found in metrics.jsonl.")

    steps = [s for s, _ in ordered]
    print(f"Parsed {n_lines} lines -> {len(ordered)} unique wandb steps.")
    print(f"  raw env-step range:  {raw_min} .. {raw_max}")
    print(f"  wandb step range:    {steps[0]} .. {steps[-1]}  (raw // {args.action_repeat})")
    eval_rows = [(s, r["eval_return"]) for s, r in ordered if "eval_return" in r]
    print(f"  rows with eval_return: {len(eval_rows)}")
    if eval_rows:
        print(f"    first: step={eval_rows[0][0]} eval_return={eval_rows[0][1]:.1f}")
        print(f"    last:  step={eval_rows[-1][0]} eval_return={eval_rows[-1][1]:.1f}")

    if args.dry_run:
        print("\nDry run -- first 3 / last 3 replay points:")
        for s, r in ordered[:3] + ordered[-3:]:
            print(f"  step={s}: { {k: r[k] for k in list(r)[:6]} }")
        return

    import wandb

    config = None
    if args.config_json:
        with open(args.config_json) as f:
            config = json.load(f)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.run_name,
        config=config,
        settings=wandb.Settings(x_disable_stats=True),
    )
    print(f"Created new run: {run.id}  ({run.url})")
    for s, r in ordered:
        wandb.log(r, step=s)
    wandb.finish()
    print(f"Replayed {len(ordered)} points into new run {run.id}.")


if __name__ == "__main__":
    main()
