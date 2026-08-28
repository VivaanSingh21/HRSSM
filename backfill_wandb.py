"""
Backfill historical training metrics into an already-live wandb run whose curve
starts partway through training -- e.g. because an untracked "zombie" process
(see memory: hrssm-multi-machine-infra, "tmux kill-window doesn't reliably kill
a docker-exec'd job") trained without --use_wandb before a checkpoint-resumed
process took over and started logging.

Source of truth: <logdir>/metrics.jsonl. tools.py's SimpleLogger/FullLogger.write()
writes this file UNCONDITIONALLY on every call -- wandb.log() is a separate,
conditional step gated on --use_wandb. So metrics.jsonl always has the complete
history regardless of whether wandb was ever attached, and can be replayed.

Safety: this script queries the LIVE run's own earliest already-logged step via
the wandb API and only backfills strictly below it -- it never trusts a hardcoded
or eyeballed cutoff, so it can't accidentally duplicate or corrupt anything the
live run has already logged, no matter exactly when the untracked-to-tracked
transition happened.

Usage (run inside the container, conda env active, from the repo root):
    # ALWAYS dry-run first and check the printed preview before committing:
    python backfill_wandb.py \
        --metrics_path log/finger_spin_dcs_seed0_v5/metrics.jsonl \
        --run_id <run_id> \
        --project hrssm-dcs \
        --entity vivaan-singh21-carnegie-mellon-university \
        --action_repeat 2 \
        --dry_run

    # then, once the preview looks right, drop --dry_run to actually log:
    python backfill_wandb.py --metrics_path ... --run_id ... --project ... --entity ... --action_repeat 2
"""
import argparse
import json

import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--project", default="hrssm-dcs")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--action_repeat", type=int, required=True)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="print what would be logged without calling wandb.log",
    )
    args = parser.parse_args()

    api = wandb.Api()
    run_path = (
        f"{args.entity}/{args.project}/{args.run_id}"
        if args.entity
        else f"{args.project}/{args.run_id}"
    )
    run = api.run(run_path)
    history = run.history(pandas=False)
    if not history:
        raise SystemExit("Run has no logged history yet -- can't determine a safe cutoff step.")
    existing_min_step = min(row["_step"] for row in history if "_step" in row)
    print(f"Live run's earliest already-logged step: {existing_min_step}")

    rows_to_backfill = []
    with open(args.metrics_path) as f:
        for line in f:
            row = json.loads(line)
            wandb_step = row["step"] // args.action_repeat
            if wandb_step < existing_min_step:
                rows_to_backfill.append((wandb_step, row))

    rows_to_backfill.sort(key=lambda x: x[0])
    print(f"Found {len(rows_to_backfill)} historical rows below step {existing_min_step} to backfill.")
    if not rows_to_backfill:
        print("Nothing to backfill.")
        return

    if args.dry_run:
        print("Dry run -- first 3 and last 3 rows that would be logged:")
        preview = rows_to_backfill[:3] + rows_to_backfill[-3:]
        for wandb_step, row in preview:
            scalars = {k: v for k, v in row.items() if k != "step"}
            print(f"  step={wandb_step}: {scalars}")
        return

    wandb.init(id=args.run_id, project=args.project, entity=args.entity, resume="must")
    for wandb_step, row in rows_to_backfill:
        scalars = {k: v for k, v in row.items() if k != "step"}
        wandb.log(scalars, step=wandb_step)
    wandb.finish()
    print(f"Backfilled {len(rows_to_backfill)} historical points into run {args.run_id}.")


if __name__ == "__main__":
    main()
