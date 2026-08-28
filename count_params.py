"""
Standalone parameter-count utility for the DreamerV3-vs-HRSSM-vs-VIBES comparison.

Builds a real WorldModel + ImagBehavior from an existing run's config.json
(no live env, no DAVIS dataset, no EGL needed -- just fake obs/act spaces with
the right shapes) and reports trainable-vs-EMA-copy parameter counts.

IMPORTANT NUANCE (don't "simplify" this by filtering on `requires_grad`):
HRSSM's EMA target networks (`WorldModel.target_dynamics`, `WorldModel.MBR.target_encoder`,
`ImagBehavior._slow_value`) are plain `copy.deepcopy()`s of the trainable originals.
Nobody sets `requires_grad = False` on them -- they stay `requires_grad=True` by
default. They only stay "frozen" because every forward pass that touches them is
wrapped in `torch.no_grad()` (models.py, WorldModel._train), so `.grad` is always
None for them and the optimizer (which does iterate `self.parameters()`, including
them) never actually updates them via gradient descent -- they only move via the
explicit `soft_update_params` EMA calls elsewhere. So `p.requires_grad` CANNOT be
used to separate "trainable" from "EMA-only" here -- this script does it by
explicit submodule reference instead.

Usage (run inside the container, conda env activated, from the repo root):
    python count_params.py --config_json log/cup_catch_dcs_seed2_v5/config.json --num_actions 2

--num_actions isn't saved in config.json (it's injected by dreamer.py at runtime
from the real env's action space, not a CLI/configs.yaml field) -- pass it explicitly.
Known DMC action dims for this campaign's tasks: cheetah_run=6, walker_walk=6,
cartpole_swingup=1, reacher_easy=2, cup_catch=2, finger_spin=2.
"""
import argparse
import json

import gym
import numpy as np

import models


def build_fake_obs_space(config):
    h, w = config.size[0], config.size[1]
    return gym.spaces.Dict(
        {"image": gym.spaces.Box(0, 255, (h, w, 3), dtype=np.uint8)}
    )


def count(module):
    return sum(p.numel() for p in module.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_json", required=True)
    parser.add_argument("--num_actions", type=int, required=True)
    args = parser.parse_args()

    with open(args.config_json) as f:
        cfg_dict = json.load(f)
    config = argparse.Namespace(**cfg_dict)
    config.num_actions = args.num_actions
    # some configs.yaml fields are only ever read as attributes at runtime by
    # networks.py/models.py -- config.json is dreamer.py's own full resolved
    # dump of exactly those fields, so this round-trips faithfully except for
    # num_actions above, which is injected from the real env at launch time.

    obs_space = build_fake_obs_space(config)
    act_space = gym.spaces.Box(-1, 1, (args.num_actions,), dtype=np.float32)

    wm = models.WorldModel(obs_space, act_space, 0, config)
    behavior = models.ImagBehavior(config, wm, config.behavior_stop_grad)

    wm_total = count(wm)
    wm_ema_only = count(wm.target_dynamics) + count(wm.MBR.target_encoder)
    wm_trainable = wm_total - wm_ema_only

    actor_params = count(behavior.actor)
    value_params = count(behavior.value)
    slow_value_params = count(behavior._slow_value) if config.slow_value_target else 0

    print(f"config: {args.config_json}")
    print(f"task: {config.task}")
    print()
    print("=== WorldModel (encoder + RSSM + reward/cont heads; no decoder) ===")
    print(f"  total (incl. EMA target_encoder + target_dynamics copy): {wm_total:,}")
    print(f"  EMA-only (target_encoder + target_dynamics, never gradient-updated): {wm_ema_only:,}")
    print(f"  trainable (total - EMA-only): {wm_trainable:,}")
    print()
    print("=== ImagBehavior (actor-critic) ===")
    print(f"  actor: {actor_params:,}")
    print(f"  value/critic: {value_params:,}")
    if config.slow_value_target:
        print(f"  slow_value (EMA target copy of critic, never gradient-updated): {slow_value_params:,}")
    print(f"  actor+value trainable total: {actor_params + value_params:,}")
    print()
    grand_trainable = wm_trainable + actor_params + value_params
    grand_total = wm_total + actor_params + value_params + slow_value_params
    print(f"=== Grand totals ===")
    print(f"  trainable (world model + actor + value): {grand_trainable:,}")
    print(f"  total incl. all EMA copies: {grand_total:,}")


if __name__ == "__main__":
    main()
