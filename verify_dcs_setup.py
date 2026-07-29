"""Standalone smoke test for the DCS (Distracting Control Suite) Easy/Dynamic pipeline.

Not part of the training loop -- run this by hand on the cluster after rsyncing DAVIS,
BEFORE launching the real 500K run, to verify:
  1. train vs eval envs actually resolve different DAVIS video pools (disjoint, 4 each).
  2. all three distraction types (background/camera/color) are genuinely active, not
     silently degraded because --ds_resource_path didn't resolve to real DAVIS content.
  3. background is genuinely dynamic (frames visibly change within an episode), camera
     framing shifts, and color differs across episode resets.

Usage (from repo root, inside the container/conda env with dm_control installed):
    python verify_dcs_setup.py --domain cheetah --task run --ds_resource_path env/data \
        --outdir dcs_verify_out

Rerun with --domain cartpole --task swingup for the other candidate task.
"""
import argparse
import ast
import contextlib
import io
import os
import re
import sys

os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np


class Tee(io.StringIO):
    """Captures written text while still echoing it to the real stdout, so build_env's
    prints remain visible live AND are parseable afterward for the runtime pool check."""

    def __init__(self, real_stdout):
        super().__init__()
        self._real_stdout = real_stdout

    def write(self, s):
        self._real_stdout.write(s)
        return super().write(s)


def save_frame(path, frame):
    from PIL import Image
    Image.fromarray(frame.astype(np.uint8)).save(path)


def rollout_and_capture(env, outdir, tag, episode_steps):
    # env here is wrapped with the same DMC2GYMWrapper dreamer.py uses, so reset()/step()
    # return plain obs dicts with an "image" key -- not a raw dm_control TimeStep. Actions
    # must come from the wrapper's *normalized* [-1, 1] action_space (env.action_space),
    # not dm_control's raw action_spec() range -- DMCWrapper asserts on the normalized
    # range before internally rescaling to the true action range.
    obs = env.reset()
    frames = {"t0": obs["image"]}

    mid = episode_steps // 2
    env.action_space.seed(0)

    for t in range(1, episode_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if t == mid:
            frames["tmid"] = obs["image"]
        if t == episode_steps - 1:
            frames["tend"] = obs["image"]
        if done:
            break

    for key, frame in frames.items():
        save_frame(os.path.join(outdir, f"{tag}_{key}.png"), frame)

    # Second episode reset, for the color-across-resets check.
    obs2 = env.reset()
    save_frame(os.path.join(outdir, f"{tag}_ep2_t0.png"), obs2["image"])

    return frames


def check_video_pool_disjoint():
    from env.distracting_control.background import (
        DAVIS17_TRAINING_VIDEOS,
        DAVIS17_VALIDATION_VIDEOS,
    )
    from env.distracting_control.suite_utils import DIFFICULTY_NUM_VIDEOS

    num_videos = DIFFICULTY_NUM_VIDEOS["0.1"]
    train_pool = DAVIS17_TRAINING_VIDEOS[:num_videos]
    val_pool = DAVIS17_VALIDATION_VIDEOS[:num_videos]
    overlap = set(train_pool) & set(val_pool)

    print(f"[verify] Easy tier num_videos={num_videos}")
    print(f"[verify] train pool: {train_pool}")
    print(f"[verify] val pool:   {val_pool}")
    print(f"[verify] overlap (must be empty): {overlap}")
    assert num_videos == 4, f"expected Easy tier to resolve 4 videos, got {num_videos}"
    assert len(train_pool) == 4 and len(val_pool) == 4
    assert not overlap, "train/val pools are NOT disjoint -- this should never happen"
    print("[verify] PASS: train/val pools are disjoint, 4 videos each.\n")


def parse_dcs_log(log_text):
    """Parse the runtime [DCS] prints emitted during the REAL gym.make() construction
    (dmcgb/wrappers.py + env/distracting_control/{suite,background}.py) -- this is
    runtime evidence tied to the actual constructed env, not a re-derivation of the
    source DAVIS17_*_VIDEOS lists."""
    parsed = {}

    m = re.search(r"background_dataset_videos=(\S+)", log_text)
    if m:
        parsed["background_dataset_videos"] = m.group(1).strip("'\"")

    m = re.search(r"resolved \d+ video path\(s\): (\[.*\])", log_text)
    if m:
        parsed["video_paths"] = ast.literal_eval(m.group(1))

    m = re.search(r"jpg counts per path: (\[.*\])", log_text)
    if m:
        parsed["jpg_counts"] = ast.literal_eval(m.group(1))

    m = re.search(r"applied_distractions=(\(.*\))", log_text)
    if m:
        parsed["applied_distractions"] = ast.literal_eval(m.group(1))

    parsed["had_empty_dir_warning"] = "WARNING: " in log_text and "zero jpg files" in log_text
    # The original (pre-existing) failure mode: ds_resource_path didn't resolve to any
    # DAVIS root at all (ground_plane_alpha still gets set, so this does NOT crash).
    parsed["had_no_dataset_paths_warning"] = (
        "no dataset paths and/or number of videos set to 0" in log_text
    )
    return parsed


def build_env(domain, task, ds_resource_path, env_split, action_repeat, image_size):
    import env.wrappers as env_wrappers
    import envs.wrappers as envs_wrappers

    print(f"\n[verify] ==== building env_split={env_split!r} for {domain}_{task} ====")
    tee = Tee(sys.stdout)
    with contextlib.redirect_stdout(tee):
        env = env_wrappers.make_env(
            domain_name=domain,
            task_name=task,
            seed=0,
            action_repeat=action_repeat,
            image_size=image_size,
            mode="distracting_cs",
            intensity=0.1,
            ds_resource_path=[ds_resource_path],
            env_split=env_split,
        )
    log_text = tee.getvalue()
    # Match dreamer.py's actual pipeline (dreamer.py:218) so this test exercises the same
    # obs shape/wrapper chain the real training run will use.
    env = envs_wrappers.DMC2GYMWrapper(env)
    return env, parse_dcs_log(log_text)


def runtime_check_pool_split(train_info, eval_info):
    print("\n" + "=" * 70)
    print("RUNTIME pool-split check (parsed from the actual gym.make() construction logs)")
    print("=" * 70)
    print(f"[verify] train background_dataset_videos = {train_info.get('background_dataset_videos')!r}")
    print(f"[verify] eval  background_dataset_videos = {eval_info.get('background_dataset_videos')!r}")
    print(f"[verify] train resolved video paths = {train_info.get('video_paths')}")
    print(f"[verify] eval  resolved video paths = {eval_info.get('video_paths')}")
    print(f"[verify] train jpg counts = {train_info.get('jpg_counts')}")
    print(f"[verify] eval  jpg counts = {eval_info.get('jpg_counts')}")
    print(f"[verify] train applied_distractions = {train_info.get('applied_distractions')}")
    print(f"[verify] eval  applied_distractions = {eval_info.get('applied_distractions')}")

    assert train_info.get("background_dataset_videos") == "train", (
        f"train env did not resolve to the 'train' split at runtime: {train_info}"
    )
    assert eval_info.get("background_dataset_videos") == "val", (
        f"eval env did not resolve to the 'val' split at runtime -- the gym-registry "
        f"caching bug may not actually be fixed: {eval_info}"
    )

    train_paths = set(train_info.get("video_paths") or [])
    eval_paths = set(eval_info.get("video_paths") or [])
    overlap = train_paths & eval_paths
    assert train_paths and eval_paths, "one of the splits resolved zero video paths"
    assert not overlap, f"train/eval resolved video paths OVERLAP at runtime: {overlap}"
    assert len(train_paths) == 4 and len(eval_paths) == 4, (
        f"expected 4 resolved paths per split (Easy tier), got "
        f"train={len(train_paths)} eval={len(eval_paths)}"
    )

    for label, info in (("train", train_info), ("eval", eval_info)):
        assert info.get("applied_distractions") == ("background", "camera", "color"), (
            f"{label} env did not apply all 3 distraction types: {info.get('applied_distractions')}"
        )
        assert not info.get("had_empty_dir_warning"), (
            f"{label} env hit an empty-directory warning -- background silently degraded"
        )
        assert not info.get("had_no_dataset_paths_warning"), (
            f"{label} env: ds_resource_path never resolved to a DAVIS root at all -- "
            f"background is running with ZERO distraction (camera/color still active)"
        )
        jpg_counts = info.get("jpg_counts") or []
        assert jpg_counts and all(n > 0 for _, n in jpg_counts), (
            f"{label} env: not all resolved video paths have jpg files: {jpg_counts}"
        )

    print("[verify] PASS: train/eval resolve disjoint 4-video pools at runtime, both apply "
          "all 3 distraction types, no empty directories.\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="cheetah")
    parser.add_argument("--task", default="run")
    parser.add_argument("--ds_resource_path", default="env/data",
                         help="parent dir containing DAVIS/JPEGImages/480p")
    parser.add_argument("--outdir", default="dcs_verify_out")
    parser.add_argument("--episode_steps", type=int, default=40)
    parser.add_argument("--action_repeat", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("=" * 70)
    print("STEP 1: static check -- DAVIS train/val source lists are disjoint")
    print("=" * 70)
    check_video_pool_disjoint()

    print("=" * 70)
    print("STEP 2: build train env (real gym.make()), roll out, capture frames")
    print("=" * 70)
    train_env, train_info = build_env(
        args.domain, args.task, args.ds_resource_path, "train",
        args.action_repeat, args.image_size,
    )
    rollout_and_capture(train_env, args.outdir, "train", args.episode_steps)

    print("=" * 70)
    print("STEP 3: build eval env (real gym.make()), roll out, capture frames")
    print("=" * 70)
    eval_env, eval_info = build_env(
        args.domain, args.task, args.ds_resource_path, "eval",
        args.action_repeat, args.image_size,
    )
    rollout_and_capture(eval_env, args.outdir, "eval", args.episode_steps)

    print("=" * 70)
    print("STEP 4: RUNTIME pool-split check (not a re-derivation of the source lists --")
    print("        this parses the logs from the actual gym.make() calls above)")
    print("=" * 70)
    runtime_check_pool_split(train_info, eval_info)

    print("=" * 70)
    print(f"ALL CHECKS PASSED. Frames saved to {args.outdir}/.")
    print("Remaining manual step -- visually compare the saved PNGs:")
    print("  - train_t0/tmid/tend.png: background should visibly change (not frozen), camera")
    print("    framing should shift")
    print("  - train_t0.png vs train_ep2_t0.png: agent geom color should differ")
    print("=" * 70)


if __name__ == "__main__":
    main()
