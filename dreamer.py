import argparse
import contextlib
import functools
import os
import pathlib
import sys
import json
import time

# Force EGL (GPU-accelerated) rendering, not OSMesa (CPU software rendering). This is a
# hard override, not setdefault: PYOPENGL_PLATFORM is set to 'osmesa' at the container/shell
# level on canebrake (Dockerfile/.bashrc), so setdefault would silently no-op against it --
# confirmed EGL works cleanly in this container (verify_dcs_setup.py, 2026-07-29).
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Pin the EGL render context to the same physical GPU as --device, before dm_control gets
# imported (transitively, below) -- dm_control's egl_renderer.py picks its rendering GPU
# from CUDA_VISIBLE_DEVICES exactly once, at import time, and falls back to whichever
# device EGL enumerates first (always GPU0) if that var is unset. --device alone only
# steers PyTorch's compute placement, not EGL's, so without this every concurrent run's
# rendering piles onto GPU0 regardless of --device (RUNTIME_CHALLENGES.md). Skipped if
# CUDA_VISIBLE_DEVICES is already set by the launcher -- that already handles both.
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    for _i, _arg in enumerate(sys.argv):
        _gpu_idx = None
        if _arg == "--device" and _i + 1 < len(sys.argv) and sys.argv[_i + 1].startswith("cuda:"):
            _gpu_idx = sys.argv[_i + 1].split(":", 1)[1]
            sys.argv[_i + 1] = "cuda:0"
        elif _arg.startswith("--device=cuda:"):
            _gpu_idx = _arg.split(":", 1)[1]
            sys.argv[_i] = "--device=cuda:0"
        if _gpu_idx is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = _gpu_idx
            break

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

import collections


to_np = lambda x: x.detach().cpu().numpy()


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        # train-vs-env wall-clock split diagnostic (see RUNTIME_CHALLENGES.md #13/#14)
        self._train_time_accum = 0.0
        # one-shot op-level CPU/CUDA profiling diagnostic (see RUNTIME_CHALLENGES.md #15)
        self._profiled_train = False
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(
            config, self._wm, config.behavior_stop_grad
        )
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if self._should_reset(step):
            state = None
        if state is not None and reset.any():
            mask = 1 - reset
            for key in state[0].keys():
                for i in range(state[0][key].shape[0]):
                    state[0][key][i] *= mask[i]
            for i in range(len(state[1])):
                state[1][i] *= mask[i]
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            if self._config.profile_train and not self._profiled_train and steps > 0:
                from torch.profiler import profile, ProfilerActivity

                with profile(
                    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    with_stack=True,
                    record_shapes=True,
                ) as prof:
                    for _ in range(5):
                        self._train(next(self._dataset))
                        self._update_count += 1
                        self._metrics["update_count"] = self._update_count
                print("=== flat op table ===")
                print(
                    prof.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=25
                    )
                )
                print("=== grouped by call stack (top 5 frames) ===")
                print(
                    prof.key_averages(group_by_stack_n=5).table(
                        sort_by="self_cpu_time_total", row_limit=40
                    )
                )
                self._profiled_train = True
            for _ in range(steps):
                _t0 = time.perf_counter()
                self._train(next(self._dataset))
                self._train_time_accum += time.perf_counter() - _t0
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            batch_size = len(obs["image" if "image" in obs else "obs"])
            latent = self._wm.dynamics.initial(len(obs["image" if "image" in obs else "obs"]))
            action = torch.zeros((batch_size, self._config.num_actions)).to(
                self._config.device
            )
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        action = self._exploration(action, training)
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _exploration(self, action, training):
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._config.actor_dist:
            probs = amount / self._config.num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


def make_env(config, mode):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        train_mode = "_".join(task.split("_")[-2:])
        if train_mode in ['color_easy', 'color_hard', 'video_easy', 'video_hard', 'distracting_cs', 'sensor_cs']:
            domain, task = task[:-len(train_mode) - 1].split("_", 1)
            import env.wrappers
            assert config.size[0] == config.size[1]
            # `mode` here is the outer train/eval selector (see make_env's caller); it must NOT be
            # confused with `train_mode` above, which is the distraction-type string parsed from
            # the task name. Threaded through as env_split so eval envs draw DAVIS val videos.
            env_split = "train" if mode == "train" else "eval"
            env = env.wrappers.make_env(domain_name=domain, task_name=task, seed=config.seed, action_repeat=config.action_repeat, image_size=config.size[0], mode=train_mode, intensity=config.distracting_cs_intensity, ds_resource_path=[config.ds_resource_path], env_split=env_split)
            env = wrappers.DMC2GYMWrapper(env)
        elif task.split("_")[-1] == "video":
            task = "_".join(task.split("_")[:-1])
            domain, task = task.split("_", 1)
            # import envs.dmc_video as dmc_video
            import distractingdmc2gym
            env = distractingdmc2gym.make(
                domain_name=domain,
                task_name=task,
                resource_files=config.resource_files,
                img_source=config.img_source,
                total_frames=config.total_frames,
                seed=config.seed,
                visualize_reward=False,
                from_pixels=True,
                height=config.size[0],
                width=config.size[1],
                frame_skip=config.action_repeat,
                grayscale=True,
            )
            env = wrappers.DMC2GYMWrapper(env)
            # env = dmc_video.DeepMindControl(task, config.action_repeat, config.size, img_source=config.img_source, resource_files=config.resource_files, total_frames=config.total_frames, seed=config.seed)
        else:
            import envs.dmc as dmc

            env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed
        )
            env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    elif suite == "mw":
        import envs.metaworld as metaworld
        from envs.tensor import TensorWrapper
        env = metaworld.make_env(task, config)
        # env = TensorWrapper(env)
    elif suite == "ms":
        import envs.maniskill as maniskill
        env = maniskill.make_env(task, config)
    elif suite == "rms":
        import envs.realistic_maniskill as realistic_maniskill
        env = realistic_maniskill.make_env(task, config)
    elif suite == "myo":
        import envs.myosuite as myosuite
        env = myosuite.make_env(task, config)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    assert suite in ["dmc", "mw", "ms", "rms", "myo"], "wrappers.ClipAction now only support these"
    env = wrappers.ClipAction(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env

def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    with open(os.path.join(logdir, 'config.json'), 'w') as f:
        json.dump(vars(config), f, sort_keys=True, indent=4)
    config.traindir = config.traindir or logdir / "train_eps"
    config.evaldir = config.evaldir or logdir / "eval_eps"
    config.videodir = config.videodir or logdir / "video"
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    if not config.simple_log:
        config.videodir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    checkpoint_path = logdir / "latest.pt"
    resume_checkpoint = None
    if checkpoint_path.exists():
        resume_checkpoint = torch.load(checkpoint_path, map_location="cpu")
        # episodes are never written to disk (see tools.save_episodes, disabled), so
        # count_steps(traindir) always reads 0 -- the checkpoint's own step count is
        # the only source of truth for how far a resumed run had actually gotten.
        step = resume_checkpoint["step"]
        print(f"Resuming from {checkpoint_path} at step {step}.")
    # step in logger is environmental step
    logger = (
        tools.SimpleLogger(logdir, config.action_repeat * step, use_wandb=config.use_wandb, action_repeat=config.action_repeat)
        if config.simple_log
        else tools.FullLogger(logdir, config.videodir, config.action_repeat * step, use_wandb=config.use_wandb, action_repeat=config.action_repeat)
    )

    print("Create envs.")
    train_eps = collections.OrderedDict()
    eval_eps = collections.OrderedDict()
    make = lambda mode: make_env(config, mode)

    # Capture the resolved DCS settings (which video split, which distraction types actually
    # fired, resolved video paths) straight from the [DCS] logs emitted during the REAL
    # gym.make() construction below -- not re-derived -- so a wandb run is self-auditable
    # without needing this repo's stdout scrollback to trust it. Only relevant for DCS tasks.
    is_dcs_task = config.use_wandb and "_distracting_cs" in config.task
    dcs_wandb_config = {}
    if is_dcs_task:
        from verify_dcs_setup import Tee, parse_dcs_log

        tee = Tee(sys.stdout)
        with contextlib.redirect_stdout(tee):
            train_envs = [make("train") for _ in range(config.envs)]
        train_dcs_info = parse_dcs_log(tee.getvalue())

        tee = Tee(sys.stdout)
        with contextlib.redirect_stdout(tee):
            eval_envs = [make("eval") for _ in range(config.envs)]
        eval_dcs_info = parse_dcs_log(tee.getvalue())

        for split, info in (("train", train_dcs_info), ("eval", eval_dcs_info)):
            dcs_wandb_config[f"dcs_{split}_background_dataset_videos"] = info.get("background_dataset_videos")
            dcs_wandb_config[f"dcs_{split}_video_paths"] = info.get("video_paths")
            dcs_wandb_config[f"dcs_{split}_applied_distractions"] = list(info.get("applied_distractions") or [])
    else:
        train_envs = [make("train") for _ in range(config.envs)]
        eval_envs = [make("eval") for _ in range(config.envs)]

    if config.use_wandb:
        import wandb

        # By this point config.traindir/evaldir/videodir have been reassigned to pathlib.Path
        # objects (see above) -- stringify everything path-like so wandb's config serialization
        # doesn't choke on non-primitive values.
        serializable_config = {
            k: (str(v) if isinstance(v, pathlib.Path) else v) for k, v in vars(config).items()
        }
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.wandb_run_name or logdir.name,
            config={
                **serializable_config,
                **dcs_wandb_config,
                # HRSSM's eval policy (dreamer.py _policy, `if not training: action = actor.mode()`)
                # is always deterministic with eval_noise=0 -- there is no separate stochastic
                # eval metric to choose between, unlike VIBES which logs both.
                "eval_policy_type": "deterministic (actor.mode(), eval_noise=0)",
                "wandb_x_axis_convention": "env steps = post-action-repeat control steps (CoRe/VIBES convention), NOT raw MuJoCo ticks",
            },
        )

    if config.parallel:
        train_envs = [Parallel(env, "process") for env in train_envs]
        eval_envs = [Parallel(env, "process") for env in eval_envs]
    else:
        train_envs = [Damy(env) for env in train_envs]
        eval_envs = [Damy(env) for env in eval_envs]
    acts = train_envs[0].action_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]

    state = None
    if not config.offline_traindir:
        prefill = max(0, config.prefill - count_steps(config.traindir))
        print(f"Prefill dataset ({prefill} steps).")
        if hasattr(acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(config.num_actions).repeat(config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.Tensor(acts.low).repeat(config.envs, 1),
                    torch.Tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        state = tools.simulate(
            random_agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=prefill,
        )
        logger.step += prefill * config.action_repeat
        print(f"Logger: ({logger.step} steps).")

    print("Simulate agent.")
    train_dataset = make_dataset(train_eps, config)
    eval_dataset = make_dataset(eval_eps, config)
    agent = Dreamer(
        train_envs[0].observation_space,
        train_envs[0].action_space,
        config,
        logger,
        train_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if resume_checkpoint is not None:
        agent.load_state_dict(resume_checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, resume_checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    # make sure eval will be executed once after config.steps
    while agent._step < config.steps + config.eval_every:
        logger.write()
        if config.eval_episode_num > 0:
            print("Start evaluation.")
            eval_policy = functools.partial(agent, training=False)
            tools.simulate(
                eval_policy,
                eval_envs,
                eval_eps,
                config.evaldir,
                logger,
                is_eval=True,
                episodes=config.eval_episode_num,
            )
            if config.video_pred_log:
                video_pred = agent._wm.video_pred(next(eval_dataset))
                logger.video("eval_openl", to_np(video_pred))
            # erase_over_episodes() only runs for the train cache (tools.simulate,
            # is_eval=False) -- eval_eps is otherwise never trimmed and grows for the
            # entire run's lifetime. This was the root cause of the OOM kills (see
            # RUNTIME_CHALLENGES.md): ~25,000 env-steps of image data retained forever
            # per eval cycle. Nothing needs eval episodes to persist across cycles.
            eval_eps.clear()
        print("Start training.")
        state = tools.simulate(
            agent,
            train_envs,
            train_eps,
            config.traindir,
            logger,
            limit=config.dataset_size,
            steps=config.eval_every,
            state=state,
        )
        torch.save(
            {
                "step": agent._step,
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
            },
            logdir / "latest.pt",
        )
    for env in train_envs + eval_envs:
        try:
            env.close()
        except Exception:
            pass
    if config.use_wandb:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )

    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value

    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])
    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    parser.add_argument('--simsr_discount', default=0.99, type=float)
    parser.add_argument('--dynamics_tau', default=0.05, type=float)
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--img_source', default='video', type=str, choices=['video'])
    parser.add_argument('--resource_files', default="../distractors/*.mp4", type=str)
    parser.add_argument('--ds_resource_path', default="..", type=str)
    parser.add_argument('--total_frames', default=1000, type=int)
    parser.add_argument('--simsr_scale', default=1, type=float)
    parser.add_argument('--mbr_scale', default=1, type=float)
    parser.add_argument('--simple_log', action='store_true')
    parser.add_argument('--nomlr', action='store_true')
    parser.add_argument('--nosimsr', action='store_true')
    parser.add_argument('--post_mlr', action='store_true')
    parser.add_argument('--profile_train', action='store_true')
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_project', default='hrssm-dcs', type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)
    main(parser.parse_args(remaining))
