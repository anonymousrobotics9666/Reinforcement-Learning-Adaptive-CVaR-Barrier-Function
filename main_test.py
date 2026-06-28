"""Test one saved PPO actor checkpoint."""

import os
from datetime import datetime

import hydra
from omegaconf import DictConfig

from config.config import Config
from crowd_sim.utils import (
    build_env,
    dump_test_config,
    load_train_config_snapshot,
    resolve_env_name,
)
from eval.eval_policy import eval_policy, parse_episode_seeds_csv
from eval.policy_builder import build_eval_actor
from rl.run_utils import resolve_device, set_global_seeds


def resolve_episode_seed_args(args):
    episode_seeds = parse_episode_seeds_csv(getattr(args, "episode_seeds", "") or "")
    test_ep = len(episode_seeds) if episode_seeds is not None else int(args.test_ep)
    return episode_seeds, test_ep


def load_test_config_snapshot(config, args):
    config_payload = {}
    config_source = "config.py"

    if args.use_current_config:
        if args.config_json:
            raise ValueError("config_json and use_current_config are mutually exclusive")
        return config_payload, config_source

    config_json = str(getattr(args, "config_json", "") or "").strip()
    if not config_json:
        raise ValueError(
            "main_test.py requires config_json for checkpoint-compatible evaluation. "
            "Set use_current_config=true to evaluate with the current YAML config."
        )

    config_path = os.path.abspath(os.path.expanduser(config_json))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config snapshot not found: {config_path}")

    config_payload = load_train_config_snapshot(config, config_path, use_current_config=False)
    if not config_payload:
        raise RuntimeError(f"Failed to load config snapshot from {config_path}")
    return config_payload, os.path.basename(config_path)


def checkpoint_tag(actor_model):
    actor_file = os.path.basename(actor_model)
    stem, _ = os.path.splitext(actor_file)
    return stem


def seed_tag(episode_seeds, test_ep, episode_seed_start):
    if episode_seeds:
        if len(episode_seeds) == 1:
            return f"seed_{episode_seeds[0]}"
        return f"seeds_n{len(episode_seeds)}_{episode_seeds[0]}_{episode_seeds[-1]}"

    if test_ep <= 1 and episode_seed_start is not None:
        return f"seed_{episode_seed_start}"
    if episode_seed_start is None:
        return f"episodes_n{test_ep}"
    seed_end = int(episode_seed_start) + int(test_ep) - 1
    return f"seeds_n{test_ep}_{episode_seed_start}_{seed_end}"


def ensure_unique_dir(parent_dir, run_name):
    candidate = run_name
    idx = 1
    while os.path.exists(os.path.join(parent_dir, candidate)):
        candidate = f"{run_name}_{idx}"
        idx += 1
    return os.path.join(parent_dir, candidate)


def prepare_test_save_dir(actor_model, episode_seeds, test_ep, episode_seed_start):
    base_dir = os.path.dirname(actor_model)
    run_name = (
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
        f"ckpt_{checkpoint_tag(actor_model)}_"
        f"{seed_tag(episode_seeds, test_ep, episode_seed_start)}"
    )
    run_dir = ensure_unique_dir(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


@hydra.main(config_path="config", config_name="trainer/test", version_base=None)
def main(cfg: DictConfig):
    args = cfg
    set_global_seeds(args.seed)
    device = resolve_device(args.device, default="cpu")

    actor_model = str(args.actor_model or "").strip()
    if not actor_model or not os.path.exists(actor_model):
        raise FileNotFoundError(f"Actor model not found: {actor_model}")

    config = Config(cfg)
    config_payload, config_source = load_test_config_snapshot(config, args)
    env_name = resolve_env_name(config, config_payload)

    episode_seeds, test_ep = resolve_episode_seed_args(args)
    test_save_dir = prepare_test_save_dir(
        actor_model=actor_model,
        episode_seeds=episode_seeds,
        test_ep=test_ep,
        episode_seed_start=args.episode_seed_start,
    )

    dump_test_config(
        test_save_dir,
        config,
        hyperparameters={
            "env_name": env_name,
            "method": args.method,
            "test_ep": test_ep,
            "episode_seeds": episode_seeds,
            "episode_seed_start": args.episode_seed_start,
            "device": str(device),
            "save_gifs": bool(args.save_gifs),
            "max_gifs": args.max_gifs,
        },
        extra={
            "script": "main_test.py",
            "config_source": config_source,
            "config_json": (
                os.path.abspath(os.path.expanduser(args.config_json))
                if args.config_json
                else None
            ),
            "use_current_config": bool(args.use_current_config),
        },
    )

    render_mode = "human" if args.render else "rgb_array"
    env = build_env(env_name, render_mode=render_mode, config=config)
    try:
        actor = build_eval_actor(config, args.method, env, actor_model, device)
        eval_policy(
            policy=actor,
            env=env,
            max_episodes=test_ep,
            save_path=test_save_dir,
            base_seed=args.episode_seed_start,
            method=args.method,
            episode_seeds=episode_seeds,
            save_gifs=bool(args.save_gifs),
            max_gifs=args.max_gifs,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
