import os
from datetime import datetime
import hydra
from omegaconf import DictConfig
from config.config import Config
from controller.robot_controller_factory import build_robot_controller
from eval.eval_policy import eval_policy
from crowd_sim.utils import build_env, dump_test_config, load_train_config_snapshot, resolve_env_name

def _prepare_save_dirs(args, robot_type):
    base_dir = os.path.join(
        "trained_models",
        args.model_folder,
        f"{robot_type}_{args.method}",
    )
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return base_dir, run_dir


def _load_test_config_snapshot(args, config):
    config_payload = {}
    config_source = "config.py"

    if args.use_current_config:
        if args.config_json:
            raise ValueError("--config_json and --use_current_config are mutually exclusive")
        return config, config_payload, config_source

    config_json = str(getattr(args, "config_json", "") or "").strip()
    if not config_json:
        raise ValueError(
            "main_opt.py requires --config_json to specify the config snapshot to load. "
            "Pass --use_current_config to use the current config.py instead."
        )
    config_path = os.path.abspath(os.path.expanduser(config_json))
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config snapshot not found: {config_path}")
    config_payload = load_train_config_snapshot(config, config_path, use_current_config=False)
    if not config_payload:
        raise RuntimeError(f"Failed to load config snapshot from {config_path}")
    config_source = os.path.basename(config_path)
    return config, config_payload, config_source



@hydra.main(config_path="config", config_name="eval_opt", version_base=None)
def main(cfg: DictConfig):
    args = cfg
    config = Config(cfg)
    _, config_payload, config_source = _load_test_config_snapshot(args, config)
    env_name = resolve_env_name(config, config_payload)

    render_mode = "human" if args.render else "rgb_array"
    env = build_env(env_name, render_mode=render_mode, config=config)
    controller = build_robot_controller(args.method, config, env)
    base_save_dir, save_dir = _prepare_save_dirs(args, config.robot.type)
    dump_test_config(
        save_dir,
        config,
        hyperparameters={
            "env_name": env_name,
            "method": args.method,
            "test_ep": args.test_ep,
            "episode_seed_start": args.episode_seed_start,
        },
        extra={
            "script": "main_opt.py",
            "config_source": config_source,
            "config_json": os.path.abspath(os.path.expanduser(args.config_json)) if args.config_json else None,
            "use_current_config": bool(args.use_current_config),
        },
    )
    # dump_train_config(save_dir, args, config)
    print(f"Evaluation base dir: {base_save_dir}", flush=True)
    print(f"Evaluation outputs: {save_dir}", flush=True)

    episode_seed_start = args.seed if getattr(args, "episode_seed_start", None) is None else args.episode_seed_start

    try:
        eval_policy(
            policy=controller,
            env=env,
            max_episodes=args.test_ep,
            save_path=save_dir,
            base_seed=episode_seed_start,
        )
    finally:
        env.close()



if __name__ == "__main__":
    main()
