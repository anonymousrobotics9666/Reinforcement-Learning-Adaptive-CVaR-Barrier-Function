"""Fast smoke test for the training/eval stack.

The test writes local artifacts under test/ so results stay out of git.
"""

from pathlib import Path
import argparse
import json
import sys
from datetime import datetime

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import Config
from crowd_sim.utils import build_env, resolve_env_name
from model.factory import build_model
from rl.run_utils import resolve_device, set_global_seeds
from scripts.eval import Evaluator, policy_obs_from_env_obs


def compose_smoke_config(args):
    overrides = [
        f"method={args.method}",
        f"env/robot={args.robot}",
        f"device={args.device}",
        f"seed={args.seed}",
        "num_envs=1",
        "total_timesteps=16",
        "eval_interval=1",
        "eval_episodes=1",
        "env.max_steps=5",
        "human.num_humans=2",
        "human.human_num_range=0",
        "env.obs_top_k=1",
    ]
    with initialize_config_dir(version_base=None, config_dir=str(REPO_ROOT / "config")):
        return compose(config_name="trainer/train", overrides=overrides)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="diffcvarbfqp", choices=["vanilla_ppo", "diffcvarbfqp"])
    parser.add_argument("--robot", default="unicycle", choices=["single_integrator", "unicycle"])
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="test/smoke")
    args = parser.parse_args()

    set_global_seeds(args.seed)
    cfg = compose_smoke_config(args)
    config = Config(cfg)
    env_name = resolve_env_name(config)
    device = resolve_device(args.device, default="cpu")

    run_dir = REPO_ROOT / args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "ckpt_00000001.pt"
    result_path = run_dir / "smoke_result.json"
    OmegaConf.save(config=cfg, f=str(run_dir / "config.yaml"), resolve=True)

    env = build_env(env_name, render_mode=None, config=config)
    try:
        obs, _info = env.reset(seed=args.seed)
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _info = env.step(action)

        obs_top_k = int(config.env_params.get("obs_top_k", config.env_params.get("max_obstacles_obs", 1)))
        obs_dim = 6 + obs_top_k * 6
        act_dim = env.action_space.shape[0]
        model = build_model(
            cfg,
            obs_dim,
            act_dim,
            action_low=env.action_space.low,
            action_high=env.action_space.high,
        ).to(device)
        model.eval()

        rel_obs = policy_obs_from_env_obs(next_obs, obs_top_k)
        obs_t = torch.tensor(rel_obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            policy_out = model.get_action_deterministic(obs_t)

        torch.save({"model": model.state_dict(), "step": 1}, ckpt_path)
    finally:
        env.close()

    eval_results = Evaluator(run_dir, episodes_per_seed=1, visualize=False).eval_all_ckpts(checkpoint=ckpt_path)
    ckpt_metrics = eval_results[ckpt_path.name]

    payload = {
        "status": "ok",
        "method": args.method,
        "robot": args.robot,
        "env_name": env_name,
        "device": str(device),
        "checkpoint": str(ckpt_path),
        "one_step": {
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "policy_output_shape": list(policy_out.shape),
        },
        "eval": ckpt_metrics,
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"smoke test passed: {result_path}")


if __name__ == "__main__":
    main()
