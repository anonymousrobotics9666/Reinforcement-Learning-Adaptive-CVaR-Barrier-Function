"""Batch-evaluate every actor checkpoint in a saved training run."""

import argparse
import glob
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.config import Config
from crowd_sim.utils import build_env, dump_test_config, load_json_dict, resolve_env_name
from eval.eval_policy import resolve_episode_seed, run_one_episode
from eval.policy_builder import build_eval_actor
from rl.run_utils import resolve_device

FIXED_EVAL_SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
EVAL_SUMMARY_FILENAME = "checkpoint_eval_all_multiseed.json"


def _extract_step(path: str) -> int:
    base = os.path.basename(path).lower()
    match = re.search(r"ckpt_(\d+)\.pt$", base)
    if not match:
        raise ValueError(f"Expected checkpoint filename format ckpt_<step>.pt: {path}")
    return int(match.group(1))


def _resolve_run_dir(actor_model: str) -> str:
    if os.path.isabs(actor_model):
        return os.path.abspath(actor_model)
    if os.path.isdir(actor_model):
        return os.path.abspath(actor_model)
    return os.path.abspath(os.path.join("trained_models", actor_model))


def _load_run_config(run_dir: str) -> tuple[Config, Dict[str, Any], str]:
    train_cfg_path = os.path.join(run_dir, "train_config.json")
    if not os.path.isfile(train_cfg_path):
        raise FileNotFoundError(
            f"Expected train_config.json in run directory: {train_cfg_path}"
        )

    payload = load_json_dict(train_cfg_path)
    cfg_dict = payload.get("config")
    if not isinstance(cfg_dict, dict):
        raise ValueError(f"Invalid train_config.json: missing top-level 'config' object")

    cfg = Config(OmegaConf.create(cfg_dict))
    return cfg, payload, train_cfg_path


def discover_checkpoints(run_dir: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    ckpt_paths = sorted(glob.glob(os.path.join(run_dir, "ckpt_*.pt")))
    for path in ckpt_paths:
        entries.append(
            {
                "path": path,
                "step": _extract_step(path),
                "is_best": False,
            }
        )
    entries.sort(key=lambda item: (item["step"], item["path"]))
    return entries


def evaluate_actor(actor, env, episodes: int, base_seed: Optional[int]) -> Dict[str, Any]:
    returns = []
    lens = []
    success_hits = 0
    collision_hits = 0
    timeout_hits = 0
    infeasible_hits = 0
    min_dists_success = []

    for ep in range(episodes):
        seed = resolve_episode_seed(base_seed, ep)
        result = run_one_episode(
            actor=actor,
            env=env,
            seed=seed,
            collect_frames=False,
        )

        returns.append(float(result["ep_ret"]))
        lens.append(int(result["ep_len"]))
        is_success = bool(result["ep_success"]) and not bool(result["ep_collision"])
        if is_success:
            success_hits += 1
            ep_min_dist = result.get("ep_min_dist", float("nan"))
            if np.isfinite(ep_min_dist):
                min_dists_success.append(float(ep_min_dist))
        if bool(result["ep_collision"]):
            collision_hits += 1
        if bool(result["ep_timeout"]):
            timeout_hits += 1
        if bool(result.get("ep_infeasible", False)):
            infeasible_hits += 1

    total = max(episodes, 1)
    return {
        "total_episodes": episodes,
        "success_rate": success_hits / total,
        "collision_rate": collision_hits / total,
        "timeout_rate": timeout_hits / total,
        "infeasible_rate": infeasible_hits / total,
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "avg_ep_len": float(np.mean(lens)) if lens else 0.0,
        "min_dist": float(np.mean(min_dists_success)) if min_dists_success else float("nan"),
    }


def _summarize_metric(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else 0.0
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci95 = float(1.96 * std / math.sqrt(n)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95": ci95,
        "min": float(np.min(arr)) if n else 0.0,
        "max": float(np.max(arr)) if n else 0.0,
    }


def _aggregate_per_seed(per_seed: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_episodes = int(sum(r["total_episodes"] for r in per_seed))
    denom = max(total_episodes, 1)

    totals = {
        "total_episodes": total_episodes,
        "success_rate": float(
            sum(float(r["success_rate"]) * int(r["total_episodes"]) for r in per_seed) / denom
        ),
        "collision_rate": float(
            sum(float(r["collision_rate"]) * int(r["total_episodes"]) for r in per_seed) / denom
        ),
        "timeout_rate": float(
            sum(float(r["timeout_rate"]) * int(r["total_episodes"]) for r in per_seed) / denom
        ),
        "infeasible_rate": float(
            sum(float(r["infeasible_rate"]) * int(r["total_episodes"]) for r in per_seed) / denom
        ),
    }

    min_dist_vals = [
        r["min_dist"] for r in per_seed if np.isfinite(r.get("min_dist", float("nan")))
    ]
    aggregate = {
        "success_rate": _summarize_metric([r["success_rate"] for r in per_seed]),
        "collision_rate": _summarize_metric([r["collision_rate"] for r in per_seed]),
        "timeout_rate": _summarize_metric([r["timeout_rate"] for r in per_seed]),
        "infeasible_rate": _summarize_metric([r["infeasible_rate"] for r in per_seed]),
        "avg_return": _summarize_metric([r["avg_return"] for r in per_seed]),
        "avg_ep_len": _summarize_metric([r["avg_ep_len"] for r in per_seed]),
        "min_dist": _summarize_metric(min_dist_vals),
    }
    return {"totals": totals, "aggregate": aggregate}


def _evaluate_over_seeds(actor, env, eval_seeds: List[int], episodes_per_seed: int):
    per_seed = []
    per_seed_full = []
    for idx, seed in enumerate(eval_seeds):
        metrics = evaluate_actor(actor, env, episodes=episodes_per_seed, base_seed=seed)
        per_seed.append(
            {
                "success_rate": metrics["success_rate"],
                "collision_rate": metrics["collision_rate"],
                "timeout_rate": metrics["timeout_rate"],
                "infeasible_rate": metrics["infeasible_rate"],
                "avg_return": metrics["avg_return"],
                "avg_ep_len": metrics["avg_ep_len"],
                "min_dist": metrics["min_dist"],
            }
        )
        per_seed_full.append(metrics)
        print(
            f"  [{idx + 1}/{len(eval_seeds)}] seed={seed} "
            f"succ={metrics['success_rate']:.3f} coll={metrics['collision_rate']:.3f} "
            f"infeas={metrics['infeasible_rate']:.3f} ret={metrics['avg_return']:.3f} "
            f"len={metrics['avg_ep_len']:.2f}"
        )

    return per_seed, _aggregate_per_seed(per_seed_full)


def _choose_best(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    valid = [r for r in results if "error" not in r]
    if not valid:
        return None

    def _key(row: Dict[str, Any]):
        agg = row["aggregate"]
        return (
            agg["success_rate"]["mean"],
            -agg["collision_rate"]["mean"],
            agg["avg_return"]["mean"],
        )

    return max(valid, key=_key)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actor_model",
        required=True,
        type=str,
        help="Saved training run directory, absolute or relative to trained_models/.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="diffcvarbfqp",
        help="RL policy method used by the saved actor checkpoints.",
    )
    parser.add_argument(
        "--episodes_per_seed",
        type=int,
        default=50,
        help="Episodes evaluated for each fixed seed.",
    )
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    if args.episodes_per_seed <= 0:
        raise ValueError("--episodes_per_seed must be > 0")

    run_dir = _resolve_run_dir(args.actor_model)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    cfg, payload, train_cfg_path = _load_run_config(run_dir)
    env_name = resolve_env_name(cfg, payload)
    checkpoints = discover_checkpoints(run_dir)
    if not checkpoints:
        raise RuntimeError(f"No actor checkpoints found in {run_dir}")

    device = resolve_device(args.device, default="cpu")
    eval_seeds = FIXED_EVAL_SEEDS

    print(
        f"[EvalConfig] method={args.method}, env={env_name}, robot={cfg.robot.type}, "
        f"obs_top_k={cfg.env.obs_top_k}, checkpoints={len(checkpoints)}, "
        f"seeds={len(eval_seeds)}, episodes/seed={args.episodes_per_seed}",
        flush=True,
    )

    env = build_env(env_name, render_mode=None, config=cfg)
    dump_test_config(
        run_dir,
        cfg,
        hyperparameters={
            "env_name": env_name,
            "method": args.method,
            "eval_seeds": eval_seeds,
            "episodes_per_seed": int(args.episodes_per_seed),
            "device": str(device),
        },
        extra={
            "script": "eval/eval.py",
            "config_source": train_cfg_path,
        },
    )

    results: List[Dict[str, Any]] = []
    try:
        for idx, ckpt in enumerate(checkpoints):
            ckpt_path = ckpt["path"]
            print(
                f"\n[{idx + 1}/{len(checkpoints)}] step={ckpt['step']} "
                f"file={os.path.basename(ckpt_path)}"
            )

            actor = build_eval_actor(cfg, args.method, env, ckpt_path, device, verbose=False)
            per_seed, agg = _evaluate_over_seeds(
                actor=actor,
                env=env,
                eval_seeds=eval_seeds,
                episodes_per_seed=args.episodes_per_seed,
            )

            row = {
                "checkpoint_path": ckpt_path,
                "checkpoint_file": os.path.basename(ckpt_path),
                "step": ckpt["step"],
                "totals": agg["totals"],
                "aggregate": agg["aggregate"],
                "per_seed": per_seed,
            }
            results.append(row)
            print(
                "  aggregate: "
                f"succ={row['aggregate']['success_rate']['mean']:.3f} "
                f"coll={row['aggregate']['collision_rate']['mean']:.3f} "
                f"ret={row['aggregate']['avg_return']['mean']:.3f}"
            )
    finally:
        env.close()

    best = _choose_best(results)
    summary = {
        "run_dir": run_dir,
        "method": args.method,
        "env_name": env_name,
        "config_source": train_cfg_path,
        "device": str(device),
        "eval_seeds": eval_seeds,
        "episodes_per_seed": int(args.episodes_per_seed),
        "num_checkpoints": len(checkpoints),
        "best_checkpoint": best,
        "checkpoints": results,
    }

    out_path = os.path.join(run_dir, EVAL_SUMMARY_FILENAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Saved: {out_path}")
    if best is not None:
        s = best["aggregate"]["success_rate"]["mean"]
        c = best["aggregate"]["collision_rate"]["mean"]
        r = best["aggregate"]["avg_return"]["mean"]
        print(f"Best checkpoint: {best['checkpoint_file']} (succ={s:.3f}, coll={c:.3f}, ret={r:.3f})")


if __name__ == "__main__":
    main()
