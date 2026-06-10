import json
import os
from typing import Any, Dict, Optional

import numpy as np
from config.config import Config
 

def is_absolute_obs_dim(obs_dim: int) -> bool:
    dim = int(obs_dim)
    return dim >= 8 and (dim - 8) % 6 == 0


def relative_obs_dim_from_env_dim(obs_dim: int) -> int:
    dim = int(obs_dim)
    if is_absolute_obs_dim(dim):
        # abs format: 8 + K*6 -> relative policy format: 6 + K*6
        return dim - 2
    return dim


def absolute_obs_to_relative(obs):
    """
    Convert observation from absolute format to relative policy format.

    Absolute format (1D):
      [rx, ry, gx, gy, rvx, rvy, rtheta, rr, (hx, hy, hvx, hvy, hr, mask)*K]

    Relative policy format (1D):
      [goal_rel_x, goal_rel_y, rvx, rvy, rtheta, rr, (rel_x, rel_y, hvx, hvy, hr, mask)*K]
    """
    x = np.asarray(obs, dtype=np.float32).reshape(-1)

    # Pass through already-relative observations.
    if x.size >= 6 and (x.size - 6) % 6 == 0:
        if not (x.size >= 8 and (x.size - 8) % 6 == 0):
            return x
        # If both checks pass (unlikely/ambiguous), prefer absolute interpretation.

    if not (x.size >= 8 and (x.size - 8) % 6 == 0):
        raise ValueError(f"Unsupported observation length for abs->rel conversion: {x.size}")

    k = (x.size - 8) // 6
    out = np.zeros((6 + 6 * k,), dtype=np.float32)

    rx, ry, gx, gy, rvx, rvy, rtheta, rr = x[:8]
    out[0] = rx - gx
    out[1] = ry - gy
    out[2] = rvx
    out[3] = rvy
    out[4] = rtheta
    out[5] = rr

    if k > 0:
        blocks = x[8:].reshape(k, 6)
        out_blocks = np.zeros((k, 6), dtype=np.float32)
        out_blocks[:, 0] = rx - blocks[:, 0]
        out_blocks[:, 1] = ry - blocks[:, 1]
        out_blocks[:, 2:6] = blocks[:, 2:6]
        out[6:] = out_blocks.reshape(-1)

    return out


def absolute_obs_batch_to_relative(obs_batch):
    """
    Batch version of absolute_obs_to_relative.
    Input can be shape (N, D) absolute observations or already-relative (N, D_rel).
    """
    arr = np.asarray(obs_batch, dtype=np.float32)

    if arr.ndim == 1:
        return absolute_obs_to_relative(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected obs batch with ndim 1 or 2, got shape {arr.shape}")

    n, d = arr.shape
    if d >= 8 and (d - 8) % 6 == 0:
        k = (d - 8) // 6
        out = np.zeros((n, 6 + 6 * k), dtype=np.float32)

        rx = arr[:, 0:1]
        ry = arr[:, 1:2]
        gx = arr[:, 2:3]
        gy = arr[:, 3:4]

        out[:, 0:1] = rx - gx
        out[:, 1:2] = ry - gy
        out[:, 2:6] = arr[:, 4:8]

        if k > 0:
            blocks = arr[:, 8:].reshape(n, k, 6)
            out_blocks = np.zeros((n, k, 6), dtype=np.float32)
            out_blocks[:, :, 0] = rx - blocks[:, :, 0]
            out_blocks[:, :, 1] = ry - blocks[:, :, 1]
            out_blocks[:, :, 2:6] = blocks[:, :, 2:6]
            out[:, 6:] = out_blocks.reshape(n, 6 * k)

        return out

    if d >= 6 and (d - 6) % 6 == 0:
        return arr

    raise ValueError(f"Unsupported batch observation width for abs->rel conversion: {d}")


def select_top_k_obs(rel_obs, top_k: int):
    """Slice the first ``top_k`` human blocks from a relative-format obs.

    The env emits all humans (sorted nearest-first) so a simple prefix slice
    is the correct top-K selection. Works on 1D or 2D arrays.

    Input shape:  (..., 6 + N_all * 6)
    Output shape: (..., 6 + top_k * 6)
    """
    k = int(top_k)
    arr = np.asarray(rel_obs)
    width = 6 + k * 6
    if arr.shape[-1] < width:
        raise ValueError(
            f"obs has {arr.shape[-1]} cols, need >= {width} for top_k={k}"
        )
    if arr.shape[-1] == width:
        return arr
    return arr[..., :width]


def parse_obstacles(obs):
    """
    Parse obstacle blocks from observation.
    Supports:
    1) New format: [robot(6), K * (rel_x, rel_y, vx, vy, radius, mask)]
    2) Legacy format: [robot(6), rel_x, rel_y, vx, vy, radius]
    Returns:
        rels: (N, 2), vels: (N, 2), radii: (N,), masks: (N,)
    """
    obs = np.asarray(obs, dtype=np.float64).reshape(-1)

    # New K-obstacle format
    if obs.size >= 12 and (obs.size - 6) % 6 == 0:
        blocks = obs[6:].reshape(-1, 6)
        rels = blocks[:, 0:2].astype(np.float64)
        vels = blocks[:, 2:4].astype(np.float64)
        radii = blocks[:, 4].astype(np.float64)
        masks = np.clip(blocks[:, 5].astype(np.float64), 0.0, 1.0)
        return (
            rels,
            vels,
            radii,
            masks,
        )

    # Single-obstacle format
    if obs.size >= 11:
        return (
            obs[6:8].reshape(1, 2).astype(np.float64),
            obs[8:10].reshape(1, 2).astype(np.float64),
            np.array([float(obs[10])], dtype=np.float64),
            np.ones((1,), dtype=np.float64),
        )

    # No obstacle info in observation
    return (
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0, 2), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
    )


def sample_point_in_disk(rng, center, radius, arena_size=None, max_tries=256):
    center = np.asarray(center, dtype=float)
    for _ in range(max_tries):
        rr = radius * np.sqrt(rng.uniform(0.0, 1.0))
        theta = rng.uniform(0.0, 2.0 * np.pi)
        p = center + rr * np.array([np.cos(theta), np.sin(theta)], dtype=float)
        if arena_size is None:
            return p
        if (-arena_size <= p[0] <= arena_size) and (-arena_size <= p[1] <= arena_size):
            return p
    if arena_size is None:
        return center.copy()
    return np.clip(center, -arena_size, arena_size)


def build_env(env_name: str, render_mode: str, config: Config):
    from crowd_sim.env.social_nav import SocialNav
    from crowd_sim.env.social_nav_var_num import SocialNavVarNum

    if env_name == "social_nav":
        return SocialNav(render_mode=render_mode, config_file=config)
    if env_name == "social_nav_var_num":
        return SocialNavVarNum(render_mode=render_mode, config_file=config)
    raise ValueError(f"Unknown env: {env_name}")


def load_json_dict(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def apply_config_snapshot(config: Config, cfg_dict: Dict[str, Any]) -> None:
    for section_name in ["env", "human", "robot", "controller", "reward"]:
        section_data = cfg_dict.get(section_name)
        if not isinstance(section_data, dict):
            continue
        section_obj = getattr(config, section_name, None)
        if section_obj is None:
            continue
        for key, value in section_data.items():
            section_obj[key] = value

    # Backward-compatible alias: older train_config.json snapshots used env.max_obstacles_obs.
    # Map it to the new env.obs_top_k so downstream code only reads one key.
    env_section = getattr(config, "env", None)
    if env_section is not None and "obs_top_k" not in env_section and "max_obstacles_obs" in env_section:
        env_section["obs_top_k"] = env_section["max_obstacles_obs"]


def load_train_config_snapshot(
    config: Config,
    train_cfg_path: Optional[str],
    use_current_config: bool = False,
) -> Dict[str, Any]:
    if use_current_config or not train_cfg_path:
        return {}

    payload = load_json_dict(train_cfg_path)
    cfg_dict = payload.get("config")
    if isinstance(cfg_dict, dict):
        apply_config_snapshot(config, cfg_dict)
    return payload


def resolve_env_name(config: Config, payload: Optional[Dict[str, Any]] = None) -> str:
    if isinstance(payload, dict):
        hyper = payload.get("hyperparameters", {})
        env_name = hyper.get("env_name") if isinstance(hyper, dict) else None
        if isinstance(env_name, str) and env_name:
            return env_name
    return str(config.env.get("name", "social_nav_var_num"))

def to_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    type_module = type(obj).__module__
    type_name = type(obj).__name__
    if type_module.startswith("torch"):
        if hasattr(obj, "detach") and hasattr(obj, "cpu"):
            return obj.detach().cpu().tolist()
        if type_name == "device":
            return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    return obj


def dump_test_config(test_save_dir, config, hyperparameters=None, extra=None):
    os.makedirs(test_save_dir, exist_ok=True)
    dst = os.path.join(test_save_dir, "test_config.json")
    config_dict = {
        "config": {
            "env": to_jsonable(dict(config.env)),
            "human": to_jsonable(dict(config.human)),
            "robot": to_jsonable(dict(config.robot)),
            "controller": to_jsonable(dict(config.controller)),
            "reward": to_jsonable(dict(config.reward)),
        },
    }
    if hyperparameters is not None:
        config_dict["hyperparameters"] = to_jsonable(hyperparameters)
    if extra is not None:
        config_dict["extra"] = to_jsonable(extra)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)
    print(f"Saved test config snapshot: {dst}", flush=True)


def dump_train_config(save_dir, args, config, hyperparameters=None, extra=None):
    os.makedirs(save_dir, exist_ok=True)

    payload = {
        "args": to_jsonable(vars(args) if hasattr(args, "__dict__") else args),
        "config": {
            "env": to_jsonable(dict(config.env)),
            "human": to_jsonable(dict(config.human)),
            "robot": to_jsonable(dict(config.robot)),
            "controller": to_jsonable(dict(config.controller)),
            "reward": to_jsonable(dict(config.reward)),
        },
    }

    if hyperparameters is not None:
        payload["hyperparameters"] = to_jsonable(hyperparameters)
    if extra is not None:
        payload["extra"] = to_jsonable(extra)

    with open(os.path.join(save_dir, "train_config.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
