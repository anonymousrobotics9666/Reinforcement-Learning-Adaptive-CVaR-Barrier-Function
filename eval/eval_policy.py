"""
Minimal evaluation utilities for trained policies.
"""

import json
import os
from typing import List, Optional

import imageio
import numpy as np
import torch
from crowd_sim.utils import absolute_obs_to_relative, select_top_k_obs


def _infer_obs_top_k_from_policy(policy):
    """Walk the policy module tree to find the first Linear and infer obs_top_k.

    Network input width is 6 + obs_top_k * 6, so obs_top_k = (in_features - 6) // 6.
    Returns None if no Linear with valid width is found (callers fall back to no-slice).
    """
    if not hasattr(policy, "modules"):
        return None
    for m in policy.modules():
        if isinstance(m, torch.nn.Linear):
            in_features = int(m.in_features)
            if in_features >= 6 and (in_features - 6) % 6 == 0:
                return (in_features - 6) // 6
            return None
    return None


class RLEvalActorAdapter:
    """Deterministic RL action mapping consistent with PPO tanh-squash."""

    def __init__(self, actor, action_space, device, obs_top_k=None):
        self.actor = actor
        self.device = device
        low = np.asarray(action_space.low, dtype=np.float32)
        high = np.asarray(action_space.high, dtype=np.float32)
        self.scale = 0.5 * (high - low)
        self.bias = 0.5 * (high + low)
        self.deterministic = True
        # If not passed in, infer from the wrapped policy's first Linear layer width.
        if obs_top_k is None:
            obs_top_k = _infer_obs_top_k_from_policy(actor)
        self.obs_top_k = int(obs_top_k) if obs_top_k is not None else None

    def get_action(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mean = self.actor(obs_t)
        if torch.is_tensor(mean):
            mean = mean.detach().cpu().numpy()
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        action = self.bias + self.scale * np.tanh(mean)
        return action, 1.0

    def __getattr__(self, name):
        return getattr(self.actor, name)


def resolve_episode_seed(base_seed, episode_index):
    if base_seed is None:
        return None
    return int(base_seed) + int(episode_index)


def parse_episode_seeds_csv(raw) -> Optional[List[int]]:
    """
    Parse --episode_seeds "472,315,100" into [472, 315, 100].
    Returns None if empty / whitespace-only.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out if out else None


def _compute_action(actor, obs):
    obs = absolute_obs_to_relative(obs)
    top_k = getattr(actor, "obs_top_k", None)
    if top_k is not None:
        obs = select_top_k_obs(obs, top_k)
    if hasattr(actor, "deterministic"):
        actor.deterministic = True
    out = actor.get_action(obs)
    action = out[0] if isinstance(out, (tuple, list)) else out

    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(-1)


def _reset_actor_episode_cache(actor):
    targets = [actor]
    seen = set()
    while len(targets) > 0:
        obj = targets.pop()
        if obj is None:
            continue
        oid = id(obj)
        if oid in seen:
            continue
        seen.add(oid)

        if hasattr(obj, "_qp_warm_start"):
            obj._qp_warm_start = None
        if hasattr(obj, "_u_prev"):
            obj._u_prev = None
        if hasattr(obj, "infeasible"):
            obj.infeasible = False

        nested_actor = getattr(obj, "actor", None)
        if nested_actor is not None and nested_actor is not obj:
            targets.append(nested_actor)
        nested_module = getattr(obj, "module", None)
        if nested_module is not None and nested_module is not obj:
            targets.append(nested_module)


def _log_summary(ep_len, ep_ret, ep_num, ep_collision, ep_success):
    print(flush=True)
    print(f"-------------------- Episode #{ep_num} --------------------", flush=True)
    print(f"Episodic Length: {round(ep_len, 2)}", flush=True)
    print(f"Episodic Return: {round(ep_ret, 2)}", flush=True)
    print(f"Collision: {'YES' if ep_collision else 'NO'}", flush=True)
    success_flag = (ep_success and (not ep_collision))
    print(f"Success: {'YES' if success_flag else 'NO'}", flush=True)
    print("------------------------------------------------------", flush=True)
    print(flush=True)


def _current_min_distance(env):
    robot = getattr(env, "robot", None)
    base = env
    if robot is None and hasattr(env, "unwrapped"):
        base = env.unwrapped
        robot = getattr(base, "robot", None)

    robot_pos = getattr(base, "robot_pos", None)
    human_positions = getattr(base, "human_positions", None)
    human_radii = getattr(base, "human_radii", None)
    robot_radius = getattr(base, "robot_radius", None)
    if robot_pos is None or human_positions is None or human_radii is None or robot_radius is None:
        return np.nan

    human_positions = np.asarray(human_positions, dtype=float)
    human_radii = np.asarray(human_radii, dtype=float).reshape(-1)
    if human_positions.size == 0 or human_radii.size == 0:
        return np.nan

    center_dists = np.linalg.norm(human_positions - np.asarray(robot_pos, dtype=float), axis=1)
    clearances = center_dists - human_radii - float(robot_radius)
    if clearances.size == 0:
        return np.nan
    return float(np.min(clearances))


def _render_step(env, frames):
    if env.render_mode is None:
        return
    if env.render_mode == "rgb_array":
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    else:
        env.render()


def run_one_episode(
    actor,
    env,
    seed=None,
    reset_options=None,
    collect_frames=False,
):
    """Run a single episode and return step-level and episode-level results."""
    if seed is None and reset_options is None:
        obs, _ = env.reset()
    else:
        obs, _ = env.reset(seed=seed, options=reset_options)
    _reset_actor_episode_cache(actor)

    done = False
    ep_len = 0
    ep_ret = 0.0
    ep_collision = False
    ep_success = False
    ep_timeout = False
    ep_infeasible = False
    ep_min_dist = float('inf')

    frames = []

    while not done:
        ep_len += 1

        action = _compute_action(actor, obs)

        if bool(getattr(actor, "infeasible", False)):
            ep_infeasible = True

        obs, rew, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)

        step_clearance = _current_min_distance(env)
        if not np.isnan(step_clearance):
            ep_min_dist = min(ep_min_dist, step_clearance)

        if collect_frames:
            _render_step(env, frames)

        ep_collision = ep_collision or bool(info.get("is_collision", False))
        ep_success = ep_success or bool(info.get("is_success", False))
        ep_timeout = ep_timeout or bool(info.get("is_timeout", False))
        ep_ret += float(rew)

    if ep_min_dist == float('inf'):
        ep_min_dist = float('nan')

    return {
        "ep_len": ep_len,
        "ep_ret": ep_ret,
        "ep_collision": ep_collision,
        "ep_success": ep_success,
        "ep_timeout": ep_timeout,
        "ep_infeasible": ep_infeasible,
        "ep_min_dist": ep_min_dist,
        "frames": frames,
    }


def eval_policy(
    policy,
    env,
    max_episodes=50,
    save_path=None,
    base_seed=None,
    method=None,
    episode_seeds: Optional[List[int]] = None,
    save_gifs=True,
    max_gifs: Optional[int] = None,
):
    total_episodes = 0
    success_count = 0
    collision_count = 0
    infeasible_count = 0

    actor = policy
   
    mode = (method or "").lower()
    if episode_seeds is not None:
        num_episodes = len(episode_seeds)
    else:
        num_episodes = max_episodes

    for ep_num in range(num_episodes):
        if episode_seeds is not None:
            seed = episode_seeds[ep_num]
        else:
            seed = resolve_episode_seed(base_seed, ep_num)
        result = run_one_episode(
            actor,
            env,
            seed=seed,
            reset_options=None,
            collect_frames=True,
        )
        ep_len = result["ep_len"]
        ep_ret = result["ep_ret"]
        ep_collision = result["ep_collision"]
        ep_success = result["ep_success"]
        ep_infeasible = result["ep_infeasible"]
        frames = result["frames"]

        _log_summary(ep_len, ep_ret, ep_num, ep_collision, ep_success)

        success_flag = ep_success and (not ep_collision)
        should_save_gif = bool(save_gifs) and len(frames) > 0 and save_path
        if max_gifs is not None and total_episodes >= int(max_gifs):
            should_save_gif = False
        if should_save_gif:
            os.makedirs(save_path, exist_ok=True)
            succ_bit = 1 if success_flag else 0
            coll_bit = 1 if ep_collision else 0
            seed_tag = f"_seed_{seed}" if seed is not None else ""
            gif_name = f"eval_ep_{ep_num}{seed_tag}_succ_{succ_bit}_coll_{coll_bit}.gif"
            full_path = os.path.join(save_path, gif_name)

            imageio.mimsave(full_path, frames, fps=10)
            print(f"Saved evaluation animation to {full_path}")

        total_episodes += 1
        if ep_success and not ep_collision:
            success_count += 1
        if ep_collision:
            collision_count += 1
        if ep_infeasible:
            infeasible_count += 1

    print("\n\n-------------------- Evaluation Summary --------------------")
    print(f"Total Episodes: {total_episodes}")
    if total_episodes > 0:
        success_rate = success_count / total_episodes
        collision_rate = collision_count / total_episodes
        infeasible_rate = infeasible_count / total_episodes
        print(f"Success Rate: {success_rate * 100:.2f}%")
        print(f"Collision Rate: {collision_rate * 100:.2f}%")
        print(f"Infeasible Rate: {infeasible_rate * 100:.2f}%")
    else:
        success_rate = None
        collision_rate = None
        infeasible_rate = None
        print("Success Rate: N/A")
    print("------------------------------------------------------------")

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        summary = {
            "total_episodes": total_episodes,
            "success_count": success_count,
            "collision_count": collision_count,
            "success_rate": success_rate,
            "collision_rate": collision_rate,
            "infeasible_count": infeasible_count,
            "infeasible_rate": infeasible_rate,
        }

        log_payload = {
            "config": {
                "policy_class": actor.__class__.__name__,
                "render_mode": getattr(env, "render_mode", None),
                "max_episodes": num_episodes,
                "base_seed": base_seed,
                "episode_seeds": episode_seeds,
                "method": mode,
                "save_gifs": bool(save_gifs),
                "max_gifs": max_gifs,
            },
            "results": summary,
        }
        with open(os.path.join(save_path, "eval_log.json"), "w", encoding="utf-8") as f:
            json.dump(log_payload, f, indent=2)
