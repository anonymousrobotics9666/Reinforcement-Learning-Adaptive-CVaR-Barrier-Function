"""Policy construction helpers shared by single and batch evaluation."""

import torch

from crowd_nav.rl_policy_factory import get_rl_policy_class
from eval.eval_policy import RLEvalActorAdapter
from rl.run_utils import get_policy_kwargs


def policy_kwargs_from_config(config, method):
    kwargs = {
        "robot_type": config.robot_params["type"],
        "safe_dist": (
            config.controller_params["safety_margin"]
            + config.human_params["radius"]
            + config.robot_params["radius"]
        ),
        "alpha": config.controller_params["cbf_alpha"],
        "beta": config.controller_params["cvar_beta"],
        "vmax": config.robot_params["vmax"],
        "amax": config.robot_params["amax"],
        "omega_max": config.robot_params["omega_max"],
    }
    kwargs.update(get_policy_kwargs(config, method))
    return kwargs


def obs_top_k_from_config(config):
    return int(config.env_params.get("obs_top_k", config.env_params.get("max_obstacles_obs", 1)))


def load_actor_state_dict(checkpoint_path, device):
    state = torch.load(checkpoint_path, map_location=device)
    if not (isinstance(state, dict) and "model" in state):
        raise ValueError(
            f"Expected diff_opt-style checkpoint with a top-level 'model' key: {checkpoint_path}"
        )
    return state["model"]


def build_eval_actor(config, method, env, actor_model, device, verbose=True):
    obs_top_k = obs_top_k_from_config(config)
    obs_dim = 6 + obs_top_k * 6
    act_dim = env.action_space.shape[0]
    policy_kwargs = policy_kwargs_from_config(config, method)

    PolicyClass = get_rl_policy_class(method)
    if verbose:
        print(f"Algorithm: {method}, Policy: {PolicyClass.__name__}", flush=True)
        print(f"Policy Args: {policy_kwargs}", flush=True)

    policy = PolicyClass(obs_dim, act_dim, **policy_kwargs).to(device)
    policy.load_state_dict(load_actor_state_dict(actor_model, device))
    policy.eval()
    return RLEvalActorAdapter(policy, env.action_space, device, obs_top_k=obs_top_k)
