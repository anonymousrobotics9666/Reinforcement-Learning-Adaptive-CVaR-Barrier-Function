from config.config import Config
from model.actor_critic import ActorCritic
from trainer.utils import get_policy_kwargs


def get_model_class(method: str):
    method = (method or "").strip().lower()
    if method == "vanilla_ppo":
        from model.ppo_base import FCNet
        return FCNet
    if method == "diffcvarbfqp":
        from model.diff_cvar import DiffCVaRBFQP
        return DiffCVaRBFQP
    raise ValueError(f"Unknown method {method}")


def _actor_kwargs(cfg):
    config = Config(cfg)
    method = str(cfg.method)
    safe_dist = (
        config.controller_params["safety_margin"]
        + config.human_params["radius"]
        + config.robot_params["radius"]
    )
    kwargs = {
        "robot_type": config.robot_params["type"],
        "safe_dist": safe_dist,
        "alpha": config.controller_params["cbf_alpha"],
        "beta": config.controller_params["cvar_beta"],
        "vmax": config.robot_params["vmax"],
        "amax": config.robot_params["amax"],
        "omega_max": config.robot_params["omega_max"],
    }
    kwargs.update(get_policy_kwargs(config, method))
    return kwargs


def build_model(config, obs_dim, act_dim, action_low=None, action_high=None):
    if action_low is None:
        action_low = [-1.0] * int(act_dim)
    if action_high is None:
        action_high = [1.0] * int(act_dim)

    method = str(config.method)
    policy_class = get_model_class(method)
    action_std_init = float(config.get("action_std_init", 0.5))
    return ActorCritic(
        policy_class=policy_class,
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
        actor_kwargs=_actor_kwargs(config),
        action_low=action_low,
        action_high=action_high,
        action_std_init=action_std_init,
    )
