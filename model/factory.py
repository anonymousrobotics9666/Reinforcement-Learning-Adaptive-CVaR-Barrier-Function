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
    method = str(cfg.model.type)
    safe_dist = (
        cfg.env.controller["safety_margin"]
        + cfg.env.humans["radius"]
        + cfg.robot["radius"]
    )
    kwargs = {
        "robot_type": cfg.robot["type"],
        "safe_dist": safe_dist,
        "alpha": cfg.env.controller["cbf_alpha"],
        "beta": cfg.env.controller["cvar_beta"],
        "vmax": cfg.robot["vmax"],
        "amax": cfg.robot["amax"],
        "omega_max": cfg.robot["omega_max"],
    }
    kwargs.update(get_policy_kwargs(cfg, method))
    return kwargs


def build_model(config, obs_dim, act_dim, action_low=None, action_high=None):
    if action_low is None:
        action_low = [-1.0] * int(act_dim)
    if action_high is None:
        action_high = [1.0] * int(act_dim)

    method = str(config.model.type)
    policy_class = get_model_class(method)
    action_std_init = float(config.model.get("action_std_init", 0.5))
    return ActorCritic(
        policy_class=policy_class,
        obs_dim=int(obs_dim),
        act_dim=int(act_dim),
        actor_kwargs=_actor_kwargs(config),
        action_low=action_low,
        action_high=action_high,
        action_std_init=action_std_init,
    )
