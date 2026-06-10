"""Factory helpers for RL policy classes."""


def get_rl_policy_class(method: str):
    method = (method or "").strip().lower()
    if method == "vanilla_ppo":
        from rl.network import FCNet
        return FCNet
    if method == "diffcvarbfqp":
        from rl.diff_cvar_bf_qp import DiffCVaRBFQP
        return DiffCVaRBFQP
    if method == "social_force":
        raise ValueError(f"method '{method}' is supported in main_opt.py, not main_vec.py")
    raise ValueError(f"Unknown method {method}")
