"""Factory helpers for RL policy classes."""


def get_rl_policy_class(method: str):
    method = (method or "").strip().lower()
    if method == "vanilla_ppo":
        from rl.network import FCNet
        return FCNet
    if method == "diffcvarbfqp":
        from rl.diff_cvar_bf_qp import DiffCVaRBFQP
        return DiffCVaRBFQP
    raise ValueError(f"Unknown method {method}")
