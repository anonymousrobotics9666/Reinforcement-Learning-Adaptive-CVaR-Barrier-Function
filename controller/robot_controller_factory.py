"""Factory for non-RL robot controllers."""

from controller.cbf_qp import CBFQPController
from controller.nominal_controller import NominalController


def build_robot_controller(method, config, env):
    if method == "nominal":
        return NominalController(config_file=config, env=env)

    if method == "cbfqp":
        return CBFQPController(config_file=config, env=env)

    raise ValueError(f"Unknown controller method: {method}")
