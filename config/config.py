"""Hydra-aware Config adapter.

The env / human / robot / controller / reward defaults live in YAML
(`config/env/env.yaml` and `config/env/robot/<type>.yaml`). This adapter
rebuilds the object-style ``Config`` / ``BaseConfig`` interface expected by
the simulator, controllers, and training code.
"""

from typing import Optional

from omegaconf import DictConfig, OmegaConf


class BaseConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def to_dict(self):
        return dict(self)


def _section(cfg_section) -> BaseConfig:
    container = OmegaConf.to_container(cfg_section, resolve=True) or {}
    return BaseConfig(**container)


class Config:
    def __init__(self, hydra_cfg: Optional[DictConfig] = None):
        if hydra_cfg is None:
            raise ValueError(
                "Config now requires a Hydra DictConfig. "
                "Call as Config(cfg) inside a @hydra.main function."
            )
        self.env = _section(hydra_cfg.env)
        self.human = _section(hydra_cfg.human)
        self.robot = _section(hydra_cfg.robot)
        self.controller = _section(hydra_cfg.controller)
        self.reward = _section(hydra_cfg.reward)

        if "obs_top_k" not in self.env and "max_obstacles_obs" in self.env:
            self.env["obs_top_k"] = self.env["max_obstacles_obs"]

        # Snake-case aliases used throughout crowd_sim, crowd_nav, controller, main_vec.
        self.env_params = self.env
        self.human_params = self.human
        self.robot_params = self.robot
        self.controller_params = self.controller
        self.reward_params = self.reward
