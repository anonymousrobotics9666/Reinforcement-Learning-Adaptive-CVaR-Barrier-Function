"""Shared training framework utilities."""

import os
import shutil
from datetime import datetime

import wandb
from gymnasium.vector import AsyncVectorEnv
from omegaconf import OmegaConf

from config.config import Config
from crowd_sim.utils import build_env, resolve_env_name
from trainer.utils import get_policy_kwargs, resolve_device, set_global_seeds


ROBOT_SHORT = {
    "single_integrator": "si",
    "unicycle": "uni",
    "unicycle_dynamic": "unid",
}


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.config_dict = OmegaConf.to_container(cfg, resolve=True)
        self.config = Config(cfg)
        self.device = resolve_device(cfg.device, default="cuda")
        self.env_name = resolve_env_name(self.config)
        self.train_envs = None
        self.eval_env = None

        set_global_seeds(int(cfg.seed))
        self.run_name = self.build_run_name()
        self.save_dir = self.prepare_save_dir()
        self.hyperparameters = self.build_hyperparameters()

        os.makedirs(self.save_dir, exist_ok=True)
        OmegaConf.save(config=self.cfg, f=os.path.join(self.save_dir, "config.yaml"), resolve=True)
        print(f"Models will be saved to: {self.save_dir}", flush=True)
        self.setup_wandb()

    def build_run_name(self):
        if self.cfg.run_name is not None and str(self.cfg.run_name):
            return (
                f"{self.cfg.run_name}-{self.cfg.method}"
                f"-bs{int(self.cfg.timesteps_per_batch)}"
                f"-ep{int(self.cfg.n_updates_per_iteration)}"
                f"-lr{float(self.cfg.lr):.1e}"
                + (f"-ent{self.cfg.ent_coef}" if float(self.cfg.ent_coef) > 0.0 else "")
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        robot_short = ROBOT_SHORT.get(self.config.robot_params["type"], self.config.robot_params["type"])
        name = f"{timestamp}_{robot_short}_{self.cfg.method}_seed{int(self.cfg.seed)}"
        if int(self.cfg.total_timesteps) != 20_000_000:
            name += f"_steps{int(self.cfg.total_timesteps) // 1_000_000}M"
        if abs(float(self.cfg.lr) - 1e-4) > 1e-9:
            lr_str = f"{float(self.cfg.lr):.0e}".replace("e-0", "e-").replace("e+0", "e")
            name += f"_lr{lr_str}"
        if int(self.config.human.num_humans) != 20:
            name += f"_humans{int(self.config.human.num_humans)}"
        return self.ensure_unique_run_name(self.train_root(), name)

    def train_root(self):
        return str(self.cfg.get("save_dir") or os.path.join("trained_models", self.cfg.model_folder))

    @staticmethod
    def ensure_unique_run_name(base_root, run_name):
        candidate = run_name
        idx = 1
        while os.path.exists(os.path.join(base_root, candidate)):
            candidate = f"{run_name}_{idx}"
            idx += 1
        return candidate

    def prepare_save_dir(self):
        save_dir = os.path.join(self.train_root(), self.run_name)
        if os.path.exists(save_dir):
            if bool(self.cfg.get("overwrite", False)):
                shutil.rmtree(save_dir)
            else:
                raise ValueError(f"Save directory {save_dir} already exists; set overwrite=true to replace it")
        return save_dir

    def build_hyperparameters(self):
        safe_dist = (
            self.config.controller_params["safety_margin"]
            + self.config.human_params["radius"]
            + self.config.robot_params["radius"]
        )
        return {
            "model_config": self.cfg,
            "max_timesteps_per_episode": self.config.env.max_steps,
            "env_name": self.env_name,
            "seed": self.cfg.seed,
            "save_dir": self.save_dir,
            "device": self.device,
            "safe_dist": safe_dist,
            "alpha": self.config.controller_params["cbf_alpha"],
            "beta": self.config.controller_params["cvar_beta"],
            "cbf_alpha": self.config.controller_params["cbf_alpha"],
            "cvar_beta": self.config.controller_params["cvar_beta"],
            "robot_type": self.config.robot_params["type"],
            "vmax": self.config.robot_params["vmax"],
            "amax": self.config.robot_params["amax"],
            "omega_max": self.config.robot_params["omega_max"],
            "obs_top_k": int(
                self.config.env_params.get("obs_top_k", self.config.env_params.get("max_obstacles_obs", 1))
            ),
            "policy_kwargs": get_policy_kwargs(self.config, self.cfg.method),
            "n_updates_per_iteration": self.cfg.n_updates_per_iteration,
            "timesteps_per_batch": self.cfg.timesteps_per_batch,
            "num_minibatches": self.cfg.num_minibatches,
            "clip": self.cfg.clip,
            "lr": self.cfg.lr,
            "gamma": self.cfg.gamma,
            "lam": self.cfg.lam,
            "ent_coef": self.cfg.ent_coef,
            "target_kl": self.cfg.target_kl,
            "max_grad_norm": self.cfg.max_grad_norm,
            "action_std_init": self.cfg.action_std_init,
            "eval_interval": self.cfg.eval_interval,
            "eval_freq_timesteps": self.cfg.eval_freq_timesteps,
            "eval_episodes": self.cfg.eval_episodes,
            "max_checkpoints": self.cfg.max_checkpoints,
            "wandb_interval": self.cfg.wandb_interval,
        }

    def setup_wandb(self):
        run = wandb.init(
            project=str(self.cfg.wandb_project),
            entity=str(self.cfg.wandb_entity),
            name=self.run_name,
            config=self.config_dict,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        return run

    def make_env_fn(self, seed_offset):
        config = self.config
        env_name = self.env_name
        seed = int(self.cfg.seed) + int(seed_offset)

        def _init():
            env = build_env(env_name, render_mode=None, config=config)
            env.reset(seed=seed)
            return env

        return _init

    def setup_env(self):
        num_envs = int(self.cfg.num_envs)
        if num_envs <= 0:
            raise ValueError("num_envs must be > 0")
        self.train_envs = AsyncVectorEnv([self.make_env_fn(i) for i in range(num_envs)])
        if self.cfg.eval_interval > 0 or self.cfg.eval_freq_timesteps > 0:
            self.eval_env = build_env(self.env_name, render_mode=None, config=self.config)
            self.hyperparameters["eval_env"] = self.eval_env

    def close(self):
        if self.train_envs is not None:
            self.train_envs.close()
            self.train_envs = None
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None
        if wandb.run is not None:
            wandb.finish()
