"""Train PPO policies with vectorized environments."""

import os
import shutil
from datetime import datetime

import hydra
import torch
import wandb
from gymnasium.vector import AsyncVectorEnv
from omegaconf import DictConfig, OmegaConf

from config.config import Config
from crowd_nav.rl_policy_factory import get_rl_policy_class
from crowd_sim.utils import build_env, dump_train_config, resolve_env_name
from rl.run_utils import get_policy_kwargs, resolve_device, set_global_seeds
from rl.vec_ppo import VecPPO


def make_env_fn(config, env_name):
    def _init():
        return build_env(env_name, render_mode=None, config=config)

    return _init


def build_base_hyperparameters(args, config, env_name, save_dir, device):
    return {
        "max_timesteps_per_episode": config.env.max_steps,
        "env_name": env_name,
        "save_after_timesteps": args.save_after_timesteps,
        "save_freq": args.save_freq,
        "seed": args.seed,
        "save_dir": save_dir,
        "device": device,
        "safe_dist": (
            config.controller_params["safety_margin"]
            + config.human_params["radius"]
            + config.robot_params["radius"]
        ),
        "alpha": config.controller_params["cbf_alpha"],
        "beta": config.controller_params["cvar_beta"],
        "cbf_alpha": config.controller_params["cbf_alpha"],
        "cvar_beta": config.controller_params["cvar_beta"],
        "robot_type": config.robot_params["type"],
        "vmax": config.robot_params["vmax"],
        "amax": config.robot_params["amax"],
        "omega_max": config.robot_params["omega_max"],
        "obs_top_k": int(
            config.env_params.get("obs_top_k", config.env_params.get("max_obstacles_obs", 1))
        ),
        "policy_kwargs": get_policy_kwargs(config, args.method),
    }


def build_ppo_hyperparameters(args, base_hyperparameters):
    hyperparameters = dict(base_hyperparameters)
    hyperparameters.update({
        "n_updates_per_iteration": args.n_updates_per_iteration,
        "timesteps_per_batch": args.timesteps_per_batch,
        "num_minibatches": args.num_minibatches,
        "clip": args.clip,
        "lr": args.lr,
        "gamma": args.gamma,
        "lam": args.lam,
        "ent_coef": args.ent_coef,
        "target_kl": args.target_kl,
        "max_grad_norm": args.max_grad_norm,
        "action_std_init": args.action_std_init,
        "eval_interval": args.eval_interval,
        "eval_freq_timesteps": args.eval_freq_timesteps,
        "eval_episodes": args.eval_episodes,
        "max_checkpoints": args.max_checkpoints,
        "wandb_interval": args.wandb_interval,
    })
    return hyperparameters


ROBOT_SHORT = {
    "single_integrator": "si",
    "unicycle": "uni",
    "unicycle_dynamic": "unid",
}


def derive_train_exp_name(
    timestamp,
    robot_type,
    method,
    actor_model,
    seed,
    total_timesteps,
    lr,
    num_humans,
):
    robot_short = ROBOT_SHORT.get(robot_type, robot_type)

    if actor_model:
        actor_file = os.path.basename(actor_model)
        parent_name = os.path.basename(os.path.dirname(os.path.abspath(actor_model)))
        if actor_file.startswith("bc_actor") and parent_name:
            base = parent_name[:-3] if parent_name.endswith("_bc") else parent_name
            return f"{base}_ft_seed{seed}"

    name = f"{timestamp}_{robot_short}_{method}_seed{seed}"
    if int(total_timesteps) != 20_000_000:
        name += f"_steps{int(total_timesteps) // 1_000_000}M"
    if abs(float(lr) - 1e-4) > 1e-9:
        lr_str = f"{float(lr):.0e}".replace("e-0", "e-").replace("e+0", "e")
        name += f"_lr{lr_str}"
    if int(num_humans) != 20:
        name += f"_humans{int(num_humans)}"
    return name


def ensure_unique_exp_name(base_root, exp_name):
    candidate = exp_name
    idx = 1
    while os.path.exists(os.path.join(base_root, candidate)):
        candidate = f"{exp_name}_{idx}"
        idx += 1
    return candidate


def build_config_run_name(args):
    run_name = args.get("run_name")
    if run_name is None or not str(run_name):
        return None

    name = (
        f"{run_name}-{args.method}"
        f"-bs{int(args.timesteps_per_batch)}"
        f"-ep{int(args.n_updates_per_iteration)}"
        f"-lr{float(args.lr):.1e}"
    )
    if float(args.ent_coef) > 0.0:
        name += f"-ent{args.ent_coef}"
    return name


def setup_wandb(args, run_name, config_dict):
    wandb_cfg = getattr(args, "wandb", {}) or {}
    if not bool(wandb_cfg.get("enabled", True)):
        return None

    project = str(args.get("wandb_project") or wandb_cfg.get("project"))
    entity = str(args.get("wandb_entity") or wandb_cfg.get("entity"))
    run = wandb.init(
        project=project,
        entity=entity,
        name=run_name,
        config=config_dict,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")
    return run


def load_warm_start(model, actor_model, critic_model, device):
    loaded_parts = []
    if actor_model:
        if not os.path.exists(actor_model):
            raise FileNotFoundError(f"Actor checkpoint not found: {actor_model}")
        model.actor.load_state_dict(torch.load(actor_model, map_location=device))
        loaded_parts.append(f"actor={actor_model}")

    if critic_model:
        if not os.path.exists(critic_model):
            raise FileNotFoundError(f"Critic checkpoint not found: {critic_model}")
        model.critic.load_state_dict(torch.load(critic_model, map_location=device))
        loaded_parts.append(f"critic={critic_model}")

    if loaded_parts:
        print(f"Warm start loaded: {', '.join(loaded_parts)}", flush=True)
    else:
        print("Training from scratch.", flush=True)


def train(env, num_envs, hyperparameters, actor_model, critic_model, method, total_timesteps):
    print(f"Training with {num_envs} vectorized environments", flush=True)
    PolicyClass = get_rl_policy_class(method)
    print(f"Algorithm: {method}, Policy: {PolicyClass.__name__}", flush=True)

    seeds = [hyperparameters["seed"] + i for i in range(num_envs)]
    env.reset(seed=seeds)

    model = VecPPO(policy_class=PolicyClass, env=env, num_envs=num_envs, **hyperparameters)
    load_warm_start(
        model=model,
        actor_model=actor_model,
        critic_model=critic_model,
        device=hyperparameters["device"],
    )
    model.learn(total_timesteps=total_timesteps)


@hydra.main(config_path="config", config_name="trainer/train", version_base=None)
def main(cfg: DictConfig):
    args = cfg
    args_dict = OmegaConf.to_container(cfg, resolve=True)
    set_global_seeds(args.seed)

    device = resolve_device(args.device, default="cuda")
    config = Config(cfg)
    env_name = resolve_env_name(config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train_root = str(args.get("save_dir") or os.path.join(".", "trained_models", args.model_folder))
    exp_name = build_config_run_name(args)
    if exp_name is None:
        exp_name = derive_train_exp_name(
            timestamp=timestamp,
            robot_type=config.robot_params["type"],
            method=args.method,
            actor_model=args.actor_model,
            seed=int(args.seed),
            total_timesteps=int(args.total_timesteps),
            lr=float(args.lr),
            num_humans=int(config.human.num_humans),
        )
        exp_name = ensure_unique_exp_name(train_root, exp_name)
    save_dir = os.path.join(train_root, exp_name)
    if os.path.exists(save_dir):
        if bool(args.get("overwrite", False)):
            shutil.rmtree(save_dir)
        else:
            raise ValueError(f"Save directory {save_dir} already exists; set overwrite=true to replace it")

    base_hyperparameters = build_base_hyperparameters(
        args=args,
        config=config,
        env_name=env_name,
        save_dir=save_dir,
        device=device,
    )
    hyperparameters = build_ppo_hyperparameters(args, base_hyperparameters)

    os.makedirs(save_dir, exist_ok=True)
    dump_train_config(
        save_dir,
        args_dict,
        config,
        hyperparameters,
        extra={"seed": args.seed, "method": args.method},
    )
    print(f"Models will be saved to: {save_dir}", flush=True)

    setup_wandb(args, exp_name, args_dict)

    num_envs = int(args.num_envs)
    if num_envs <= 0:
        raise ValueError("num_envs must be > 0")

    env_fns = [make_env_fn(config, env_name) for _ in range(num_envs)]
    vec_env = AsyncVectorEnv(env_fns)
    eval_env = None

    if args.eval_interval > 0 or args.eval_freq_timesteps > 0:
        eval_env = build_env(env_name, render_mode=None, config=config)
        hyperparameters["eval_env"] = eval_env

    try:
        train(
            env=vec_env,
            num_envs=num_envs,
            hyperparameters=hyperparameters,
            actor_model=args.actor_model,
            critic_model=args.critic_model,
            method=args.method,
            total_timesteps=args.total_timesteps,
        )
    finally:
        vec_env.close()
        if eval_env is not None:
            eval_env.close()
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
