# RL Adaptive CVaR-CBF

This repository contains reinforcement-learning policies for crowd navigation with control barrier function (CBF) and CVaR-CBF safety constraints.

The current codebase is organized around two entrypoints:

| Entrypoint | Purpose |
| --- | --- |
| `scripts/run_ppo_base.py` | PPO training for RL policies |
| `scripts/eval.py` | Evaluate one or all saved RL checkpoints |

## Repository Layout

```text
config/       Hydra configs and the Config adapter
controller/   Batched trajectory prediction helper used by DiffCVaR-CBF
crowd_nav/    Policy helpers and RL policy factory
crowd_sim/    Gymnasium navigation environments
model/        Actor-critic wrapper and model factory
rl/           PPO algorithm and policy networks
scripts/      Runnable training scripts
test/         Smoke tests; generated artifacts are ignored
trainer/      Training framework, W&B setup, environments, and run lifecycle
```

## Installation

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate rl-cvar-cbf
```

Experiment outputs and checkpoints are ignored by git:

```text
trained_models/
outputs/
results/
```

## Configuration

The project uses Hydra YAML configs. `config/config.py` adapts a Hydra `DictConfig` into the object-style config used by the simulator and training code.

Main config files:

| File | Purpose |
| --- | --- |
| `config/trainer/train.yaml` | PPO training entry config for `scripts/run_ppo_base.py` |
| `config/trainer/ppo.yaml` | PPO hyperparameters |
| `config/env/env.yaml` | Environment, human, controller, and reward defaults |
| `config/env/robot/single_integrator.yaml` | Single-integrator robot preset |
| `config/env/robot/unicycle.yaml` | Unicycle robot preset |

Override config values on the command line with Hydra syntax:

```bash
python scripts/run_ppo_base.py method=diffcvarbfqp total_timesteps=1000000
```

Use a different robot preset:

```bash
python scripts/run_ppo_base.py method=diffcvarbfqp env/robot=unicycle
```

## Observation Pipeline

The environment emits absolute observations:

```text
[rx, ry, gx, gy, rvx, rvy, theta, robot_radius,
 (hx, hy, hvx, hvy, human_radius, mask) * human_cap]
```

where:

```text
human_cap = human.num_humans + human.human_num_range
```

PPO converts this to the relative policy input:

```text
[rx-gx, ry-gy, rvx, rvy, theta, robot_radius,
 (rx-hx, ry-hy, hvx, hvy, human_radius, mask) * obs_top_k]
```

`env.obs_top_k` controls the number of nearest obstacle blocks passed to the actor. A checkpoint must be evaluated with matching structural settings, especially:

```text
robot.type
env.obs_top_k
policy method
```

## Training PPO Policies

Available RL methods:

```text
vanilla_ppo
diffcvarbfqp
```

Train a vanilla PPO MLP policy:

```bash
python scripts/run_ppo_base.py method=vanilla_ppo
```

Train the differentiable CVaR-BF-QP policy:

```bash
python scripts/run_ppo_base.py method=diffcvarbfqp
```

Weights & Biases logging follows the `diff_opt` defaults:

```bash
wandb login
python scripts/run_ppo_base.py method=diffcvarbfqp wandb_entity=xinywa_umich wandb_project=diff_cvar
```

Short debug run:

```bash
python scripts/run_ppo_base.py \
  method=diffcvarbfqp \
  num_envs=4 \
  total_timesteps=200000 \
  eval_interval=1 \
  model_folder=debug
```

Training outputs are written to:

```text
trained_models/<model_folder>/<timestamp>_<robot>_<method>_seed<seed>/
```

The run folder contains `config.yaml`, `ckpt_<step>.pt` checkpoints, and `ckpt_manifest.json`.

## Smoke Test

Run this after code changes to check config loading, environment stepping, policy construction, checkpoint loading, and one short eval episode:

```bash
python test/smoke_test.py
```

Smoke-test artifacts are written under `test/smoke/`, which is ignored by git.

## Eval

Find a saved run and checkpoint:

```bash
ls trained_models/default
ls trained_models/default/<run>/ckpt_*.pt
```

Evaluate all checkpoints:

```bash
python scripts/eval.py \
  --save-dir trained_models/default/<run>
```

Evaluate one checkpoint:

```bash
python scripts/eval.py \
  --save-dir trained_models/default/<run> \
  --checkpoint trained_models/default/<run>/ckpt_<step>.pt
```

Save rollout GIFs during eval:

```bash
python scripts/eval.py \
  --save-dir trained_models/default/<run> \
  --checkpoint trained_models/default/<run>/ckpt_<step>.pt \
  --visualize
```

Eval writes `eval_results.json` in the run directory. Videos are saved under:

```text
trained_models/default/<run>/visualize_<checkpoint_name>/
```

## Outputs

| Command | Output |
| --- | --- |
| `scripts/run_ppo_base.py` | `trained_models/<model_folder>/<run>/config.yaml`, `ckpt_<step>.pt`, `ckpt_manifest.json` |
| `scripts/eval.py` | `eval_results.json`, optional `visualize_<checkpoint_name>/` GIFs |

## Notes

- `config.env.rl_xy_to_unicycle=false`: unicycle actions are interpreted as `[v, omega]`.
- `config.env.rl_xy_to_unicycle=true`: policy actions are interpreted as `[vx, vy]` and converted to unicycle controls inside the environment.
- Weights & Biases uses `wandb_entity`, `wandb_project`, and `wandb_interval`, matching `diff_opt`.
