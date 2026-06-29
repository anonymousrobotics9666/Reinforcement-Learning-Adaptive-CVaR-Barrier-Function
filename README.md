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
crowd_nav/    Social-force policy helpers
crowd_sim/    Gymnasium navigation environments
model/        PPO policy networks, DiffCVaR-CBF-QP model, QP solver, trajectory predictor, and model factory
scripts/      Runnable training scripts
test/         Smoke tests; generated artifacts are ignored
trainer/      PPO algorithm, vectorized PPO, W&B setup, environments, and run lifecycle
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
python scripts/run_ppo_base.py run_name=diffcvarbfqp method=diffcvarbfqp total_timesteps=1000000
```

Use a different robot preset:

```bash
python scripts/run_ppo_base.py run_name=diffcvarbfqp method=diffcvarbfqp env/robot=unicycle
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
python scripts/run_ppo_base.py run_name=vanilla_ppo method=vanilla_ppo
```

Train the differentiable CVaR-BF-QP policy:

```bash
python scripts/run_ppo_base.py run_name=diffcvarbfqp method=diffcvarbfqp
```

Or use the shell launcher:

```bash
bash scripts/run_ppo.sh
```

The launcher defaults to `METHOD=diffcvarbfqp`, `ROBOT=unicycle`, and forwards extra Hydra overrides:

```bash
WANDB_MODE=offline RUN_NAME=debug bash scripts/run_ppo.sh total_timesteps=100000
```

Weights & Biases logging follows the `diff_opt` defaults:

```bash
wandb login
python scripts/run_ppo_base.py run_name=diffcvarbfqp method=diffcvarbfqp wandb_entity=xinywa_umich wandb_project=diff_cvar
```

Short debug run:

```bash
python scripts/run_ppo_base.py \
  run_name=debug \
  method=diffcvarbfqp \
  num_envs=4 \
  total_timesteps=200000 \
  eval_interval=1
```

Training outputs are written to:

```text
outputs/<env.name>/runs/<run_name>-<method>-bs<timesteps_per_batch>-ep<n_updates_per_iteration>-lr<lr>-ent<ent_coef>/
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
ls outputs/social_nav_var_num/runs
ls outputs/social_nav_var_num/runs/<run>/ckpt_*.pt
```

Evaluate all checkpoints:

```bash
python scripts/eval.py \
  --save-dir outputs/social_nav_var_num/runs/<run>
```

Evaluate one checkpoint:

```bash
python scripts/eval.py \
  --save-dir outputs/social_nav_var_num/runs/<run> \
  --checkpoint outputs/social_nav_var_num/runs/<run>/ckpt_<step>.pt
```

Save rollout GIFs during eval:

```bash
python scripts/eval.py \
  --save-dir outputs/social_nav_var_num/runs/<run> \
  --checkpoint outputs/social_nav_var_num/runs/<run>/ckpt_<step>.pt \
  --visualize
```

Eval writes `eval_results.json` in the run directory. Videos are saved under:

```text
outputs/social_nav_var_num/runs/<run>/visualize_<checkpoint_name>/
```

## Outputs

| Command | Output |
| --- | --- |
| `scripts/run_ppo_base.py` | `outputs/<env.name>/runs/<run>/config.yaml`, `ckpt_<step>.pt`, `ckpt_manifest.json` |
| `scripts/eval.py` | `eval_results.json`, optional `visualize_<checkpoint_name>/` GIFs |

## Notes

- `config.env.rl_xy_to_unicycle=false`: unicycle actions are interpreted as `[v, omega]`.
- `config.env.rl_xy_to_unicycle=true`: policy actions are interpreted as `[vx, vy]` and converted to unicycle controls inside the environment.
- Weights & Biases uses `wandb_entity`, `wandb_project`, and `wandb_interval`, matching `diff_opt`.
