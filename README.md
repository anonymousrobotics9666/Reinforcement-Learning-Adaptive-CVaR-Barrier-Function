# RL Adaptive CVaR-CBF

This repository contains reinforcement-learning and optimization-based controllers for crowd navigation with control barrier function (CBF) and CVaR-CBF safety constraints.

The current codebase is organized around four entrypoints:

| Entrypoint | Purpose |
| --- | --- |
| `main_vec.py` | PPO training for RL policies |
| `main_test.py` | Single-checkpoint RL policy evaluation |
| `main_opt.py` | Evaluation for optimization/controller baselines |
| `eval/eval.py` | Batch evaluation of saved checkpoints across multiple seeds |

## Repository Layout

```text
config/       Hydra configs and the Config adapter
controller/   Nominal/social-force helpers and the CBF-QP baseline controller
crowd_nav/    Policy helpers and RL policy factory
crowd_sim/    Gymnasium navigation environments
eval/         Rollout helpers and checkpoint evaluation
rl/           PPO trainer and policy networks
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

The project uses Hydra YAML configs. `config/config.py` adapts a Hydra `DictConfig` into the object-style config used by the simulator and controllers.

Main config files:

| File | Purpose |
| --- | --- |
| `config/trainer/train.yaml` | PPO training entry config for `main_vec.py` |
| `config/trainer/ppo.yaml` | PPO hyperparameters |
| `config/trainer/test.yaml` | Single-checkpoint test config for `main_test.py` |
| `config/eval_opt.yaml` | Controller evaluation config for `main_opt.py` |
| `config/env/env.yaml` | Environment, human, controller, and reward defaults |
| `config/env/robot/single_integrator.yaml` | Single-integrator robot preset |
| `config/env/robot/unicycle.yaml` | Unicycle robot preset |

Override config values on the command line with Hydra syntax:

```bash
python main_vec.py method=diffcvarbfqp total_timesteps=1000000
```

Use a different robot preset:

```bash
python main_vec.py method=diffcvarbfqp env/robot=unicycle
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
python main_vec.py method=vanilla_ppo
```

Train the differentiable CVaR-BF-QP policy:

```bash
python main_vec.py method=diffcvarbfqp
```

Weights & Biases logging is disabled by default. Enable it explicitly when needed:

```bash
python main_vec.py method=diffcvarbfqp wandb.enabled=true
```

Short debug run:

```bash
python main_vec.py \
  method=diffcvarbfqp \
  num_envs=4 \
  total_timesteps=200000 \
  eval_freq_timesteps=0 \
  model_folder=debug
```

Training outputs are written to:

```text
trained_models/<model_folder>/<timestamp>_<robot>_<method>_seed<seed>/
```

The run folder contains `train_config.json` and PPO actor/critic checkpoints.

## Testing One RL Checkpoint

Use the checkpoint's saved config for reproducible evaluation:

```bash
python main_test.py \
  method=diffcvarbfqp \
  actor_model=trained_models/default/<run>/ppo_actor_step_7000000.pth \
  config_json=trained_models/default/<run>/train_config.json \
  test_ep=100 \
  episode_seed_start=100
```

Run exactly one seed:

```bash
python main_test.py \
  method=diffcvarbfqp \
  actor_model=trained_models/default/<run>/ppo_actor_step_7000000.pth \
  config_json=trained_models/default/<run>/train_config.json \
  episode_seeds=472
```

Use the current YAML config instead of a saved snapshot:

```bash
python main_test.py \
  method=diffcvarbfqp \
  actor_model=trained_models/default/<run>/ppo_actor_step_7000000.pth \
  use_current_config=true \
  test_ep=20
```

By default, test mode saves GIFs using `rgb_array`. Set `render=true` to open a live window instead.
Use `save_gifs=false` to skip GIF writing, or `max_gifs=N` to cap saved animations.

## Evaluating Controller Baselines

Controller baselines are run with `main_opt.py`:

```bash
python main_opt.py \
  method=cbfqp \
  config_json=trained_models/default/<run>/train_config.json \
  test_ep=100 \
  episode_seed_start=100
```

Supported controller methods include:

```text
nominal
cbfqp
```

## Batch Checkpoint Evaluation

Evaluate all actor checkpoints in a run directory:

```bash
python eval/eval.py \
  --actor_model trained_models/default/<run> \
  --method diffcvarbfqp \
  --episodes_per_seed 100
```

The script writes:

```text
checkpoint_eval_all_multiseed.json
```

## Outputs

| Command | Output |
| --- | --- |
| `main_vec.py` | `trained_models/<model_folder>/<run>/train_config.json`, actor checkpoints, critic checkpoints |
| `main_test.py` | `<checkpoint-dir>/<timestamp>_ckpt_.../test_config.json`, `eval_log.json`, GIFs |
| `main_opt.py` | `trained_models/<model_folder>/<robot>_<method>/<timestamp>/test_config.json`, `eval_log.json`, GIFs |
| `eval/eval.py` | `checkpoint_eval_all_multiseed.json` under the evaluated run or compare directory |

## Notes

- `config.env.rl_xy_to_unicycle=false`: unicycle actions are interpreted as `[v, omega]`.
- `config.env.rl_xy_to_unicycle=true`: policy actions are interpreted as `[vx, vy]` and converted to unicycle controls inside the environment.
- Weights & Biases logging is optional and disabled by default through `wandb.enabled=false`.
