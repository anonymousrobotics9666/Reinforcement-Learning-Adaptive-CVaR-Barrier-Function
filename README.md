# Reinforcement Learning Adaptive CVaR Barrier Function

Crowd navigation rollout

The project trains PPO policies for crowd navigation with differentiable CVaR-CBF-QP safety layers.

## Install

Clone the repository and create the environment:

```bash
git clone https://github.com/anonymousrobotics9666/Reinforcement-Learning-Adaptive-CVaR-Barrier-Function.git
cd Reinforcement-Learning-Adaptive-CVaR-Barrier-Function
conda env create -f environment.yml
conda activate rl-cvar-cbf
```

For a specific CUDA build, install the matching PyTorch wheel for your machine after activating the environment. 

## Quick Start

Run a short environment rollout to verify the install and save a GIF:

```bash
python scripts/test_env.py
```

Expected ending:

```text
env test passed: outputs/social_nav_var_num/env_test/<run>/summary.json
```

## Train

Default DiffCVaR-CBF-QP training:

```bash
bash scripts/run_ppo.sh
```

Useful short debug run:

```bash
WANDB_MODE=offline RUN_NAME=debug bash scripts/run_ppo.sh \
  trainer.total_timesteps=100000 \
  trainer.num_envs=4 \
  trainer.eval_interval=1
```

Train the vanilla PPO baseline:

```bash
MODEL=ppo_base RUN_NAME=ppo_base bash scripts/run_ppo.sh
```

Outputs are saved under:

```text
outputs/social_nav_var_num/runs/<run_name>-<model>-bs<batch>-ep<epochs>-lr<lr>/
```

Each run contains `config.yaml`, `ckpt_<step>.pt`, and `ckpt_manifest.json`.

## W&B Logging

For offline logging:

```bash
WANDB_MODE=offline bash scripts/run_ppo.sh
```

For online logging:  

```bash
wandb login
WANDB_PROJECT=<project_name> WANDB_ENTITY=<user_or_team> bash scripts/run_ppo.sh
```

## Evaluate

List checkpoints:

```bash
ls outputs/social_nav_var_num/runs
ls outputs/social_nav_var_num/runs/<run>/ckpt_*.pt
```

Evaluate one checkpoint:

```bash
python scripts/eval.py \
  --save-dir outputs/social_nav_var_num/runs/<run> \
  --checkpoint outputs/social_nav_var_num/runs/<run>/ckpt_<step>.pt
```

Save rollout GIFs:

```bash
python scripts/eval.py \
  --save-dir outputs/social_nav_var_num/runs/<run> \
  --checkpoint outputs/social_nav_var_num/runs/<run>/ckpt_<step>.pt \
  --visualize
```

## Repository Layout

```text
config/      Hydra configs
crowd_sim/   Gymnasium crowd navigation environments
model/       PPO and DiffCVaR-CBF-QP models
trainer/     PPO training loop and checkpointing
scripts/     Train, eval, and environment test entrypoints
```

