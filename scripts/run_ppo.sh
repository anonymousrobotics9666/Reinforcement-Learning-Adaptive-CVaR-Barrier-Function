#!/usr/bin/env bash
set -euo pipefail

# Optional: uncomment for a one-off login/key injection.
# Do not commit a real API key to a public repo.
# export WANDB_API_KEY="..."

export WANDB_MODE="${WANDB_MODE:-online}"
WANDB_ENTITY="${WANDB_ENTITY:-xinywa_umich}"
WANDB_PROJECT="${WANDB_PROJECT:-diff_cvar}"
METHOD="${METHOD:-diffcvarbfqp}"
ROBOT="${ROBOT:-unicycle}"
RUN_NAME="${RUN_NAME:-${METHOD}}"
DEVICE="${DEVICE:-auto}"
NUM_ENVS="${NUM_ENVS:-8}"

cd "$(dirname "$0")/.."

python scripts/run_ppo_base.py \
  method="${METHOD}" \
  "env/robot=${ROBOT}" \
  run_name="${RUN_NAME}" \
  wandb_entity="${WANDB_ENTITY}" \
  wandb_project="${WANDB_PROJECT}" \
  device="${DEVICE}" \
  num_envs="${NUM_ENVS}" \
  "$@"
