"""Hydra entrypoint for PPO training."""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trainer.ppo_trainer import PPOTrainer


@hydra.main(version_base=None, config_path=str(REPO_ROOT / "config"), config_name="config")
def main(cfg: DictConfig):
    trainer = PPOTrainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
