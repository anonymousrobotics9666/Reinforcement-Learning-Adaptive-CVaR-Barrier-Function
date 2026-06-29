"""PPO training framework wrapper."""

import os
import torch

from model.factory import get_model_class
from trainer.trainer import Trainer
from trainer.vec_ppo import VecPPO


class PPOTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.setup_env()
        self.model = self.build_model()

    def build_model(self):
        print(f"Training with {int(self.cfg.num_envs)} vectorized environments", flush=True)
        policy_class = get_model_class(self.cfg.method)
        print(f"Algorithm: {self.cfg.method}, Policy: {policy_class.__name__}", flush=True)

        seeds = [int(self.hyperparameters["seed"]) + i for i in range(int(self.cfg.num_envs))]
        self.train_envs.reset(seed=seeds)
        model = VecPPO(
            env=self.train_envs,
            num_envs=int(self.cfg.num_envs),
            **self.hyperparameters,
        )
        self.load_warm_start(model)
        return model

    def load_warm_start(self, model):
        loaded_parts = []
        actor_model = str(self.cfg.actor_model or "").strip()

        if actor_model:
            if not os.path.exists(actor_model):
                raise FileNotFoundError(f"Model checkpoint not found: {actor_model}")
            state = torch.load(actor_model, map_location=self.device, weights_only=False)
            if not (isinstance(state, dict) and "model" in state):
                raise ValueError(f"Expected checkpoint with a top-level 'model' key: {actor_model}")
            model.model.load_state_dict(state["model"], strict=True)
            loaded_parts.append(f"model={actor_model}")

        if loaded_parts:
            print(f"Warm start loaded: {', '.join(loaded_parts)}", flush=True)
        else:
            print("Training from scratch.", flush=True)

    def train(self):
        try:
            self.model.learn(total_timesteps=self.cfg.total_timesteps)
        finally:
            self.close()
