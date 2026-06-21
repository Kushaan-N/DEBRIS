import json
import torch
from pathlib import Path

from training.policy import PolicyNet


class CurriculumManager:
    def __init__(self, metadata_path: str | Path, checkpoint_root: str | Path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        self.stages: list[dict] = metadata["curriculum"]
        self.stage_names: list[str] = [s["name"] for s in self.stages]
        self.checkpoint_root = Path(checkpoint_root)

    def get_stage_config(self, stage_name: str) -> dict:
        for s in self.stages:
            if s["name"] == stage_name:
                return s
        raise ValueError(f"Unknown stage: {stage_name}. Available: {self.stage_names}")

    def get_checkpoint_path(self, stage_name: str) -> Path:
        return self.checkpoint_root / f"stage_{stage_name}" / "policy_final.pt"

    def get_prev_stage(self, stage_name: str) -> str | None:
        idx = self.stage_names.index(stage_name)
        return self.stage_names[idx - 1] if idx > 0 else None

    def save_checkpoint(self, policy: PolicyNet, optimizer: torch.optim.Optimizer,
                        step: int, stage_name: str, extra_meta: dict | None = None):
        path = self.get_checkpoint_path(stage_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_state": policy.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "stage": stage_name,
            **(extra_meta or {}),
        }, path)
        print(f"[ckpt] saved stage={stage_name} step={step} -> {path}")

    def load_checkpoint(self, policy: PolicyNet, optimizer: torch.optim.Optimizer,
                        stage_name: str) -> int:
        path = self.get_checkpoint_path(stage_name)
        if not path.exists():
            raise FileNotFoundError(f"No checkpoint for stage {stage_name} at {path}")
        ckpt = torch.load(path, map_location="cpu")
        policy.load_state_dict(ckpt["policy_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        print(f"[ckpt] loaded stage={stage_name} step={ckpt['step']}")
        return ckpt["step"]

    def checkpoint_exists(self, stage_name: str) -> bool:
        return self.get_checkpoint_path(stage_name).exists()
