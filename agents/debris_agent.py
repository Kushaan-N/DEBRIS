import numpy as np
import torch

from hud.agents.robot import RobotAgent, Model, Adapter

# Action scale factors — policy outputs tanh ∈ [-1,1], scaled to physical ranges
ACT_SCALE = np.array([1.5, 1.0, 1.0], dtype=np.float32)  # vx, vy, wz


class DebrisModel(Model):
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        from training.policy import load_policy
        self.policy = load_policy(checkpoint_path, device)
        self.policy.eval()
        self.device = device

    def reset(self) -> None:
        pass  # MLP is stateless

    def infer(self, batch: dict) -> np.ndarray:
        obs = torch.from_numpy(batch["observation/state"]).float().to(self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        with torch.no_grad():
            action, _, _ = self.policy.get_action(obs, deterministic=True)
        return action.cpu().numpy().astype(np.float32)  # shape (1, 3)


class RandomModel(Model):
    def reset(self) -> None:
        pass

    def infer(self, batch: dict) -> np.ndarray:
        raw = np.random.uniform(-1.0, 1.0, size=(1, 3)).astype(np.float32)
        return raw * ACT_SCALE


class DebrisAdapter(Adapter):
    def adapt_observation(self, obs_dict: dict, spaces) -> dict:
        return {"observation/state": obs_dict["observation/state"]}

    def adapt_action(self, action_array: np.ndarray, spaces) -> np.ndarray:
        return action_array


class DebrisAgent(RobotAgent):
    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        self.model = DebrisModel(checkpoint_path, device)
        self.adapter = DebrisAdapter()


class RandomAgent(RobotAgent):
    def __init__(self):
        self.model = RandomModel()
        self.adapter = DebrisAdapter()
