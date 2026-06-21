import torch
import torch.nn as nn
from torch.distributions import Normal
from pathlib import Path

OBS_DIM = 71
ACT_DIM = 3
ACT_SCALE = torch.tensor([1.5, 1.0, 1.0])


def _orthogonal_init(layer: nn.Linear, gain: float) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        sqrt2 = (2.0 ** 0.5)
        self.shared = nn.Sequential(
            _orthogonal_init(nn.Linear(OBS_DIM, 256), gain=sqrt2),
            nn.LayerNorm(256),
            nn.Tanh(),
            _orthogonal_init(nn.Linear(256, 256), gain=sqrt2),
            nn.LayerNorm(256),
            nn.Tanh(),
        )
        self.actor_mean = _orthogonal_init(nn.Linear(256, ACT_DIM), gain=0.01)
        self.actor_log_std = nn.Parameter(torch.zeros(ACT_DIM))
        self.critic = _orthogonal_init(nn.Linear(256, 1), gain=1.0)

    def forward(self, obs: torch.Tensor):
        h = self.shared(obs)
        mean = self.actor_mean(h)
        std = self.actor_log_std.exp().expand_as(mean)
        value = self.critic(h).squeeze(-1)
        return mean, std, value

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        mean, std, value = self.forward(obs)
        scale = ACT_SCALE.to(obs.device)
        if deterministic:
            action_scaled = torch.tanh(mean) * scale
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            action_scaled = torch.tanh(action) * scale
        return action_scaled, log_prob, value

    def evaluate_actions(self, obs: torch.Tensor, actions_scaled: torch.Tensor):
        mean, std, value = self.forward(obs)
        scale = ACT_SCALE.to(obs.device)
        actions_pretanh = torch.atanh((actions_scaled / scale).clamp(-1 + 1e-6, 1 - 1e-6)).clamp(-3.0, 3.0)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(actions_pretanh).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


def save_policy(policy: PolicyNet, path: str | Path, metadata: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": policy.state_dict(), "metadata": metadata or {}}, path)


def load_policy(path: str | Path, device: str = "cpu") -> PolicyNet:
    ckpt = torch.load(path, map_location=torch.device(device), weights_only=False)
    policy = PolicyNet()
    # Handle both key names — older saves used 'state_dict', newer use 'policy_state'
    state_dict = ckpt.get("policy_state", ckpt.get("state_dict", ckpt))
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy
