import numpy as np
import torch
from torch.optim import Adam

from training.policy import PolicyNet


class RolloutBuffer:
    def __init__(self, n_envs: int, rollout_len: int, obs_dim: int = 71, act_dim: int = 3):
        self.T = rollout_len
        self.rollout_len = rollout_len   # alias used by collect_rollout
        self.N = n_envs
        self.obs = np.zeros((rollout_len, n_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((rollout_len, n_envs, act_dim), dtype=np.float32)
        self.rewards = np.zeros((rollout_len, n_envs), dtype=np.float32)
        self.dones = np.zeros((rollout_len, n_envs), dtype=np.float32)
        self.values = np.zeros((rollout_len, n_envs), dtype=np.float32)
        self.log_probs = np.zeros((rollout_len, n_envs), dtype=np.float32)
        self.ptr = 0

    def add(self, obs, action, reward, done, value, log_prob):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.ptr += 1

    def compute_returns(self, last_values: np.ndarray, gamma: float = 0.99, lam: float = 0.95):
        advantages = np.zeros_like(self.rewards)
        gae = np.zeros(self.N, dtype=np.float32)
        for t in reversed(range(self.T)):
            next_values = last_values if t == self.T - 1 else self.values[t + 1]
            next_non_terminal = 1.0 - (self.dones[t] if t == self.T - 1 else self.dones[t])
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
        returns = advantages + self.values
        return returns, advantages

    def reset(self):
        self.ptr = 0


class PPOUpdater:
    def __init__(self, policy: PolicyNet, lr: float = 3e-4):
        self.policy = policy
        self.optimizer = Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.clip_eps = 0.2
        self.vf_coef = 0.5
        self.ent_coef = 0.05          # annealed toward ent_coef_final during training
        self.ent_coef_final = 0.005
        self.max_grad = 0.5
        self.n_epochs = 10
        self.batch_size = 256

    def update(self, buffer: RolloutBuffer, last_values: np.ndarray, device: str,
               ent_coef: float | None = None) -> dict:
        if ent_coef is None:
            ent_coef = self.ent_coef
        returns, advantages = buffer.compute_returns(last_values)

        # flatten (T, N, ...) -> (T*N, ...)
        obs_flat = buffer.obs.reshape(-1, buffer.obs.shape[-1])
        actions_flat = buffer.actions.reshape(-1, buffer.actions.shape[-1])
        old_log_probs_flat = buffer.log_probs.reshape(-1)
        returns_flat = returns.reshape(-1)
        advantages_flat = advantages.reshape(-1)

        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        obs_t = torch.tensor(obs_flat, device=device)
        actions_t = torch.tensor(actions_flat, device=device)
        old_lp_t = torch.tensor(old_log_probs_flat, device=device)
        returns_t = torch.tensor(returns_flat, device=device)
        adv_t = torch.tensor(advantages_flat, device=device)

        n_samples = obs_t.shape[0]
        stats = {"pg_loss": [], "vf_loss": [], "ent_loss": [], "approx_kl": []}

        for _ in range(self.n_epochs):
            indices = torch.randperm(n_samples, device=device)
            for start in range(0, n_samples, self.batch_size):
                idx = indices[start: start + self.batch_size]
                new_lp, entropy, new_val = self.policy.evaluate_actions(obs_t[idx], actions_t[idx])

                log_ratio = new_lp - old_lp_t[idx]
                ratio = log_ratio.exp()

                adv_batch = adv_t[idx]
                pg_loss = torch.max(-adv_batch * ratio,
                                    -adv_batch * ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)).mean()
                vf_loss = 0.5 * (new_val - returns_t[idx]).pow(2).mean()
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()

                stats["pg_loss"].append(pg_loss.item())
                stats["vf_loss"].append(vf_loss.item())
                stats["ent_loss"].append(ent_loss.item())
                stats["approx_kl"].append(approx_kl)

        return {k: float(np.mean(v)) for k, v in stats.items()}
