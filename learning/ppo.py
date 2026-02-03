import torch
import torch.nn.functional as F
from torch.distributions import Normal


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0.0
    values = values + [0.0]

    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages.insert(0, gae)

    return advantages


def ppo_update(
    policy,
    optimizer,
    buffer,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    epochs=4,
):
    obs = torch.tensor(buffer.obs, dtype=torch.float32)
    actions = torch.tensor(buffer.actions, dtype=torch.float32)
    old_log_probs = torch.tensor(buffer.log_probs, dtype=torch.float32)
    rewards = buffer.rewards
    dones = buffer.dones

    with torch.no_grad():
        _, _, values = policy(obs)
        values = values.squeeze().tolist()

    advantages = compute_gae(rewards, values, dones)
    returns = [a + v for a, v in zip(advantages, values)]

    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(epochs):
        mean, std, values = policy(obs)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1).mean()

        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(), returns)

        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
