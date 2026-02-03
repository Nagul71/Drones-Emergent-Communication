import torch
from torch.distributions import Normal


class PPOAgent:
    def __init__(self, policy):
        self.policy = policy

    def act(self, obs):
        obs_t = torch.tensor(obs, dtype=torch.float32)

        with torch.no_grad():
            mean, std, value = self.policy(obs_t)

        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum().item()

        return action.numpy(), log_prob, value.item()
