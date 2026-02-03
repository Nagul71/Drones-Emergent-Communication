import os
import torch

from environment.scenario_loader import Scenario
from environment.env import Environment
from environment.pygame_renderer import PygameRenderer

from learning.policy import PolicyNet
from learning.buffer import RolloutBuffer
from learning.agent import PPOAgent
from learning.ppo import ppo_update


coverage_history = []
reward_history = []



# ================= SETTINGS =================
TRAIN = False       # 🔴 True = train, False = demo
MODE = "trained"   # "trained" or "random"
EPISODES = 300         # only used when TRAIN=True
STEPS_PER_UPDATE = 1024

CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = f"{CHECKPOINT_DIR}/policy_latest.pth"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# ============================================


# ----------- ENV SETUP -----------
scenario = Scenario("configs/scenario.yaml")
env = Environment(scenario)
renderer = PygameRenderer(scenario)

policy = PolicyNet(obs_dim=9, act_dim=2)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

agent = PPOAgent(policy)
buffer = RolloutBuffer()


# ----------- LOAD MODEL IF EXISTS -----------
if os.path.exists(CHECKPOINT_PATH):
    policy.load_state_dict(torch.load(CHECKPOINT_PATH))
    print("✅ Loaded trained policy")
else:
    print("🆕 No trained policy found")


# ================= MAIN LOOP =================
for episode in range(EPISODES if TRAIN else 1):

    obs = env.reset()
    done = False
    step_count = 0
    episode_reward = 0.0

    while not done:
        actions = {}

        # ---- POLICY ACTIONS ----
        for i in obs:
            if MODE == "trained":
                action, log_prob, value = agent.act(obs[i])
            else:
                action = env.random_actions()[i]

            actions[i] = action


            if TRAIN:
                buffer.obs.append(obs[i])
                buffer.actions.append(action)
                buffer.log_probs.append(log_prob)
                buffer.values.append(value)

        obs, rewards, done, _ = env.step(actions)

        if TRAIN:
            for i in rewards:
                buffer.rewards.append(rewards[i])
                buffer.dones.append(done)
                episode_reward += rewards[i]

        renderer.draw(env)
        coverage = env.coverage_grid.get_coverage_percentage()
        coverage_history.append(coverage)

        step_count += 1

        # ---- PPO UPDATE ----
        if TRAIN and step_count >= STEPS_PER_UPDATE:
            ppo_update(policy, optimizer, buffer)
            buffer.clear()
            step_count = 0

    if TRAIN:
        torch.save(policy.state_dict(), CHECKPOINT_PATH)
        reward_history.append(episode_reward)
        print(f"Episode {episode} | Total Reward: {episode_reward:.2f}")

print("✅ Simulation finished")

import numpy as np

if MODE == "trained":
    np.save("coverage_trained.npy", np.array(coverage_history))
    print("📊 Saved trained coverage")
else:
    np.save("coverage_random.npy", np.array(coverage_history))
    print("📊 Saved random coverage")


