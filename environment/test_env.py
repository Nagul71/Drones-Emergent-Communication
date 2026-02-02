from environment.env import Environment
from environment.config import EnvironmentConfig
import random

config = EnvironmentConfig()
env = Environment(config)

obs = env.reset()
print("Initial Observations:", obs)

for step in range(10):
    actions = {}
    for i in obs.keys():
        actions[i] = (
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        )

    obs, rewards, done, info = env.step(actions)

    print(f"Step {step+1}")
    print("Coverage:", info["coverage"])
    print("Observations:", obs)

    if done:
        break
