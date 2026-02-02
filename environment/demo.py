import random
from environment.env import Environment
from environment.config import EnvironmentConfig
from environment.renderer import EnvironmentRenderer

config = EnvironmentConfig()
env = Environment(config)
renderer = EnvironmentRenderer(config)

obs = env.reset()

done = False
while not done:
    actions = {}
    for i in obs.keys():
        actions[i] = (
            random.uniform(-1, 1),
            random.uniform(-1, 1)
        )

    obs, rewards, done, info = env.step(actions)
    renderer.render(env)
