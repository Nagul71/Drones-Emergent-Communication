from environment.scenario_loader import Scenario
from environment.env import Environment
from environment.pygame_renderer import PygameRenderer

scenario = Scenario("configs/scenario.yaml")
env = Environment(scenario)
renderer = PygameRenderer(scenario)

env.reset()
done = False

while not done:
    actions = env.random_actions()
    dt = renderer.clock.get_time() / 1000.0  # milliseconds → seconds

    _, _, done, _ = env.step(actions, dt)
    renderer.draw(env)



