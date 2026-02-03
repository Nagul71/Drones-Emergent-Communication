import random
import numpy as np
from environment.coverage_grid import CoverageGrid
import math


class Environment:
    def __init__(self, scenario):
        self.scenario = scenario

        # ---- Drone model (ALL drone properties live here) ----
        self.drone_model = scenario.drone_model
        self.num_drones = scenario.drones["count"]

        # ---- World ----
        self.width = scenario.world["width"]
        self.height = scenario.world["height"]

        # ---- Coverage grid ----
        self.coverage_grid = CoverageGrid(
            self.width,
            self.height,
            scenario.coverage["grid_rows"],
            scenario.coverage["grid_cols"]
        )

        # ---- Simulation ----
        self.max_steps = scenario.simulation["max_steps"]

        self.current_step = 0
        self.drones = []
        self.obstacles = []

    def reset(self):
        self.current_step = 0
        self.coverage_grid.reset()
        self.drones.clear()
        self.obstacles.clear()

        # ---- Obstacles ----
        if self.scenario.obstacles["enabled"]:
            for _ in range(self.scenario.obstacles["count"]):
                self.obstacles.append((
                    random.uniform(10, self.width - 10),
                    random.uniform(10, self.height - 10)
                ))

        # ---- Drones ----
        for _ in range(self.num_drones):
            d = {
                "x": random.uniform(0, self.width),
                "y": random.uniform(0, self.height),
                "battery": self.drone_model["max_battery"],
                "heading": 0.0,
                "angular_velocity": 0.0,
                "active": True,
                "path": []
            }
            self.drones.append(d)
            self.coverage_grid.mark_covered(d["x"], d["y"])

        return self._get_observations()

    def step(self, actions):
        self.current_step += 1
        rewards = {}

        prev_coverage = self.coverage_grid.get_coverage_percentage()

        for i, action in actions.items():
            d = self.drones[i]

            if not d["active"]:
                rewards[i] = -1.0   # dead drone penalty
                continue

            dx, dy = action
            norm = np.hypot(dx, dy) + 1e-8
            dx /= norm
            dy /= norm

            d["heading"] = np.arctan2(dy, dx)

            nx = d["x"] + dx * self.drone_model["move_step"]
            ny = d["y"] + dy * self.drone_model["move_step"]

            collided = False
            for ox, oy in self.obstacles:
                if np.hypot(nx - ox, ny - oy) < self.scenario.obstacles["radius"]:
                    collided = True
                    break

            if not collided:
                d["x"] = np.clip(nx, 0, self.width)
                d["y"] = np.clip(ny, 0, self.height)

            # Battery cost
            d["battery"] -= self.drone_model["move_cost"]
            if d["battery"] <= 0:
                d["active"] = False

            # Coverage
            self.coverage_grid.mark_covered(d["x"], d["y"])

            # ---------- Reward components ----------
            reward = 0.0

            # Battery penalty
            reward -= 0.01

            # Collision penalty
            if collided:
                reward -= 0.2

            rewards[i] = reward

        # ---------- Global coverage reward ----------
        new_coverage = self.coverage_grid.get_coverage_percentage()
        coverage_gain = new_coverage - prev_coverage

        for i in rewards:
            rewards[i] += 100.0 * coverage_gain   # shared team reward

        done = (
            self.current_step >= self.max_steps
            or all(not d["active"] for d in self.drones)
            or new_coverage >= self.scenario.coverage["target_percentage"]
        )

        self.last_rewards = rewards

        return self._get_observations(), rewards, done, {}

    
    def _get_observations(self):
        observations = {}
        radius = self.drone_model["sensing_radius"]

        for i, d in enumerate(self.drones):
            # Inactive drone → zero observation
            if not d["active"]:
                observations[i] = np.zeros(9, dtype=np.float32)
                continue

            x, y = d["x"], d["y"]

            # ---------- Nearest drone ----------
            ndx, ndy = 0.0, 0.0
            min_dist = radius

            for j, other in enumerate(self.drones):
                if i == j or not other["active"]:
                    continue

                dx = other["x"] - x
                dy = other["y"] - y
                dist = math.hypot(dx, dy)

                if dist < min_dist:
                    min_dist = dist
                    ndx = dx / radius
                    ndy = dy / radius

            # ---------- Nearest obstacle ----------
            odx, ody = 0.0, 0.0
            min_obs_dist = radius

            for ox, oy in self.obstacles:
                dx = ox - x
                dy = oy - y
                dist = math.hypot(dx, dy)

                if dist < min_obs_dist:
                    min_obs_dist = dist
                    odx = dx / radius
                    ody = dy / radius

            # ---------- Local coverage ----------
            local_cov = self.coverage_grid.local_coverage(x, y, radius)

            observations[i] = np.array([
                x / self.width,
                y / self.height,
                d["heading"] / math.pi,
                d["battery"] / self.drone_model["max_battery"],

                ndx,
                ndy,

                odx,
                ody,

                local_cov
            ], dtype=np.float32)

        return observations

    def random_actions(self):
        return {
            i: (random.uniform(-1, 1), random.uniform(-1, 1))
            for i in range(len(self.drones))
        }
