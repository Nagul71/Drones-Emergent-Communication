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

    def step(self, actions,dt):
        self.current_step += 1
        model = self.drone_model

        for i, action in actions.items():
            d = self.drones[i]
            if not d["active"]:
                continue

            dx, dy = action
            norm = np.hypot(dx, dy) + 1e-8
            dx /= norm
            dy /= norm


            desired_heading = math.atan2(dy, dx)
            # --- Smooth rotation ---
            max_turn_rate = math.radians(120)  # degrees/sec → rad/sec

            angle_diff = (desired_heading - d["heading"] + math.pi) % (2 * math.pi) - math.pi

            turn = np.clip(angle_diff, -max_turn_rate * dt, max_turn_rate * dt)
            d["heading"] += turn


            nx = d["x"] + dx * model["move_step"]
            ny = d["y"] + dy * model["move_step"]

            # ---- Obstacle collision ----
            blocked = False
            for ox, oy in self.obstacles:
                if np.hypot(nx - ox, ny - oy) < self.scenario.obstacles["radius"]:
                    blocked = True
                    break

            if not blocked:
                d["x"] = np.clip(nx, 0, self.width)
                d["y"] = np.clip(ny, 0, self.height)

            d["path"].append((d["x"], d["y"]))
            d["battery"] -= model["move_cost"]

            if d["battery"] <= 0:
                d["active"] = False

            self.coverage_grid.mark_covered(d["x"], d["y"])

        done = (
            self.current_step >= self.max_steps
            or all(not d["active"] for d in self.drones)
            or self.coverage_grid.get_coverage_percentage()
               >= self.scenario.coverage["target_percentage"]
        )

        return self._get_observations(), {}, done, {}

    def _get_observations(self):
        observations = {}
        radius = self.drone_model["sensing_radius"]

        for i, d in enumerate(self.drones):
            if not d["active"]:
                observations[i] = np.zeros(9, dtype=np.float32)
                continue

            x, y = d["x"], d["y"]

            # ---- Nearest drone ----
            nearest_dx, nearest_dy = 0.0, 0.0
            min_dist = radius

            for j, other in enumerate(self.drones):
                if i == j or not other["active"]:
                    continue

                dx = other["x"] - x
                dy = other["y"] - y
                dist = np.hypot(dx, dy)

                if dist < min_dist:
                    min_dist = dist
                    nearest_dx = dx / radius
                    nearest_dy = dy / radius

            # ---- Nearest obstacle ----
            obs_dx, obs_dy = 0.0, 0.0
            min_obs_dist = radius

            for ox, oy in self.obstacles:
                dx = ox - x
                dy = oy - y
                dist = np.hypot(dx, dy)

                if dist < min_obs_dist:
                    min_obs_dist = dist
                    obs_dx = dx / radius
                    obs_dy = dy / radius

            # ---- Local coverage ----
            local_coverage = self.coverage_grid.local_coverage(x, y, radius)

            observations[i] = np.array([
                x / self.width,
                y / self.height,
                d["heading"] / math.pi,
                d["battery"] / self.drone_model["max_battery"],

                nearest_dx,
                nearest_dy,

                obs_dx,
                obs_dy,

                local_coverage
            ], dtype=np.float32)

        return observations


    def random_actions(self):
        return {
            i: (random.uniform(-1, 1), random.uniform(-1, 1))
            for i in range(len(self.drones))
        }
