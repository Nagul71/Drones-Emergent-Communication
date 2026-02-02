import yaml
import os

class Scenario:
    def __init__(self, path):
        base_dir = os.path.dirname(os.path.abspath(path))

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        required = [
            "world", "map", "drone_model",
            "drones", "coverage",
            "obstacles", "simulation"
        ]
        for key in required:
            if key not in data:
                raise ValueError(f"Missing '{key}' in scenario.yaml")

        self.world = data["world"]
        self.map = data["map"]
        self.drones = data["drones"]
        self.coverage = data["coverage"]
        self.obstacles = data["obstacles"]
        self.simulation = data["simulation"]

        # ✅ Resolve drone_model path relative to scenario.yaml
        drone_model_path = os.path.join(base_dir, data["drone_model"])

        if not os.path.exists(drone_model_path):
            raise FileNotFoundError(f"Drone model not found: {drone_model_path}")

        with open(drone_model_path, "r") as f:
            model = yaml.safe_load(f)

        if "drone" not in model:
            raise ValueError("drone_model.yaml must contain 'drone' block")

        self.drone_model = model["drone"]
