import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.transforms import Affine2D

class EnvironmentRenderer:
    def __init__(self, scenario):
        self.scenario = scenario
        plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_xlim(0, scenario.world["width"])
        self.ax.set_ylim(0, scenario.world["height"])
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # ---- Load real map image ----
        self.map_img = mpimg.imread(scenario.map["image_path"])
        self.ax.imshow(
            self.map_img,
            extent=[0, scenario.world["width"], 0, scenario.world["height"]],
            origin="lower"
        )

        self.drone_img = mpimg.imread("assests/drones/drone.png")
        self.drone_artists = []

        self.drone_cache = []
        for _ in range(scenario.drones["count"]):
            img = self.ax.imshow(
                self.drone_img,
                extent=[0,0,0,0],
                alpha=0.9,
                zorder=5
            )
            self.drone_cache.append(img)


    def render(self, env):
        # Clear old drones
        for a in self.drone_artists:
            a.remove()
        self.drone_artists.clear()

        for d in env.drones:
            if not d["active"]:
                continue

            trans = (
                Affine2D()
                .rotate_around(d["x"], d["y"], d["heading"])
                .translate(d["x"], d["y"])
            )

            img = self.ax.imshow(
                self.drone_img,
                extent=[
                    d["x"] - 2, d["x"] + 2,
                    d["y"] - 2, d["y"] + 2
                ],
                transform=trans + self.ax.transData,
                alpha=0.9
            )
            self.drone_artists.append(img)

        self.ax.set_title(
            f"Coverage: {env.coverage_grid.get_coverage_percentage():.2%}"
        )
        plt.pause(1 / self.scenario.simulation["render_fps"])
