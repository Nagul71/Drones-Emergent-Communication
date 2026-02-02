import pygame
import math
import sys
import time


class PygameRenderer:
    def __init__(self, scenario):
        pygame.init()

        self.scenario = scenario
        self.scale = 8  # pixels per world unit
        self.paused = False

        # ---- Fonts ----
        self.font = pygame.font.SysFont("arial", 14)

        # ---- Screen ----
        self.width = int(scenario.world["width"] * self.scale)
        self.height = int(scenario.world["height"] * self.scale)

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Drone Swarm Simulation")

        self.clock = pygame.time.Clock()

        # ---- Load map ----
        self.map = pygame.image.load(
            scenario.map["image_path"]
        ).convert()
        self.map = pygame.transform.scale(self.map, (self.width, self.height))

        # ---- Load & scale drone image ----
        raw_drone_img = pygame.image.load(
            "assests/drones/drone.png"   
        ).convert_alpha()

        drone_size_world = scenario.drone_model["size"]
        drone_size_px = int(drone_size_world * self.scale)

        self.drone_img = pygame.transform.smoothscale(
            raw_drone_img,
            (drone_size_px, drone_size_px)
        )

    # -------------------------------------------------
    def draw_text(self, text, x, y, color=(255, 255, 255)):
        surface = self.font.render(text, True, color)
        self.screen.blit(surface, (x, y))

    # -------------------------------------------------
    def draw(self, env):
        # ---- Events ----
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    self.paused = not self.paused

                if event.key == pygame.K_s:
                    filename = f"snap_{int(time.time())}.png"
                    pygame.image.save(self.screen, filename)
                    print(f"[SNAPSHOT SAVED] {filename}")

        # ---- Draw map ----
        self.screen.blit(self.map, (0, 0))

        # ---- Coverage overlay ----
        grid = env.coverage_grid.grid
        rows, cols = grid.shape
        cell_w = self.width // cols
        cell_h = self.height // rows

        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        for r in range(rows):
            for c in range(cols):
                if grid[r, c]:
                    pygame.draw.rect(
                        overlay,
                        (0, 255, 0, 60),
                        (c * cell_w, r * cell_h, cell_w, cell_h)
                    )

        self.screen.blit(overlay, (0, 0))

        # ---- Draw drones ----
        for i, d in enumerate(env.drones):
            if not d["active"]:
                continue

            rotated = pygame.transform.rotate(
                self.drone_img,
                -math.degrees(d["heading"])
            )

            x = int(d["x"] * self.scale)
            y = int((self.scenario.world["height"] - d["y"]) * self.scale)

            rect = rotated.get_rect(center=(x, y))
            self.screen.blit(rotated, rect)

            # Drone ID
            self.draw_text(f"D{i}", x + 10, y - 12, (255, 255, 0))

            # Battery bar
            b_ratio = d["battery"] / self.scenario.drone_model["max_battery"]
            b_ratio = max(0.0, min(1.0, b_ratio))

            bar_w, bar_h = 22, 4
            bar_x = x - bar_w // 2
            bar_y = y + rect.height // 2 + 4

            pygame.draw.rect(
                self.screen, (120, 0, 0),
                (bar_x, bar_y, bar_w, bar_h)
            )
            pygame.draw.rect(
                self.screen, (0, 220, 0),
                (bar_x, bar_y, int(bar_w * b_ratio), bar_h)
            )

        # ---- Legend panel (TOP RIGHT) ----
        total = len(env.drones)
        active = sum(1 for d in env.drones if d["active"])
        avg_battery = (
            sum(d["battery"] for d in env.drones if d["active"]) / active
            if active > 0 else 0
        )
        coverage = env.coverage_grid.get_coverage_percentage() * 100

        panel_w, panel_h = 240, 160
        panel_x = self.width - panel_w - 15
        panel_y = 15

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 160))
        self.screen.blit(panel, (panel_x, panel_y))

        tx = panel_x + 10
        ty = panel_y + 10

        self.draw_text("Drone Swarm Simulation", tx, ty)
        self.draw_text(f"Active Drones : {active}/{total}", tx, ty + 22)
        self.draw_text(f"Coverage      : {coverage:.1f}%", tx, ty + 44)
        self.draw_text(f"Avg Battery   : {avg_battery:.1f}%", tx, ty + 66)
        self.draw_text(f"Step          : {env.current_step}", tx, ty + 88)

        self.draw_text("Legend:", tx, ty + 112)
        self.draw_text("Covered Area", tx, ty + 132)

        # ---- Pause indicator ----
        if self.paused:
            self.draw_text("PAUSED", self.width // 2 - 30, 20, (255, 80, 80))

        pygame.display.flip()
        self.clock.tick(self.scenario.simulation["render_fps"])
