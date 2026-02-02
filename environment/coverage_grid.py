import numpy as np


class CoverageGrid:
    def __init__(self, width, height, rows, cols):
        self.width = width
        self.height = height
        self.rows = rows
        self.cols = cols

        # ---- Cell size (THIS WAS MISSING) ----
        self.cell_width = width / cols
        self.cell_height = height / rows

        # Use one cell_size for radius math (square cells assumed)
        self.cell_size = min(self.cell_width, self.cell_height)

        self.grid = np.zeros((rows, cols), dtype=np.bool_)

    # --------------------------------------
    def reset(self):
        self.grid[:] = False

    # --------------------------------------
    def world_to_cell(self, x, y):
        """
        Convert world coordinates (x, y) to grid indices (row, col)
        """
        col = int(x / self.cell_width)
        row = int(y / self.cell_height)

        row = max(0, min(self.rows - 1, row))
        col = max(0, min(self.cols - 1, col))

        return row, col

    # --------------------------------------
    def mark_covered(self, x, y):
        r, c = self.world_to_cell(x, y)
        self.grid[r, c] = True

    # --------------------------------------
    def get_coverage_percentage(self):
        return np.sum(self.grid) / self.grid.size

    # --------------------------------------
    def local_coverage(self, x, y, radius):
        """
        Percentage of covered cells within sensing radius
        """
        r, c = self.world_to_cell(x, y)
        rad = int(radius / self.cell_size)

        total = 0
        covered = 0

        for i in range(r - rad, r + rad + 1):
            for j in range(c - rad, c + rad + 1):
                if 0 <= i < self.rows and 0 <= j < self.cols:
                    total += 1
                    if self.grid[i, j]:
                        covered += 1

        return covered / total if total > 0 else 0
