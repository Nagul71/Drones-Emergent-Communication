class EnvironmentConfig:
    def __init__(self):
        self.WORLD_WIDTH = 100
        self.WORLD_HEIGHT = 100

        self.GRID_ROWS = 60
        self.GRID_COLS = 60

        self.NUM_DRONES = 4
        self.MAX_STEPS = 800

        self.BATTERY_CAPACITY = 100.0
        self.MOVE_STEP = 1.2
        self.MOVE_COST = 0.4

        # Obstacles
        self.NUM_OBSTACLES = 6
        self.OBSTACLE_RADIUS = 5
