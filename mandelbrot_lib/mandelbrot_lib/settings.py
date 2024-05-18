DEFAULT_ITER = 100
DEFAULT_ESCAPE_RADIUS = 2
DEFAULT_WIDTH = 100
DEFAULT_HEIGHT = 100

BASE_VECTORS = [(0, 0), (1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1), (0, -1), (-1, 0)]
BASE_COMPLEX = [complex(*v) for v in BASE_VECTORS]

BASE_GRID_VECTORS = [((-2, -2), (2, 2)), ((-1, -1), (1, 1)), ((-3, -3), (3, 3)), ((-2, -1), (2, 1)), ((-1, -2), (1, 2))]
BASE_GRID_COMPLEX = [(complex(*v_min), complex(*v_max)) for v_min, v_max in BASE_GRID_VECTORS]


