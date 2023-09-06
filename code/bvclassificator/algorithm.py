import numpy as np
import simulation
import math


DIMENSION = 3
i = 0

x_range = np.arange(0, DIMENSION)
y_range = np.arange(0, DIMENSION)
grid_points = np.array([(x, y) for x in x_range for y in y_range])


def build_structure(profile_list):
    structure = np.zeros(
        shape=(len(profile_list[0].time_frames), DIMENSION, DIMENSION))
    for p in profile_list:
        structure[:, p.y, p.x] = p.profile
    return structure


def random_nuclei(n: int):
    nuclei = []

    while len(nuclei) < n:
        x_coord = np.random.randint(low = 0, high = DIMENSION)
        y_coord = np.random.randint(low = 0, high = DIMENSION)

        if (x_coord, y_coord) not in nuclei:
            nuclei.append((x_coord, y_coord))

    return np.array(nuclei)


def cos(arr):
    return np.array([i for _ in range(arr.shape[0])])


def euclidean_distance(point1: tuple, point2: tuple):
    """
    Calcola la distanza euclidea tra due tuple di coordinate.
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def max_min_distance(coordinates: list):
    """
    Calcola la distanza euclidea massima e minima tra tutte le coordinate nella lista.
    """

    max_dist = float('-inf')
    min_dist = float('inf')

    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = euclidean_distance(coordinates[i], coordinates[j])
            max_dist = max(max_dist, distance)
            min_dist = min(min_dist, distance)

    return max_dist, min_dist


def init_grid():
    x_range = np.arange(0, DIMENSION)
    y_range = np.arange(0, DIMENSION)
    grid_points = np.array([(x, y) for x in x_range for y in y_range])
    return grid_points


def nuclei_mapping(grid_points: np.ndarray, nuclei: np.ndarray):
    mapping = []
    for grid_point in grid_points:
        min_distance = float('inf')
        nearest_user_point = None
        for i, user_point in enumerate(nuclei):
            distance = np.linalg.norm(grid_point - user_point)
            if distance < min_distance:
                min_distance = distance
                nearest_user_point = i
        mapping.append(nearest_user_point)
    return mapping


# def cluster_now(n: int, k: int, p: int):


if __name__ == "__main__":
    grid = init_grid()
    nuclei = random_nuclei(3)
    mapping = nuclei_mapping(grid, nuclei)
    print(nuclei)
    print(mapping)

    # Create structure
    profile_list = []
    for y in y_range:
        for x in x_range:
            c = simulation.Profile(x, y, n_frames=60, time_period=60)
            c.simulate(sim_params={"in_control":True,
                                   "function":cos},
                                   loc=0,
                                   scale=1,
                                   with_noise=False)
            profile_list.append(c)
            i += 1

    s = build_structure(profile_list)


