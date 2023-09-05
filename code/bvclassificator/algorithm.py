import numpy as np
import simulate

DIMENSION = 3
i = 0

x_range = np.arange(0, DIMENSION)
y_range = np.arange(0, DIMENSION)
grid_points = np.array([(x, y) for x in x_range for y in y_range])


def create_structure(profile_list):
    structure = np.zeros(
        shape=(len(profile_list[0].time_frames), DIMENSION, DIMENSION))
    for p in profile_list:
        structure[:, p.y, p.x] = p.profile
    return structure


def random_nuclei(n: int):
    x_coord = np.random.randint(low=0, high=DIMENSION, size=n)
    y_coord = np.random.randint(low=0, high=DIMENSION, size=n)

    return [(i, j) for i, j in zip(x_coord, y_coord)]


def cos(arr):
    return np.array([i for _ in range(arr.shape[0])])


if __name__ == "__main__":
    cell_list = []
    for y in y_range:
        for x in x_range:
            c = simulate.Profile(x, y, 60, 60)
            c.simulate({"in_control": True,
                        "function": cos}, 0, 1, with_noise=False)
            cell_list.append(c)
            i += 1

    s = create_structure(cell_list)
    print(s[0])
