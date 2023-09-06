import numpy as np
import simulation
import math
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix

DIMENSION = 3
i = 0


class Lattice:
    def __init__(self, dimension: int, time_period: float, n_frames: int) -> None:
        x_range = np.arange(0, dimension)
        y_range = np.arange(0, dimension)
        self.grid_points = np.array([(x, y) for x in x_range for y in y_range])
        self.label_matrix = np.zeros(shape=(dimension,
                                            dimension))
        self.structure = np.zeros(
            shape=(n_frames,
                   dimension,
                   dimension))

        self.labels = np.zeros(shape=(dimension,
                                      dimension))

        self.percentage = np.zeros(shape=(dimension,
                                          dimension))

        self.final_label = np.zeros(shape=(dimension,
                                           dimension))

        self.spatial_entropy = np.zeros(shape=(dimension,
                                               dimension))

        self.normalized_spatial_entropy = np.zeros(shape=(dimension,
                                                          dimension))

        self.dimension = dimension
        self.k = None

    def build(self, profile_list: list):
        for p in profile_list:
            self.structure[:, p.y, p.x] = p.profile
            self.label_matrix[p.y, p.x] = p.label

    def find_final_label(self):
        for i in range(self.dimension):
            for j in range(self.dimension):
                for k in range(self.k):
                    self.percentage[k, i, j] = np.sum(
                        self.labels[:, i, j] == k)

        self.percentage = self.percentage/self.labels.shape[0]
        self.final_label = np.argmax(self.percentage, axis=0)

    def do_cluster_matching(self):
        baseline = self.labels[:, 0, 0]
        unique_baseline = np.sort(np.unique(baseline))

        for i in range(self.dimension):
            for j in range(self.dimension):
                new_labels = _cluster_mapping(baseline, self.labels[:, i, j])
                unique_other = np.sort(np.unique(self.labels[:, i, j]))
                m_d = _mapping_dict(unique_baseline, unique_other, new_labels)
                self.labels[:, i, j] = _change_label(self.labels[:, i, j], m_d)

    def find_entropy(self):
        for i in range(self.dimension):
            for j in range(self.dimension):
                arr = self.percentage[:, i, j] + 1e-15  #  For log
                self.spatial_entropy[i,
                                     j] = round(-np.sum(arr * np.log(arr)), 5)


def _mapping_dict(ub: np.ndarray, ua: np.ndarray, l: np.ndarray) -> dict:
    d = dict()
    for a in ua:
        d[a] = ub[l[np.where(ua == a)]]
    return d


def _change_label(element: np.ndarray, d: dict) -> np.ndarray:
    final = np.zeros_like(element)
    for i, e in enumerate(element):
        for key in d.keys():
            if e == key:
                final[i] = d[key]
    return final


def _cluster_mapping(array: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    matrix = contingency_matrix(array, baseline)
    new_label = np.argmax(matrix, axis=1)
    return new_label


def random_nuclei(n: int, dim: int) -> np.ndarray:
    """Fuction to select n random nuclei for the Voronoi tesselation.

    Args:
        n (int): Cardinality of the sampling
        dim (int): Dimension of the lattice

    Returns:
        np.ndarray: Array with n random nuclei
    """
    nuclei = []
    while len(nuclei) < n:
        x_coord = np.random.randint(low=0, high=dim)
        y_coord = np.random.randint(low=0, high=dim)
        if (x_coord, y_coord) not in nuclei:
            nuclei.append((x_coord, y_coord))
    return np.array(nuclei)


def cos(arr):
    return np.array([i for _ in range(arr.shape[0])])


def euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """Compute euclidean distance between two points.

    Args:
        point1 (np.ndarray): First point
        point2 (np.ndarray): Second point

    Returns:
        float: _description_
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def max_min_distance(coordinates: np.ndarray):
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


def init_lattice():
    x_range = np.arange(0, DIMENSION)
    y_range = np.arange(0, DIMENSION)
    grid_points = np.array([(x, y) for x in x_range for y in y_range])
    return grid_points


def indexing_coordinates_array(coordinates_list: list):
    ax_1 = []
    ax_0 = []
    for c in coordinates_list:
        ax_0.append(c[0])
        ax_1.append(c[1])
    return ax_0, ax_1


def nuclei_mapping(grid_points: np.ndarray, nuclei: np.ndarray) -> np.ndarray:
    """Function to map each point of the lattice to the closest nucleus.

    Args:
        grid_points (np.ndarray): Grid points ad array of coordinates
        nuclei (np.ndarray): Array of nuclei coordinates

    Returns:
        np.ndarray: An array containing the index of the respective nucleus in the list
    """
    mapping = []
    for grid_point in grid_points:
        min_distance = float('inf')
        user_point = None
        for i, user_point in enumerate(nuclei):
            distance = np.linalg.norm(grid_point - user_point)
            if distance < min_distance:
                min_distance = distance
                user_point = i
        mapping.append(user_point)
    return np.array(mapping)


def group(nuclei: np.ndarray, mapping: np.ndarray, grid_points: np.ndarray, s: np.ndarray) -> dict:
    """Function to compute weighted average representative for each nuclues and its relative neighborhood.

    Args:
        nuclei (np.ndarray): Array of nuclei coordinates
        mapping (np.ndarray): Mapping of each point in the grid to relative nuleus index
        grid_points (np.ndarray): Lattice grid_points
        s (np.ndarray): Lattice structure

    Returns:
        dict: Return a dictionary in the form {int label of the nucleus, np.ndarray of the weighted average}
    """
    d = dict()
    max_dist, min_dist = max_min_distance(nuclei)

    # Find sigma <- CONTROLLARE SE È STD O VAR
    sigma = max_dist/min_dist

    # Compute covariance matrix for bivariate normal distribution
    cov_matrix = sigma**2*np.identity(2)

    for n in range(np.max(mapping)+1):
        selected_points = grid_points[mapping == n]
        ax_0, ax_1 = indexing_coordinates_array(selected_points)
        weights = compute_gaussian_weight(loc=nuclei[n],
                                          covmatrix=cov_matrix,
                                          selected_points=selected_points)

        selected_profiles = s[:, ax_1, ax_0]

        for i, w in enumerate(weights):
            selected_profiles[:, i] = selected_profiles[:, i]*w

        d[n] = np.sum(selected_profiles, axis=1)/np.sum(weights)

    return d


def compute_gaussian_weight(loc: np.ndarray, covmatrix: np.ndarray, selected_points: np.ndarray) -> np.ndarray:
    """Function to compute gaussian weight for a given nucleo and a given array of selected coordinates.

    Args:
        loc (np.ndarray): Loc of the bivariate normal distribution, in our case the nucleus.
        covmatrix (np.ndarray): Covariance matrix of the bivariate normal distribution
        selected_points (np.ndarray): Selected points from the neighborhood

    Returns:
        np.ndarray: array with weights found
    """
    w = []
    gaussian_dist = multivariate_normal(
        mean=loc, cov=covmatrix)

    for point in selected_points:
        w.append(gaussian_dist.pdf(point))

    return np.array(w)


def cluster_now(lattice, n: int, k: int, p: int):
    lattice.k = k

    # Select n random nuclei from the lattice
    nuclei = random_nuclei(n=n,
                           dim=lattice.dimension)

    # Mapping each gridpoint to nearest nucleus
    nuclei_map = nuclei_mapping(grid_points=lattice.grid_points,
                                nuclei=nuclei)

    nuclei = np.array([(0, 0), (2, 1)])
    nuclei_map = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1])

    # Compute representative
    rep_functions = group(nuclei=nuclei,
                          mapping=nuclei_map,
                          grid_points=lattice.grid_points,
                          s=lattice.structure)

    print(rep_functions)


if __name__ == "__main__":
    dimension = 3
    x_range = np.arange(0, dimension)
    y_range = np.arange(0, dimension)
    grid = Lattice(dimension=dimension,
                   time_period=60,
                   n_frames=60)

    # Create structure
    profile_list = []
    for y in y_range:
        for x in x_range:
            c = simulation.Profile(x, y, n_frames=60, time_period=60)
            c.simulate(sim_params={"label": 1,
                                   "function": cos},
                       loc=0,
                       scale=1,
                       with_noise=False)
            profile_list.append(c)
            i += 1

    grid.build(profile_list=profile_list)

    cluster_now(lattice=grid,
                n=3,
                k=5,
                p=10)
