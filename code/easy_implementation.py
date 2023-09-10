import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math

from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
import random

import skfda
from skfda.misc.hat_matrix import (
    KNeighborsHatMatrix,
    LocalLinearRegressionHatMatrix,
    NadarayaWatsonHatMatrix,
)
from skfda.misc.kernels import uniform
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.preprocessing.smoothing.validation import SmoothingParameterSearch


from skfda.exploratory.visualization import FPCAPlot
from skfda.preprocessing.dim_reduction import FPCA
from skfda.representation.basis import (
    BSplineBasis,
    FourierBasis,
    MonomialBasis,
)


# Global parameters
DIMENSION = 3  #  Dimension of the grid


class Profile:
    def __init__(self, x: int, y: int, n_frames: int) -> None:
        self.x = x
        self.y = y
        self.time_frames = np.arange(0, n_frames)
        self.label = None
        self.profile = None
        self.errors = None
        self.func_profile = None
        self.err_params = None

    def simulate(self, loc: float, scale: float, with_noise: bool, label: int, in_control: bool) -> None:
        """_summary_

        Args:
            sim_params (dict): A dictionary containing a int variable "label" and a "funciton" simulation function
            loc (float): Mean of error normal distribution
            scale (float): Standard deviation of error normal distribution
        """
        self.label = label
        if in_control:
            self.func_profile = linear(self.time_frames)

        else:
            self.func_profile = quadratic(self.time_frames)

        if with_noise:
            self.errors = np.random.normal(
                loc, scale, self.time_frames.shape[0])
            self.profile = self.func_profile + self.errors
            self.err_params = (loc, scale)

        else:
            self.profile = self.func_profile

    def plot(self) -> None:
        gs1 = gridspec.GridSpec(2, 6)
        ax1 = plt.subplot(gs1[0, :-2])
        ax2 = plt.subplot(gs1[1, :-2])
        ax3 = plt.subplot(gs1[0, -2:])
        ax4 = plt.subplot(gs1[1, -2:])

        ax1.plot(self.time_frames, self.profile, marker="D",
                 markerfacecolor="white", markersize=4, markeredgewidth=1)
        ax1.set_xticks([])
        ax1.set_title("Temperature Profile")

        ax2.scatter(self.time_frames, self.errors, s=12)
        ax2.axhline(y=self.err_params[0], color="orange",
                    linestyle="--", linewidth=1)
        ax2.set_title(
            f"Errors from a N({self.err_params[0]},{self.err_params[1]})")
        ax3.plot(self.func_profile)
        ax3.set_xticks([])
        ax3.yaxis.tick_right()
        ax3.set_title("Profile Function")

        ax4.hist(self.errors, "auto", edgecolor="black",
                 alpha=0.8, orientation="horizontal", density=True)
        ax4.set_title("Error Distribution")
        ax4.set_yticks([])

        plt.tight_layout()
        plt.show()

    @property
    def index(self) -> int:
        return self.y + self.x * DIMENSION

    @property
    def coordinates(self) -> tuple:
        return (self.x, self.y)


def linear(arr: np.ndarray):
    return arr


def quadratic(arr: np.ndarray):
    return np.power(arr, 2)


class Lattice:
    def __init__(self, dimension: int, n_frames: int, b: int, k) -> None:
        x_range = np.arange(0, dimension)
        y_range = np.arange(0, dimension)
        self.grid_points = np.array([(x, y) for x in x_range for y in y_range])
        self.label_matrix = np.zeros(shape=(dimension,
                                            dimension))
        self.structure = np.zeros(
            shape=(n_frames,
                   dimension,
                   dimension))

        self.labels = np.zeros(shape=(b,
                                      dimension,
                                      dimension))

        self.percentage = np.zeros(shape=(k,
                                          dimension,
                                          dimension))

        self.final_label = np.zeros(shape=(dimension,
                                           dimension))

        self.spatial_entropy = np.zeros(shape=(dimension,
                                               dimension))

        self.normalized_spatial_entropy = np.zeros(shape=(dimension,
                                                          dimension))

        self.dimension = dimension

        self.time_frames = np.arange(0, n_frames)

        self.k = k
        self.b = b

    @property
    def average_normalized_entropy(self):
        return np.sum(self.spatial_entropy)/(np.log(self.k)*(self.dimension * self.dimension))

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
        baseline = self.labels[0, :, :]
        for i in range(self.labels.shape[0]):
            new_labels = _cluster_mapping(self.labels[i, :, :], baseline)
            self.labels[i, :, :] = _change_label(self.labels[i, :, :], new_labels)

    def find_entropy(self):
        for i in range(self.dimension):
            for j in range(self.dimension):
                arr = self.percentage[:, i, j] + 1e-15  #  For log
                self.spatial_entropy[i,
                                     j] = np.round(-np.sum(arr * np.log(arr)), 5)

        self.normalized_spatial_entropy = np.round(
            self.spatial_entropy/np.log(self.k), 4)


def _change_label(arr: np.ndarray, new_label) -> np.ndarray:
    final = np.zeros_like(arr)
    for i, l in enumerate(new_label):
        final[arr == i] = l
    return final


def _cluster_mapping(array: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    matrix = contingency_matrix(array, baseline)
    new_label = np.argmax(matrix, axis=1)
    return new_label


def random_nuclei(args: tuple) -> np.ndarray:
    """Fuction to select n random nuclei for the Voronoi tesselation.

    Args:
        n (int): Cardinality of the sampling
        dim (int): Dimension of the lattice

    Returns:
        np.ndarray: Array with n random nuclei
    """
    n = args[0]
    dim = args[1]
    assert n <= dim**2, "Error: selected nuclei must be less or equal to available points number."
    nuclei = []
    while len(nuclei) < n:
        x_coord = random.randint(0, dim-1)
        y_coord = random.randint(0, dim-1)
        if (x_coord, y_coord) not in nuclei:
            nuclei.append((x_coord, y_coord))
    return np.array(nuclei)


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
    assignments = []

    for grid_point in grid_points:
        min_distance = float('inf')
        nearest_user_point = None
        for i, user_point in enumerate(nuclei):
            distance = np.linalg.norm(grid_point - user_point)
            if distance < min_distance:
                min_distance = distance
                nearest_user_point = i
        assignments.append(nearest_user_point)
    return np.array(assignments)


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
    d = np.zeros(shape=(nuclei.shape[0], s.shape[0]))
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

        d[n, :] = np.sum(selected_profiles, axis=1)/np.sum(weights)

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


def kmeans_clustering(matrix: np.ndarray, k: int) -> np.ndarray:
    """Function that perform kmeans clustering and return the labels of the cluster. 
    Take as argument a bidimensional numpy array in the form [[p11,p12,p13], [p21,p22,p23], [p31,p32,p33], ... [pn1,pn2,pn3],]


    Args:
        matrix (np.ndarray): A matrix with the observations to be clustered
        k (int): Number of cluster

    Returns:
        np.ndarray: ordered cluster label
    """
    kmeans = KMeans(n_clusters=k, n_init="auto").fit(matrix)
    return kmeans.labels_


def do_fda(arr: np.ndarray, frames: np.ndarray) -> None:
    fd = skfda.FDataGrid(
        data_matrix=arr,
        grid_points=frames,
    )
    fpca = FPCA(n_components=4)
    fd_score = fpca.fit_transform(fd)
    return fd_score


def cluster_now(lattice, n: int, k: int, p: int):
    lattice.k = k

    for i in range(lattice.b):
        # Select n random nuclei from the lattice
        nuclei = random_nuclei((n, lattice.dimension))
        # print(f"Nuclei are: {nuclei}")

        # Mapping each gridpoint to nearest nucleus
        nuclei_map = nuclei_mapping(grid_points=lattice.grid_points,
                                    nuclei=nuclei)

        # print(f"Mapping is: {nuclei_map}")

        # Compute representative
        rep_functions = group(nuclei=nuclei,
                              mapping=nuclei_map,
                              grid_points=lattice.grid_points,
                              s=lattice.structure)

        # Compute scores of FPCA
        scores = do_fda(rep_functions, lattice.time_frames)

        # Cluster the score
        cluster_label = kmeans_clustering(scores, 2)
        # print(f"Nuclei labels: {cluster_label}")

        #  Remap the cluster to original observation
        unfold = unfold_clusters(cluster_label, nuclei_map)
        # print(f"Unfolded labels: {unfold}")
        print(f"Reshaped labels:\n {unfold.reshape(DIMENSION, DIMENSION)}")

        lattice.labels[i, :, :] = unfold.reshape(DIMENSION, DIMENSION)


def unfold_clusters(labels: list, mapping: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mapping)
    for i, e in enumerate(mapping):
        result[i] = labels[e]

    return result


# if __name__ == "__main__":
#     grid = Lattice(DIMENSION, 60, 7, 5)

#     profile_list = []
#     for i in range(DIMENSION):
#         for j in range(DIMENSION):
#             c = Profile(i, j, 60)
#             if c.index in [0, 1, 3, 4]:
#                 c.simulate(0, 1, False, 1, in_control=False)
#             else:
#                 c.simulate(0, 1, False, 0, in_control=True)
#             # print(f"(x,y) = {(i,j)}")
#             # print(f"Index = {c.index}")

#             profile_list.append(c)

#     grid.build(profile_list=profile_list)
#     #print(f"Grid points is {grid.grid_points}")

#     cluster_now(grid, 8, 2, 1)
#     # print("Before cluster matching")
#     # print(grid.labels, end = "\n\n")
#     grid.do_cluster_matching()

#     print("After cluster matching")
#     print(grid.labels)

#     grid.find_final_label()
#     print("Percentages")
#     print(grid.percentage)

#     print("Final Label")
#     print(grid.final_label)

#     grid.find_entropy()

#     print("Entropy")
#     print(grid.normalized_spatial_entropy)
