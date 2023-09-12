import numpy as np
import math
from scipy.stats import multivariate_normal
import random


import os
from PIL import Image
import shutil
import json
import time


# Machine Learning
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import precision_score, recall_score

# FDA
from skfda.preprocessing.smoothing import KernelSmoother
from skfda.misc.hat_matrix import NadarayaWatsonHatMatrix
from skfda.representation.grid import FDataGrid
from skfda.exploratory import stats


# FPCA
from skfda.preprocessing.dim_reduction import FPCA


# Plots and graphics
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib import colormaps
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


class Lattice:
    def __init__(self, dimension: int, time_period: float, fps: int, simulation: bool, id_: str, run: None) -> None:

        #  Dimension of the lattice
        self.dimension = dimension

        # Lattice id
        self.id_ = id_

        # Grid points in the lattice in the form (x, y)
        self.grid_points = np.array([(x, y) for x in np.arange(
            0, self.dimension) for y in np.arange(0, self.dimension)])

        # Matrix to store gorund truth labels of profiles
        if simulation:
            self.label_matrix = np.zeros(shape=(dimension,
                                                dimension))

        self.simulation = simulation

        # Compute time frames
        self.time_frames = np.linspace(0, time_period-1, fps*time_period)

        self.structure = np.zeros(
            shape=(self.time_frames.shape[0],
                   dimension,
                   dimension))

        self.final_label = np.zeros(shape=(dimension,
                                           dimension))

        self.spatial_entropy = np.zeros(shape=(dimension,
                                               dimension))

        self.normalized_spatial_entropy = np.zeros(shape=(dimension,
                                                          dimension))

        # Funct object
        self.func_object = None
        self.p = None

        #  Attributes initialized by calling init_algo()
        self.n = None
        self.k = None
        self.b = None
        self.labels = None
        self.percentage = None
        self.explained_variance_pct = None

        # Attribute to verify that clustering has been done
        self.clustered = False
        self.execution_time = None

        # Attributo per tenere traccia delle diverse tun
        self.run = run

    def init_algo(self, n: int, k: int, explained_variance_pct:float, b: int) -> None:
        """Init algorithm parameters.

        Args:
            n (int): Number of select point over the grid to be used as nuclei
            k (int): Number of clusters
            p (int): Number of FPCs to be retained
            b (int): Number of bootstrap replications

        Returns:
            None
        """
        self.labels = np.zeros(shape=(b,
                                      self.dimension,
                                      self.dimension))

        self.percentage = np.zeros(shape=(k,
                                          self.dimension,
                                          self.dimension))
        self.n = n
        self.k = k
        self.b = b
        self.explained_variance_pct = explained_variance_pct

    @property
    def average_normalized_entropy_(self):
        """Find average normalized entropy in order to assess results quality

        Returns:
            float: If available return entropy, otherwise raise an exception. 
        """
        if self.clustered:
            return np.sum(self.spatial_entropy)/(np.log(self.k)*(self.dimension * self.dimension))
        else:
            raise Exception(
                "Error: you must first perform clustering and cluster matching.")

    @property
    def classification_rate_(self):
        """Find classification rate.

        Returns:
            float: If available return classification rate, otherwise raise an exception. 
        """
        assert self.simulation, "You can compute classification rate only on simulated data."
        if self.clustered:
            new_labels = _cluster_mapping(self.final_label, self.label_matrix)
            temp = _change_label(self.final_label, new_labels)
            return np.sum(temp == self.label_matrix)/(self.dimension*self.dimension)

        else:
            raise Exception(
                "Error: you must first perform clustering and cluster matching.")

    @property
    def precision_score_(self):
        """Find precision score in binary clustering.

        Returns:
            float: If available return precision score, otherwise raise an exception. 
        """
        assert self.simulation, "You can compute classification rate only on simulated data."
        assert self.k == 2, "Precision score is available only in binary clusterization"
        if self.clustered:
            new_labels = _cluster_mapping(self.final_label, self.label_matrix)
            temp = _change_label(self.final_label, new_labels)
            return precision_score(self.label_matrix.reshape((self.dimension*self.dimension, 1)), temp.reshape((self.dimension*self.dimension, 1)), average='weighted', labels=np.unique(temp))

        else:
            raise Exception(
                "Error: you must first perform clustering and cluster matching.")

    @property
    def recall_score_(self):
        """Find recall score in binary clustering.

        Returns:
            float: If available return recall score, otherwise raise an exception. 
        """
        assert self.simulation, "You can compute classification rate only on simulated data."
        assert self.k == 2, "Recall score is available only in binary clusterization"
        if self.clustered:
            new_labels = _cluster_mapping(self.final_label, self.label_matrix)
            temp = _change_label(self.final_label, new_labels)
            return recall_score(self.label_matrix.reshape((self.dimension*self.dimension, 1)), temp.reshape((self.dimension*self.dimension, 1)))

        else:
            raise Exception(
                "Error: you must first perform clustering and cluster matching.")

    @property
    def labels_(self) -> np.ndarray:
        """Return final label as a matrix.

        Returns:
            np.ndarray: If available return final label matrix, otherwise raise an exception. 
        """
        if self.clustered:
            return self.final_label
        else:
            raise Exception(
                "Error: you must first perform clustering and cluster matching.")

    def build(self, profile_list: list):
        arr = []
        if self.simulation:
            for p in profile_list:
                arr.append(p.profile)
                self.structure[:, p.y, p.x] = p.profile
                self.label_matrix[p.x, p.y] = p.label

            self.func_object = FDataGrid(
                data_matrix=arr,
                grid_points=self.time_frames
            )
        else:
            for p in profile_list:
                self.structure[:, p.y, p.x] = p.profile

    def build_smooth_func(self, bandwidth: float):
        kernel_estimator = NadarayaWatsonHatMatrix(bandwidth=bandwidth)
        smoother = KernelSmoother(kernel_estimator=kernel_estimator)
        self.func_object = smoother.fit_transform(self.func_object)

    def find_final_label(self):
        self.clustered = True
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
            self.labels[i, :, :] = _change_label(
                self.labels[i, :, :], new_labels)

    def find_entropy(self):
        for i in range(self.dimension):
            for j in range(self.dimension):
                arr = self.percentage[:, i, j] + 1e-15  #  For log
                self.spatial_entropy[i,
                                     j] = np.round(-np.sum(arr * np.log(arr)), 5)

        self.normalized_spatial_entropy = np.round(
            self.spatial_entropy/np.log(self.k), 4)

    def cluster_now(self):
        start_time = time.time()
        self.set_p()
        for i in range(self.b):
            # Select n random nuclei from the lattice
            nuclei = _random_nuclei(self.n, self.dimension)
            # print(f"Nuclei are: {nuclei}")

            # Mapping each gridpoint to nearest nucleus
            nuclei_map = _nuclei_mapping(grid_points=self.grid_points,
                                         nuclei=nuclei)

            # Compute representative
            rep_functions = _group(nuclei=nuclei,
                                   mapping=nuclei_map,
                                   grid_points=self.grid_points,
                                   fd=self.func_object)
            
            # Compute scores of FPCA
            scores = _do_fda(fd=rep_functions,
                             p=self.p)

            # Cluster the score
            cluster_label = _kmeans_clustering(scores, self.k)

            #  Remap the cluster to original observation
            unfold = _unfold_clusters(cluster_label, nuclei_map)
            self.labels[i, :, :] = unfold.reshape(
                self.dimension, self.dimension)

        self.do_cluster_matching()
        self.find_final_label()

        self.find_entropy()

        self.execution_time = time.time()-start_time

    def save_log_json(self, root: str):
        location = _build_path(root, self)

        output_file = os.path.join(location, self.run + ".json")
        p = self.precision_score_
        r = self.recall_score_
        if p==0 or r==0:
            f1=0
        else:
            f1=2*(r*p)/(r+p)

        dictionary = dict(scenario=self.id_,
                          run=self.run,
                          k=self.k,
                          n=self.n,
                          b=self.b,
                          p=self.p,
                          average_normalized_entropy=self.average_normalized_entropy_,
                          classification_rate=self.classification_rate_,
                          precision=p,
                          recall=r,
                          f1score=f1,
                          execution_time=self.execution_time,
                          ground_truth=self.label_matrix.tolist(),
                          final_label=self.final_label.tolist(),
                          entropy_map=self.spatial_entropy.tolist())

        with open(output_file, "w") as file:
            json.dump(dictionary, file, indent=4)

    def plot_profiles(self) -> None:
        fig, ax = plt.subplots()

        if self.simulation:
            if np.unique(self.label_matrix).shape[0] <= 8:
                colors = ["#1f78b4", "#ff7f00", "#33a02c", "#e31a1c",
                          "#a6cee3", "#fdbf6f", "#b2df8a", "#e31a1c"]
                my_cmap = ListedColormap(
                    colors, name="my_cmap").resampled(np.unique(self.label_matrix).shape[0])

            else:
                my_cmap = colormaps["viridis"].resampled(
                    np.unique(self.label_matrix).shape[0])

            for i in range(self.dimension):
                for j in range(self.dimension):
                    l = f"Label {self.label_matrix[i,j]}"
                    col = my_cmap.colors[int(self.label_matrix[i, j])]

                    ax.plot(
                        self.time_frames, self.structure[:, i, j], color=col)

            legend_elements = [Line2D([0], [0], color=my_cmap.colors[int(
                e)], label=f"Label {int(e)}") for e in np.unique(self.label_matrix)]
            ax.legend(handles=legend_elements)

        else:
            my_cmap = colormaps["viridis"].resampled(
                self.dimension * self.dimension)
            for i in range(self.dimension):
                for j in range(self.dimension):
                    col = my_cmap.colors[i*self.dimension + j]

                    ax.plot(
                        self.time_frames, self.structure[:, i, j], color=col)
        ax.set_title("Observed Profiles", size=20, pad=10)
        plt.show()

    def plot(self) -> None:
        """Function to plot clusters map and entropy map if available, otherwise raise an exception.
        """

        if not self.clustered:
            raise Exception(
                "Error: you must first perform clustering and cluster matching.")

        if self.k <= 8:
            colors = ["#1f78b4", "#ff7f00", "#33a02c", "#e31a1c",
                      "#a6cee3", "#fdbf6f", "#b2df8a", "#e31a1c"]
            my_cmap = ListedColormap(colors, name="my_cmap").resampled(self.k)

        else:
            my_cmap = colormaps["viridis"].resampled(self.k)

        fig = plt.figure(layout="constrained", figsize=(10, 5))

        gs = gridspec.GridSpec(2, 2, figure=fig)

        ax1 = plt.subplot(gs[:, :1])
        ax2 = plt.subplot(gs[:, 1:])

        ax1.imshow(self.final_label, cmap=my_cmap, rasterized=True)
        im = ax2.imshow(self.spatial_entropy, cmap="YlOrBr_r", rasterized=True)
        cbar = ax2.figure.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.set_xticks([])
        ax2.set_yticks([])

        ax1.set_title("Cluster Map", size=20, pad=15)
        ax2.set_title("Normalized Entropy Map", size=20, pad=15)

        legend_elements = []
        for i in range(self.k):
            legend_elements.append(
                Patch(facecolor=my_cmap.colors[i], edgecolor='k', label=f"Cluster {i}"))

        ax1.legend(handles=legend_elements,
                   bbox_to_anchor=(-0.3, 0.5), loc="center left")

        plt.show()

    def func_plot(self) -> None:
        self.func_object.plot()
        plt.show()

    def save_in_gif(self, root: str, cmap_string: str = "plasma", milliseconds: int = 50):
        # Build location
        location = _build_path(root, self)

        # Build temporary location
        temp_location = os.path.join(location, "temp")
        if not os.path.exists(temp_location):
            # If not, then create it
            os.makedirs(temp_location)

        temp = (self.structure - np.min(self.structure)) / \
            (np.max(self.structure)-np.min(self.structure))

        _min, _max = np.amin(temp), np.amax(temp)

        for i, f in enumerate(temp):
            name = str(i) + ".png"
            save_location = os.path.join(temp_location, name)
            plt.imsave(save_location, f.T, vmax=_max,
                       vmin=_min, cmap=cmap_string)

        frames_gif = []
        for i in range(temp.shape[0]):
            name = str(i) + ".png"
            saved_location = os.path.join(temp_location, name)
            img = Image.open(saved_location)
            frames_gif.append(img)

        name = self.id_ + "_run" + self.run + ".gif"

        # Save as GIF
        frames_gif[0].save(os.path.join(location, name), save_all=True, append_images=frames_gif[1:], loop=0,
                           duration=milliseconds)

        shutil.rmtree(temp_location)
        del temp
        return None
    

    def set_p(self):
        fpca = FPCA(n_components=20)
        fpca.fit(self.func_object)
        a = np.cumsum(fpca.explained_variance_ratio_)
        self.p= min(a[a<self.explained_variance_pct].shape[0]+1,self.n-1)



def _build_path(root, lattice):
    location = os.path.join(root, lattice.id_)
    if not os.path.exists(location):
        os.mkdir(location)
    return location


def _change_label(arr: np.ndarray, new_label) -> np.ndarray:
    final = np.zeros_like(arr)
    for i, l in enumerate(new_label):
        final[arr == i] = l
    return final


def _cluster_mapping(array: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    matrix = contingency_matrix(array, baseline)
    new_label = np.argmax(matrix, axis=1)
    return new_label


def _random_nuclei(n: int, dim: int) -> np.ndarray:
    """Fuction to select n random nuclei for the Voronoi tesselation.

    Args:
        n (int): Cardinality of the sampling
        dim (int): Dimension of the lattice

    Returns:
        np.ndarray: Array with n random nuclei
    """
    assert n <= dim**2, "Error: selected nuclei must be less or equal to available points number."
    nuclei = []
    while len(nuclei) < n:
        x_coord = random.randint(0, dim-1)
        y_coord = random.randint(0, dim-1)
        if (x_coord, y_coord) not in nuclei:
            nuclei.append((x_coord, y_coord))
    return np.array(nuclei)


def _euclidean_distance(point1: np.ndarray, point2: np.ndarray) -> float:
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


def _max_min_distance(coordinates: np.ndarray):
    """
    Calcola la distanza euclidea massima e minima tra tutte le coordinate nella lista.
    """

    max_dist = float('-inf')
    min_dist = float('inf')

    for i in range(len(coordinates)):
        for j in range(i + 1, len(coordinates)):
            distance = _euclidean_distance(coordinates[i], coordinates[j])
            max_dist = max(max_dist, distance)
            min_dist = min(min_dist, distance)

    return max_dist, min_dist


# def _indexing_coordinates_array(coordinates_list: list):
#     ax_1 = []
#     ax_0 = []
#     for c in coordinates_list:
#         ax_0.append(c[0])
#         ax_1.append(c[1])
#     return ax_0, ax_1


def _nuclei_mapping(grid_points: np.ndarray, nuclei: np.ndarray) -> np.ndarray:
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


def _group(nuclei: np.ndarray, mapping: np.ndarray, grid_points: np.ndarray, fd) -> dict:
    """Function to compute weighted average representative for each nuclues and its relative neighborhood.

    Args:
        nuclei (np.ndarray): Array of nuclei coordinates
        mapping (np.ndarray): Mapping of each point in the grid to relative nuleus index
        grid_points (np.ndarray): Lattice grid_points
        structure (np.ndarray): Lattice structure

    Returns:
        dict: Return a dictionary in the form {int label of the nucleus, np.ndarray of the weighted average}
    """
    
    max_dist, min_dist = _max_min_distance(nuclei)

    # Find sigma
    sigma = max_dist/min_dist

    # Compute covariance matrix for bivariate normal distribution
    cov_matrix = sigma**2*np.identity(2)


    n = 0
    selected_points = grid_points[mapping == n]
    
    weights = _compute_gaussian_weight(loc=nuclei[n],
                                       covmatrix=cov_matrix,
                                       selected_points=selected_points)
    avg_function = stats.mean(fd[mapping == n], weights=weights)


    for n in range(1, np.max(mapping)+1):
        selected_points = grid_points[mapping == n]
        weights = _compute_gaussian_weight(loc=nuclei[n],
                                           covmatrix=cov_matrix,
                                           selected_points=selected_points)
        temp = stats.mean(fd[mapping == n], weights=weights)
        avg_function  = avg_function.concatenate(temp)
    
    return avg_function


def _compute_gaussian_weight(loc: np.ndarray, covmatrix: np.ndarray, selected_points: np.ndarray) -> np.ndarray:
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


def _kmeans_clustering(matrix: np.ndarray, k: int) -> np.ndarray:
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


def _do_fda(fd, p: int) -> np.ndarray:
    fpca = FPCA(n_components=p)
    fd_score = fpca.fit_transform(fd)
    return fd_score


def _unfold_clusters(labels: list, mapping: np.ndarray) -> np.ndarray:
    result = np.zeros_like(mapping)
    for i, e in enumerate(mapping):
        result[i] = labels[e]

    return result
