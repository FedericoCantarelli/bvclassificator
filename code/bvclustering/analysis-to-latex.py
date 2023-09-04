import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import math
from skfda.preprocessing.dim_reduction import FPCA
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans
from skfda.representation.grid import FDataGrid
from scipy.stats import multivariate_normal


def plot_results(df):
    """_summary_

    Args:
        df (pd.DataFrame): Dataframe in the format (perc_i | label | entropy | normalized_entropy | id | x | y | t | is_in_control)
    """

    matrix_entropy = df.pivot(index='y', columns='x',
                              values='entropy').values

    matrix_cluster = df.pivot(index='y', columns='x', values='label').values

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)

    cmap = plt.cm.get_cmap('Dark2', np.max(matrix_cluster) + 1)
    im = ax1.imshow(matrix_cluster, cmap=cmap)

    ax1.set_xticks([], labels=[])
    ax1.set_yticks([], labels=[])

    ax1.set_title("Clustering Result")

    im2 = ax2.imshow(matrix_entropy)

    ax2.set_xticks([], labels=[])
    ax2.set_yticks([], labels=[])

    cbar2 = ax2.figure.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.remove()
    ax2.set_title("Spatial Entropy")

    fig.tight_layout()
    plt.show()


def get_spatial_entropy(df):
    """_summary_

    Args:
        df (pd.DataFrame): Dataframe in the format (perc_i | label) where perc_i is the frequency with which the site is labelled as cluster i

    Returns:
        _type_: _description_
    """
    columns = df.columns.drop("label")
    entropies = []

    for i in range(df.shape[0]):
        p_vector = np.array(df[columns].iloc[i].values) + 1e-10
        log_p = np.log10(p_vector)
        e = -np.sum(p_vector*log_p)
        entropies.append(e)

    return entropies


def progress_bar(percent, width=40):
    """
    Function to print a simple progress bar in the output console

    Args:
        percent (int): Percentage of the completed work.
        width (int, optional): Width of the progress bar. Defaults to 40.
    """
    percent = int(percent*100)
    left = width * percent // 100
    right = width - left
    tags = "=" * left
    spaces = " " * right
    percents = f"{percent:.0f}%"
    print("\r[", tags, spaces, "] ", percents, sep="", end="", flush=True)


def euclidean_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        x (np.ndarray): First one-dimensional binary array with coordinates tuples. 
        y (np.ndarray): Second one-dimensional binary array with coordinates tuples.

    Returns:
        np.ndarray: Euclidean distance matrix of all the points in the given vectors.
    """

    n = len(x)
    m = len(y)
    distance = np.zeros(shape=(n, m))

    for i in range(n):
        for j in range(m):
            distance[i, j] = math.sqrt(
                (x[i][0] - y[j][0]) ** 2 + (x[i][1] - y[j][1]) ** 2)
    return distance


def return_cluster_mapping(reference, testing):
    """_summary_

    Args:
        reference (pd.Series): The label results of the first clustering
        testing (pd.Series): The label results of the second clustering

    Returns:
        dict: Return the mapping_dict in the format {bootstra_label:reference_label}
    """

    # Contingency matrix automatically reorder cluster labels
    matrix = contingency_matrix(testing, reference)

    mapping_dict = dict()

    unique_reference = np.asarray(list(set(reference)))
    unique_testing = np.asarray(list(set(testing)))

    indexes_min = np.argmax(matrix, axis=1)

    for key, item in zip(unique_testing, unique_reference[indexes_min]):
        mapping_dict[key] = item

    return mapping_dict


class Profile:
    def __init__(self, profile_id: str, x: float, y: float, time: float, fps: float, profile: np.ndarray = None) -> None:
        self.profile_id = profile_id
        self.x = x
        self.y = y
        self.profile = profile

        self.time = time
        self.fps = fps

    @property
    def position(self):
        return (self.x, self.y)

    def dump_to_json(self, path: str):
        location = os.path.join(path, "json_output")
        if not os.path.exists(location):
            os.makedirs(location)

        dictionary = self.__dict__
        for k in dictionary.keys():
            if type(dictionary[k]) == np.ndarray:
                print(f"{k} is array")
                dictionary[k] = dictionary[k].tolist()

        with open(os.path.join(location, self.profile_id + ".json"), "w") as f:
            json.dump(dictionary, f, indent=4)

    @classmethod
    def fill_from_json(cls, path: str):
        with open(path, "r") as f:
            dictionary = json.load(f)

        return cls(dictionary["profile_id"],
                   dictionary["x"],
                   dictionary["y"],
                   np.ndarray(dictionary["profile"]))


class Structure:
    def __init__(self, profiles: list) -> None:
        self.profiles = profiles
        self.coordinates = dict(id=[],
                                x=[],
                                y=[],
                                t=[])

        for each_profile in profiles:
            self.coordinates["id"].append(each_profile.profile_id)
            self.coordinates["x"].append(each_profile.x)
            self.coordinates["y"].append(each_profile.y)
            self.coordinates["t"].append(each_profile.position)

        self.df_coordinates = pd.DataFrame.from_dict(self.coordinates)
        self.distance_matrix = euclidean_distance_matrix(
            self.df_coordinates.t, self.df_coordinates.t)

        self.time = self.profiles[0].time
        self.fps = self.profiles[0].fps

    # def print_df(self):
    #     print(self.df_coordinates)

    # def print_dm(self):
    #     print(self.distance_matrix)

    def index_from_coordinates(self, coordinates: tuple):
        return (int(self.df_coordinates[self.df_coordinates.t == coordinates].index.values[0]))

    def id_from_coordinates(self, coordinates: tuple):
        return (self.df_coordinates[self.df_coordinates.t == coordinates].id.values[0])

    def coordinates_from_id(self, search_id: int):
        return (self.df_coordinates[self.df_coordinates.id == str(search_id)].t.values[0])

    def cluster_now(self, n: int, bootstrap: int, k: int, p: int):
        """Function to perform a bagging-voronoi clustering algorithm.

        Args:
            n (int): Number of initial centroid for the Voronoi tessellation
            b (int): Number of bootstrap of the algorithm
            k (int): Nuber of cluster
            p (int): Number of principal components.

        Returns:
            pd.DataFrame, float: Return a dataframe with clustered observations and space entropy in the form 
            (perc_i | label | entropy | normalized_entropy | id | x | y | t ) and average normalized entropy. 
        """

        df = pd.DataFrame()

        print("Bootstrapping...")

        # Iterate the procedure b times
        for b in range(bootstrap):

            # Print progress bar as UI
            progress_bar(b/bootstrap)

            # List for selected centroid
            selected_centroid_voronoi = []

            # Select n centroid for voronoi tesselation
            for i in range(n):
                # Select random index
                num = np.random.randint(len(self.profiles))

                # If selected index is selected for the first time, append to the list
                if num not in selected_centroid_voronoi:
                    selected_centroid_voronoi.append(num)

                #  If it is already in the list, iterate untill a new centroid is selected
                else:
                    while num in selected_centroid_voronoi:
                        num = np.random.randint(len(self.profiles))
                    selected_centroid_voronoi.append(num)

            # Dictionary for grouping profiles to the closest centroid
            # format {profile:closest centroid}
            dictionary = dict()

            #  Iterate over all profiles and assign them to the closest centroid
            for i in range(len(self.profiles)):
                index_min = np.argmin(
                    self.distance_matrix[i, selected_centroid_voronoi])
                dictionary[i] = selected_centroid_voronoi[index_min]

            # Group the previous dictionary in the format
            # {centroid: closest profiles}
            grouped_dict = dict()

            # Perform grouping
            for key, value in dictionary.items():
                if value not in grouped_dict:
                    grouped_dict[value] = [key]
                else:
                    grouped_dict[value].append(key)

            # Compute representative function for each centroid
            avg_dictionary = dict()

            # Compute
            centroid_coordinates_list = [self.coordinates_from_id(
                key) for key in grouped_dict.keys()]

            # Find maximum distance and minimum distance between al the centroids
            temp_matrix = euclidean_distance_matrix(
                centroid_coordinates_list, centroid_coordinates_list)
            dist_max_centroid = np.max(temp_matrix)

            temp_matrix = temp_matrix[temp_matrix != 0]
            dist_min_centroid = np.min(temp_matrix)

            del temp_matrix

            #  Compute sigma
            sigma = dist_max_centroid/dist_min_centroid

            # Compute covariance matrix for bivariate normal distribution
            cov_matrix = sigma**2*np.identity(2)

            # Compute representetive function for each centroid
            for j in grouped_dict.keys():

                # Gaussian weights centered in Zi
                means = list(self.coordinates_from_id(j))

                # Compute the weight
                bivariate_gaussian = multivariate_normal(
                    mean=means, cov=cov_matrix)

                # Keeo track of profile
                temp_list = []

                # Keep track of weights
                weights_list = []

                # Multiply each profile for the correspondent weight
                for index in grouped_dict[j]:
                    pos = self.coordinates_from_id(index)
                    w = bivariate_gaussian.pdf(pos)
                    weights_list.append(w)
                    temp_list.append(self.profiles[index].profile*w)

                temp_array = np.array(temp_list)

                # Compute final dictionary
                avg_dictionary[j] = np.sum(
                    temp_array, axis=0)/sum(weights_list)

            # Create a FData object
            fda_matrix = np.zeros(
                (len(avg_dictionary.keys()), self.time * self.fps))
            for i, key in enumerate(avg_dictionary.keys()):
                fda_matrix[i] = avg_dictionary[key]

            fd = FDataGrid(fda_matrix,
                           np.arange(0, self.time, 1/self.fps))

            #  Compute FPCA
            fpca = FPCA(n_components=p)

            # Compute scores
            fd_score = fpca.fit_transform(fd)

            df_score = pd.DataFrame(fd_score, columns=[
                                    "s" + str(i+1) for i in range(p)])

            # Keep track of centroid for debugging reason
            df_score["centroid"] = avg_dictionary.keys()

            # Find clusterin cluster
            kmeans = KMeans(n_clusters=k,
                            n_init="auto").fit(df_score[df_score.columns[:-1]])

            # Find the mapping dictionary for centroid and cluster labels
            # The format is {centroid:cluster label}
            mapping_dict = dict()
            for key, cluster in zip(grouped_dict.keys(), kmeans.labels_):
                mapping_dict[key] = cluster

            #  Find the single site to centroid mapping dictionary

            cluster_dict = dict()
            for key in grouped_dict.keys():
                for i in grouped_dict[key]:
                    cluster_dict[i] = key

            #  Sort the dictionary according in order to have the all the sites in order
            sorted_cluster_dict = dict(sorted(cluster_dict.items()))

            df["boot_" + str(b)] = sorted_cluster_dict.values()
            df["b_" + str(b)] = df["boot_" + str(b)].replace(mapping_dict)

            df.drop(["boot_" + str(b)], axis=1, inplace=True)

        # Perform cluster matching using contingency matrix
        print("\nCluster matching...")

        # Save first bootstrap as reference
        reference = df.loc[:, "b_0"]

        # For all the remaining bootstraps do clusters matching
        for col in df.columns[1:]:

            # Find the map
            mapping = return_cluster_mapping(reference, df[col])

            # Replace the labels of the old cluster to the label of the new cluster
            df[col] = df[col].replace(mapping)

        df_clust = pd.DataFrame()

        #  Find percentage of each site to be labeled with that specific label
        for j in range(k):
            list_temp = []
            for i in range(df.shape[0]):
                list_temp.append(
                    sum(df.iloc[i].values == j)/len(df.iloc[i].values))
            df_clust["perc_" + str(j)] = list_temp

        final_cluster = []

        for i in range(df_clust.shape[0]):
            final_cluster.append(np.argmax(df_clust.iloc[i].values))

        #  Add final label to the dataframe
        df_clust["label"] = final_cluster

        #  Compute spatial entropy and add column
        df_clust["entropy"] = get_spatial_entropy(df_clust)

        #  Compute normalized spatial entropy and add column
        df_clust["normalized_entropy"] = df_clust["entropy"]/np.log10(k)

        #  Add coordinates to final dataframe
        df_clust = pd.concat([df_clust, self.df_coordinates], axis=1)

        # Return clustered dataframe and average normalized entropy
        return df_clust, np.sum(df_clust.normalized_entropy)/len(self.profiles)

    def evaluate(self, ground_truth: list) -> tuple:
        pass
