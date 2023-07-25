from typing import Any
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
import warnings
warnings.filterwarnings('ignore')


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def plot_result(df):

    matrix = df.pivot(index='y', columns='x', values='entropy').values

    fig, ax = plt.subplots()

    x_tick = [str(i) for i in range(matrix.shape[1])]
    y_tick = [str(i) for i in range(matrix.shape[0])]

    im, cbar = heatmap(matrix, x_tick, y_tick, ax=ax,
                       cmap="viridis", cbarlabel="")

    fig.tight_layout()
    plt.show()

    ax.set_xticks(np.arange(matrix.shape[0]), labels=[
                  str(i) for i in range(matrix.shape[0])])
    ax.set_yticks(np.arange(matrix.shape[1]), labels=[
                  str(i) for i in range(matrix.shape[1])])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    fig.colorbar(matrix, ax=ax, orientation='vertical', fraction=.1)

    plt.show()


def get_spatial_entropy(df):
    columns = df.columns.drop("label")
    entropies = []

    for i in range(df.shape[0]):
        p_vector = np.array(df[columns].iloc[i].values)
        if any(p_vector == 1):
            entropies.append(0)

        else:
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
    def __init__(self, profile_id: str, x: float, y: float, time: float, fps: float, profile: np.ndarray = None, is_in_control: bool = True) -> None:
        self.profile_id = profile_id
        self.is_in_control = is_in_control
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
                   np.ndarray(dictionary["profile"]),
                   dictionary["is_in_control"])


class Structure:
    def __init__(self, profiles: list) -> None:
        self.profiles = profiles
        self.coordinates = dict(id=[],
                                x=[],
                                y=[],
                                t=[],
                                is_in_control=[])

        for each_profile in profiles:
            self.coordinates["id"].append(each_profile.profile_id)
            self.coordinates["x"].append(each_profile.x)
            self.coordinates["y"].append(each_profile.y)
            self.coordinates["t"].append(each_profile.position)
            self.coordinates["is_in_control"].append(
                each_profile.is_in_control)

        self.df_coordinates = pd.DataFrame.from_dict(self.coordinates)
        self.distance_matrix = euclidean_distance_matrix(
            self.df_coordinates.t, self.df_coordinates.t)

        self.time = self.profiles[0].time
        self.fps = self.profiles[0].fps

    def print_df(self):
        print(self.df_coordinates)

    def print_dm(self):
        print(self.distance_matrix)

    def index_from_coordinates(self, coordinates: tuple):
        return (int(self.df_coordinates[self.df_coordinates.t == coordinates].index.values[0]))

    def id_from_coordinates(self, coordinates: tuple):
        return (self.df_coordinates[self.df_coordinates.t == coordinates].id.values[0])

    def cluster_now(self, n: int, bootstrap: int, k: int, p: int):
        """Function to perform a bagging-voronoi clustering algorithm.

        Args:
            n (int): Number of initial centroid for the Voronoi tessellation
            b (int): Number of bootstrap of the algorithm
            k (int): Nuber of cluster
            p (int): Number of principal componets

        Returns:
            tuple: Return a dataframe with clustered observations and space entropy in the form (x | y | cluster | space entropy) and average normalized entropy. 
        """

        # Dataframe for bootstrapping result
        df = pd.DataFrame()

        print("Bootstrapping...")
        for b in range(bootstrap):
            progress_bar(b/bootstrap)
            selected_centroid_voronoi = []
            for i in range(n):
                num = np.random.randint(len(self.profiles))
                if num not in selected_centroid_voronoi:
                    selected_centroid_voronoi.append(num)

                else:
                    while num in selected_centroid_voronoi:
                        num = np.random.randint(len(self.profiles))
                    selected_centroid_voronoi.append(num)

            dictionary = dict()

            for i in range(len(self.profiles)):
                index_min = np.argmin(
                    self.distance_matrix[i, selected_centroid_voronoi])
                dictionary[i] = selected_centroid_voronoi[index_min]

            grouped_dict = dict()
            for key, value in dictionary.items():
                if value not in grouped_dict:
                    grouped_dict[value] = [key]
                else:
                    grouped_dict[value].append(key)

            avg_dictionary = dict()

            for j in grouped_dict.keys():
                temp_list = []
                for index in grouped_dict[j]:
                    temp_list.append(self.profiles[index].profile)

                temp_array = np.array(temp_list)
                avg_dictionary[j] = np.mean(temp_array, axis=0)

            # Create a FData object
            fda_matrix = np.zeros(
                (len(avg_dictionary.keys()), self.time * self.fps))
            for i, key in enumerate(avg_dictionary.keys()):
                fda_matrix[i] = avg_dictionary[key]

            fd = FDataGrid(fda_matrix,
                           np.arange(0, self.time, 1/self.fps))

            #  Compute FPCA
            fpca = FPCA(n_components=p)
            fd_score = fpca.fit_transform(fd)
            df_score = pd.DataFrame(fd_score, columns=[
                                    "s" + str(i+1) for i in range(p)])
            df_score["centroid"] = avg_dictionary.keys()

            # Compute cluster
            kmeans = KMeans(n_clusters=k, random_state=0,
                            n_init="auto").fit(df_score[df_score.columns[:-1]])

            mapping_dict = dict()

            for key, cluster in zip(grouped_dict.keys(), kmeans.labels_):
                mapping_dict[key] = cluster

            cluster_dict = dict()
            for key in grouped_dict.keys():
                for i in grouped_dict[key]:
                    cluster_dict[i] = key

            sorted_cluster_dict = dict(sorted(cluster_dict.items()))

            df["boot_" + str(b)] = sorted_cluster_dict.values()
            df["b_" + str(b)] = df["boot_" + str(b)].replace(mapping_dict)

            df.drop(["boot_" + str(b)], axis=1, inplace=True)

        print("\nBegin clustering matching")
        reference = df.loc[:, "b_0"]

        for col in df.columns:
            mapping = return_cluster_mapping(reference, df[col])
            df[col] = df[col].replace(mapping)

        df_clust = pd.DataFrame()

        for j in range(k):
            list_temp = []
            for i in range(df.shape[0]):
                list_temp.append(
                    sum(df.iloc[i].values == j)/len(df.iloc[i].values))
            df_clust["perc_" + str(j)] = list_temp

        final_cluster = []

        for i in range(df_clust.shape[0]):
            final_cluster.append(np.argmax(df_clust.iloc[i].values))

        df_clust["label"] = final_cluster

        df_clust["entropy"] = get_spatial_entropy(df_clust)

        df_clust["normalized_entropy"] = df_clust["entropy"]/np.log10(k)

        df_clust = pd.concat([df_clust, self.df_coordinates], axis=1)

        return df_clust, np.sum(df_clust.normalized_entropy)/len(self.profiles)


class Result:
    def __init__(self) -> None:
        pass


def main():
    a = Profile(profile_id="a",
                x=0,
                y=0,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    b = Profile(profile_id="b",
                x=1,
                y=0,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=False)

    c = Profile(profile_id="c",
                x=2,
                y=0,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    d = Profile(profile_id="d",
                x=3,
                y=0,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    e = Profile(profile_id="e",
                x=0,
                y=1,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    f = Profile(profile_id="f",
                x=1,
                y=1,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=False)

    g = Profile(profile_id="g",
                x=2,
                y=1,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    h = Profile(profile_id="h",
                x=3,
                y=1,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    i = Profile(profile_id="i",
                x=0,
                y=2,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    l = Profile(profile_id="l",
                x=1,
                y=2,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=False)

    m = Profile(profile_id="m",
                x=2,
                y=2,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    n = Profile(profile_id="n",
                x=3,
                y=2,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    o = Profile(profile_id="o",
                x=0,
                y=3,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    p = Profile(profile_id="p",
                x=1,
                y=3,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=False)

    q = Profile(profile_id="q",
                x=2,
                y=3,
                time=60,
                fps=1,
                profile=np.arange(0, 60),
                is_in_control=True)

    r = Profile(profile_id="r",
                x=3,
                y=3,
                time=60,
                fps=1,
                profile=np.arange(10, 70),
                is_in_control=True)

    strc = Structure([a, b, c, d, e, f, g, h, i, l, m, n, o, p, q, r])

    df, avg_entropy = strc.cluster_now(10, 10, 2, 3)

    print(df)
    print(avg_entropy)

    # df = pd.DataFrame(dict(a=[0.25, 1, 0, 0.4],
    #                        b=[0.25, 0, 1, 0.2],
    #                        c=[0.25, 0, 0, 0.2],
    #                        d=[0.25, 0, 0, 0.2],
    #                        label=[1, 0, 1, 0]))

    # print(get_spatial_entropy(df))

    print(plot_grid(df))


if __name__ == "__main__":
    main()
