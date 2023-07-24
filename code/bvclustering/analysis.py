import numpy as np
import matplotlib.pyplot as plt
import json
import os
import pandas as pd
import math
import skfda
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans


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


class Profile:
    def __init__(self, profile_id: str, x: float, y: float, profile: np.ndarray = None, is_in_control: bool = True) -> None:
        self.profile_id = profile_id
        self.is_in_control = is_in_control
        self.x = x
        self.y = y
        self.profile = profile

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

    def print_df(self):
        print(self.df_coordinates)

    def print_dm(self):
        print(self.distance_matrix)

    def index_from_coordinates(self, coordinates: tuple):
        return (int(self.df_coordinates[self.df_coordinates.t == coordinates].index.values[0]))

    def id_from_coordinates(self, coordinates: tuple):
        return (self.df_coordinates[self.df_coordinates.t == coordinates].id.values[0])

    def cluster_now(self, n: int, bootstrap: int, k: int, fpca_percentage: float):
        """Function to perform a bagging-voronoi clustering algorithm.

        Args:
            n (int): Number of initial centroid for the Voronoi tessellation
            b (int): Number of bootstrap of the algorithm
            k (int): Nuber of cluster
            fpca_percentage (float): Percentage of variance to retain in the FPCA

        Returns:
            tuple: Return a dataframe with clustered observations and space entropy in the form (x | y | cluster | space entropy). 
        """
        assert n < len(self.profiles)//3
        df = pd.DataFrame()

        for b in range(bootstrap):
            selected_centroid_voronoi = []
            for i in range(n):
                num = np.random.randint(len(self.profiles))
                if num not in selected_centroid_voronoi:
                    selected_centroid_voronoi.append(num)
                
                else:
                    while num in selected_centroid_voronoi:
                        num = np.random.randint(len(self.profiles))
                    selected_centroid_voronoi.append(num)
        

            grouped_dict = dict()

            for i in range(len(self.profiles)):
                index_min = 
                grouped_dict[c] = 


        df["clustering_n_" + str(b)] = # clustering result


        print("Begin cluster matching...")


def main():
    a = Profile(profile_id="a",
                x=0,
                y=0,
                profile=np.arange(0, 60, 1800),
                is_in_control=True)

    b = Profile(profile_id="b",
                x=4,
                y=0,
                profile=np.arange(0, 60, 1800),
                is_in_control=False)

    c = Profile(profile_id="c",
                x=5,
                y=2,
                profile=np.arange(0, 60, 1800),
                is_in_control=True)

    strc = Structure([a, b, c])
    print(strc.index_from_coordinates((5, 2)))


if __name__ == "__main__":
    main()
