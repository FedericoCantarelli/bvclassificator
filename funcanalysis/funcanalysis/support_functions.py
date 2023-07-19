###########################################################################################
# Support function package                                                                #
# Package to provide usefull functions for the package Lattice simulation package - V 1.0 #
# This package was made to simulate data for a new approach of QC of functional data      #
#                                                                                         #
# This is part of my MSc thesis project in Management Engineer @ POLIMI                   #
# A.Y. 2022/2023                                                                          #
#                                                                                         #
# Copyright by Federico Cantarelli                                                        #
# Maintained by Federico Cantarelli                                                       #
# Reach me if you have suggestion of if you feel lonely: fede.cantarelli98@gmail.com      #
#                                                                                         #
# Dreamed, designed, implemented and ran in 2023 in ["Casalmaggiore", "Milano"]           #
#                                                                                         #
# Project end date: ??/??/??                                                              #
###########################################################################################
import numpy as np
import math
import hashlib

def euclidean_distance_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        x (np.ndarray): First one-dimensional binary array with coordinates tuples. 
        y (np.ndarray): Second one-dimensional binary array with coordinates tuples.

    Returns:
        np.ndarray: Euclidean distance matrix of all the points in the given vectors.
    """

    n = len(x[0])
    m = len(y[0])
    distance = np.zeros(shape=(n, m))

    for i in range(n):
        for j in range(m):
            distance[i, j] = math.sqrt((x[0][i] - y[0][j]) ** 2 + (x[1][i] - y[1][j]) ** 2)
    return distance


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


def get_array_hash(array: np.ndarray):

    """
    Function to compute the hash of a numpy array.

    Args:
        array (np.ndarray): Array to convert in hash

    Returns:
        hashlib.Hash: Hash of the numpy array given as argument
    """

    array_bytes = array.tobytes()
    hash_object = hashlib.md5(array_bytes)
    hash_object.digest()
    return hash_object


def get_layer_max(array: np.ndarray) -> np.ndarray:
    """
    Function to compute the maximum of each layer and return the array.

    Args:
        array (np.ndarray): Numpy array, suggested Hausdorff deviation map for the single cell.

    Returns:
        np.ndarray: Array with max of each layer.
    """
   
    result = np.zeros(shape=array.shape[0])
    for i in range(array.shape[0]):
        result[i] = np.max(array[i])

    return result


def get_layer_min(array: np.ndarray) -> np.ndarray:
    """
    Function to compute the minimum of each layer and return the array.

    Args:
        array (np.ndarray): Numpy array, suggested Hausdorff deviation map for the single cell.

    Returns:
        np.ndarray: Array with min of each layer.
    """
    
    result = np.zeros(shape=array.shape[0])
    for i in range(array.shape[0]):
        result[i] = np.min(array[i])
    return result



def get_layer_avg(array: np.ndarray) -> np.ndarray:
    """
    Function to compute the avg deviation of each layer and return the array.

    Args:
        array (np.ndarray): Numpy array, suggested Hausdorff deviation map for the single cell.

    Returns:
        np.ndarray: Array with avg deviation of each layer.
    """
    
    result = np.zeros(shape=array.shape[0])
    for i in range(array.shape[0]):
        result[i] = np.mean(array[i])
 


def get_layer_count(array: np.ndarray) -> np.ndarray:
    """
    Function to count deviation pixel for each layer and return the array.

    Args:
        array (np.ndarray): Numpy array, suggested tme or tml deviation map for the single cell.

    Returns:
        np.ndarray: Array with count of deviation pixels for each layer.
    """
    result = np.zeros(shape=array.shape[0])
    for i in range(array.shape[0]):
        result[i] = np.sum(array[i].flatten())
    return result


def convert_centroid_to_id(coordinates:list, order:int) -> int:
    """
    Function to convert coordinates cetroid into a cell id during structure simulation.

    Args:
        coordinates (list): Tuple of coordinates in the space. Remember, it starts from zero.
        order (int): Order of the lattice structure.

    Returns:
        int: Cell_id
    """    

    cell_id = coordinates[0]*order**2 + coordinates[2] * order+ coordinates[1] + 1 
    return cell_id


def distance_3d(x:tuple, y:tuple):
    """
    Compute Euclidean distance in 3d given two tuples of coordinates.

    Args:
        x (tuple): First point coordinates
        y (tuple): Second point coordinates
    """
    
    return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2+(x[2]-y[2])**2)