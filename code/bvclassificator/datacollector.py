import numpy as np
import matplotlib.pyplot as plt
import os
import json


def _generate_defect(dimension: int, defect_dimension: int, width: int):

    matrix = np.zeros((dimension, dimension), dtype=int)

    central_row = np.random.randint(
        defect_dimension // 2, dimension - defect_dimension // 2)
    central_column = np.random.randint(
        defect_dimension // 2, dimension - defect_dimension // 2)

    half_defect = defect_dimension // 2
    half_width = width // 2

    matrix[central_row-half_defect:central_row+half_defect+1, central_column -
            half_width:central_column+half_width+1] = 1
    matrix[central_row-half_width:central_row+half_width +
            1, central_column-half_defect:central_column+half_defect+1] = 1

    plt.imshow(matrix)
    plt.show()

    r = np.where(matrix==1)

    return [(i,j) for i,j in zip(r[0], r[1])]



class DataCollector:
    def __init__(self, root:str) -> None:
        pass




if __name__=="__main__":
    while True:
        prova = _generate_defect(50, 9, 3)
        print(prova)
