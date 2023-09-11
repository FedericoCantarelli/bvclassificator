import numpy as npcd ..


class Scenarios:
    def __init__(self):
        pass


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

    matrix = np.zeros((3, 3))
    matrix[:2, :2] = 1

    return np.where(matrix == 1)
