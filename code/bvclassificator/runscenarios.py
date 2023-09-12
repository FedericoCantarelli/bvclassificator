import numpy as np
import os


def generate_defect(dimension: int, defect_dimension: int, v_width: int, h_width: int):
    matrix = np.zeros((dimension, dimension), dtype=int)

    central_row = np.random.randint(
        defect_dimension // 2, dimension - defect_dimension // 2)

    central_column = np.random.randint(
        defect_dimension // 2, dimension - defect_dimension // 2)

    half_defect = defect_dimension // 2
    half_v_width = v_width // 2
    half_h_width = h_width // 2

    matrix[central_row-half_defect:central_row+half_defect+1,
           central_column - half_h_width:central_column+half_h_width+1] = 1
    matrix[central_row-half_v_width:central_row+half_v_width + 1,
           central_column-half_defect:central_column+half_defect+1] = 1

    axis = np.where(matrix == 1)

    return matrix, [j + i * dimension for i, j in zip(axis[0], axis[1])]


def generate_combinations(scenario: str, n_list: list, b_list: list):
    params = []
    for n in n_list:
        for b in b_list:
            params.append((scenario, n, b))
    return params


if __name__ == "__main__":
    matrix, combination = generate_defect(dimension=5,
                                          defect_dimension=2,
                                          v_width=2,
                                          h_width=2)
