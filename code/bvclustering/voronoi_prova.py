from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt
import time

DIMENSION = 3

def init_index():
    index = [i + DIMENSION *
             j for j in range(DIMENSION) for i in range(DIMENSION)]
    return index


# Punti utente scelti
user_points = np.array([(0, 0), (2,3)])

# # Punti sulla griglia
x_range = np.arange(0, 5)
y_range = np.arange(0, 5)
grid_points = np.array([(x, y) for x in x_range for y in y_range])

# starttime=time.time()
# # Calcola le assegnazioni di Voronoi
# vor = Voronoi(user_points)

# # Trova il punto utente più vicino a ciascun punto sulla griglia
assignments = []
for grid_point in grid_points:
    min_distance = float('inf')
    nearest_user_point = None
    for i, user_point in enumerate(user_points):
        distance = np.linalg.norm(grid_point - user_point)
        if distance < min_distance:
            min_distance = distance
            nearest_user_point = i
    assignments.append(nearest_user_point)

# duration = time.time() - starttime
# print(f"Time: {duration} s")
# # Disegna le celle di Voronoi e i punti sulla griglia
# voronoi_plot_2d(vor)
# plt.scatter(grid_points[:, 0], grid_points[:, 1],
#             c=assignments, cmap='viridis', marker='.')
# plt.scatter(user_points[:, 0], user_points[:, 1], c='red', marker='o')
# plt.xlim(0, 127)
# plt.ylim(0, 127)
# plt.gca().set_aspect('equal', adjustable='box')
# plt.show()

print(assignments)