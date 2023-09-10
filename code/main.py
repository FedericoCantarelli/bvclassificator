from bvclassificator.simulation import Profile
from bvclassificator.algorithm import Lattice
# from bvclassificator.algorithm import cluster_now
from easy_implementation import cluster_now

# Global Parameters
DIMENSION = 3

grid = Lattice(DIMENSION, 60, 7, 5)

profile_list = []
for i in range(DIMENSION):
    for j in range(DIMENSION):
        c = Profile(i, j, 60)
        if c.index in [0, 1, 3, 4]:
            c.simulate(0, 1, False, 1, in_control=False)
        else:
            c.simulate(0, 1, False, 0, in_control=True)
            profile_list.append(c)

grid.build(profile_list=profile_list)
# print(f"Grid points is {grid.grid_points}")

cluster_now(grid, 8, 2, 1)
# print("Before cluster matching")
# print(grid.labels, end = "\n\n")
grid.do_cluster_matching()

print("After cluster matching")
print(grid.labels)

grid.find_final_label()
print("Percentages")
print(grid.percentage)

print("Final Label")
print(grid.final_label)

grid.find_entropy()

print("Entropy")
print(grid.normalized_spatial_entropy)
