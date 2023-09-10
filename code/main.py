from bvclassificator.simulation import Profile
from bvclassificator.algorithm import Lattice

# Global Parameters
DIMENSION = 3

if __name__ == "__main__":
    grid = Lattice(DIMENSION,
                   time_period=60,
                   fps=1,
                   b=7,
                   k=5)

    profile_list = []
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            c = Profile(x=i,
                        y=j,
                        time_period=60,
                        fps=1)
            if c.index(DIMENSION) in [0, 1, 3, 4]:
                c.simulate(0, 1, False, 1, in_control=False)
            else:
                c.simulate(0, 1, False, 0, in_control=True)

            profile_list.append(c)

    grid.build(profile_list=profile_list)

    grid.init_algo(n=8,
                   k=2,
                   p=2,
                   b=100)

    grid.cluster_now()
    # print("Before cluster matching")
    # print(grid.labels, end = "\n\n")
    grid.do_cluster_matching()

    # print("After cluster matching")
    #  print(grid.labels)

    grid.find_final_label()
    # print("Percentages")
    #  print(grid.percentage)

    # print("Final Label")
    # print(grid.final_label)

    print("Clustering result:")
    print(grid.labels_)

    grid.find_entropy()
    print("Average normalized entropy")
    print(grid.average_normalized_entropy_)

    print(grid.labels)
    
    print("Classification rate")
    print(grid.classification_rate_)
    grid.plot()
