from bvclassificator.simulation import Profile
from bvclassificator.algorithm import Lattice
import time


# Global Parameters
DIMENSION = 50

if __name__ == "__main__":
    grid = Lattice(DIMENSION,
                   time_period=60,
                   fps=1,
                   simulation=True)

    profile_list = []
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            c = Profile(x=i,
                        y=j,
                        time_period=60,
                        fps=1)
            if c.index(DIMENSION) in [0, 1, 2, 10, 11, 12, 20, 21, 22]:
                c.simulate(0, 1, True, 0, tau=50, h=0.95)

            else:
                c.simulate(0, 1, True, 1, tau=0, h=0.95)

            profile_list.append(c)

    grid.build(profile_list=profile_list)

    grid.init_algo(n=30,
                   k=2,
                   p=3,
                   b=10)

    start_time = time.time()
    grid.cluster_now()

    print("Time: {} seconds".format(time.time()-start_time))

    print("Classification rate")
    print(grid.classification_rate_)

    grid.plot()
