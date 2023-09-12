from bvclassificator.simulation import Profile
from bvclassificator.algorithm import Lattice
from bvclassificator.runscenarios import generate_defect, generate_combinations

# Global Parameters
DIMENSION = 10
SCENARIOS = 3
SCENARIO_NAME = "S1"
TAU = 60
DEFECT_PARAMS = (4, 2, 2)
ROOT = "/Users/federicocantarelli/Documents/bvclassificator/output"


def create_list(index):
    profile_list = []
    for i in range(DIMENSION):
        for j in range(DIMENSION):
            c = Profile(x=i,
                        y=j,
                        time_period=60,
                        fps=1)

            if c.index(DIMENSION) in index:
                c.simulate(loc=0,
                           scale=1,
                           with_noise=True,
                           label=1,
                           tau=TAU,
                           h=0.95)

            else:
                c.simulate(loc=0,
                           scale=1,
                           with_noise=True,
                           label=0,
                           tau=0,
                           h=0.95)

            profile_list.append(c)

    return profile_list


if __name__ == "__main__":

    combinations = generate_combinations(scenario=SCENARIO_NAME,
                                         n_list=[50],
                                         b_list=[100])

    for i, c in enumerate(combinations):
        for run in range(SCENARIOS):
            matrix, index = generate_defect(dimension=DIMENSION,
                                            defect_dimension=DEFECT_PARAMS[0],
                                            v_width=DEFECT_PARAMS[1],
                                            h_width=DEFECT_PARAMS[2])

            grid = Lattice(dimension=DIMENSION,
                           time_period=60,
                           fps=1,
                           simulation=True,
                           id_=c[0] + "_params" + str(i),
                           run=str(run))

            grid.build(profile_list=create_list(index))
            grid.init_algo(n=c[1],
                           k=2,
                           p=2,
                           b=c[2])
            grid.cluster_now()
            grid.save_in_gif(root=ROOT)
            grid.save_log_json(root=ROOT)

            grid.plot()

            print("\n\n\n Run {}".format(run))
            print("Defect matrix:\n {}".format(matrix))

            print("GT label:\n {}".format(grid.label_matrix))

            print("Final labels:\n {}".format(grid.final_label))

            print("Entropy: {}".format(grid.average_normalized_entropy_))
            print("Classrate: {}".format(grid.classification_rate_))
