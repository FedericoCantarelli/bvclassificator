from bvclassificator.simulation import Profile
from bvclassificator.algorithm import Lattice
from bvclassificator.runscenarios import generate_defect, generate_combinations

# Global Parameters
DIMENSION = 50
SCENARIOS = 30
SCENARIO_NAME = "S10"
TAU = 5
TAU_1 = 20
TAU_2 = 50
TAU_3 = 20 

DEFECT_PARAMS = (2, 2, 2)
DEFECT_PARAMS_1 = (4, 2, 2 )
DEFECT_PARAMS_2 = (4, 2, 2)
DEFECT_PARAMS_3 = (10, 4, 4)
ROOT = "/Users/federicocantarelli/Desktop/GIF"


def create_list(index, index_1, index_2, index_3):
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
              
              elif c.index(DIMENSION) in index_1:
                     c.simulate(loc=0,
                                scale=1,
                                with_noise=True,
                                label=2,
                                tau=TAU_1,
                                h=0.95)
              
              elif c.index(DIMENSION) in index_2:
                    c.simulate(loc=0,
                               scale=1,
                               with_noise=True,
                               label=3,
                               tau=TAU_2,
                               h=0.95)
                    
              elif c.index(DIMENSION) in index_3:
                    c.simulate(loc=0,
                               scale=1,
                               with_noise=True,
                               label=4,
                               tau=TAU_3,
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
       matrix, index = generate_defect(dimension=DIMENSION,
                                            defect_dimension=DEFECT_PARAMS[0],
                                            v_width=DEFECT_PARAMS[1],
                                            h_width=DEFECT_PARAMS[2])
            
       matrix_1, index_1 = generate_defect(dimension=DIMENSION,
                                           defect_dimension=DEFECT_PARAMS_1[0],
                                           v_width=DEFECT_PARAMS_1[1],
                                           h_width=DEFECT_PARAMS_1[2])

            
       matrix_1, index_2 = generate_defect(dimension=DIMENSION,
                                           defect_dimension=DEFECT_PARAMS_1[0],
                                           v_width=DEFECT_PARAMS_2[1],
                                           h_width=DEFECT_PARAMS_2[2])
       

       matrix_1, index_3 = generate_defect(dimension=DIMENSION,
                                           defect_dimension=DEFECT_PARAMS_1[0],
                                           v_width=DEFECT_PARAMS_3[1],
                                           h_width=DEFECT_PARAMS_3[2])
            
            
       grid = Lattice(dimension=DIMENSION,
                      time_period=60,
                      fps=1,
                      simulation=True,
                      id_="prova",
                      run="0")

       grid.build(profile_list=create_list(index, index_1, index_2, index_3))
       grid.plot_profiles()

       grid.func_plot()

       grid.save_in_gif(root=ROOT)
       grid.build_smooth_func(bandwidth=1.5)
       grid.init_algo(n=1500,
                      k=5,
                      explained_variance_pct=0.95,
                      b=100)
       grid.cluster_now()

       print("Classification rate {}".format(grid.classification_rate_))
       print("Entropy {}".format(grid.average_normalized_entropy_))

       grid.plot()


