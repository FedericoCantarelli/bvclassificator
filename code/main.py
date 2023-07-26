import bvclustering as bvc

b_0 = (5.421, 0.0105)
b_1 = (-2.165e-4, 1.773e-5)
b_2 = (-0.0000001, -0.000523)
error = (0, 0.0005)

profile_list = []

h = 0
for i in range(4):
    for j in range(4):
        if i == 4 and j==4:
            dev_ent = 3
        else:
            dev_ent = None
        prova = bvc.Simulation(str(h), i, j, b_0, b_1, b_2[0], b_2[1],
                           error, deviation_entity=dev_ent)
        prova.run()
        h += 1

        profile_list.append(prova)

strc = bvc.analysis.Structure(profile_list)

df, avg_entropy = strc.cluster_now(10, 1000, 2, 0.9)

print(df)

bvc.analysis.plot_results(df)
