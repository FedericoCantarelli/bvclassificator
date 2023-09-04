import bvclustering as bvc

# Original
# b_0 = (5.421, 0.0105)
# b_1 = (-2.165e-4, 1.773e-5)
# b_2 = (-0.0000001, -0.000523)
# error = (0, 0.0005)

b_0 = (5.421, 0.0105)
b_1 = (-2.165e-3, 1.773e-4)
b_2 = (-0.0000001, -0.000523)
error = (0, 0.00005)

profile_list = []

h = 0
for i in range(4):
    for j in range(4):
        if i == 3 and j == 3:
            dev_ent = 3

        elif i == 2 and j == 3:
            dev_ent = 3

        elif i == 3 and j == 2:
            dev_ent = 3

        else:
            dev_ent = None
        prova = bvc.Simulation(str(h), i, j, b_0, b_1, b_2[0], b_2[1],
                               error, deviation_entity=dev_ent)
        prova.run()
        h += 1

        profile_list.append(prova)

strc = bvc.analysis.Structure(profile_list)

df, avg_entropy = strc.cluster_now(12, 1000, 2, 3)

print(df)

bvc.analysis.plot_results(df)
