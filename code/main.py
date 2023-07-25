import bvclustering as bvc

b_0 = (5.421, 0.0105)
b_1 = (-2.165e-4, 1.773e-5)
b_2 = (-0.0000001, -0.000523)
error = (0, 0.0005)

prova = bvc.Simulation("demo", 0, 0, b_0, b_1, b_2[0], b_2[1],
                       error, False, deviation_entity=3)  # Se do un'entity allora sicuramente non è in controllo
# l'in-control ce l'ho solo se sono delle simulazioni altrimenti no...

prova.run()

prova.dump_to_json("./")

print(prova.position)

prova.plot()
