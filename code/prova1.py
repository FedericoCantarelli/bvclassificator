import os
import pandas as pd
import json
from bvclassificator import algorithm
from sklearn.metrics._classification import recall_score
import numpy as np

# Directory contenente i file JSON
for i in range(9):
    cartella = '/Users/federicocantarelli/Desktop/output/S10_params' + \
        str(i)

# Inizializza una lista per memorizzare i dati dai file JSON
    d = dict(scenario=[],
             run=[],
             n=[],
             b=[],
             def_class=[],
             execution_time=[],
             entropy=[])

# Scansiona la cartella per trovare i file JSON
    for nome_file in os.listdir(cartella):
        if nome_file.endswith('.json'):
            percorso_file = os.path.join(cartella, nome_file)

        # Apre il file JSON e legge i dati
            with open(percorso_file, 'r') as file_json:
                dati_json = json.load(file_json)
                d["scenario"].append(dati_json["scenario"])
                d["run"].append(dati_json["run"])
                d["n"].append(dati_json["n"])
                d["b"].append(dati_json["b"])
                d["execution_time"].append(dati_json["execution_time"])
                d["entropy"].append(dati_json["average_normalized_entropy"])


                truth = np.array(dati_json["ground_truth"])
                pred = np.array(dati_json["final_label"])

                new_labels = algorithm._cluster_mapping(pred, truth)
                temp = algorithm._change_label(pred, new_labels)


                def_class = np.sum(
                    truth[truth != 0] == temp[truth != 0])/np.sum(truth!=0)
                d["def_class"] = def_class

# Crea un DataFrame utilizzando i dati
    df = pd.DataFrame.from_dict(d)

# Salva il DataFrame in un file CSV
    # Il parametro index=False evita di salvare l'indice
    df.to_csv(cartella + '/dati' + str(i) + '.csv', index=False)
