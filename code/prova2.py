import os
import pandas as pd
import json
from bvclassificator import algorithm
from sklearn.metrics._classification import recall_score
import numpy as np

# Directory contenente i file JSON
df_list = []
cartella = '/Users/federicocantarelli/Desktop/output1'


for nome_file in os.listdir(cartella):
    if nome_file.endswith('.csv'):
        percorso_file = os.path.join(cartella, nome_file)

        df = pd.read_csv(percorso_file)
        df_list.append(df)

df_final = pd.concat(df_list)
            
# Salva il DataFrame in un file CSV
df_final.to_csv(cartella + '/dati_finali.csv', index=False)  # Il parametro index=False evita di salvare l'indice

