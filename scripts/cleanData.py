from scrapData import data2325, data2223, data2122, data2021, data1920, data1619
import re
import pandas as pd
import numpy as np

def nettoyage(presentation, classement, resultats, evolution_classement, forme):
    #on enlève les notes
    presentation["Capacité"] = presentation["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
    #on remplace 1er,2eme,.. par 1,2,..
    presentation["Classement précédent"] = presentation["Classement précédent"].apply(lambda x: str(int(re.search(r"\d+", x).group()) + 14) if isinstance(x, str) and "(Pro D2)" in x else re.sub(r"[^\d]", "", x))
    #on enlève les valeurs aberrantes
    presentation["Capacité"] = presentation["Capacité"].apply(lambda x: np.nan if x > 100000 else x)

    #on garde seulement le nom de l'équipe
    presentation["Club"] = presentation["Club"].apply(lambda x: re.sub(r"\[\d+\]$", "", x) if isinstance(x, str) else x)
    presentation["Club"] = presentation["Club"].apply(lambda x: x.rstrip(" C1") if x.endswith(" C1") else x)
    presentation["Club"] = presentation["Club"].apply(lambda x: x.rstrip(" C2") if x.endswith(" C2") else x)
    presentation["Club"] = presentation["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
    presentation["Club"] = presentation["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)

    #on garde seulement le nom de l'équipe
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" C1") if x.endswith(" C1") else x)
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" C2") if x.endswith(" C2") else x)
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" (T)") if x.endswith(" (T)") else x)
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" (P)") if x.endswith(" (P)") else x)

    #on complète les blancs par -
    resultats.replace('', '-', inplace=True)

    #il y'a des fois des notes après le classement, on garde que le classement
    for col in evolution_classement.columns[1:]:
        evolution_classement[col] = evolution_classement[col].apply(lambda x: int(float(re.sub(r"[^\d.]", "", str(x)))) if pd.notna(x) and isinstance(x, (float, int, str)) and re.match(r"^\d+(\.\d+)?$", re.sub(r"[^\d.]", "", str(x))) else x)
    
    forme.iloc[:, 1:] = forme.iloc[:, 1:].replace({'G': 'V', 'P': 'D'}, regex=False)
