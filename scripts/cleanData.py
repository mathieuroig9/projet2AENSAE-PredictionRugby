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
    evolution_classement.iloc[:, 1:] = evolution_classement.iloc[:, 1:].applymap(lambda x: int(float(re.sub(r"[^\d.]", "", str(x)))) if pd.notna(x) and isinstance(x, (float, int, str)) and re.match(r"^\d+(\.\d+)?$", re.sub(r"[^\d.]", "", str(x))) else x)

    forme.iloc[:, 1:] = forme.iloc[:, 1:].replace({'G': 'V', 'P': 'D'}, regex=False)

tab=data2325("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2024-2025")
#tab=data2325("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2023-2024")
#tab=data2223("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2022-2023")
#tab=data2122("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2021-2022")
#tab=data2021("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2020-2021")
#tab=data1920("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2019-2020")
#tab=data1619("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2018-2019")
#tab=data1619("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2017-2018")
#tab=data1619("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2016-2017")

nettoyage(*tab)
pr=tab[0]
cl=tab[1]
re=tab[2]
ev=tab[3]
fo=tab[4]


#pour vscode sur mac
import code
code.interact(local=locals())