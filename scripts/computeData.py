from scrapData import data2325, data2223, data2122, data2021, data1920, data1619
from cleanData import nettoyage
import pandas as pd

tab2425=data2325("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2024-2025")
tab2324=data2325("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2023-2024")
tab2223=data2223("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2022-2023")
#tab2122=data2122("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2021-2022")
#tab2021=data2021("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2020-2021")
#tab1920=data1920("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2019-2020")
#tab1819=data1619("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2018-2019")
#tab1718=data1619("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2017-2018")
#tab1617=data1619("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2016-2017")

nettoyage(*tab2425)
nettoyage(*tab2324)
nettoyage(*tab2223)
#nettoyage(*tab2122)
#nettoyage(*tab2021)
#nettoyage(*tab1920)
#nettoyage(*tab1819)
#nettoyage(*tab1718)
#nettoyage(*tab1617)

def club(nom, t2425, t2324, t2223):
    # Filtrer les tableaux pour le club spécifié
    ft2425_0 = t2425[0][t2425[0]["Club"] == nom]
    ft2425_1 = t2425[1][t2425[1]["Club"] == nom]
    ft2324_0 = t2324[0][t2324[0]["Club"] == nom]
    ft2324_1 = t2324[1][t2324[1]["Club"] == nom]
    ft2223_0 = t2223[0][t2223[0]["Club"] == nom]
    ft2223_1 = t2223[1][t2223[1]["Club"] == nom]

    # Construire le tableau récapitulatif
    tab_recap = pd.DataFrame({
        "": ["Budget en M€", "Entraîneur en chef", "Classement précédent", "Rang"],
        "24/25": [
            ft2425_0["Budget en M€"].values[0], 
            ft2425_0["Entraîneur en chef"].values[0], 
            ft2425_0["Classement précédent"].values[0], 
            ft2425_1["Rang"].values[0]
        ],
        "23/24": [
            ft2324_0["Budget en M€"].values[0], 
            ft2324_0["Entraîneur en chef"].values[0], 
            ft2324_0["Classement précédent"].values[0], 
            ft2324_1["Rang"].values[0]
        ],
        "22/23": [
            ft2223_0["Budget en M€"].values[0], 
            ft2223_0["Entraîneur en chef"].values[0], 
            ft2223_0["Classement précédent"].values[0], 
            ft2223_1["Rang"].values[0]
        ]
    })
    
    return tab_recap


t=club("USA Perpignan", tab2425, tab2324,tab2223)

#pour vscode sur mac
import code
code.interact(local=locals())