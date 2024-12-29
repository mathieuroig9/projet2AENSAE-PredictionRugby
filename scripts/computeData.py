import pandas as pd
from scrapData import data2325, data2223, data2122, data2021, data1920, data1619
from cleanData import nettoyage
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

tab2425=data2325("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2024-2025")
tab2324=data2325("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2023-2024")
tab2223=data2223("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2022-2023")
tab2122=data2122("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2021-2022")
tab2021=data2021("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2020-2021")
tab1920=data1920("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2019-2020")
tab1819=data1619("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2018-2019")
tab1718=data1619("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2017-2018")
tab1617=data1619("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2016-2017")

nettoyage(*tab2425)
nettoyage(*tab2324)
nettoyage(*tab2223)
nettoyage(*tab2122)
nettoyage(*tab2021)
nettoyage(*tab1920)
nettoyage(*tab1819)
nettoyage(*tab1718)
nettoyage(*tab1617)

# On va maintenant fusionner les tableaux on change le nom pour faciliter la boucle :
tab1 = tab1617
tab2 = tab1718
tab3 = tab1819
tab4 = tab1920
tab5 = tab2021
tab6 = tab2122
tab7 = tab2223
tab8 = tab2324
tab9 = tab2425


# On crée une fonction qui permet d'ajouter une colonne année à chaque tableau pour bien pouvoir les séparer par année si besoin après
def ajout_an(df,i):
   df["année"] = 2016 + i
   return df

# On va faire maintenant une boucle pour concaténer les tableaux entre eux :
tableauglobal = [pd.DataFrame() for _ in range(5)]
# On remarquera que l'on ne prend pas le tableau résusltat global car les équipes du championnat changent chaque année donc ce n'est pas possible de le concaténer

for j in range(1, 10):
    for i in range(5):
        if i == 2:
            pass  # on ne récupère pas le tableau résultat
        else:
         tableauglobal[i] = pd.concat([tableauglobal[i], ajout_an(eval(f"tab{j}")[i], j - 1)], ignore_index=True)

tab_presentation_global = tableauglobal[0]
tab_classement_global = tableauglobal[1]
tab_evolution_classement_global = tableauglobal[3]
tab_forme_global = tableauglobal[4]
tab_presentation_global['Budget en M€'] = pd.to_numeric(tab_presentation_global['Budget en M€'], errors='coerce')
tab_presentation_global['Classement précédent'] = pd.to_numeric(tab_presentation_global['Classement précédent'], errors='coerce')

def club(nom, tab_presentation_global, tab_classement_global):
    # Filtrer les tableaux pour le club spécifié
    ft_pr = tab_presentation_global[tab_presentation_global["Club"] == nom]
    ft_cl = tab_classement_global[tab_classement_global["Club"] == nom]

    # Créer un dictionnaire pour stocker les données du tableau tab_recap
    tab_recap_data = {
        "24/25": [],
        "23/24": [],
        "22/23": [],
        "21/22": [],
        "20/21": [],
        "19/20": [],
        "18/19": [],
        "17/18": [],
        "16/17": []
    }

    # Mapper les années aux colonnes du tableau recap
    year_to_season = {
        2024: "24/25",
        2023: "23/24",
        2022: "22/23",
        2021: "21/22",
        2020: "20/21",
        2019: "19/20",
        2018: "18/19",
        2017: "17/18",
        2016: "16/17"
    }

    # Remplir les informations depuis ft_pr pour "Budget en M€" et "Entraîneur en chef"
    for year, season in year_to_season.items():
        row = ft_pr[ft_pr["année"] == year]
        if not row.empty:
            tab_recap_data[season].append(row.iloc[0].get("Budget en M€", float("nan")))
            # Limiter "Entraîneur en chef" à un seul nom et prénom
            coach = row.iloc[0].get("Entraîneur en chef", "")
            if isinstance(coach, str):
                coach = " ".join(coach.split()[:2])
                coach = coach.rstrip(".,")
                coach = coach if coach.strip() else float("nan")  # Remplacer les chaînes vides par NaN
            else:
                coach = float("nan")
            tab_recap_data[season].append(coach)
        else:
            tab_recap_data[season].extend([float("nan"), float("nan")])  # Ajouter NaN pour le budget et l'entraîneur

    # Remplir les informations depuis ft_cl pour "Rang", "J", "V", "D", "Diff"
    for year, season in year_to_season.items():
        row = ft_cl[ft_cl["année"] == year]
        if not row.empty:
            tab_recap_data[season].append(row.iloc[0].get("Rang", float("nan")))
            tab_recap_data[season].append(row.iloc[0].get("J", float("nan")))
            tab_recap_data[season].append(row.iloc[0].get("V", float("nan")))
            tab_recap_data[season].append(row.iloc[0].get("D", float("nan")))
            tab_recap_data[season].append(row.iloc[0].get("Diff", float("nan")))
        else:
            tab_recap_data[season].extend([float("nan")] * 5)  # Ajouter NaN pour Rang, J, V, D, Diff

    # Convertir les données en DataFrame avec les années en colonnes et les catégories en lignes
    tab_recap = pd.DataFrame(
        {
            "24/25": tab_recap_data["24/25"],
            "23/24": tab_recap_data["23/24"],
            "22/23": tab_recap_data["22/23"],
            "21/22": tab_recap_data["21/22"],
            "20/21": tab_recap_data["20/21"],
            "19/20": tab_recap_data["19/20"],
            "18/19": tab_recap_data["18/19"],
            "17/18": tab_recap_data["17/18"],
            "16/17": tab_recap_data["16/17"]
        },
        index=["Budget en M€", "Entraîneur en chef", "Rang", "J", "V", "D", "Diff"]
    )
    return tab_recap


def plot_club_evolution(nom, tab_classement, tab_resultats):
    tab_resultats = tab_resultats.iloc[:, :-3]
    # Partie 1 : Evolution du classement
    club_classement = tab_classement[tab_classement["Club"] == nom]

    if club_classement.empty:
        print(f"Aucune donnée de classement trouvée pour le club : {nom}")
        return

    years_classement = club_classement["année"].values
    classement_transposed = club_classement.drop(columns=["Club", "année"]).transpose()
    classement_transposed.columns = years_classement

    plt.figure(figsize=(12, 6))
    for year in classement_transposed.columns:
        plt.plot(classement_transposed.index, classement_transposed[year], label=f"{year}")

    plt.title(f"Évolution du classement pour {nom}")
    plt.xlabel("Journées")
    plt.ylabel("Classement")
    plt.legend(title="Année", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.gca().invert_yaxis()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Partie 2 : Evolution cumulée des résultats
    club_resultats = tab_resultats[tab_resultats["Club"] == nom]

    if club_resultats.empty:
        print(f"Aucune donnée de résultats trouvée pour le club : {nom}")
        return

    years_resultats = club_resultats["année"].values
    resultats_data = club_resultats.drop(columns=["Club", "année"])

    plt.figure(figsize=(12, 6))
    for i, year in enumerate(years_resultats):
        # Remplacer les résultats par des valeurs numériques et gérer les valeurs manquantes
        results_numeric = resultats_data.iloc[i].replace({'V': 1, 'D': -1, 'N': 0, 'R': 0, '': 0}).fillna(0).astype(int)
        
        # Calculer les résultats cumulés en commençant directement à J1
        cumulative_results = [results_numeric.iloc[0]]  # Le score initial est la valeur de J1
        for result in results_numeric.iloc[1:]:  # On commence après J1
            cumulative_results.append(cumulative_results[-1] + result)

        # Tracer la courbe cumulée
        journees = resultats_data.columns[:]  # Les colonnes contiennent les journées (J1, J2, ..., J26)
        plt.plot(journees, cumulative_results, label=f"{year}")

    plt.title(f"Évolution cumulée des résultats pour {nom}")
    plt.xlabel("Journées")
    plt.ylabel("Score cumulatif (V=+1, D=-1, N=0)")
    plt.axhline(0, color='black', linestyle='--', alpha=0.7)
    plt.legend(title="Année", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.xticks(journees, rotation=45)  # Affiche les journées sur l'axe des abscisses avec une rotation de 45° si nécessaire
    plt.show()

#plot_club_evolution("Stade toulousain", tab_evolution_classement_global, tab_forme_global)
#plot_club_evolution("USA Perpignan", tab_evolution_classement_global, tab_forme_global)

#toulouse=club("Stade toulousain", tab_presentation_global, tab_classement_global)
#perpignan=club("USA Perpignan", tab_presentation_global, tab_classement_global)