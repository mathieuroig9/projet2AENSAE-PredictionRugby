import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


def club(nom, tab_presentation_global, tab_classement_global):
    ft_pr = tab_presentation_global[tab_presentation_global["Club"] == nom]
    ft_cl = tab_classement_global[tab_classement_global["Club"] == nom]

    tab_recap_data = {s: [] for s in [
    "24/25","23/24","22/23","21/22","20/21",
    "19/20","18/19","17/18","16/17","15/16",
    "14/15","13/14","12/13","11/12","10/11",
    "09/10","08/09","07/08","06/07","05/06"
    ]}

    year_to_season = {
        2024:"24/25", 2023:"23/24", 2022:"22/23", 2021:"21/22", 2020:"20/21",
        2019:"19/20", 2018:"18/19", 2017:"17/18", 2016:"16/17", 2015:"15/16",
        2014:"14/15", 2013:"13/14", 2012:"12/13", 2011:"11/12", 2010:"10/11",
        2009:"09/10", 2008:"08/09", 2007:"07/08", 2006:"06/07", 2005:"05/06"
    }

    for year, season in year_to_season.items():
        row = ft_pr[ft_pr["année"] == year]
        if not row.empty:
            coach = row.iloc[0].get("Entraîneur en chef", "")
            coach = " ".join(coach.split()[:2]).rstrip(".,") if isinstance(coach, str) and coach.strip() else float("nan")
            tab_recap_data[season] += [row.iloc[0].get("Budget en M€", float("nan")), coach]
        else:
            tab_recap_data[season] += [float("nan"), float("nan")]

    for year, season in year_to_season.items():
        row = ft_cl[ft_cl["année"] == year]
        tab_recap_data[season] += [row.iloc[0].get(k, float("nan")) if not row.empty else float("nan") for k in ["Rang","J","V","D","Diff"]]

    tab_recap = pd.DataFrame({s: tab_recap_data[s] for s in tab_recap_data},
                            index=["Budget en M€","Entraîneur en chef","Rang","J","V","D","Diff"])
    tab_recap = tab_recap.T.rename_axis("Saison").reset_index()

    for col in ["Rang","J","V","D","Diff"]:
        tab_recap[col] = tab_recap[col].astype("Int64")

    tab_recap = tab_recap.astype(object).where(pd.notnull(tab_recap), np.nan)

    return tab_recap


def plot_club_evolution(nom, evolution, forme):
    def journees(df):
        j = [c for c in df.columns if re.fullmatch(r'J\d+', str(c))]
        return sorted(j, key=lambda x: int(re.search(r'\d+', x).group()))

    # --- Partie 1 : Evolution du classement (evolution) ---
    club_evo = evolution[(evolution["Club"] == nom) & (evolution["année"] >= 2016)]
    jcols = journees(evolution)

    if not club_evo.empty and "année" in club_evo.columns:
        pivot = (club_evo.set_index("année")[jcols]
                 .apply(pd.to_numeric, errors="coerce"))
        pivot = pivot.T  # index = J*, colonnes = années

        plt.figure(figsize=(12, 6))
        for year in pivot.columns:
            plt.plot(pivot.index, pivot[year], marker="o", label=str(year))
        plt.title(f"Évolution du classement pour {nom}")
        plt.xlabel("Journées"); plt.ylabel("Classement")
        plt.gca().invert_yaxis()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Année", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout(); plt.show()
    else:
        print(f"Aucune donnée de classement pour {nom} (ou colonne 'année' absente).")

    # --- Partie 2 : Evolution cumulée des résultats (forme) ---
    club_forme = forme[(forme["Club"] == nom) & (forme["année"] >= 2016)]
    jcols_f = journees(forme)

    if not club_forme.empty and "année" in club_forme.columns:
        num = (club_forme.set_index("année")[jcols_f]
               .replace({'V': 1, 'D': -1, 'N': 0, 'R': 0, '': 0})
               .fillna(0))

        plt.figure(figsize=(12, 6))
        for year, row in num.iterrows():
            cumu = row.astype(int).cumsum()
            plt.plot(jcols_f, cumu.values, marker="o", label=str(year))
        plt.title(f"Évolution cumulée des résultats pour {nom}")
        plt.xlabel("Journées"); plt.ylabel("Score cumulatif (V=+1, D=-1, N/R=0)")
        plt.axhline(0, linestyle="--", alpha=0.7)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(title="Année", bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout(); plt.xticks(jcols_f, rotation=45); plt.show()
    else:
        print(f"Aucune donnée de résultats pour {nom} (ou colonne 'année' absente).")