from scrapData import data2325, data2223, data2122, data2021, data1920, data1619
from cleanData import nettoyage
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

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
         tableauglobal[i] = pd.concat([tableauglobal[i], ajout_an(eval(f"tab{j}")[i], j - 1)], ignore_index=True)

tab_presentation_global = tableauglobal[0]
tab_classement_global = tableauglobal[1]
tab_resultat_global = tableauglobal[2]
tab_evolution_classement_global = tableauglobal[3]
tab_forme_global = tableauglobal[4]
tab_presentation_global['Budget en M€'] = pd.to_numeric(tab_presentation_global['Budget en M€'], errors='coerce')
tab_presentation_global['Classement précédent'] = pd.to_numeric(tab_presentation_global['Classement précédent'], errors='coerce')

def uniforme_tab(tab):
    # Copie du tableau pour éviter de modifier l'original
    tab_uniforme = tab.copy()
    tab_uniforme = tab_uniforme.iloc[:, :-1]

    # Remplace les en-têtes des colonnes avec les valeurs de la première colonne à partir de la ligne 1
    tab_uniforme.iloc[0, 1:] = tab_uniforme.iloc[1:, 0].values 
    # Remplace la case (0, 0) par "Club"
    tab_uniforme.iloc[0, 0] = "Club"
    
    return tab_uniforme

t9=uniforme_tab(tab9[2])
t8=uniforme_tab(tab8[2])
t7=uniforme_tab(tab7[2])
t6=uniforme_tab(tab6[2])
t5=uniforme_tab(tab5[2])
t4=uniforme_tab(tab4[2])
t3=uniforme_tab(tab3[2])
t2=uniforme_tab(tab2[2])
t1=uniforme_tab(tab1[2])

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def complete_t9_resultats(t5,t6,t7,t8,t9):
    """
    Complète les cases JX de t9 avec des scores prédits basés sur les tableaux historiques t1 à t8.

    Arguments :
    - t1, ..., t8 : DataFrames contenant les données historiques des scores.
    - t9 : DataFrame représentant le tableau à compléter.

    Retourne :
    - t9 complété avec les scores sous la forme "XX-YY".
    """
    # Fusionner les tableaux historiques pour créer une base de données unique
    historical_tables = [t5,t6,t7,t8]
    historical_data = []

    for table in historical_tables:
        for i, row in table.iterrows():
            for j, value in enumerate(row):
                if isinstance(value, str) and '-' in value:
                    score_a, score_b = map(int, value.split('-'))
                    historical_data.append({
                        'team_a': table.iloc[i, 0],
                        'team_b': table.columns[j],
                        'score_a': score_a,
                        'score_b': score_b,
                    })

    historical_df = pd.DataFrame(historical_data)

    # Feature engineering
    def calculate_features(team_a, team_b, historical_df):
        """Calcule les caractéristiques pour une paire d'équipes."""
        history_a = historical_df[historical_df['team_a'] == team_a]
        history_b = historical_df[historical_df['team_b'] == team_b]
        
        features = {
            'mean_score_a': history_a['score_a'].mean() if not history_a.empty else 0,
            'mean_score_b': history_b['score_b'].mean() if not history_b.empty else 0,
            'diff_mean_scores': history_a['score_a'].mean() - history_b['score_b'].mean() if not history_a.empty and not history_b.empty else 0
        }
        return list(features.values())

    # Préparer les données d'entraînement pour le modèle
    features = []
    labels_a = []
    labels_b = []

    for _, row in historical_df.iterrows():
        team_a = row['team_a']
        team_b = row['team_b']
        features.append(calculate_features(team_a, team_b, historical_df))
        labels_a.append(row['score_a'])
        labels_b.append(row['score_b'])

    features = np.array(features)
    labels_a = np.array(labels_a)
    labels_b = np.array(labels_b)

    # Entraîner deux modèles : un pour score_a et un pour score_b
    X_train, X_test, y_train_a, y_test_a = train_test_split(features, labels_a, test_size=0.2, random_state=42)
    _, _, y_train_b, y_test_b = train_test_split(features, labels_b, test_size=0.2, random_state=42)

    model_a = RandomForestRegressor(random_state=42)
    model_b = RandomForestRegressor(random_state=42)

    model_a.fit(X_train, y_train_a)
    model_b.fit(X_train, y_train_b)

    for i, row in t9.iterrows():
        for j, value in enumerate(row):
            if isinstance(value, str) and value.startswith('J'):
                team_a = t9.iloc[i, 0]
                team_b = t9.columns[j]

                # Vérifier si l'équipe est absente des données historiques
                is_team_a_missing = not (np.isin(team_a, historical_df['team_a'].values) | np.isin(team_a, historical_df['team_b'].values)).any()
                is_team_b_missing = not (np.isin(team_b, historical_df['team_a'].values) | np.isin(team_b, historical_df['team_b'].values)).any()

                if is_team_a_missing:
                    # Calculer les scores pour team_a à partir des données déjà dans t9
                    existing_scores = [
                        list(map(int, val.split('-'))) for val in t9.iloc[i, 1:] if isinstance(val, str) and '-' in val
                    ]
                    if existing_scores:
                        avg_score_a = int(np.mean([score[0] for score in existing_scores]))
                        avg_score_b = int(np.mean([score[1] for score in existing_scores]))
                        t9.iloc[i, j] = f"{avg_score_a}-{avg_score_b}"
                elif is_team_b_missing:
                    # Calculer les scores pour team_b à partir des données déjà dans t9
                    existing_scores = [
                        list(map(int, val.split('-'))) for val in t9.iloc[:, j] if isinstance(val, str) and '-' in val
                    ]
                    if existing_scores:
                        avg_score_a = int(np.mean([score[0] for score in existing_scores]))
                        avg_score_b = int(np.mean([score[1] for score in existing_scores]))
                        t9.iloc[i, j] = f"{avg_score_a}-{avg_score_b}"
                else:
                    # Utiliser les données historiques si les deux équipes y sont présentes
                    input_features = calculate_features(team_a, team_b, historical_df)
                    score_a = int(model_a.predict([input_features])[0])
                    score_b = int(model_b.predict([input_features])[0])
                    t9.iloc[i, j] = f"{score_a}-{score_b}"

    #pour plus de lisibilité du tableau, enlever les #
    #column_names = list(tab9[2].iloc[0, 1:-1])
    #t9.iloc[0, 1:] = column_names

    return t9

import pandas as pd

def resultat_equipe(club, t):
    """
    Met à jour le tableau t en conservant uniquement les résultats pour la ligne et la colonne
    correspondant au club donné. Toutes les autres cases sont remplacées par '-'.

    :param club: Nom du club (chaîne de caractères) à analyser
    :param t: Tableau sous forme de DataFrame Pandas
    :return: DataFrame Pandas modifié
    """
    # Trouver les indices de la ligne et de la colonne correspondant au club
    try:
        ligne_idx = t[t[0] == club].index[0]  # Ligne où le club est trouvé dans la première colonne
        colonne_idx = t.iloc[0, :].tolist().index(club)  # Colonne où le club est trouvé dans la première ligne
    except IndexError:
        raise ValueError(f"Le club '{club}' n'existe pas dans le tableau.")
    
    # Créer une copie du tableau pour la mise à jour
    t_resultat = t.copy()
    
    # Parcourir tout le tableau et appliquer la règle
    for i in range(1, t.shape[0]):
        for j in range(1, t.shape[1]):
            if i != ligne_idx and j != colonne_idx:  # Ni la ligne ni la colonne du club
                t_resultat.iloc[i, j] = '-'
    
    column_names = list(tab9[2].iloc[0, 1:-1])
    t_resultat.iloc[0, 1:] = column_names
    
    return t_resultat

t9=complete_t9_resultats(t5,t6,t7,t8,t9)

r=resultat_equipe("Stade toulousain",t9)

import code
code.interact(local=locals())