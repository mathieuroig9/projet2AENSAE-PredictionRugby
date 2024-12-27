
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from computeData import tab_presentation_global, tab_classement_global, tab_evolution_classement_global, tab_forme_global 
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Charger vos données
presentation = tab_presentation_global
classement = tab_classement_global
evolution = tab_evolution_classement_global
forme = tab_forme_global

# Fusion des différentes données en un seul data frame
# On va fusionner les tableaux en se servant du club et de l'année comme clef
# On commence déjà par uniformiser le nom de la colonne 'Club':
evolution.rename(columns={'Equipes/Journées': 'Club'}, inplace=True)
forme.rename(columns={'Equipes/Journées': 'Club'}, inplace=True)

# En fait en faisant la fusion on se rend compte qu'un même club est appelé de manière différentes selon le tableau
# On va donc harmoniser ça
# On trouve les différences d'appelations grâce au code suivant : 
# Clés uniques dans chaque DataFrame
# keys_presentation = set(presentation[['Club', 'année']].itertuples(index=False, name=None))
# keys_classement = set(classement[['Club', 'année']].itertuples(index=False, name=None))
# keys_evolution = set(evolution[['Club', 'année']].itertuples(index=False, name=None))
# keys_forme = set(forme[['Club', 'année']].itertuples(index=False, name=None))
# Lignes présentes uniquement dans presentation
# diff_presentation = keys_presentation - keys_classement
# print("Lignes dans presentation mais pas dans classement :", diff_presentation)
#  # Lignes présentes uniquement dans classement
# diff_classement = keys_classement - keys_presentation
#  print("Lignes dans classement mais pas dans presentation :", diff_classement)
# Le premier tableau utilise la dénomination 'Stade français' tandis que le second 'Stade français Paris'
# On procède de même avec les différents tableaux pour observer toutes les appelations différentes (on le fait par rapport à classement)
# On répertorie alors ces appelations différentes pour pouvoir par la suite uniformiser tout les noms : 

mapping = {
    'Paris' : 'Stade français Paris',
    'Stade français': 'Stade français Paris',
    'Clermont': 'ASM Clermont',
    'La Rochelle': 'Stade rochelais',
    'Toulouse': 'Stade toulousain',
    'Bayonne': 'Aviron bayonnais',
    'Brive': 'CA Brive',
    'Montpellier': 'Montpellier HR',
    'Toulon': 'RC Toulon',
    'Castres': 'Castres olympique',
    'Pau': 'Section paloise',
    'Agen': 'SU Agen',
    'Grenoble': 'FC Grenoble',
    'Oyonnax': 'US Oyonnax',
    'Perpignan': 'USA Perpignan',
    'Bordeaux-Bègles': 'Union Bordeaux Bègles',
    'Bordeaux Bègles' : 'Union Bordeaux Bègles',
    'Lyon' : 'Lyon OU'
}
# On a plus qu'à uniformiser les noms dans tout les tableaux
presentation['Club'] = presentation['Club'].replace(mapping)
classement['Club'] = classement['Club'].replace(mapping)
forme['Club'] = forme['Club'].replace(mapping)
evolution['Club'] = evolution['Club'].replace(mapping)

data = pd.merge(presentation, classement, on=['Club', 'année'], how='inner')
data = pd.merge(data, evolution, on=['Club', 'année'], how='inner')
data = pd.merge(data, forme, on=['Club', 'année'], how='inner')


# Préparation des données
X = data.drop(columns=['Rang','J26_x'])  # On retire la varibale à prédire, J26_x est le classement le jour 26 qui est le même que le classement final
y = data['Rang']  # Classement final On prend la variable que l'on souhaite prédire

# On prend un jeu d'entraînement et de test (pour l'année 2023/2024, on s'entraîne sur le passé)
X_train = X[data['année'] < 2023]
X_test = X[data['année'] >= 2023]
y_train = y[data['année'] < 2023]
y_test = y[data['année'] >= 2023]

# On sépare les variables numériques et catégorielles
variables_numériques = X_train.select_dtypes(include=['int64', 'float64']).columns
variables_catégorielles = X_train.select_dtypes(include=['object']).columns

# ¨Pipeline permet de faire plusieurs modifications sur les données en une seule fois :
# Pour les variables numériques
var_num_ajustées = Pipeline(steps=[
    ('valeur_manquantes', SimpleImputer(strategy='mean')),  # On remplace les valeurs manquantes par la moyenne de la variable
    ('standardiser', StandardScaler())  # On standardise les variables
])

# Pour les variables catégorielles
var_cat_ajustées = Pipeline(steps=[
    ('valeur_manquantes', SimpleImputer(strategy='most_frequent')),  # Lorsqu'il manque des données on les remplace par la catégorie la plus fréquente
    ('transfomration_binaire', OneHotEncoder(handle_unknown='ignore'))  # Transformation en variables binaires
])

# Combiner les variables ajustées grâce à columnTransformer qui permet d'effectuer des modifications différentes sur des variables d'un même datatset
# Ici on sépare catégorielle et numérique et on applique les 2 transformations qu'on a définie précédement
var_combinée = ColumnTransformer(
    transformers=[
        ('num', var_num_ajustées, variables_numériques),
        ('cat', var_cat_ajustées, variables_catégorielles)
    ])

# Modélisation avec RandomForestRegressor
# Pipeline complet avec préprocesseur et modèle Random Forest
modele = Pipeline(steps=[
    ('données_utilisées', var_combinée),
    ('forêt_aléatoire', RandomForestRegressor(n_estimators=100, random_state=2024))  # On fixe la seed à 2024
])

# Entraîner le modèle
modele.fit(X_train, y_train)

y_pred = modele.predict(X_test) # on fait nos prédictions avec le modèle

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Prédire le classement des équipes pour 2023/2024 et 2024/2025


# future_data22 = X[X['année'] == 2022].reset_index(drop=True) # on sélectionne les données pour 2022/2023 
# # et on initialise l'indexation à 0 pour pouvoir fusionner par la suite

# predictions22 = pd.Series(modele.predict(future_data22), name='Rang') # On transforme le array en Series et on rajoute le nom Rang à la colonne

# pred22 = pd.concat([future_data22['Club'], predictions22], axis=1) # On concatène les 2 tableaux

# pred22['Rang_comparatif'] = pred22['Rang'].rank(method='min') # On crée un rang comparatif

# print("Predictions pour 2022/2023:", pred22) # On affiche la prédiction

# On fait de même pour les deux autres années : 

future_data23 = X[X['année'] == 2023].reset_index(drop=True) 
predictions23 = pd.Series(modele.predict(future_data23), name='Rang_prédit')
pred23 = pd.concat([future_data23['Club'], predictions23], axis=1)
pred23['Rang_prédit_ajusté'] = pred23['Rang_prédit'].rank(method='min')
print("Predictions pour 2023/2024:", pred23)


future_data24 = X[X['année'] == 2024].reset_index(drop=True) 
predictions24 = pd.Series(modele.predict(future_data24), name='Rang_prédit')
pred24 = pd.concat([future_data24['Club'], predictions24], axis=1)
pred24['Rang_prédit_ajusté'] = pred24['Rang_prédit'].rank(method='min')
print("Predictions pour 2024/2025:", pred24)


# Comparons avec une prédiction ajustée en fonction du rang des autres
a=[pred23['Rang_prédit_ajusté'], pred24['Rang_prédit_ajusté']]
a = np.array(a)
a = a.flatten()
print("Mean Absolute Error:", mean_absolute_error(y_test, a))
print("Root Mean Squared Error:", np.sqrt(mean_squared_error(y_test, a)))
# Avec le rang ajusté les prédictions sont encore meilleures les 2 ont diminuées

# On veut maintenant voir comment évolue la prédiction lorsque l'on retire les jours au fur et à mesure :
# Pour ce faire on va stocker les valeurs prédites et les erreurs dans 3 tableaux :
PREDICT=[]
MAE=[]
RMSE= []

PREDICT.append( y_pred)
MAE.append(mean_absolute_error(y_test, y_pred))
RMSE.append(np.sqrt(mean_squared_error(y_test, y_pred)))

# On crée une copie de notre jeu de test pour le garder intact

X_test_bis = X_test

# On va faire comme ci on avait pas accès aux journée en enlevant une par une les données du jour 26 jusqu'au jour 2 de l'année que l'on veut prédire(on le fait en même temps pour les deux années)
# Pas besoinde de supprimer JR1 à JR3 car 2023 et 2024 ne sont pas concernées (pas de valeur)

# Suppression des colonnes J25_x, J25_y, ..., J2_x, J2_y
for i in range(25, 1, -1):  # Commence à 25 et descend jusqu'à 2
    X_test_bis[f'J{i}_x'] = np.nan
    X_test_bis[f'J{i}_y'] = np.nan
    # X_test_bis.drop(columns=[f'J{i}_x', f'J{i}_y'], inplace=True)
    y_pred = modele.predict(X_test_bis)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    PREDICT.append(y_pred)
    MAE.append(mae)
    RMSE.append(rmse)
# La précision du modèle est de moins en moins bonne mais au final cela reste quand même bon
# Cela s'exlique peut-être par le fait que le nom du club joue est ce qui importe le plus 
# Et qu'au final les données obtenues sur les différents jours sont très corrélées au nom du club