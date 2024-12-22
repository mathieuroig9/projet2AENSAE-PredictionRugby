# IMPORTATION BIBLIOTHEQUES
from io import StringIO
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import sys
import subprocess

# Installer le module lxml
#subprocess.check_call([sys.executable, "-m", "pip", "install", "lxml"])

#les virgules ne sont pas prises en compte pour les chiffres (expl : 27,3 lu comme 273)
def clean_table(table_html):
    table_str = str(table_html) 
    table_str = table_str.replace(',', '.') 
    return pd.read_html(StringIO(table_str))[0]

# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2024/2025
def data2325(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})

    # il y'a 37 tables mais seulement 5 nous intéressent
    presentation = clean_table(tables[0])
    presentation = presentation.iloc[:, :-1]  #on enlève la coupe d'europe
    classement = pd.read_html(StringIO(str(tables[2])))[0]
    resultats = pd.read_html(StringIO(str(tables[3])))[0]
    evolution_classement = pd.read_html(StringIO(str(tables[30])))[0]
    forme = pd.read_html(StringIO(str(tables[31])))[0]

    # === ETAPE 2 : SÉLECTION DES DONNEES ===
    # presentation : je rectifie les colonnes
    presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement précédent', 'Entraîneur en chef', 'Stade', 'Capacité']
    # evolution : j'enlève les 3 dernières colonnes non utiles et je donne un nom aux colonnes qui n'en ont pas
    evolution_classement = evolution_classement.iloc[:, :-3]
    evolution_classement.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']
    # forme : je donne un nom aux colonnes qui n'en ont pas
    forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

    return presentation, classement, resultats, evolution_classement, forme

#data23("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2024-2025")
#data23("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2023-2024")

# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2022/2023
# 3 modifs :
# evolution_classement = pd.read_html(StringIO(str(tables[30])))[0] -> evolution_classement = pd.read_html(StringIO(str(tables[31])))[0]

# forme = pd.read_html(StringIO(str(tables[31])))[0] -> forme = pd.read_html(StringIO(str(tables[32])))[0]

# Pour les 2 on doit décaler table de 1 car il y a un tableau "cumul des points" avant ces tableaux qu'il n'y avait pas sur les autres pages

# evolution_classement = evolution_classement.iloc[:, :-3] -> evolution_classement = evolution_classement.iloc[:, :-2] ( A partir de cette année et des précédents il manque la colonne barrage)


def data2223(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})

    # il y a 37 tables mais seulement 5 nous intéressent
    presentation = clean_table(tables[0])
    presentation = presentation.iloc[:, :-1]  #on enlève la coupe d'europe
    classement= pd.read_html(StringIO(str(tables[2])))[0]
    resultats = pd.read_html(StringIO(str(tables[3])))[0]
    evolution_classement = pd.read_html(StringIO(str(tables[31])))[0]
    forme = pd.read_html(StringIO(str(tables[32])))[0]

    # === ETAPE 2 : NETTOYAGE DES DONNEES ===
    # presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
    presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement précédent', 'Entraîneur en chef', 'Stade', 'Capacité']

    # evolution : j'enlève les 3 dernières colonnes non utiles et je donne un nom aux colonnes qui n'en ont pas
    evolution_classement = evolution_classement.iloc[:, :-2]
    evolution_classement.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

    # forme : je donne un nom aux colonnes qui n'en ont pas
    forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

    return presentation, classement,resultats, evolution_classement, forme
# data2223("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2022-2023")


# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2021/2022, modification pareil que pour 2022/2023 + il y a eu des journées de rattrapages nommées R1, R2,R3 donc il y a 3 colonnes de plus (a voir plus tard comment intégrer ça au modèle) :
# forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26'] -> forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'JR1', 'J18', 'J19', 'J20', 'J21','JR2', 'JR3', 'J22', 'J23', 'J24', 'J25', 'J26']
def data2122(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})

    # il y a 37 tables mais seulement 5 nous intéressent
    presentation = clean_table(tables[0])
    presentation = presentation.iloc[:, :-1]  #on enlève la coupe d'europe
    classement = pd.read_html(StringIO(str(tables[2])))[0]
    resultats = pd.read_html(StringIO(str(tables[3])))[0]
    evolution_classement = pd.read_html(StringIO(str(tables[31])))[0]
    forme = pd.read_html(StringIO(str(tables[32])))[0]

    # === ETAPE 2 : NETTOYAGE DES DONNEES ===
    # presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
    presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement précédent', 'Entraîneur en chef', 'Stade', 'Capacité']

    # evolution : je donne un nom aux colonnes qui n'en ont pas
    evolution_classement = evolution_classement.iloc[:, :-2]
    evolution_classement.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

    # forme : je donne un nom aux colonnes qui n'en ont pas
    forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'JR1', 'J18', 'J19', 'J20', 'J21','JR2', 'JR3', 'J22', 'J23', 'J24', 'J25', 'J26']# Remarque : On passe de V/D(victoire/défaite) à G/P(gagné/perdu) dans le tableau forme à prendre en compte lorsqu'on fera le travail sur les données.
    return presentation, classement,resultats, evolution_classement, forme
# data2122("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2021-2022")

# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2020/2021
# tableau résultat plus bas que d'habitude : 
# resultats = pd.read_html(StringIO(str(tables[3])))[0] -> resultats = pd.read_html(StringIO(str(tables[4])))[0]


# Tableau présentation n'a que 7 colonnes, pas de "compétition européenne …"
# (modulo les années à adapter
# presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2023-2024', 'Entraîneur en chef', 'Stade', 'Capacité', 'Compétition européenne 2024-2025'] -> presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2023-2024', 'Entraîneur en chef', 'Stade', 'Capacité']

# evolution_classement = pd.read_html(StringIO(str(tables[30])))[0] -> evolution_classement = pd.read_html(StringIO(str(tables[31])))[0]

# forme = pd.read_html(StringIO(str(tables[31])))[0] -> forme = pd.read_html(StringIO(str(tables[32])))[0]


# Le tableau évolution du classement ne comporte pas de colonne après le jour 26, donc pas besoin de supprimer les colonnes
def data2021(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})

    # il y a 37 tables mais seulement 5 nous intéressent
    presentation = clean_table(tables[0])
    classement = pd.read_html(StringIO(str(tables[2])))[0]
    resultats = pd.read_html(StringIO(str(tables[4])))[0]
    evolution_classement = pd.read_html(StringIO(str(tables[31])))[0]
    forme = pd.read_html(StringIO(str(tables[32])))[0]

    # === ETAPE 2 : NETTOYAGE DES DONNEES ===
    # presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
    presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement précédent', 'Entraîneur en chef', 'Stade', 'Capacité']

    # evolution : je donne un nom aux colonnes qui n'en ont pas
    evolution_classement.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

    # forme : je donne un nom aux colonnes qui n'en ont pas
    forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']
    return presentation, classement,resultats, evolution_classement, forme
#data2021("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2020-2021")

# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2019/2020
# Tableau présentation n'a que 7 colonnes, pas de "compétition européenne …"
# (modulo les années à adapter
# presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2023-2024', 'Entraîneur en chef', 'Stade', 'Capacité', 'Compétition européenne 2024-2025'] -> presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2023-2024', 'Entraîneur en chef', 'Stade', 'Capacité']
# tableau classement, le tableau classemnt est plus tôt dans les tables :
# classement = pd.read_html(StringIO(str(tables[2])))[0] -> classement = pd.read_html(StringIO(str(tables[1])))[0]

# évolution classement, le tableau d'évolution du classement est plsu tôt dans les tables : evolution_classement = pd.read_html(StringIO(str(tables[30])))[0] -> evolution_classement = pd.read_html(StringIO(str(tables[28])))[0]

# de même pour forme :
# forme = pd.read_html(StringIO(str(tables[31])))[0] -> forme = pd.read_html(StringIO(str(tables[29])))[0]

# Remarque : Pour cette année à cause du covid les journées s'arrêtent à 17
def data1920(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})

    # il y a 37 tables mais seulement 5 nous intéressent
    presentation = clean_table(tables[0])
    classement = pd.read_html(StringIO(str(tables[1])))[0]
    # resultats20 = pd.read_html(StringIO(str(tables[3])))[0]
    evolution_classement = pd.read_html(StringIO(str(tables[28])))[0]
    evolution_classement = evolution_classement.iloc[:, :18]
    forme = pd.read_html(StringIO(str(tables[29])))[0]
    forme = forme.iloc[:, :18]

    # === ETAPE 2 : NETTOYAGE DES DONNEES ===
    # presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
    presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement précédent', 'Entraîneur en chef', 'Stade', 'Capacité']

    # evolution : je donne un nom aux colonnes qui n'en ont pas
    evolution_classement.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17']

    # forme : je donne un nom aux colonnes qui n'en ont pas
    forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17']

    soup = BeautifulSoup(response.content, "html.parser")

    tables = soup.find_all("table")

    # Étape 3 : Sélectionner la 7ème table de n'importe qu'elle nature
    table_7 = tables[7]  # Les indices commencent à 0

    # Étape 4 : Convertir la table HTML en DataFrame
    data = []
    rows = table_7.find_all('tr')
    for row in rows:
        cols = row.find_all(['th', 'td'])
        cols = [col.text.strip() for col in cols]  # Nettoyer le texte
        data.append(cols)

    # Créer un DataFrame pandas
    df = pd.DataFrame(data)

    # # Étape 5 : Enregistrer en CSV ou manipuler les données
    # df.to_csv("table_7.csv", index=False)


    return presentation, classement,df,evolution_classement, forme
#data1920("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2019-2020")

# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2018/2019
# classement = pd.read_html(StringIO(str(tables[2])))[0] -> classement = pd.read_html(StringIO(str(tables[1])))[0]

# evolution_classement = pd.read_html(StringIO(str(tables[29])))[0]
# forme = pd.read_html(StringIO(str(tables[30])))[0]
def data1619(url):
    # url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2018-2019"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})
    # il y a 37 tables mais seulement 5 nous intéressent
    presentation = clean_table(tables[0])
    classement = pd.read_html(StringIO(str(tables[1])))[0]
    # resultats19 = pd.read_html(StringIO(str(tables[3])))[0]
    evolution_classement = pd.read_html(StringIO(str(tables[29])))[0]
    forme = pd.read_html(StringIO(str(tables[30])))[0]

    # === ETAPE 2 : NETTOYAGE DES DONNEES ===
    # presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
    presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement précédent', 'Entraîneur en chef', 'Stade', 'Capacité']

    # evolution : je donne un nom aux colonnes qui n'en ont pas
    evolution_classement.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

    # forme : je donne un nom aux colonnes qui n'en ont pas
    forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table")

    # Étape 3 : Sélectionner la 7ème table de n'importe qu'elle nature
    table_7 = tables[9]  # Les indices commencent à 0

    # Étape 4 : Convertir la table HTML en DataFrame
    data = []
    rows = table_7.find_all('tr')
    for row in rows:
        cols = row.find_all(['th', 'td'])
        cols = [col.text.strip() for col in cols]  # Nettoyer le texte
        data.append(cols)

    # Créer un DataFrame pandas
    df = pd.DataFrame(data)

    # Étape 5 : Enregistrer en CSV ou manipuler les données
    #df.to_csv("table_7.csv", index=False)
    return presentation, classement, df, evolution_classement, forme
# data18("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2018-2019")
# data18("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2017-2018")
# data18("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2016-2017")



# # Homogénéisation des tableaux formes entre les années : changer V D des années 2022/2023 à 2024/2025

# # forme25 = forme25.replace({'V':'G', 'D':'P'})


# # Résolution du problème de l'exctraction du tableau résultat à intégrer à data18
# url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2016-2017"
# response = requests.get(url)
# soup = BeautifulSoup(response.content, "html.parser")
# # tables = soup.find_all("table", {"class": "wikitable"})
# # for i, table in enumerate(tables):
# #     print(f"Table {i}:")
# #     print(pd.read_html(str(table))[0].head())
# #     print("\n")
# tables = soup.find_all("table")

# # Étape 3 : Sélectionner la 7ème table de n'importe qu'elle nature
# table_7 = tables[9]  # Les indices commencent à 0

# # Étape 4 : Convertir la table HTML en DataFrame
# data = []
# rows = table_7.find_all('tr')
# for row in rows:
#     cols = row.find_all(['th', 'td'])
#     cols = [col.text.strip() for col in cols]  # Nettoyer le texte
#     data.append(cols)

# # Créer un DataFrame pandas
# df = pd.DataFrame(data)

# # Étape 5 : Enregistrer en CSV ou manipuler les données
# df.to_csv("table_7.csv", index=False)
# print(df)

# Résolution du problème de l'exctraction du tableau résultat à intégrer à data1920
# url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2019-2020"
# response = requests.get(url)
# soup = BeautifulSoup(response.content, "html.parser")

# tables = soup.find_all("table")

# Étape 3 : Sélectionner la 7ème table de n'importe qu'elle nature
# table_7 = tables[7]  # Les indices commencent à 0

# # Étape 4 : Convertir la table HTML en DataFrame
# data = []
# rows = table_7.find_all('tr')
# for row in rows:
#     cols = row.find_all(['th', 'td'])
#     cols = [col.text.strip() for col in cols]  # Nettoyer le texte
#     data.append(cols)

# # Créer un DataFrame pandas
# df = pd.DataFrame(data)

# # Étape 5 : Enregistrer en CSV ou manipuler les données
# df.to_csv("table_7.csv", index=False)
# print(df)

