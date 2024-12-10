# IMPORTATION BIBLIOTHEQUES
from io import StringIO
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup
import sys
import subprocess

# Installer le module lxml
subprocess.check_call([sys.executable, "-m", "pip", "install", "lxml"])


# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2024/2025
def data23(url, year):
    # url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2024-2025"
    a = f'Classement{year}{year+1}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})

    # il y'a 37 tables mais seulement 5 nous intéressent
    presentation = pd.read_html(StringIO(str(tables[0])))[0]
    classement = pd.read_html(StringIO(str(tables[2])))[0]
    resultats = pd.read_html(StringIO(str(tables[3])))[0]
    evolution_classement = pd.read_html(StringIO(str(tables[30])))[0]
    forme = pd.read_html(StringIO(str(tables[31])))[0]

    # === ETAPE 2 : SÉLECTION DES DONNEES ===
    # presentation : j'enleve les notes de la page et je rectifie 1e,2e,.. en 1,2,..
    presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'a', 'Entraîneur en chef', 'Stade', 'Capacité', 'Compétition européenne 2024-2025']
    presentation["Capacité"] = presentation["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
    presentation["a"] = presentation["a"].apply(lambda x: re.sub(r"[^0-9]", "", x))
    # classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)
    # evolution : j'enlève les 3 dernières colonnes non utiles et je donne un nom aux colonnes qui n'en ont pas
    evolution_classement = evolution_classement.iloc[:, :-3]
    evolution_classement.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']
    # forme : je donne un nom aux colonnes qui n'en ont pas
    forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

    # # Affichage
    # print("Présentation des données :")
    # print(presentation25)
    # print("\nClassement des équipes :")
    # print(classement25)
    # print("\nÉvolution du classement :")
    # print(evolution_classement25)
    # print("\nForme des équipes :")
    # print(forme25)
    return presentation, classement, resultats, evolution_classement, forme

#data23("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2024-2025", 2024)

# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2023/2024, exactement comme pour 2024/2025
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2023-2024"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})

# il y a 37 tables mais seulement 5 nous intéressent
presentation24 = pd.read_html(StringIO(str(tables[0])))[0]
classement24 = pd.read_html(StringIO(str(tables[2])))[0]
resultats24 = pd.read_html(StringIO(str(tables[3])))[0]
evolution_classement24 = pd.read_html(StringIO(str(tables[30])))[0]
forme24 = pd.read_html(StringIO(str(tables[31])))[0]

# === ETAPE 2 : SÉLECTION DES DONNEES ===
# presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
presentation24.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2022-2023', 'Entraîneur en chef', 'Stade', 'Capacité', 'Compétition européenne 2023-2024']
presentation24["Capacité"] = presentation24["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
presentation24["Classement 2022-2023"] = presentation24["Classement 2022-2023"].apply(lambda x: re.sub(r"[^0-9]", "", x))

# classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
classement24["Club"] = classement24["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
classement24["Club"] = classement24["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)

# evolution : j'enlève les 3 dernières colonnes non utiles et je donne un nom aux colonnes qui n'en ont pas
evolution_classement24 = evolution_classement24.iloc[:, :-3]
evolution_classement24.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

# forme : je donne un nom aux colonnes qui n'en ont pas
forme24.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']



# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2022/2023
# 3 modifs :
# evolution_classement = pd.read_html(StringIO(str(tables[30])))[0] -> evolution_classement = pd.read_html(StringIO(str(tables[31])))[0]

# forme = pd.read_html(StringIO(str(tables[31])))[0] -> forme = pd.read_html(StringIO(str(tables[32])))[0]

# Pour les 2 on doit décaler table de 1 car il y a un tableau "cumul des points" avant ces tableaux qu'il n'y avait pas sur les autres pages

# evolution_classement = evolution_classement.iloc[:, :-3] -> evolution_classement = evolution_classement.iloc[:, :-2] ( A partir de cette année et des précédents il manque la colonne barrage)




url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2022-2023"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})

# il y a 37 tables mais seulement 5 nous intéressent
presentation23 = pd.read_html(StringIO(str(tables[0])))[0]
classement23 = pd.read_html(StringIO(str(tables[2])))[0]
resultats23 = pd.read_html(StringIO(str(tables[3])))[0]
evolution_classement23 = pd.read_html(StringIO(str(tables[31])))[0]
forme23 = pd.read_html(StringIO(str(tables[32])))[0]

# === ETAPE 2 : SÉLECTION DES DONNEES ===
# presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
presentation23.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2021-2022', 'Entraîneur en chef', 'Stade', 'Capacité', 'Compétition européenne 2022-2023']
presentation23["Capacité"] = presentation23["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
presentation23["Classement 2021-2022"] = presentation23["Classement 2021-2022"].apply(lambda x: re.sub(r"[^0-9]", "", x))

# classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
classement23["Club"] = classement23["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
classement23["Club"] = classement23["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)

# evolution : j'enlève les 3 dernières colonnes non utiles et je donne un nom aux colonnes qui n'en ont pas
evolution_classement23 = evolution_classement23.iloc[:, :-2]
evolution_classement23.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

# forme : je donne un nom aux colonnes qui n'en ont pas
forme23.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']




# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2021/2022, modification pareil que pour 2022/2023 + il y a eu des journées de rattrapages nommées R1, R2,R3 donc il y a 3 colonnes de plus (a voir plus tard comment intégrer ça au modèle) :
# forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26'] -> forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'JR1', 'J18', 'J19', 'J20', 'J21','JR2', 'JR3', 'J22', 'J23', 'J24', 'J25', 'J26']

url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2021-2022"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})

# il y a 37 tables mais seulement 5 nous intéressent
presentation22 = pd.read_html(StringIO(str(tables[0])))[0]
classement22 = pd.read_html(StringIO(str(tables[2])))[0]
resultats22 = pd.read_html(StringIO(str(tables[3])))[0]
evolution_classement22 = pd.read_html(StringIO(str(tables[31])))[0]
forme22 = pd.read_html(StringIO(str(tables[32])))[0]

# === ETAPE 2 : SÉLECTION DES DONNEES ===
# presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
presentation22.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2020-2021', 'Entraîneur en chef', 'Stade', 'Capacité', 'Compétition européenne 2021-2022']
presentation22["Capacité"] = presentation22["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
presentation22["Classement 2020-2021"] = presentation22["Classement 2020-2021"].apply(lambda x: re.sub(r"[^0-9]", "", x))

# classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
classement22["Club"] = classement22["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
classement22["Club"] = classement22["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)

# evolution : je donne un nom aux colonnes qui n'en ont pas
evolution_classement22 = evolution_classement22.iloc[:, :-2]
evolution_classement22.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

# forme : je donne un nom aux colonnes qui n'en ont pas
forme22.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'JR1', 'J18', 'J19', 'J20', 'J21','JR2', 'JR3', 'J22', 'J23', 'J24', 'J25', 'J26']# Remarque : On passe de V/D(victoire/défaite) à G/P(gagné/perdu) dans le tableau forme à prendre en compte lorsqu'on ra le travail sur les données.


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

url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2020-2021"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})

# il y a 37 tables mais seulement 5 nous intéressent
presentation21 = pd.read_html(StringIO(str(tables[0])))[0]
classement21 = pd.read_html(StringIO(str(tables[2])))[0]
resultats21 = pd.read_html(StringIO(str(tables[4])))[0]
evolution_classement21 = pd.read_html(StringIO(str(tables[31])))[0]
forme21 = pd.read_html(StringIO(str(tables[32])))[0]

# === ETAPE 2 : SÉLECTION DES DONNEES ===
# presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
presentation21.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2019-2020', 'Entraîneur en chef', 'Stade', 'Capacité']
presentation21["Capacité"] = presentation21["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
presentation21["Classement 2019-2020"] = presentation21["Classement 2019-2020"].apply(lambda x: re.sub(r"[^0-9]", "", x))

# classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
classement21["Club"] = classement21["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
classement21["Club"] = classement21["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)

# evolution : je donne un nom aux colonnes qui n'en ont pas
evolution_classement21.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

# forme : je donne un nom aux colonnes qui n'en ont pas
forme21.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']


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
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2019-2020"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})

# il y a 37 tables mais seulement 5 nous intéressent
presentation20 = pd.read_html(StringIO(str(tables[0])))[0]
classement20 = pd.read_html(StringIO(str(tables[1])))[0]
# resultats20 = pd.read_html(StringIO(str(tables[3])))[0]
evolution_classement20 = pd.read_html(StringIO(str(tables[28])))[0]
forme20 = pd.read_html(StringIO(str(tables[29])))[0]

# === ETAPE 2 : SÉLECTION DES DONNEES ===
# presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
presentation20.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2018-2019', 'Entraîneur en chef', 'Stade', 'Capacité']
presentation20["Capacité"] = presentation20["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
presentation20["Classement 2018-2019"] = presentation20["Classement 2018-2019"].apply(lambda x: re.sub(r"[^0-9]", "", x))

# classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
classement20["Club"] = classement20["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
classement20["Club"] = classement20["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)

# evolution : je donne un nom aux colonnes qui n'en ont pas
evolution_classement20.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

# forme : je donne un nom aux colonnes qui n'en ont pas
forme20.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']


# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2018/2019
# classement = pd.read_html(StringIO(str(tables[2])))[0] -> classement = pd.read_html(StringIO(str(tables[1])))[0]

# evolution_classement = pd.read_html(StringIO(str(tables[29])))[0]
# forme = pd.read_html(StringIO(str(tables[30])))[0]

url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2018-2019"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})

# il y a 37 tables mais seulement 5 nous intéressent
presentation19 = pd.read_html(StringIO(str(tables[0])))[0]
classement19 = pd.read_html(StringIO(str(tables[1])))[0]
# resultats19 = pd.read_html(StringIO(str(tables[3])))[0]
evolution_classement19 = pd.read_html(StringIO(str(tables[29])))[0]
forme19 = pd.read_html(StringIO(str(tables[30])))[0]

# === ETAPE 2 : SÉLECTION DES DONNEES ===
# presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
presentation19.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2017-2018', 'Entraîneur en chef', 'Stade', 'Capacité']
presentation19["Capacité"] = presentation19["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
presentation19["Classement 2017-2018"] = presentation19["Classement 2017-2018"].apply(lambda x: re.sub(r"[^0-9]", "", x))

# classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
classement19["Club"] = classement19["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
classement19["Club"] = classement19["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)

# evolution : je donne un nom aux colonnes qui n'en ont pas
evolution_classement19.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

# forme : je donne un nom aux colonnes qui n'en ont pas
forme19.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']


# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2017/2018
# Pareil que 2018/2019
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2017-2018"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})

# il y a 37 tables mais seulement 5 nous intéressent
presentation18 = pd.read_html(StringIO(str(tables[0])))[0]
classement18 = pd.read_html(StringIO(str(tables[1])))[0]
# resultats18 = pd.read_html(StringIO(str(tables[3])))[0]
evolution_classement18 = pd.read_html(StringIO(str(tables[29])))[0]
forme18 = pd.read_html(StringIO(str(tables[30])))[0]

# === ETAPE 2 : SÉLECTION DES DONNEES ===
# presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
presentation18.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2016-2017', 'Entraîneur en chef', 'Stade', 'Capacité']
presentation18["Capacité"] = presentation18["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
presentation18["Classement 2016-2017"] = presentation18["Classement 2016-2017"].apply(lambda x: re.sub(r"[^0-9]", "", x))

# classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
classement18["Club"] = classement18["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
classement18["Club"] = classement18["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)

# evolution : j'enlève les 3 dernières colonnes non utiles et je donne un nom aux colonnes qui n'en ont pas

evolution_classement18.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

# forme : je donne un nom aux colonnes qui n'en ont pas
forme18.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']


# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB 2016/2017
# Pareil que 2018/2019
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2016-2017"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})

# il y a 37 tables mais seulement 5 nous intéressent
presentation17 = pd.read_html(StringIO(str(tables[0])))[0]
classement17 = pd.read_html(StringIO(str(tables[1])))[0]
# resultats17 = pd.read_html(StringIO(str(tables[3])))[0]
evolution_classement17 = pd.read_html(StringIO(str(tables[29])))[0]
forme17 = pd.read_html(StringIO(str(tables[30])))[0]

# === ETAPE 2 : SÉLECTION DES DONNEES ===
# presentation : j'enleve les notes de la page et je rectifie 1e, 2e,.. en 1, 2,..
presentation17.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2015-2016', 'Entraîneur en chef', 'Stade', 'Capacité']
presentation17["Capacité"] = presentation17["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
presentation17["Classement 2015-2016"] = presentation17["Classement 2015-2016"].apply(lambda x: re.sub(r"[^0-9]", "", x))

# classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
classement17["Club"] = classement17["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
classement17["Club"] = classement17["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)

# evolution : je donne un nom aux colonnes qui n'en ont pas
evolution_classement17.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

# forme : je donne un nom aux colonnes qui n'en ont pas
forme17.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']
