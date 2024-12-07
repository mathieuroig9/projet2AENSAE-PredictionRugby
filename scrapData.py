# IMPORTATION BIBLIOTHEQUES
from io import StringIO
import pandas as pd
import requests
import re
from bs4 import BeautifulSoup

# === ETAPE 1 : IMPORTATION DONNEES ===
# IMPORTATION DE LA PAGE WEB
url = "https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2024-2025"
response = requests.get(url)
soup = BeautifulSoup(response.content, "html.parser")
tables = soup.find_all("table", {"class": "wikitable"})

#il y'a 37 tables mais seulement 5 nous intéressent
presentation = pd.read_html(StringIO(str(tables[0])))[0]
classement = pd.read_html(StringIO(str(tables[2])))[0]
resultats = pd.read_html(StringIO(str(tables[3])))[0]
evolution_classement = pd.read_html(StringIO(str(tables[30])))[0]
forme = pd.read_html(StringIO(str(tables[31])))[0]

# === ETAPE 2 : NETTOYAGE DES DONNEES ===
#presentation : j'enleve les notes de la page et je rectifie 1e,2e,.. en 1,2,..
presentation.columns = ['Club', 'Dernière montée', 'Budget en M€', 'Classement 2023-2024', 'Entraîneur en chef', 'Stade', 'Capacité', 'Compétition européenne 2024-2025']
presentation["Capacité"] = presentation["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
presentation["Classement 2023-2024"] = presentation["Classement 2023-2024"].apply(lambda x: re.sub(r"[^0-9]", "", x))
#classement : j'enleve champion et promu de l'année précédente pour avoir juste le nom des équipes
classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)
#evolution : j'enlève les 3 dernières colonnes non utiles et je donne un nom aux colonnes qui n'en ont pas
evolution_classement = evolution_classement.iloc[:, :-3]
evolution_classement.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']
#forme : je donne un nom aux colonnes qui n'en ont pas
forme.columns = ['Equipes/Journées', 'J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7', 'J8', 'J9', 'J10', 'J11', 'J12', 'J13', 'J14', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20', 'J21', 'J22', 'J23', 'J24', 'J25', 'J26']

# Affichage
print("Présentation des données :")
print(presentation)
print("\nClassement des équipes :")
print(classement)
print("\nÉvolution du classement :")
print(evolution_classement)
print("\nForme des équipes :")
print(forme)
