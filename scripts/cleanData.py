from scrapData import data23

def nettoyage(presentation, classement, resultats, evolution_classement, forme):
    #on enlève les notes
    presentation["Capacité"] = presentation["Capacité"].apply(lambda x: int(x.replace(" ", "").replace("\xa0", "").split("[")[0]))
    #on remplace 1er,2eme,.. par 1,2,..
    presentation["Classement précédent"] = presentation["Classement précédent"].apply(lambda x: re.sub(r"[^0-9]", "", x))
    #on enlève les valeurs aberrantes
    presentation["Capacité"] = presentation["Capacité"].apply(lambda x: np.nan if x > 100000 else x)
    presentation["Classement précédent"] = presentation["Classement précédent"].apply(lambda x: re.sub(r"[^0-9]", "", x) if "Pro D2" not in x else str(int(re.sub(r"[^0-9]", "", x)) + 14)

    #on garde seulement le nom de l'équipe
    classement24["Club"] = classement24["Club"].apply(lambda x: x.rstrip(" C1") if x.endswith(" C1") else x)
    classement24["Club"] = classement24["Club"].apply(lambda x: x.rstrip(" C2") if x.endswith(" C2") else x)
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
    classement["Club"] = classement["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)
    
    forme = forme.replace({'V':'G', 'D':'P'})

tab25=data23("https://fr.wikipedia.org/wiki/Championnat_de_France_de_rugby_%C3%A0_XV_2024-2025")
pr25=tab25[0]
cl25=tab25[1]
re25=tab25[2]
ev25=tab25[3]
fo25=tab25[4]


#pour vscode sur mac
import code
code.interact(local=locals())