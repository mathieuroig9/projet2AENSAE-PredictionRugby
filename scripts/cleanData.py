import re
import pandas as pd
import numpy as np

def nettoyage(presentation, classement, resultats, evolution_classement, forme, do_pre=True, do_evo=True, do_for=True):
    
    if do_pre:
        def clean_capacite(x):
            if pd.isna(x):
                return np.nan
            s = str(x).replace("\xa0", "").replace(" ", "")
            # si plusieurs nombres séparés par "et", on les additionne
            if "et" in s:
                nums = re.findall(r"\d+", s)
                return sum(int(n) for n in nums)
            # sinon on prend le premier nombre trouvé
            nums = re.findall(r"\d+", s)
            return int(nums[0]) if nums else np.nan

        # on enlève les notes et on nettoie capacité
        presentation["Capacité"] = presentation["Capacité"].apply(clean_capacite)

        # on enlève les valeurs aberrantes
        presentation["Capacité"] = presentation["Capacité"].apply(
            lambda x: np.nan if pd.notna(x) and int(x) > 100000 else x
        )

        # --- Budget en M€ : supprimer les [ ... ] éventuels et convertir en float ---
        presentation["Budget en M€"] = (
            presentation["Budget en M€"]
            .astype(str)
            .str.replace(r"\[.*?\]", "", regex=True)  # on enlève les [1], [Note 2] etc.
            .str.replace(",", ".", regex=False)       # si jamais il y a une virgule
        )
        presentation["Budget en M€"] = pd.to_numeric(presentation["Budget en M€"], errors="coerce")


        #on remplace 1er,2eme,.. par 1,2,..
        presentation["Classement précédent"] = presentation["Classement précédent"].apply(lambda x: str(int(re.search(r"\d+", x).group()) + 14) if isinstance(x, str) and "(Pro D2)" in x else re.sub(r"[^\d]", "", x))
        
        #on garde seulement le nom de l'équipe
        presentation["Club"] = presentation["Club"].apply(lambda x: re.sub(r"\[.*?\]", "", x).strip() if isinstance(x, str) else x)
        presentation["Club"] = presentation["Club"].apply(lambda x: x.rstrip(" C1") if x.endswith(" C1") else x)
        presentation["Club"] = presentation["Club"].apply(lambda x: x.rstrip(" C2") if x.endswith(" C2") else x)
        presentation["Club"] = presentation["Club"].apply(lambda x: x.rstrip(" T") if x.endswith(" T") else x)
        presentation["Club"] = presentation["Club"].apply(lambda x: x.rstrip(" P") if x.endswith(" P") else x)
        
    # --- Noms des clubs : retire les [notes], et les suffixes "C1 C2 T P (T) (P)" en fin de chaîne
    classement["Club"] = (
        classement["Club"]
        .astype(str)
        .str.replace(r"\[.*?\]", "", regex=True)          # enlève [ ... ]
        .str.replace(r"\s+(C1|C2|T|P|\(T\)|\(P\))$", "", regex=True)
        .str.strip()
    )

    # --- Colonnes numériques (classement) : nettoie et force en entiers (nullable Int64)
    int_cols = [c for c in ["Pts", "Diff", "J", "Rang", "V", "N", "D", "Pm", "Pe", "Bo", "Bd"]
                if c in classement.columns]

    for c in int_cols:
        s = (
            classement[c]
            .astype(str)
            .str.replace("\u00a0", " ", regex=False)        # NBSP -> espace
            .str.replace(",", ".", regex=False)             # virgule -> point
            .str.replace(r"[^\d\.\-\+]", "", regex=True)    # garde chiffres/signe/point (enlève ¹, *)
            .str.replace(r"\..*$", "", regex=True)          # coupe tout après un point (évite "6.0")
            .str.strip()
        )
        s = s.replace("", pd.NA)                            # <<< ICI : vide -> NaN (OK)
        classement[c] = pd.to_numeric(s, errors="coerce").astype("Int64")


    # on complète les blancs par -
    resultats.replace('', '-', inplace=True)

    if do_evo:
        #il y'a des fois des notes après le classement, on garde que le classement
        cols = evolution_classement.columns[1:]

        # 1) extraire l'entier en début de cellule (ou NaN)
        tmp = (
            evolution_classement[cols]
                .astype(str)
                .apply(lambda s: s.str.extract(r'^\s*(\d+)', expand=False))
        )
        tmp = tmp.apply(pd.to_numeric, errors='coerce')
        tmp = tmp.where((tmp >= 0) & (tmp <= 16))
        evolution_classement[cols] = tmp

    if do_for:
        forme.iloc[:, 1:] = forme.iloc[:, 1:].replace({'G': 'V', 'P': 'D'}, regex=False)
