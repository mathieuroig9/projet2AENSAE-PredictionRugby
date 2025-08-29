import unicodedata
import pandas as pd

def make_calBIN_from_mapped(cal_norm: pd.DataFrame,
                            res_norm: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un tableau type calBIN (V/D/N) à partir de :
      - cal_norm : calendrier mappé (cellules = adversaire canonisé, tout-MAJ = domicile, tout-min = extérieur)
      - res_norm : résultats mappés (ligne 0 = visiteurs canonisés, colonne 0 = domiciles canonisés)

    Retour : DataFrame même structure que cal_norm (col 0 = "Journée X"), cellules = 'V'/'D'/'N'
    """

    def _strip_accents(s: str) -> str:
        return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

    def _norm(s: str) -> str:
        s = (s or "").strip()
        s = _strip_accents(s).lower()
        s = ' '.join(s.split())
        return s

    def _is_upper_word(s: str) -> bool:
        letters = [c for c in str(s) if c.isalpha()]
        return len(letters) > 0 and all(c.isupper() for c in letters)

    # --- préparer la matrice de scores ---
    away_names = list(res_norm.iloc[0, 1:])   # colonnes = extérieurs
    home_names = list(res_norm.iloc[1:, 0])   # lignes = domiciles

    scores = res_norm.iloc[1:, 1:].copy()
    scores.columns = away_names
    scores.index = home_names

    norm_home_to_real = {_norm(h): h for h in home_names}
    norm_away_to_real = {_norm(a): a for a in away_names}

    # --- construire la sortie ---
    out = pd.DataFrame(columns=cal_norm.columns)
    out.iloc[:, 0] = cal_norm.iloc[:, 0]  # "Journée X"

    team_cols = cal_norm.columns[1:]

    for i in range(len(cal_norm)):
        for team in team_cols:
            opp_cell = cal_norm.at[i, team]
            if pd.isna(opp_cell) or str(opp_cell).strip() == "":
                out.at[i, team] = ""
                continue

            opponent_str = str(opp_cell).strip()
            col_team_is_home = _is_upper_word(opponent_str)

            team_norm = _norm(team)
            opp_norm = _norm(opponent_str)

            team_home = norm_home_to_real.get(team_norm, None)
            team_away = norm_away_to_real.get(team_norm, None)
            opp_home = norm_home_to_real.get(opp_norm, None)
            opp_away = norm_away_to_real.get(opp_norm, None)

            if col_team_is_home:
                home_name = team_home or team
                away_name = opp_away or opp_home
            else:
                home_name = opp_home or opp_away
                away_name = team_away or team

            if home_name is None or away_name is None:
                raise KeyError(
                    f"Impossible d'apparier les équipes à la ligne {i+1} (team '{team}' vs '{opponent_str}')."
                )

            score_str = str(scores.at[home_name, away_name]).strip()
            if score_str in {"-", ""} or pd.isna(score_str):
                out.at[i, team] = ""
                continue

            try:
                hs, as_ = map(int, score_str.split("-"))
            except Exception as e:
                raise ValueError(f"Score invalide '{score_str}' pour (dom='{home_name}', ext='{away_name}').") from e

            my_pts, opp_pts = (hs, as_) if col_team_is_home else (as_, hs)
            out.at[i, team] = "V" if my_pts > opp_pts else ("D" if my_pts < opp_pts else "N")

    out.rename(columns={out.columns[0]: "Journées"}, inplace=True)
    return out

def build_points_diff_table(calBIN: pd.DataFrame, scores_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construit un tableau cumulatif (points, diff) à partir d'un calBIN (V/D/N) 
    et de la matrice des scores issue de normalized_res.
    
    - V = 4 pts, N = 2 pts, D = 0 pts
    - diff = points marqués - encaissés
    - Les valeurs sont cumulées par journée.
    """

    # --- nettoyage de la matrice de scores (pour éviter KeyError) ---
    teams = scores_raw.iloc[0, 1:].tolist()
    scores = scores_raw.iloc[1:, 1:].copy()
    scores.index = scores_raw.iloc[1:, 0].tolist()
    scores.columns = teams

    # --- préparation ---
    teams = calBIN.columns[1:]  # toutes les équipes (hors colonne Journée)
    n_days = len(calBIN)
    table = pd.DataFrame(index=calBIN.index, columns=teams)

    # cumuls
    cum_points = {team: 0 for team in teams}
    cum_diff   = {team: 0 for team in teams}

    # --- calcul journée par journée ---
    for i in range(n_days):
        for team in teams:
            result = calBIN.at[i, team]
            if pd.isna(result) or result == "":
                table.at[i, team] = (cum_points[team], cum_diff[team])
                continue

            # trouver l’adversaire = l’équipe qui a aussi joué ce jour-là
            row_results = calBIN.loc[i, teams]
            possible_opponents = row_results.index[row_results != ""]
            opponents = [opp for opp in possible_opponents if opp != team]

            opponent = None
            for opp in opponents:
                try:
                    _ = scores.at[team, opp]
                    opponent = opp
                    break
                except KeyError:
                    continue

            if opponent is None:
                raise KeyError(f"Impossible de trouver l’adversaire de {team} à la journée {i+1}")

            # points du match
            if result == "V":
                pts_match = 4
            elif result == "N":
                pts_match = 2
            else:
                pts_match = 0

            # différentiel
            score_str = str(scores.at[team, opponent])
            if "-" not in score_str:
                scored, conceded = 0, 0
            else:
                scored, conceded = map(int, score_str.split("-"))
            diff_match = scored - conceded

            # cumuls
            cum_points[team] += pts_match
            cum_diff[team]   += diff_match
            table.at[i, team] = (cum_points[team], cum_diff[team])

    return table

def build_rank_table(points_diff_table: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme un tableau (points, diff) en un tableau de classements (1 à 14).
    
    - Tri décroissant sur points puis diff
    - En cas d'égalité stricte → même rang (classement dense : 1,2,2,4,...)
    """
    teams = points_diff_table.columns
    n_days = len(points_diff_table)
    rank_table = pd.DataFrame(index=points_diff_table.index, columns=teams)

    for i in range(n_days):
        # récupérer (points, diff) de toutes les équipes à la journée i
        stats = {team: points_diff_table.at[i, team] for team in teams}

        # trier par points puis diff (ordre décroissant)
        sorted_stats = sorted(stats.items(), key=lambda x: (x[1][0], x[1][1]), reverse=True)

        # assigner les rangs
        rank = 1
        last_val = None
        for j, (team, val) in enumerate(sorted_stats):
            if last_val is not None and val == last_val:
                # même rang que l'équipe précédente
                rank_table.at[i, team] = rank
            else:
                # nouveau rang (dense)
                rank = j + 1
                rank_table.at[i, team] = rank
                last_val = val

    return rank_table

def _strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn')

def reshape_table(df, label_col="Equipes/Journées", day_prefix="J"):
    """
    Reformate un DataFrame (calBIN ou rank) :
      - Équipes en lignes
      - Journées en colonnes renommées J1..Jn
      - 1ère colonne = 'Equipes/Journées'
    """
    df_clean = df.copy()

    # 1) détecter la colonne Journées si elle existe (cas calBIN)
    jour_col = None
    for c in df_clean.columns:
        s = _strip_accents(str(c)).strip().lower()
        if s.startswith("journee") or s.startswith("journees") or s.startswith("journ"):
            jour_col = c
            break

    if jour_col is not None:
        # ----- CAS calBIN -----
        # On garde l'ordre des journées tel qu'il apparaît (ligne 0..n-1)
        n_days = len(df_clean)
        # on enlève la colonne 'Journées' et on transpose
        teams_only = df_clean.drop(columns=[jour_col])
        reshaped = teams_only.T.reset_index().rename(columns={"index": label_col})
        # renommer les colonnes de journées en J1..Jn
        day_cols = [c for c in reshaped.columns if c != label_col]
        new_names = {old: f"{day_prefix}{i+1}" for i, old in enumerate(day_cols)}
        reshaped.rename(columns=new_names, inplace=True)
        # s'assurer de l'ordre J1..Jn
        reshaped = reshaped[[label_col] + [f"{day_prefix}{i+1}" for i in range(n_days)]]
    else:
        # ----- CAS rank -----
        # l'index représente déjà les journées (0..n-1)
        n_days = df_clean.shape[0]
        reshaped = df_clean.T.reset_index().rename(columns={"index": label_col})
        # genère J1..Jn
        old_day_cols = [c for c in reshaped.columns if c != label_col]
        new_names = {old: f"{day_prefix}{i+1}" for i, old in enumerate(old_day_cols)}
        reshaped.rename(columns=new_names, inplace=True)
        reshaped = reshaped[[label_col] + [f"{day_prefix}{i+1}" for i in range(n_days)]]

    return reshaped
