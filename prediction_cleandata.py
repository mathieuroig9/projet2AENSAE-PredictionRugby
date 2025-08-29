from typing import Optional, Dict, Iterable
import re
import numpy as np
import pandas as pd

# -------- utilitaires --------
def _week_cols(df: pd.DataFrame, prefix="J") -> list:
    cols = [c for c in df.columns if isinstance(c, str) and c.startswith(prefix) and c[len(prefix):].isdigit()]
    return sorted(cols, key=lambda x: int(x[len(prefix):]))

_dash_re = re.compile(r'^\s*(\d+)\s*[-–—]\s*(\d+)\s*$')
def _parse_score(v):
    if pd.isna(v): return None
    m = _dash_re.match(str(v))
    return (int(m.group(1)), int(m.group(2))) if m else None

# -------- 1) Winrates depuis la MATRICE (sans mapping) --------
def compute_winrates_from_matrix_simple(
    resultats_year: pd.DataFrame,
    classement_year: pd.DataFrame,
) -> pd.DataFrame:
    """
    Suppose que:
      - la 1ère ligne 'entêtes' est identifiable par Club == 'Clubs' OU 'Résultats (▼dom., ►ext.)'
        (sinon on prend la 1ère ligne comme entêtes par défaut)
      - les colonnes adverses sont '1'..'14' (ou, à défaut, toutes les colonnes sauf 'Club'/'année')
      - les libellés de clubs sont identiques entre entêtes, lignes et classement
    Retourne: [Club, %_victoiredom, %_victoireext, %_victoiretop6]
    """
    # repère la ligne entêtes
    mask_hdr = resultats_year["Club"].astype(str).str.strip().str.lower().isin(
        ["clubs", "résultats (▼dom., ►ext.)".lower()]
    )
    if mask_hdr.any():
        header_row = resultats_year.loc[mask_hdr].iloc[0]
        body = resultats_year.loc[~mask_hdr].copy()
    else:
        header_row = resultats_year.iloc[0]
        body = resultats_year.iloc[1:].copy()

    # colonnes adverses
    digit_cols = [c for c in resultats_year.columns if str(c).isdigit()]
    cand_cols  = digit_cols if digit_cols else [c for c in resultats_year.columns if c not in ("Club","année")]

    # mapping colonne -> adversaire (sans normalisation)
    col_to_opp = {col: str(header_row[col]).strip() for col in cand_cols}
    opp_to_col = {opp: col for col, opp in col_to_opp.items()}

    # lignes: clubs à domicile
    body["Club"] = body["Club"].astype(str).str.strip()
    clubs = body["Club"].tolist()
    row_by_club = {row["Club"]: row for _, row in body.iterrows()}

    # top-6
    cls = classement_year.copy()
    cls["Rang"] = pd.to_numeric(cls["Rang"], errors="coerce")
    top6_set = set(cls.sort_values("Rang").head(6)["Club"].astype(str).str.strip())

    rows = []
    for club in clubs:
        row = row_by_club[club]

        # domicile
        home_games = home_wins = home_vs_top6_games = home_vs_top6_wins = 0
        for col, opp in col_to_opp.items():
            if opp == club: 
                continue
            sc = _parse_score(row.get(col))
            if sc is None: 
                continue
            home_games += 1
            if opp in top6_set:
                home_vs_top6_games += 1
            if sc[0] > sc[1]:
                home_wins += 1
                if opp in top6_set:
                    home_vs_top6_wins += 1

        # extérieur
        away_games = away_wins = away_vs_top6_games = away_vs_top6_wins = 0
        club_col = opp_to_col.get(club, None)
        if club_col is not None:
            for opp, opp_row in row_by_club.items():
                if opp == club:
                    continue
                sc = _parse_score(opp_row.get(club_col))
                if sc is None:
                    continue
                away_games += 1
                if opp in top6_set:
                    away_vs_top6_games += 1
                if sc[1] > sc[0]:
                    away_wins += 1
                    if opp in top6_set:
                        away_vs_top6_wins += 1

        tot6_g = home_vs_top6_games + away_vs_top6_games
        tot6_w = home_vs_top6_wins  + away_vs_top6_wins

        rows.append({
            "Club": club,
            "%_victoiredom":  (home_wins / home_games) if home_games else np.nan,
            "%_victoireext":  (away_wins / away_games) if away_games else np.nan,
            "%_victoiretop6": (tot6_w / tot6_g)        if tot6_g    else np.nan,
        })

    return pd.DataFrame(rows)

# -------- 2) Extract features (sans mapping de noms) --------
def extract_features_top14(
    classement: pd.DataFrame,
    evolution: pd.DataFrame,
    forme: pd.DataFrame,
    df_champions: pd.DataFrame,
    resultats: Optional[pd.DataFrame] = None,
    weeks_total: int = 26,
    mapping_forme: Optional[Dict[str, int]] = None,  # V=4, N=2, D=0 par défaut
) -> pd.DataFrame:

    def ensure_year_col(df: pd.DataFrame, name: str) -> pd.DataFrame:
        if "année" in df.columns: return df
        if "annee" in df.columns: return df.rename(columns={"annee": "année"})
        raise KeyError(f"[{name}] colonne 'annee'/'année' manquante.")

    def longest_win_streak(values: Iterable[str]) -> int:
        best = cur = 0
        for v in values:
            if v == "V": cur += 1; best = max(best, cur)
            else: cur = 0
        return best

    if mapping_forme is None:
        mapping_forme = {"V": 4, "N": 2, "D": 0}

    # Harmonise juste la colonne année
    classement = ensure_year_col(classement.copy(), "classement")
    evolution  = ensure_year_col(evolution.copy(),  "evolution")
    forme      = ensure_year_col(forme.copy(),      "forme")
    champs     = ensure_year_col(df_champions.copy(), "df_champions")
    if resultats is not None:
        resultats = ensure_year_col(resultats.copy(), "resultats")

    # base des paires (Club, année)
    base = (pd.concat(
                [classement[["Club","année"]],
                 evolution[["Club","année"]],
                 forme[["Club","année"]]],
                axis=0, ignore_index=True)
            .dropna(subset=["Club","année"])
            .drop_duplicates(subset=["Club","année"]))

    # classement -> Pts/Diff/Rang, Pm/Pe ramenés à 26
    required_cls = {"Pts","Diff","J","Rang","Pm","Pe"}
    miss = required_cls - set(classement.columns)
    if miss:
        raise KeyError(f"[classement] colonnes manquantes: {sorted(miss)}")

    cls = classement[["Club","Rang","année","Pts","Diff","J","Pm","Pe"]].copy()
    for col in ["Pts","Diff","J","Rang","Pm","Pe"]:
        cls[col] = pd.to_numeric(cls[col], errors="coerce")

    factor = np.where(cls["J"]>0, weeks_total/cls["J"], np.nan)
    cls["Pts_adj_26"]    = cls["Pts"]  * factor
    cls["Diff_adj_26"]   = cls["Diff"] * factor
    cls["pts_marques"]   = cls["Pm"]   * factor
    cls["pts_encaisses"] = cls["Pe"]   * factor
    cls["rank_J26"]      = cls["Rang"]
    cls_feats = cls[["Club","année","Pts_adj_26","Diff_adj_26","rank_J26","pts_marques","pts_encaisses"]]

    # forme / séries / momentum
    frm = forme.copy()
    frm_weeks = _week_cols(frm)
    for c in frm_weeks:
        frm[c] = frm[c].astype(str).str.upper().replace({"NAN":"", "NONE":"", "": np.nan})

    def last5_form_sum(row):
        seq = [row[c] for c in frm_weeks if pd.notna(row[c])]
        if not seq: return np.nan
        return float(sum(mapping_forme.get(v, 0) for v in seq[-5:]))

    def pts10prem_vs_10fin(row):
        seq = [row[c] for c in frm_weeks if pd.notna(row[c])]
        if not seq: return np.nan
        s1 = sum(mapping_forme.get(v, 0) for v in seq[:10])
        s2 = sum(mapping_forme.get(v, 0) for v in seq[-10:])
        return float(s2 - s1)

    frm["win_streak_max"]   = frm.apply(lambda r: longest_win_streak([r[c] for c in frm_weeks if pd.notna(r[c])]), axis=1)
    frm["form_last5_sum"]   = frm.apply(last5_form_sum, axis=1)
    frm["pts10premvs10fin"] = frm.apply(pts10prem_vs_10fin, axis=1)
    frm_feats = frm[["Club","année","win_streak_max","form_last5_sum","pts10premvs10fin"]]

    # winrates depuis la matrice (si fournie)
    if resultats is not None:
        pieces = []
        for year, grp in resultats.groupby("année", sort=False):
            cls_y = classement[classement["année"] == year][["Club","Rang"]]
            piece = compute_winrates_from_matrix_simple(grp.reset_index(drop=True), cls_y)
            piece["année"] = year
            pieces.append(piece)
        winrates = (pd.concat(pieces, ignore_index=True)
                    if pieces else pd.DataFrame(columns=["Club","année","%_victoiredom","%_victoireext","%_victoiretop6"]))
    else:
        winrates = pd.DataFrame(columns=["Club","année","%_victoiredom","%_victoireext","%_victoiretop6"])

    # historique champion/vice
    champs = champs[["année","champion","vice_champion"]].copy()
    champions_by_year = champs.set_index("année")["champion"].to_dict()
    vice_by_year      = champs.set_index("année")["vice_champion"].to_dict()

    def hist_features(row):
        y, c = int(row["année"]), row["Club"]
        last = 1.0 if champions_by_year.get(y-1) == c else 0.0
        w3 = sum(1 for d in (1,2,3) if champions_by_year.get(y-d) == c)
        v3 = sum(1 for d in (1,2,3) if vice_by_year.get(y-d) == c)
        return pd.Series({
            "championAnDernier": last,
            "champion3dernieresSaisons": float(w3),
            "viceChampion3dernieresSaisons": float(v3),
        })

    hist = base.copy()
    hist = pd.concat([hist, hist.apply(hist_features, axis=1)], axis=1)
    hist = hist[["Club","année","championAnDernier","champion3dernieresSaisons","viceChampion3dernieresSaisons"]]

    # assemblage
    out = (base.merge(cls_feats, on=["Club","année"], how="left")
                .merge(frm_feats, on=["Club","année"], how="left")
                .merge(winrates, on=["Club","année"], how="left")
                .merge(hist, on=["Club","année"], how="left")
                .sort_values(["année","Club"]).reset_index(drop=True))

    cols = [
        "année","Club",
        "Pts_adj_26","Diff_adj_26","rank_J26",
        "form_last5_sum","win_streak_max",
        "pts_marques","pts_encaisses",
        "%_victoiretop6","%_victoiredom","%_victoireext",
        "pts10premvs10fin",
        "championAnDernier","champion3dernieresSaisons","viceChampion3dernieresSaisons",
    ]
    for c in cols:
        if c not in out.columns: out[c] = np.nan
    return out[cols]