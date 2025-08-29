import pandas as pd
import numpy as np
import re
from collections import Counter
import random
from math import log
from time import time
import math


def find_best_parameters(classement,resultat):
    # -----------------------------
    # Config I/O
    # -----------------------------

    # Random search
    TRIALS_STATIC     = 200   # augmente si besoin (300-500)
    TRIALS_SEQUENTIAL = 100   # optionnel, plus lent
    RANDOM_SEED       = 42
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # -----------------------------
    # Chargement des données
    # -----------------------------
    classement_df = classement
    resultat_df   = resultat

    # Assure la colonne année (fallbacks si besoin)
    if "année" not in classement_df.columns and "annee" in classement_df.columns:
        classement_df = classement_df.rename(columns={"annee": "année"})
    if "année" not in resultat_df.columns and "annee" in resultat_df.columns:
        resultat_df = resultat_df.rename(columns={"annee": "année"})

    # -----------------------------
    # Parsing "matrice" -> matches
    # -----------------------------
    SCORE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

    def parse_score(cell):
        if not isinstance(cell, str):
            return None
        m = SCORE_RE.match(cell)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def season_clubs(df, year):
        return set(df[df["année"] == year]["Club"].dropna().unique())

    def build_opponent_mapping_for_season(df_season, clubs_y):
        """
        Pour chaque colonne adverse (non 'Club'/'année'), regarde la valeur la plus fréquente
        qui correspond à un nom de club connu cette saison. Sert de mapping col -> club.
        """
        cols = list(df_season.columns)
        key_cols = [c for c in ["Club", "club", "CLUB"] if c in cols]
        year_cols = [c for c in ["année", "annee", "year"] if c in cols]
        if not key_cols or not year_cols:
            return {}
        club_col = key_cols[0]
        year_col = year_cols[0]
        opp_cols = [c for c in cols if c not in (club_col, year_col)]
        mapping = {}
        for oc in opp_cols:
            vals = df_season[oc].dropna().astype(str)
            candidates = [v for v in vals if v in clubs_y]
            mapping[oc] = Counter(candidates).most_common(1)[0][0] if candidates else None
        return mapping

    def extract_matches_smart(resultat_df, classement_df):
        """
        Reconstruit la liste des matchs avec scores à partir du format matrice.
        Retour: DataFrame [season, home, away, hs, as]
        """
        all_matches = []
        if "année" not in resultat_df.columns:
            raise ValueError("Column 'année' missing in resultat.csv")
        for y in sorted(resultat_df["année"].dropna().unique()):
            block = resultat_df[resultat_df["année"] == y].copy()
            clubs_y = season_clubs(classement_df, y)
            if not clubs_y:
                continue
            cols = list(block.columns)
            key_cols = [c for c in ["Club", "club", "CLUB"] if c in cols]
            if not key_cols:
                continue
            club_col = key_cols[0]
            opp_map = build_opponent_mapping_for_season(block, clubs_y)
            for _, row in block.iterrows():
                home = row[club_col]
                if home not in clubs_y:
                    continue
                for oc, away in opp_map.items():
                    score = parse_score(row[oc])
                    if score is None or away is None or away == home:
                        continue
                    hs, a_s = score
                    all_matches.append({"season": int(y), "home": home, "away": away, "hs": hs, "as": a_s})
        return pd.DataFrame(all_matches)

    matches_df = extract_matches_smart(resultat_df, classement_df)
    matches_df = matches_df[(matches_df["home"] != matches_df["away"])].reset_index(drop=True)

    print(f"Parsed matches: {len(matches_df)} | seasons sample: {sorted(matches_df['season'].unique())[:8]} ...")

    # -----------------------------
    # Probabilités V/N/D (Davidson)
    # -----------------------------
    def davidson_probs(elo_home, elo_away, H=60.0, nu=0.02, s=400.0):
        """
        H: avantage domicile (Elo)
        nu: paramètre "tie" (nul) — augmente P(nul) quand les forces sont proches
        s : "pente" de l'échelle Elo (400 par défaut)
        Retourne (P_home_win, P_draw, P_away_win)
        """
        A = 10 ** ((elo_home + H) / s)
        B = 10 ** (elo_away / s)
        D = A + B + 2.0 * nu * (A * B) ** 0.5
        p_home = A / D
        p_away = B / D
        p_draw = 2.0 * nu * (A * B) ** 0.5 / D
        # sécurité numérique
        p_home = max(p_home, 1e-12); p_draw = max(p_draw, 1e-12); p_away = max(p_away, 1e-12)
        ssum = p_home + p_draw + p_away
        return p_home/ssum, p_draw/ssum, p_away/ssum

    def match_label(hs, a_s):
        return 'H' if hs > a_s else ('A' if hs < a_s else 'D')

    def logloss_3way(pH, pD, pA, label):
        if label == 'H': return -np.log(pH)
        if label == 'D': return -np.log(pD)
        return -np.log(pA)

    def margin_factor(delta_elo, margin_points, scale=2.2):
        """
        Optionnel: facteur marge (type Elo football)
        Si tu ne veux pas l'utiliser, renvoie 1.0.
        """
        if margin_points is None:
            return 1.0
        return log(margin_points + 1.0) * (2.2 / (0.001 * abs(delta_elo) + 2.2))

    # -----------------------------
    # Construction Elo pré-saison t
    # -----------------------------
    def build_prior_scores(classement_df, year_t1, year_t2, M, w_recent=0.7):
        """
        Prior multi-saisons (SR uniquement), z-scores par saison, combine rang & diff (0.6 / 0.4)
        """
        def season_score(df):
            s = df.copy()
            s["score_rang"] = M - s["Rang"]
            for col in ["score_rang", "Diff"]:
                mu = s[col].mean(); sd = s[col].std()
                s[col+"_z"] = 0.0 if (sd == 0 or pd.isna(sd)) else (s[col] - mu) / sd
            s["score_z"] = 0.6*s["score_rang_z"] + 0.4*s["Diff_z"]
            return s[["Club", "score_z"]]

        sr_1 = classement_df[classement_df["année"] == year_t1][["Club","Rang","Diff"]]
        sr_2 = classement_df[classement_df["année"] == year_t2][["Club","Rang","Diff"]] if (year_t2 in classement_df["année"].values) else pd.DataFrame(columns=["Club","Rang","Diff"])

        sc1 = season_score(sr_1) if len(sr_1) else pd.DataFrame(columns=["Club","score_z"])
        sc2 = season_score(sr_2) if len(sr_2) else pd.DataFrame(columns=["Club","score_z"])

        prior_scores = pd.merge(sc1, sc2, on="Club", how="outer", suffixes=("_t1","_t2")).fillna(0.0)
        prior_scores["prior_z"] = w_recent*prior_scores["score_z_t1"] + (1.0 - w_recent)*prior_scores["score_z_t2"]
        return prior_scores[["Club","prior_z"]]

    def build_preseason_elos_for_year(year_t, params, classement_df):
        """
        Construit l'Elo pré-saison pour l'année t en utilisant seulement t-1 et t-2.
        Gère promus multiples via pseudo-Diff ancré bas de tableau t-1.
        """
        alpha = params.get("alpha", 15.0)
        beta  = params.get("beta", 0.10)
        lam   = params.get("lam", 0.20)
        w_recent = params.get("w_recent", 0.7)
        promo_penalty = params.get("promo_penalty", 0.25)  # fraction de sigma_bot
        bottom_k = int(params.get("bottom_k", 3))          # nb clubs pour mu/sigma bas de tableau

        year_t1 = year_t - 1
        year_t2 = year_t - 2

        clubs_t  = season_clubs(classement_df, year_t)
        clubs_t1 = season_clubs(classement_df, year_t1)
        M = len(clubs_t) if len(clubs_t) > 0 else 14

        promoted = sorted(clubs_t - clubs_t1)    # 0, 1 ou 2
        relegated = sorted(clubs_t1 - clubs_t)

        # Diff bas de tableau t-1
        sr_t1 = classement_df[classement_df["année"] == year_t1].copy().sort_values("Rang")
        bot = sr_t1.tail(min(bottom_k, len(sr_t1)))
        mu_bot = bot["Diff"].mean() if len(bot) else 0.0
        sigma_bot = bot["Diff"].std() if len(bot) else 0.0
        diff_prom = mu_bot - promo_penalty * (0.0 if (np.isnan(sigma_bot)) else sigma_bot)

        # Elo "fin t-1" (hybride SR: rang + diff)
        elo_fin_t1 = {}
        for _, row in sr_t1.iterrows():
            club = row["Club"]
            rank = int(row["Rang"]) if pd.notna(row["Rang"]) else M
            diff = float(row["Diff"]) if pd.notna(row["Diff"]) else 0.0
            elo_fin_t1[club] = alpha * (M - rank) + beta * diff

        # Ajout promus (rang=M, pseudo-diff bas de tableau)
        for club in promoted:
            elo_fin_t1[club] = alpha * (M - M) + beta * diff_prom

        # Prior z-scored (t-1 & t-2), reprojeté sur l'échelle d'Elo fin t-1
        prior_scores = build_prior_scores(classement_df, year_t1, year_t2, M, w_recent=w_recent)
        mu_elo = np.mean(list(elo_fin_t1.values())) if len(elo_fin_t1) else 0.0
        sd_elo = np.std(list(elo_fin_t1.values())) if len(elo_fin_t1) else 1.0
        mu_p   = prior_scores["prior_z"].mean() if len(prior_scores) else 0.0
        sd_p   = prior_scores["prior_z"].std() if len(prior_scores) else 1.0
        sd_p   = sd_p if sd_p != 0 else 1.0

        def proj(z):  # projection linéaire z -> échelle Elo
            return mu_elo + ( (z - mu_p) * (sd_elo / sd_p) )

        prior_scores["prior_like_elo"] = prior_scores["prior_z"].apply(proj)
        prior_map = dict(zip(prior_scores["Club"], prior_scores["prior_like_elo"]))
        bottom_prior = np.percentile(list(prior_map.values()), 8) if len(prior_map) else (mu_elo - sd_elo)

        # Shrink léger vers prior club-spécifique
        elo_pre_t = {}
        for club in clubs_t:
            last  = elo_fin_t1.get(club, alpha*(M-M) + beta*diff_prom)
            prior = prior_map.get(club, bottom_prior)
            elo_pre_t[club] = (1.0 - lam) * last + lam * prior

        info = {"promoted": promoted, "relegated": relegated, "mu_bot": mu_bot, "sigma_bot": sigma_bot}
        return elo_pre_t, info

    # -----------------------------
    # Evaluation saison t
    # -----------------------------
    def evaluate_season(year_t, params, classement_df, matches_df, update_mode=False):
        H  = params.get("H", 60.0)
        s  = params.get("s", 400.0)          # <-- AJOUT
        nu = max(1e-4, params.get("nu", 0.02))
        K  = params.get("K", 20.0)
        margin_scale = params.get("margin_scale", 2.2)

        elos, info = build_preseason_elos_for_year(year_t, params, classement_df)
        m = matches_df[matches_df["season"] == year_t].copy()
        if m.empty:
            return None, info

        m = m.sort_values(by=["home","away"]).reset_index(drop=True)

        ll_sum, n = 0.0, 0
        for _, row in m.iterrows():
            home, away, hs, a_s = row["home"], row["away"], int(row["hs"]), int(row["as"])
            if (home not in elos) or (away not in elos):
                continue
            pH, pD, pA = davidson_probs(elos[home], elos[away], H=H, nu=nu, s=s)  # <-- s passé ici
            label = match_label(hs, a_s)
            ll_sum += logloss_3way(pH, pD, pA, label)
            n += 1

            if update_mode:
                s_home = 1.0 if label=='H' else (0.0 if label=='A' else 0.5)
                e_home = pH + 0.5*pD
                delta  = s_home - e_home
                g = margin_factor((elos[home]+H) - elos[away], abs(hs - a_s), scale=margin_scale)
                elos[home] += K * g * delta
                elos[away] -= K * g * delta

        if n == 0:
            return None, info
        return ll_sum / n, info


    # -----------------------------
    # LOSO & Random search
    # -----------------------------
    SEASONS_ALL = sorted(set(classement_df["année"].unique()).intersection(set(matches_df["season"].unique())))
    # On commence à t=année quand t-1 et t-2 existent
    SEASONS = [t for t in SEASONS_ALL if (t-1 in SEASONS_ALL and t-2 in SEASONS_ALL)]

    print("\nPromotion / relégation détectées par saison:")
    for t in SEASONS:
        _, info_t = build_preseason_elos_for_year(t, {
            "alpha":15.0, "beta":0.10, "lam":0.20, "w_recent":0.7, "promo_penalty":0.25, "bottom_k":3
        }, classement_df)
        print(f"{t}: promus={info_t['promoted']}, relegues={info_t['relegated']}")

    def loso_logloss(params, update_mode=False):
        scores = []
        for t in SEASONS:
            ll, _ = evaluate_season(t, params, classement_df, matches_df, update_mode=update_mode)
            if ll is not None and np.isfinite(ll):
                scores.append(ll)
        return np.mean(scores) if scores else None

    # Plages de recherche
    RANGES = {
        "H": (40.0, 70.0),
        "s": (240.0, 800.0),      # <-- AJOUT : pente de l'échelle Elo
        "nu": (0.005, 0.05),
        "K": (14.0, 28.0),
        "margin_scale": (1.8, 2.6),
        "alpha": (10.0, 25.0),
        "beta": (0.05, 0.15),
        "lam": (0.15, 0.30),
        "w_recent": (0.5, 0.8),
        "promo_penalty": (0.0, 0.5),
        "bottom_k": 3,
    }

    def sample_params():
        p = {}
        for k, v in RANGES.items():
            if isinstance(v, tuple):
                p[k] = float(np.random.uniform(*v))
            else:
                p[k] = v
        return p

    # --- Recherche STATIC (rapide) ---
    start = time()
    best_static = (float("inf"), None)
    for _ in range(TRIALS_STATIC):
        p = sample_params()
        score = loso_logloss(p, update_mode=False)
        if score is None:
            continue
        if score < best_static[0]:
            best_static = (score, p)
    dur_static = time() - start

    print("\n=== Best (STATIC pre-season Elo only) ===")
    print("Log-loss:", round(best_static[0], 5))
    print("Params :", {k: (round(v,3) if isinstance(v,float) else v) for k,v in (best_static[1] or {}).items()})
    print(f"Trials: {TRIALS_STATIC} | Time: {dur_static:.1f}s")

    # --- Recherche SEQUENTIAL (optionnelle) ---
    start = time()
    best_seq = (float("inf"), None)
    for _ in range(TRIALS_SEQUENTIAL):
        p = sample_params()
        score = loso_logloss(p, update_mode=True)  # updates intra-saison (ordre déterministe)
        if score is None:
            continue
        if score < best_seq[0]:
            best_seq = (score, p)
    dur_seq = time() - start

    print("\n=== Best (SEQUENTIAL updates — deterministic order) ===")
    print("Log-loss:", round(best_seq[0], 5))
    print("Params :", {k: (round(v,3) if isinstance(v,float) else v) for k,v in (best_seq[1] or {}).items()})
    print(f"Trials: {TRIALS_SEQUENTIAL} | Time: {dur_seq:.1f}s")

def elo_preseason_with_mercato(
    classement: pd.DataFrame,
    forme: pd.DataFrame,
    resultat: pd.DataFrame | None = None,   # fournir pour calculer la table attendue
    # --- saison de référence (terminée) ---
    base_year: int = 2023,                  # ex: 2023 => on prédit la saison 2024 (24/25)
    season_for_expected: int | None = None, # par défaut: base_year + 1
    # --- pondération des composantes du score saison ---
    rank_scale: float = 60.0,
    diff_scale: float = 22.0,
    form_scale: float = 10.0,        # poids modéré sur la forme
    use_form_in_prev: bool = False,  # pas de forme sur base_year-1 par défaut
    # --- pré-saison: mélange + régression vers moyenne ---
    w_last: float = 0.75,            # poids base_year
    w_prev: float = 0.25,            # poids base_year-1
    rho: float = 0.80,               # régression vers la moyenne
    mu: float = 1500.0,              # offset global (moyenne ligue)
    # --- promus / relégués (manuels) ---
    promo_penalty: float = 0.451,
    bottom_k: int = 3,
    delta_niveau: float = -35.0,     # handicap division appliqué aux promus
    promu: str = "RC Vannes",
    relegue: str = "US Oyonnax",
    m_equipes: int = 14,
    # --- mercato ---
    gamma: float = 10.0,
    notes_mercato: dict | None = None,
    # --- shrink par étages ---
    apply_tier_compress: bool = True,
    tau_top_1_3: float = 0.90,
    tau_4_7: float   = 0.75,
    tau_8_13: float  = 0.60,
    tau_14: float    = 0.95,
    # --- proba match (pour table attendue) ---
    H: float = 69.0,                 # avantage domicile (Elo)
    s: float = 400.0,                # pente de l'échelle Elo
    nu: float = 0.035,               # paramètre de nul (Davidson)
    compute_expected_table: bool = True,
    do_print: bool = True
):
    """
    Construit des ratings de pré-saison pour base_year+1 à partir de base_year et base_year-1 :
      - latent par saison = rank/diff (+ forme pour base_year)
      - mélange w_last/w_prev + régression vers la moyenne
      - ajustement promu (delta_niveau)
      - ajustement mercato (gamma*(note-5))
      - shrink par étages (1–3, 4–7, 8–13, 14)
      - (optionnel) table attendue via Davidson (H, s, nu)

    Si la saison season_for_expected est absente de `resultat`, un calendrier aller/retour
    est généré à partir des clubs de base_year en remplaçant `relegue` par `promu`.

    Retourne: elo_with_notes, elo_preseason_df, classement_corrige_t1, table_attendue (ou None)
    """

    # saison prédite par défaut
    if season_for_expected is None:
        season_for_expected = base_year + 1

    # --- harmonisation noms de colonnes 'année' ---
    for df in (classement, forme, resultat) if resultat is not None else (classement, forme):
        if df is None:
            continue
        if "année" not in df.columns and "annee" in df.columns:
            df.rename(columns={"annee": "année"}, inplace=True)

    # --- helpers ---
    def _safe_std(x, fallback=1.0):
        v = float(np.std(x)) if len(x) else fallback
        return v if v != 0 and not np.isnan(v) else fallback

    def _safe_mean(x, fallback=0.0):
        return float(np.mean(x)) if len(x) else fallback

    def _davidson_probs(elo_home, elo_away, H, nu, s):
        A = 10 ** ((elo_home + H) / s)
        B = 10 ** (elo_away / s)
        D = A + B + 2.0 * nu * (A * B) ** 0.5
        pH = A / D
        pA = B / D
        pD = 2.0 * nu * (A * B) ** 0.5 / D
        ssum = pH + pD + pA
        return pH / ssum, pD / ssum, pA / ssum

    def _extract_matches_smart(resultat_df: pd.DataFrame, classement_df: pd.DataFrame):
        """
        Reconstruit la liste des matchs à partir d'un format matrice 'résultats' OU 'calendrier'.
        - Gère les cellules type '12-9' (score) OU 'J05' (calendrier) OU '—' (pas de match).
        - Retour: DataFrame [season, home, away, hs, as] ; hs/as peuvent être None si calendrier.
        """
        SCORE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

        def season_clubs(df, year):
            return set(df[df["année"] == year]["Club"].dropna().unique())

        all_rows = []

        if "année" not in resultat_df.columns:
            raise ValueError("Column 'année' manquante dans resultat.")

        for y in sorted(resultat_df["année"].dropna().unique()):
            block = resultat_df[resultat_df["année"] == y].copy()
            clubs_y = season_clubs(classement_df, y)
            if not clubs_y or block.empty:
                continue

            # détecter la colonne des clubs (1ère colonne non 'année' la plus plausible)
            cols = list(block.columns)
            candidates = [c for c in cols if str(c).lower() in {"club", "équipe", "equipes/journées", "resultats", "résultats"}]
            if candidates:
                club_col = candidates[0]
            else:
                non_year_cols = [c for c in cols if c != "année"]
                club_col = non_year_cols[0] if non_year_cols else cols[0]

            opp_cols = [c for c in cols if c not in (club_col, "année")]

            for _, row in block.iterrows():
                home = str(row[club_col]).strip()
                if home not in clubs_y:
                    continue
                for oc in opp_cols:
                    away = str(oc).strip()
                    if away not in clubs_y or away == home:
                        continue
                    cell = row[oc]
                    if pd.isna(cell):
                        continue
                    s_cell = str(cell).strip()
                    if s_cell in {"", "—", "-", "–"}:
                        continue

                    # Essayer de parser un éventuel score "x-y"
                    m = SCORE_RE.match(s_cell)
                    if m:
                        hs, a_s = int(m.group(1)), int(m.group(2))
                    else:
                        hs, a_s = None, None  # calendrier type "J05"

                    all_rows.append({"season": int(y), "home": home, "away": away, "hs": hs, "as": a_s})

        return pd.DataFrame(all_rows)

    def _generate_calendar_from_previous_year(clubs_prev: list[str], promu: str, relegue: str,
                                              season: int, double: bool = True) -> pd.DataFrame:
        """
        Crée un calendrier à partir des clubs de l'année précédente, en remplaçant le relégué par le promu.
        - double=True => aller/retour (n-1 matchs à domicile + n-1 à l'extérieur par équipe)
        Retourne: DataFrame [season, home, away]
        """
        clubs = [promu if c == relegue else c for c in clubs_prev]
        if promu not in clubs:
            clubs.append(promu)
        # dédoublonne en conservant l'ordre
        seen = set()
        clubs = [c for c in clubs if not (c in seen or seen.add(c))]

        rows = []
        n = len(clubs)
        if double:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    rows.append({"season": season, "home": clubs[i], "away": clubs[j]})
        else:
            for i in range(n):
                for j in range(i+1, n):
                    rows.append({"season": season, "home": clubs[i], "away": clubs[j]})
        return pd.DataFrame(rows)

    # --- 1) Liste clubs base_year (sans hard-code) ---
    c_t1 = classement.loc[classement["année"] == base_year, ["Club", "Rang", "Diff"]].dropna(subset=["Club"]).copy()
    if c_t1.empty:
        raise ValueError(f"Aucune ligne pour 'année' == {base_year} dans 'classement'.")

    # Clubs base_year → liste de la saison suivante (remplacer relégué par promu)
    clubs_t1_ordered = [x for _, x in sorted(zip(c_t1["Rang"].values, c_t1["Club"].values))]
    clubs_next = [promu if x == relegue else x for x in clubs_t1_ordered]
    if promu not in clubs_next:
        clubs_next.append(promu)

    classement_corrige_t1 = pd.DataFrame({
        "Rang_final": list(range(1, len(clubs_next) + 1)),
        "Club": clubs_next
    })

    # --- 2) Pseudo-diff promu (bas de tableau base_year) ---
    bot = c_t1.sort_values("Rang").tail(min(bottom_k, len(c_t1)))
    mu_bot = float(bot["Diff"].mean()) if len(bot) else 0.0
    sd_bot = float(bot["Diff"].std() or 0.0)
    pseudo_diff_promu = mu_bot - promo_penalty * (0.0 if np.isnan(sd_bot) else sd_bot)

    # --- 3) Forme J22..J26 base_year (score 4/2/0) ---
    f_t1 = forme.loc[forme["année"] == base_year].set_index("Club").copy()
    last_cols = ["J22", "J23", "J24", "J25", "J26"]

    def _map_outcome(x):
        if not isinstance(x, str):
            return 0
        t = x.strip().upper()
        if t.startswith("V"): return 4
        if t.startswith("N"): return 2
        if t.startswith("D"): return 0
        try:
            return int(x)
        except:
            return 0

    form_scores = {}
    for club in clubs_next:
        if club in f_t1.index and all(col in f_t1.columns for col in last_cols):
            vals = [_map_outcome(f_t1.loc[club, col]) for col in last_cols]
            form_scores[club] = float(np.nansum(vals))
        else:
            form_scores[club] = np.nan

    known_forms = [v for v in form_scores.values() if not np.isnan(v)]
    muF = _safe_mean(known_forms, fallback=10.0)
    sdF = _safe_std(known_forms, fallback=1.0)

    # --- 4) Score latent par saison (base_year & base_year-1) ---
    def season_latent(df_season, year, include_form: bool):
        dfy = df_season.loc[df_season["année"] == year, ["Club", "Rang", "Diff"]].copy()
        if dfy.empty:
            return {}
        M = m_equipes
        denom_rank = max(1, (M - 1))
        muD = float(dfy["Diff"].mean()); sdD = float(dfy["Diff"].std() or 1.0)
        lat = {}
        for _, row in dfy.iterrows():
            club = row["Club"]; rang = int(row["Rang"])
            rank_norm = (M - rang) / denom_rank
            diff = float(row["Diff"])
            diff_z = (diff - muD) / (sdD if sdD != 0 else 1.0)
            part_form = 0.0
            if include_form:
                fscore = form_scores.get(club, np.nan)
                if np.isnan(fscore): fscore = muF
                form_z = (fscore - muF) / (sdF if sdF != 0 else 1.0)
                part_form = form_scale * form_z
            lat[club] = rank_scale * rank_norm + diff_scale * diff_z + part_form
        return lat

    lat_t1 = season_latent(classement, base_year, include_form=True)
    lat_t2 = season_latent(classement, base_year - 1, include_form=use_form_in_prev)

    # Ajouter le promu si absent en base_year
    if promu not in lat_t1:
        M = m_equipes
        rank_norm_promu = 0.0
        muD_t1 = float(c_t1["Diff"].mean() if len(c_t1) else 0.0)
        sdD_t1 = float(c_t1["Diff"].std() or 1.0)
        diff_z_promu = ((pseudo_diff_promu + delta_niveau) - muD_t1) / (sdD_t1 if sdD_t1 != 0 else 1.0)
        lat_t1[promu] = rank_scale * rank_norm_promu + diff_scale * diff_z_promu

    # --- 5) Mélange w_last/w_prev puis régression vers la moyenne ---
    clubs_all = set(clubs_next)
    default_t1 = _safe_mean(list(lat_t1.values()), fallback=0.0)
    default_t2 = _safe_mean(list(lat_t2.values()), fallback=0.0)

    r_mix = {}
    for club in clubs_all:
        r1 = lat_t1.get(club, default_t1)
        r2 = lat_t2.get(club, default_t2)
        r_mix[club] = w_last * r1 + w_prev * r2

    mu_mix = _safe_mean(list(r_mix.values()), fallback=0.0)
    elo_pre_next = {club: (mu + rho * (r - mu_mix)) for club, r in r_mix.items()}

    # --- 6) DataFrame pré-saison (avant mercato) ---
    elo_preseason_df = pd.DataFrame(sorted(elo_pre_next.items(), key=lambda x: -x[1]),
                                    columns=["Club", "Elo_pre"])

    # --- 7) Mercato ---
    if notes_mercato is None:
        notes_mercato = {club: 5.0 for club in elo_preseason_df["Club"]}  # neutre
    notes_df = pd.DataFrame(list(notes_mercato.items()), columns=["Club", "NoteMercato"])
    elo_with_notes = elo_preseason_df.merge(notes_df, on="Club", how="left")
    elo_with_notes["NoteMercato"] = elo_with_notes["NoteMercato"].fillna(5.0)
    elo_with_notes["Elo_pre_adj"] = (
        elo_with_notes["Elo_pre"] + gamma * (elo_with_notes["NoteMercato"] - 5.0)
    )

    # --- 8) Shrink par étages (1–3, 4–7, 8–13, 14) ---
    if apply_tier_compress:
        df = elo_with_notes.sort_values("Elo_pre_adj", ascending=False).reset_index(drop=True)
        df["Rang_pre"] = np.arange(1, len(df) + 1)
        df["Elo_pre_adj_raw"] = df["Elo_pre_adj"].values

        def _shrink_block(df_in, start, end, tau):
            mask = (df_in["Rang_pre"] >= start) & (df_in["Rang_pre"] <= end)
            if not mask.any():
                return df_in
            mu_g = df_in.loc[mask, "Elo_pre_adj"].mean()
            df_in.loc[mask, "Elo_pre_adj"] = mu_g + tau * (df_in.loc[mask, "Elo_pre_adj"] - mu_g)
            return df_in

        df = _shrink_block(df,  1,  3, tau_top_1_3)
        df = _shrink_block(df,  4,  7, tau_4_7)
        df = _shrink_block(df,  8, 13, tau_8_13)
        df = _shrink_block(df, 14, 14, tau_14)
        elo_with_notes = df.drop(columns=["Rang_pre"]).reset_index(drop=True)

    elo_with_notes = elo_with_notes.sort_values("Elo_pre_adj", ascending=False).reset_index(drop=True)

    # --- 9) Table attendue (Davidson) avec fallback calendrier généré ---
    table_attendue = None
    if compute_expected_table:
        ratings_final = dict(zip(elo_with_notes["Club"], elo_with_notes["Elo_pre_adj"]))

        # 1) Essayer d'extraire la saison demandée depuis resultat.csv
        m_season = pd.DataFrame()
        if resultat is not None:
            matches_df = _extract_matches_smart(resultat, classement)
            if not matches_df.empty:
                m_season = matches_df[matches_df["season"] == season_for_expected].copy()

        # 2) Fallback: pas de lignes pour la saison demandée -> on génère un calendrier A/R
        if m_season.empty:
            m_season = _generate_calendar_from_previous_year(
                clubs_prev=list(c_t1.sort_values("Rang")["Club"].values),  # clubs de base_year, ordonnés
                promu=promu,
                relegue=relegue,
                season=season_for_expected,
                double=True  # Top 14 = aller/retour
            )

        # 3) Points attendus via proba (Davidson)
        pts = {team: 0.0 for team in ratings_final.keys()}
        for _, r in m_season.iterrows():
            h, a = r["home"], r["away"]
            if h not in ratings_final or a not in ratings_final:
                continue
            pH, pD, pA = _davidson_probs(ratings_final[h], ratings_final[a], H=H, nu=nu, s=s)
            pts[h] += 3.0 * pH + 1.0 * pD
            pts[a] += 3.0 * pA + 1.0 * pD

        table_attendue = pd.DataFrame([{"Club": k, "Pts_attendus": v} for k, v in pts.items()]) \
                            .sort_values("Pts_attendus", ascending=False).reset_index(drop=True)
        table_attendue["Rang_attendu"] = np.arange(1, len(table_attendue) + 1)

    if do_print:
        print(f"\n=== Elo pré-saison (base_year={base_year} -> saison attendue {season_for_expected}) ===")
        cols = ["Club", "Elo_pre", "NoteMercato"]
        if "Elo_pre_adj_raw" in elo_with_notes.columns:
            cols += ["Elo_pre_adj_raw"]
        cols += ["Elo_pre_adj"]
        print(elo_with_notes[cols])
        if table_attendue is not None:
            src = "résultats.csv" if (resultat is not None and not m_season.empty) else "calendrier généré"
            print(f"\n=== Classement attendu (points attendus) — calendrier utilisé: {src} ===")
            print(table_attendue[["Rang_attendu", "Club", "Pts_attendus"]])

    # Nettoyage de la colonne raw si présente
    if "Elo_pre_adj_raw" in elo_with_notes.columns:
        elo_with_notes = elo_with_notes.drop(columns=["Elo_pre_adj_raw"])

    return elo_with_notes, elo_preseason_df, classement_corrige_t1, table_attendue

def extract_fixtures_from_resultat(resultat_df, classement_df, season, use_scores=True):
    """
    Construit la liste des matches (home, away) pour 'season' à partir du format matrice.
    - use_scores=True : ne retient que les cellules qui ressemblent à 'xx-yy'
    - use_scores=False : crée les paires home/away pour toutes les colonnes mappées, même sans score
    Retour: DataFrame avec colonnes ['home','away'] (doublons supprimés).
    """
    # harmonise 'année'
    if "année" not in resultat_df.columns and "annee" in resultat_df.columns:
        resultat_df = resultat_df.rename(columns={"annee":"année"})
    if "année" not in classement_df.columns and "annee" in classement_df.columns:
        classement_df = classement_df.rename(columns={"annee":"année"})

    SCORE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

    def _parse_score(cell):
        if not isinstance(cell, str):
            return None
        m = SCORE_RE.match(cell)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def _season_clubs(df, year):
        col_year = "année" if "année" in df.columns else ("annee" if "annee" in df.columns else None)
        if col_year is None:
            raise ValueError("Colonne 'année'/'annee' absente dans 'classement'.")
        return set(df[df[col_year] == year]["Club"].dropna().unique())

    def _build_opponent_mapping_for_season(block, clubs_y):
        """
        Essaie d'abord de mapper les colonnes directement par leur nom (si ce sont des clubs),
        sinon, utilise la valeur la plus fréquente dans la colonne correspondant à un club connu.
        """
        cols = list(block.columns)
        club_col = next((c for c in ["Club","club","CLUB"] if c in cols), None)
        year_col = next((c for c in ["année","annee","year"] if c in cols), None)
        if club_col is None or year_col is None:
            return {}
        opp_cols = [c for c in cols if c not in (club_col, year_col)]
        mapping = {}
        for oc in opp_cols:
            # 1) si l'en-tête est déjà un nom de club de la saison, prends-le
            if oc in clubs_y:
                mapping[oc] = oc
                continue
            # 2) sinon, devine par la valeur la plus fréquente qui est un club
            vals = block[oc].dropna().astype(str)
            candidates = [v for v in vals if v in clubs_y]
            mapping[oc] = Counter(candidates).most_common(1)[0][0] if candidates else None
        return mapping

    block = resultat_df[resultat_df["année"] == season].copy()
    if block.empty:
        raise ValueError(f"Aucun bloc 'resultat' pour l'année {season}.")
    clubs_y = _season_clubs(classement_df, season)
    cols = list(block.columns)
    club_col = next((c for c in ["Club","club","CLUB"] if c in cols), None)
    if club_col is None:
        raise ValueError("Colonne 'Club' absente dans 'resultat'.")

    opp_map = _build_opponent_mapping_for_season(block, clubs_y)

    matches = []
    for _, row in block.iterrows():
        home = row[club_col]
        if home not in clubs_y:
            continue
        for oc, away in opp_map.items():
            if away is None or away == home:
                continue
            if use_scores:
                sc = _parse_score(row[oc])
                if sc is None:
                    continue
            matches.append((home, away))

    fixtures = pd.DataFrame(matches, columns=["home","away"]).drop_duplicates().reset_index(drop=True)
    return fixtures

def print_probas_journee(
    elo_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    journee: int,
    H: float = 69.935,   # avantage domicile
    s: float = 400.0,    # pente échelle Elo
    nu: float = 0.035,   # paramètre de nul (Davidson)
    journee_col: str = "Journée",
    home_col: str = "home",
    away_col: str = "away",
    include_draw: bool = False         # True -> affiche aussi P(nul)
) -> None:
    """
    Imprime les probabilités de victoire pour chaque match d'une journée donnée,
    en utilisant l'Elo **de la journée précédente** s'il est disponible.
    """
    # --- choisir la bonne colonne de rating ---
    preferred_cols = [
        f"Elo_J{journee-1}",           # ex: J=2 -> 'Elo_J1'
        f"Elo_J({journee-1})",         # ex: J=1 -> 'Elo_J(0)'
        "Elo_pre_adj",
        "Elo_pre",
    ]
    rating_col = next((c for c in preferred_cols if c in elo_df.columns), None)
    if rating_col is None:
        raise ValueError(
            f"Aucune colonne Elo trouvée parmi {preferred_cols}. "
            f"Colonnes dispo: {list(elo_df.columns)}"
        )

    ratings = dict(zip(elo_df["Club"], elo_df[rating_col]))

    # --- filtre journée ---
    def day_num(x: str) -> int | None:
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None

    day_games = calendar_df.loc[calendar_df[journee_col].map(day_num) == journee, [journee_col, home_col, away_col]]
    if day_games.empty:
        print(f"Aucun match trouvé pour la journée {journee}.")
        return

    # --- Davidson (pH, pD, pA) ---
    def davidson_probs(elo_home: float, elo_away: float) -> tuple[float, float, float]:
        A = 10 ** ((elo_home + H) / s)
        B = 10 ** (elo_away / s)
        D = A + B + 2.0 * nu * (A * B) ** 0.5
        pH = A / D
        pA = B / D
        pD = 2.0 * nu * (A * B) ** 0.5 / D
        ssum = pH + pD + pA
        return pH / ssum, pD / ssum, pA / ssum

    # --- impression ---
    unknown = set()
    for _, row in day_games.iterrows():
        h, a = row[home_col], row[away_col]
        if h not in ratings: unknown.add(h)
        if a not in ratings: unknown.add(a)
        if h not in ratings or a not in ratings:
            continue
        pH, pD, pA = davidson_probs(ratings[h], ratings[a])
        if include_draw:
            print(f"{h} ({pH*100:.1f}%) vs {a} ({pA*100:.1f}%)  |  nul {pD*100:.1f}%")
        else:
            print(f"{h} ({pH*100:.1f}%) vs {a} ({pA*100:.1f}%)")

    if unknown:
        print("\n[warning] clubs sans rating:", ", ".join(sorted(unknown)))

def add_scores_for_journee(cal_df: pd.DataFrame,
                           journee: int,
                           scores_list: list[str],
                           journee_col: str = "Journée",
                           score_col: str = "score",
                           allow_partial: bool = False,
                           overwrite: bool = True,
                           inplace: bool = True) -> pd.DataFrame | None:
    """
    Ajoute / met à jour les scores pour la journée `journee` SANS toucher aux autres journées.
    - Crée la colonne `score` si elle n'existe pas.
    - Remplit les lignes de la journée ciblée dans l'ordre courant du DataFrame.
    - `allow_partial=True` autorise moins de scores que de matchs (le reste reste NA).
    - `overwrite=False` n'écrase pas un score déjà présent dans la journée.
    - `inplace=True` modifie cal_df et ne retourne rien ; sinon retourne une copie mise à jour.
    """
    df = cal_df if inplace else cal_df.copy()

    # 1) colonne score
    if score_col not in df.columns:
        df[score_col] = pd.NA

    # 2) filtrer la journée
    def day_num(x: str) -> int | None:
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None

    mask = df[journee_col].map(day_num) == journee
    idx = df.index[mask]
    n_games = len(idx)
    if n_games == 0:
        raise ValueError(f"Aucun match trouvé pour la journée {journee}.")

    # 3) check longueur
    if not allow_partial and len(scores_list) != n_games:
        raise ValueError(
            f"Incohérence: {len(scores_list)} scores fournis pour {n_games} matchs (journée {journee}). "
            f"Passe allow_partial=True pour autoriser une liste partielle."
        )

    # 4) normaliser 'xx-xx' -> 'NN-NN'
    def norm_score(s: str) -> str:
        m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(s))
        if not m:
            raise ValueError(f"Format invalide: {s!r} (attendu 'xx-xx').")
        h, a = int(m.group(1)), int(m.group(2))
        if not (0 <= h <= 99 and 0 <= a <= 99):
            raise ValueError(f"Valeurs hors borne (0..99): {s!r}")
        return f"{h:02d}-{a:02d}"

    k = min(len(scores_list), n_games)
    if k > 0:
        normalized = [norm_score(s) for s in scores_list[:k]]
        fill_idx = list(idx[:k])

        if not overwrite:
            # ne remplis que les cases NA
            fill_idx = [i for i in fill_idx if pd.isna(df.at[i, score_col])]
            normalized = normalized[:len(fill_idx)]

        df.loc[fill_idx, score_col] = normalized

    return None if inplace else df


def update_elo_after_journee(
    elo_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    J: int,
    club_col: str = "Club",
    journee_col: str = "Journée",
    home_col: str = "home",
    away_col: str = "away",
    score_col: str = "score",
    # hyperparamètres (tes meilleurs “sequential”)
    H: float = 63.837,
    s: float = 259.415,
    nu: float = 0.031,
    K_factor: float = 27.936,
    margin_scale: float = 2.485,      # <— au lieu de margin_scale_cap
    allow_partial: bool = True
) -> pd.DataFrame:
    """
    Met à jour l’Elo d’entrée de journée avec les scores de la journée J.
    Retourne un tableau trié par Elo après J: [Club, Elo_J(J-1), Elo_JJ, Delta].

    Détection automatique de la colonne de départ:
      - 'Elo_J{J-1}' si disponible (ex: J=2 -> 'Elo_J1')
      - sinon 'Elo_J({J-1})' (ex: J=1 -> 'Elo_J(0)')
      - sinon 'Elo_pre_adj'
      - sinon 'Elo_pre'
    """

    # 0) choisir la colonne de rating de départ selon J
    candidates = [f"Elo_J{J-1}", f"Elo_J({J-1})", "Elo_pre_adj", "Elo_pre"]
    start_col = next((c for c in candidates if c in elo_df.columns), None)
    if start_col is None:
        raise ValueError(
            f"Aucune colonne Elo trouvée pour démarrer J={J}. "
            f"Requiert l'une de: {candidates}. Colonnes dispo: {list(elo_df.columns)}"
        )

    ratings_in = dict(zip(elo_df[club_col], elo_df[start_col]))

    # helpers
    def day_num(x: str) -> int | None:
        m = re.search(r"(\d+)", str(x))
        return int(m.group(1)) if m else None

    def parse_score(sc: str):
        m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", str(sc))
        return (int(m.group(1)), int(m.group(2))) if m else None

    def davidson_probs(elo_home: float, elo_away: float):
        A = 10 ** ((elo_home + H) / s)
        B = 10 ** (elo_away / s)
        D = A + B + 2.0 * nu * (A * B) ** 0.5
        pH = A / D; pA = B / D
        pD = 2.0 * nu * (A * B) ** 0.5 / D
        ssum = pH + pD + pA
        return pH / ssum, pD / ssum, pA / ssum

    def g_margin(delta_elo: float, margin_points: int) -> float:
        # g = max(1, ln(m+1)) * (scale / (0.001*|ΔE| + scale))
        base = max(1.0, math.log(margin_points + 1.0))
        return base * (margin_scale / (0.001 * abs(delta_elo) + margin_scale))

    # 1) matchs de la journée J avec score
    block = calendar_df[calendar_df[journee_col].map(day_num) == J]
    if block.empty:
        raise ValueError(f"Aucun match trouvé pour la journée {J}.")
    if not allow_partial and block[score_col].isna().any():
        raise ValueError(f"Des scores manquent pour la journée {J} (allow_partial=False).")
    block = block.dropna(subset=[score_col])
    if block.empty:
        raise ValueError(f"Aucun score exploitable pour la journée {J}.")

    # 2) calcul des deltas avec ratings gelés (pas de biais d’ordre)
    delta_by_team = {team: 0.0 for team in ratings_in.keys()}
    unknown_teams = set()

    for _, r in block.iterrows():
        h, a = str(r[home_col]), str(r[away_col])
        sc = str(r[score_col])
        if h not in ratings_in or a not in ratings_in:
            unknown_teams.update([t for t in [h, a] if t not in ratings_in])
            continue
        parsed = parse_score(sc)
        if parsed is None:
            continue

        hs, as_ = parsed
        pH, pD, pA = davidson_probs(ratings_in[h], ratings_in[a])
        E_h = pH + 0.5 * pD
        S_h = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)

        m_pts = abs(hs - as_)
        deltaE = (ratings_in[h] + H) - ratings_in[a]
        g = g_margin(deltaE, m_pts)

        delta = K_factor * g * (S_h - E_h)
        delta_by_team[h] += delta
        delta_by_team[a] -= delta

    # 3) appliquer en bloc
    rows = []
    for team in ratings_in.keys():
        before = ratings_in[team]
        after = before + delta_by_team.get(team, 0.0)
        rows.append((team, before, after, after - before))

    out = (
        pd.DataFrame(rows, columns=[club_col, f"Elo_J({J-1})", f"Elo_J{J}", "Delta"])
        .sort_values(f"Elo_J{J}", ascending=False)
        .reset_index(drop=True)
    )

    if unknown_teams:
        print("[warning] clubs sans rating ignorés: " + ", ".join(sorted(unknown_teams)))

    return out

