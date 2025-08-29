import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss

def round_sig(x, sig=2):
    """Arrondi à 2 chiffres significatifs."""
    if x == 0 or np.isnan(x):
        return 0
    return float(f"{x:.{sig}g}")

def fit_predict_logit(features: pd.DataFrame, df_champions: pd.DataFrame):
    """
    Régression logistique + LOSO, avec résultats enrichis :
      - predicted, actual, correct
      - predicted_proba, actual_proba
      - second_best, second_proba
    """

    # Harmoniser colonnes année/annee
    if "annee" in features.columns and "année" in df_champions.columns:
        df_champions = df_champions.rename(columns={"année": "annee"})
    elif "année" in features.columns and "annee" in df_champions.columns:
        df_champions = df_champions.rename(columns={"annee": "année"})

    year_col = "annee" if "annee" in features.columns else "année"

    # --- préparation données ---
    df = features.merge(df_champions, on=year_col, how="left")
    df["is_champion"] = (df["Club"] == df["champion"]).astype(int)

    # colonnes possibles
    feat_cols_all = ["Pts_adj_26","Diff_adj_26","rank_J26",
                     "win_streak_max","form_last5_sum"]
    feat_cols = [c for c in feat_cols_all if c in df.columns]

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    y = df["is_champion"]
    clubs = df["Club"]
    seasons = df[year_col]

    results = []
    all_probs = []

    # --- Leave-One-Season-Out ---
    for season in sorted(seasons.unique()):
        train_mask = seasons != season
        test_mask = seasons == season

        if df.loc[test_mask, "is_champion"].sum() == 0:
            continue  # saison sans champion (ex: 2019 None)

        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_test = X.loc[test_mask]
        clubs_test = clubs.loc[test_mask]

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500))
        ])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:,1]  # proba champion
        df_probs = pd.DataFrame({
            year_col: season,
            "Club": clubs_test.values,
            "proba": proba
        })
        all_probs.append(df_probs)

        # --- analyse des probabilités ---
        order = np.argsort(proba)[::-1]  # indices triés décroissants
        best_idx, second_idx = order[0], order[1]

        predicted = clubs_test.iloc[best_idx]
        predicted_proba = proba[best_idx]

        second_best = clubs_test.iloc[second_idx]
        second_proba = proba[second_idx]

        actual = df.loc[test_mask & (df["is_champion"]==1), "Club"].values[0]
        actual_proba = proba[clubs_test.tolist().index(actual)]

        results.append({
            year_col: season,
            "predicted": predicted,
            "predicted_proba": round_sig(predicted_proba),
            "second_best": second_best,
            "second_proba": round_sig(second_proba),
            "actual": actual,
            "actual_proba": round_sig(actual_proba),
            "correct": int(predicted == actual)
        })

    results = pd.DataFrame(results)
    probs = pd.concat(all_probs, ignore_index=True)
    probs["proba"] = probs["proba"].apply(round_sig)

    return results, probs

def evaluate_results(results: pd.DataFrame, probs: pd.DataFrame) -> pd.DataFrame:
    """
    Évalue les performances du modèle à partir des résultats LOSO.

    results : DataFrame
        Colonnes attendues :
        ['année','predicted','predicted_proba','second_best','second_proba',
         'actual','actual_proba','correct']
    probs : DataFrame
        Colonnes attendues :
        ['année','Club','proba']

    Retourne :
        Un DataFrame résumé avec accuracy, top2_accuracy et brier_score.
    """

    # --- Accuracy ---
    accuracy = results["correct"].mean()

    # --- Top-2 accuracy ---
    top2_correct = (
        (results["predicted"] == results["actual"]) |
        (results["second_best"] == results["actual"])
    ).mean()

    # --- Brier score ---
    # proba attribuée au vrai champion chaque année
    brier_vals = []
    for _, row in results.iterrows():
        y = 1
        p = row["actual_proba"]
        brier_vals.append((p - y) ** 2)
    brier_score = np.mean(brier_vals)

    summary = pd.DataFrame({
        "metric": ["Accuracy", "Top-2 Accuracy", "Brier Score"],
        "value": [accuracy, top2_correct, brier_score]
    })
    return summary

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


def run_random_forest_simple(
    features: pd.DataFrame,
    df_champions: pd.DataFrame,
    calibrate: bool = False,
    blend_prior: bool = True,
    prior_kind: str = "rank",
    prior_temp: float = 0.8,
    alpha: float = 0.30,
    random_state: int = 42,
    # ↓↓↓ hyperparams RF que l'on veut tuner ↓↓↓
    n_estimators: int = 1200,
    max_depth=None,
    max_features="sqrt",
    min_samples_leaf: int = 2,
    class_weight="balanced",
    criterion: str = "gini",
):
    # label binaire champion vs non
    data = features.merge(df_champions[["année","champion"]], on="année", how="left")
    data = data[~data["champion"].isna()].copy()  # exclut 2019
    data["y"] = (data["Club"] == data["champion"]).astype(int)

    feat_cols = [
        "Pts_adj_26","Diff_adj_26","rank_J26",
        "form_last5_sum","win_streak_max",
        "pts_marques","pts_encaisses",
        "%_victoiretop6","%_victoiredom","%_victoireext",
        "pts10premvs10fin",
        "championAnDernier","champion3dernieresSaisons","viceChampion3dernieresSaisons",
    ]
    feat_cols = [c for c in feat_cols if c in data.columns]

    X_full = data[feat_cols].fillna(data[feat_cols].median(numeric_only=True)).values
    y_full = data["y"].values
    years  = data["année"].values
    clubs  = data["Club"].values

    def season_prior(df_season: pd.DataFrame) -> np.ndarray:
        if prior_kind == "rank":
            x = -prior_temp * df_season["rank_J26"].to_numpy(dtype=float)
        else:  # "points"
            x =  prior_temp * df_season["Pts_adj_26"].to_numpy(dtype=float)
        x = x - x.max()
        p = np.exp(x); p = p / p.sum()
        return p

    all_years = sorted(np.unique(years))
    out_rows, per_year_summary = [], []

    for y_test in all_years:
        te_mask = (years == y_test)
        tr_mask = ~te_mask

        X_tr, y_tr = X_full[tr_mask], y_full[tr_mask]
        X_te, y_te = X_full[te_mask], y_full[te_mask]
        clubs_te   = clubs[te_mask]
        df_te      = data.loc[te_mask, ["année","Club","rank_J26","Pts_adj_26"]]

        base_rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            criterion=criterion,
            n_jobs=-1,
            random_state=random_state,
        )

        if calibrate:
            clf = CalibratedClassifierCV(base_rf, method="sigmoid", cv=5)
        else:
            clf = base_rf

        clf.fit(X_tr, y_tr)
        proba_te = clf.predict_proba(X_te)[:, 1]

        # Renormalisation intra-saison
        eps = 1e-9
        proba_te = np.clip(proba_te, eps, 1 - eps)
        p_norm = proba_te / proba_te.sum()

        # Mélange prior
        if blend_prior:
            p0 = season_prior(df_te.reset_index(drop=True))
            p_mix = (1.0 - alpha) * p_norm + alpha * p0
            p_norm = p_mix / p_mix.sum()

        # stock proba par club
        for c, p, yt in zip(clubs_te, p_norm, y_te):
            out_rows.append({"année": y_test, "Club": c, "y_true": int(yt), "proba_norm": float(p)})

        # top-1 / top-2
        order = np.argsort(-p_norm)
        top1 = clubs_te[order[0]]
        top2 = clubs_te[order[1]] if len(order) > 1 else None
        per_year_summary.append({"année": y_test, "predicted": top1, "second_best": top2})

    pred_df = pd.DataFrame(out_rows)
    year_pred = pd.DataFrame(per_year_summary)
    real = data.loc[data["y"]==1, ["année","Club"]].drop_duplicates().rename(columns={"Club":"actual"})
    year_pred = year_pred.merge(real, on="année", how="left")

    top1 = (year_pred["predicted"] == year_pred["actual"]).mean()

    def is_top2(grp):
        g = grp.sort_values("proba_norm", ascending=False)
        real_c = grp.loc[grp["y_true"]==1, "Club"]
        if real_c.empty: return np.nan
        return float(real_c.iloc[0] in g["Club"].head(2).tolist())
    top2 = pred_df.groupby("année").apply(is_top2).dropna().mean()

    # Brier & LogLoss
    eps = 1e-12
    brier = np.mean((pred_df["proba_norm"] - pred_df["y_true"])**2)
    logloss = -np.mean(
        pred_df["y_true"] * np.log(np.clip(pred_df["proba_norm"], eps, 1-eps)) +
        (1 - pred_df["y_true"]) * np.log(np.clip(1 - pred_df["proba_norm"], eps, 1-eps))
    )

    metrics = {"top1": top1, "top2": top2, "brier": brier, "logloss": logloss}
    return pred_df, year_pred, metrics


from xgboost import XGBClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def run_xgb_loso_simple(
    df_features: pd.DataFrame,
    df_champions: pd.DataFrame,
    target: str = "is_champion",
    n_estimators: int = 400,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    subsample: float = 0.9,
    colsample_bytree: float = 0.9,
    reg_lambda: float = 1.0,
    random_state: int = 42,
):
    # Merge labels
    df = df_features.merge(
        df_champions[["année", "champion", "vice_champion"]],
        on="année", how="left"
    )
    df["is_champion"] = (df["Club"] == df["champion"]).astype(int)
    df["is_finalist"] = ((df["Club"] == df["champion"]) | (df["Club"] == df["vice_champion"])).astype(int)

    SEASON_COL, TEAM_COL, TARGET_COL = "année", "Club", target
    y = df[TARGET_COL].astype(int)

    # Features
    drop_cols = [TARGET_COL, SEASON_COL, "champion", "vice_champion"]
    numeric_candidates, categorical_candidates = [], []
    for c in df.columns:
        if c in drop_cols: 
            continue
        if df[c].dtype.kind in "biufc":
            numeric_candidates.append(c)
        else:
            categorical_candidates.append(c)
    X = df[numeric_candidates + categorical_candidates]

    # Preprocess
    num_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", ohe),
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_candidates),
            ("cat", cat_transformer, categorical_candidates),
        ],
        remainder="drop",
    )

    # Model (mêmes defaults que ta version)
    xgb = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model = Pipeline([("prep", preprocess), ("xgb", xgb)])

    # LOSO
    seasons = sorted(df[SEASON_COL].dropna().unique())
    hits_top1, hits_top2 = [], []
    results = []

    for s in seasons:
        train_idx = df[SEASON_COL] != s
        test_idx  = df[SEASON_COL] == s

        if df.loc[test_idx, TARGET_COL].sum() != 1:
            continue

        X_train, y_train = X[train_idx], y[train_idx]
        X_test,  y_test  = X[test_idx],  y[test_idx]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        tbl = df.loc[test_idx, [SEASON_COL, TEAM_COL, TARGET_COL]].copy()
        tbl["proba"] = proba
        tbl = tbl.sort_values("proba", ascending=False).reset_index(drop=True)

        predicted       = tbl.loc[0, TEAM_COL]
        second_best     = tbl.loc[1, TEAM_COL]
        predicted_proba = float(tbl.loc[0, "proba"])
        second_proba    = float(tbl.loc[1, "proba"])

        actual       = tbl.loc[tbl[TARGET_COL] == 1, TEAM_COL].iloc[0]
        actual_proba = float(tbl.loc[tbl[TARGET_COL] == 1, "proba"].iloc[0])

        correct = int(predicted == actual)
        hits_top1.append(correct)
        hits_top2.append(int(actual in tbl.iloc[:2][TEAM_COL].tolist()))

        results.append({
            "année": int(s),
            "predicted": predicted,
            "predicted_proba": round(predicted_proba, 3),
            "second_best": second_best,
            "second_proba": round(second_proba, 3),
            "actual": actual,
            "actual_proba": round(actual_proba, 3),
            "correct": correct
        })

    year_table = pd.DataFrame(results).sort_values("année").reset_index(drop=True)
    metrics = pd.DataFrame({
        "precision_top1": [np.mean(hits_top1)],
        "precision_top2": [np.mean(hits_top2)]
    })
    return year_table, metrics