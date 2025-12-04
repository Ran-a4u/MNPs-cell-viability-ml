#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Group-level feature importance on raw data with native categorical features.

This script:
- Fits a LightGBM model under a Leave-One-Group-Out (LOGO) scheme.
- Computes group importances using three methods:
    1) Group SHAP (sum of |SHAP| values within each group).
    2) Group Drop-Column (LOGO re-training after dropping a group of features).
    3) Group Gain (sum of LightGBM "gain" feature importances within each group).
- Optionally computes conditional permutation importance (CPI) at the group level.

Outputs:
- group_importance_SHAP.csv
- group_importance_DropColumn.csv
- group_importance_gain.csv
- [optional] group_importance_CPI.csv
- group_importance_rank_correlation.csv (Spearman/Kendall between methods)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, kendalltau


# ----------------------------------------------------------------------
# Default configuration (edit here if your columns/groups change)
# ----------------------------------------------------------------------
DEFAULT_TARGET_COL = "CV"          # Regression target
DEFAULT_GROUP_LOGO_COL = "Study_ID"  # Group column for LOGO (not fed to the model)

# Feature columns used by the model (excluding the target)
DEFAULT_FEATURE_COLS = [
    "CT",
    "TYP",
    "Z-Ave",
    "CO",
    "Time",
    "SHP",
]

# Categorical columns (native categories)
DEFAULT_CAT_COLS = ["CT", "TYP"]

# Group definition: group name -> list of feature columns belonging to this group
DEFAULT_GROUP_MAP = {
    "CT": ["CT"],
    "TYP": ["TYP"],
    "Z-Ave": ["Z-Ave"],
    "CO": ["CO"],
    "Time": ["Time"],
    "SHP": ["SHP"],
}

# CPI settings: for each group specify permutation type & strategy
# - type = "categorical" or "continuous"
# - for categorical, use "strata" to specify stratification columns
# - for continuous, use "n_bins" to quantile-bin the feature and permute within bins
DEFAULT_CPI_SETTINGS = {
    "CT": {
        "type": "categorical",
        "strata": ["TYP", "Study_ID"],  # permute CT within (TYP, Study_ID)
    },
    "TYP": {
        "type": "categorical",
        "strata": ["Study_ID"],        # permute TYP within Study_ID
    },
    "Z-Ave": {
        "type": "continuous",
        "n_bins": 5,                   # quantile bins; permute within bins
    },
    "CO": {
        "type": "continuous",
        "n_bins": 5,
    },
    "Time": {
        "type": "continuous",
        "n_bins": 5,
    },
    "SHP": {
        "type": "continuous",
        "n_bins": 5,
    },
}

# Additional columns that may be used as strata in CPI
DEFAULT_STRATA_EXTRA_COLS = ["TYP", "CT", "Study_ID"]

DEFAULT_RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Metric & model factory
# ----------------------------------------------------------------------
def metric_func(y_true, y_pred) -> float:
    """
    Performance metric used throughout the script.
    You can swap this to MAE, CCC, etc. if needed.
    """
    return r2_score(y_true, y_pred)


def make_model(random_state: int = DEFAULT_RANDOM_STATE) -> lgb.LGBMRegressor:
    """
    Factory that returns a fresh LightGBM regressor each time.
    Hyperparameters are fixed here for reproducibility.
    """
    return lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
    )


# ----------------------------------------------------------------------
# Step 1: Fit baseline models with LOGO and store fold information
# ----------------------------------------------------------------------
def fit_baseline_models(
    df: pd.DataFrame,
    feature_cols,
    target_col: str,
    cat_cols,
    group_col_logo: str,
    random_state: int = DEFAULT_RANDOM_STATE,
    strata_extra_cols=None,
):
    """
    Fit a baseline LightGBM model in a Leave-One-Group-Out scheme.

    For each fold, we store:
    - fitted model
    - validation features/targets
    - baseline score
    - optional strata values (for CPI)
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    groups = df[group_col_logo].copy()

    if strata_extra_cols is not None:
        strata_df = df[strata_extra_cols].copy()
    else:
        strata_df = None

    logo = LeaveOneGroupOut()
    splits = list(logo.split(X, y, groups))

    fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(splits):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()

        # Ensure categorical columns are pandas 'category'
        for col in cat_cols:
            if col in X_train.columns:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")

        model = make_model(random_state=random_state)

        model.fit(
            X_train,
            y_train,
            categorical_feature=cat_cols,
        )

        y_pred = model.predict(X_val)
        score = metric_func(y_val, y_pred)

        fold_info = {
            "fold_id": fold_id,
            "model": model,
            "X_val": X_val.reset_index(drop=True),
            "y_val": y_val.reset_index(drop=True),
            "train_idx": train_idx,
            "val_idx": val_idx,
            "baseline_score": score,
        }

        if strata_df is not None:
            fold_info["strata_val"] = strata_df.iloc[val_idx].reset_index(drop=True)

        fold_results.append(fold_info)

    return fold_results


# ----------------------------------------------------------------------
# Step 2: Group SHAP importance
# ----------------------------------------------------------------------
def compute_group_shap_importance(fold_results, group_map):
    """
    Compute group-level SHAP importance.

    For each fold:
    - Compute SHAP values using TreeExplainer.
    - Sum absolute SHAP values over all features within each group.
    - Average across samples and folds.
    """
    group_scores = {g: [] for g in group_map}

    for fr in fold_results:
        model = fr["model"]
        X_val = fr["X_val"]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_val, check_additivity=False)

        # For regression, LightGBM shap_values is usually (n_samples, n_features)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_abs = np.abs(shap_values)
        col_to_idx = {col: i for i, col in enumerate(X_val.columns)}

        for g, cols in group_map.items():
            idxs = [col_to_idx[c] for c in cols if c in col_to_idx]
            if not idxs:
                continue
            # Mean of total |SHAP| within group
            group_imp_fold = shap_abs[:, idxs].sum(axis=1).mean()
            group_scores[g].append(group_imp_fold)

    rows = []
    for g, vals in group_scores.items():
        if len(vals) == 0:
            continue
        vals = np.array(vals)
        rows.append(
            {
                "group": g,
                "importance_mean": vals.mean(),
                "importance_std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
            }
        )

    df_imp = pd.DataFrame(rows).set_index("group")
    df_imp["rank"] = df_imp["importance_mean"].rank(
        ascending=False, method="average"
    ).astype(int)
    df_imp = df_imp.sort_values("importance_mean", ascending=False)
    return df_imp


# ----------------------------------------------------------------------
# Step 3: Group Drop-Column importance
# ----------------------------------------------------------------------
def compute_group_drop_column_importance(
    df: pd.DataFrame,
    feature_cols,
    target_col: str,
    cat_cols,
    group_col_logo: str,
    group_map,
    fold_results,
    random_state: int = DEFAULT_RANDOM_STATE,
):
    """
    Group Drop-Column importance.

    For each group g:
    - For each LOGO fold, drop columns in that group from train/val.
    - Re-train a fresh model on the reduced feature set.
    - Compare performance to the baseline (same fold).
    - Delta = baseline_score - new_score (larger = more important).
    """
    X_all = df[feature_cols].copy()
    y_all = df[target_col].copy()
    groups = df[group_col_logo].copy()

    logo = LeaveOneGroupOut()
    splits = list(logo.split(X_all, y_all, groups))
    baseline_scores = np.array([fr["baseline_score"] for fr in fold_results])

    group_importances = {g: [] for g in group_map}

    for g, cols in group_map.items():
        for fold_id, (train_idx, val_idx) in enumerate(splits):
            X_train = X_all.iloc[train_idx].drop(columns=cols, errors="ignore").copy()
            X_val = X_all.iloc[val_idx].drop(columns=cols, errors="ignore").copy()
            y_train = y_all.iloc[train_idx].copy()
            y_val = y_all.iloc[val_idx].copy()

            # Adjust categorical columns for this fold (some may have been dropped)
            cat_cols_fold = [c for c in cat_cols if c in X_train.columns]
            for col in cat_cols_fold:
                X_train[col] = X_train[col].astype("category")
                X_val[col] = X_val[col].astype("category")

            model = make_model(random_state=random_state)
            model.fit(
                X_train,
                y_train,
                categorical_feature=cat_cols_fold,
            )

            y_pred = model.predict(X_val)
            score = metric_func(y_val, y_pred)

            delta = baseline_scores[fold_id] - score  # larger delta => more important
            group_importances[g].append(delta)

    rows = []
    for g, deltas in group_importances.items():
        deltas = np.array(deltas)
        rows.append(
            {
                "group": g,
                "delta_mean": deltas.mean(),
                "delta_std": deltas.std(ddof=1) if len(deltas) > 1 else 0.0,
            }
        )

    df_imp = pd.DataFrame(rows).set_index("group")
    df_imp["importance_mean"] = df_imp["delta_mean"]
    df_imp["rank"] = df_imp["importance_mean"].rank(
        ascending=False, method="average"
    ).astype(int)
    df_imp = df_imp.sort_values("importance_mean", ascending=False)
    return df_imp


# ----------------------------------------------------------------------
# Step 4: Group CPI (conditional permutation importance)
# ----------------------------------------------------------------------
def stratified_permutation(values, strata_keys, random_state=None):
    """
    Permute values within strata defined by strata_keys.

    Parameters
    ----------
    values : array-like of shape (n_samples,)
        The values to permute.
    strata_keys : array-like
        Keys defining strata. Same length as values.
        Can be 1D or 2D (multiple stratification columns).
    """
    rng = np.random.RandomState(random_state)
    values = np.asarray(values)
    strata_keys = np.asarray(strata_keys)

    if strata_keys.ndim == 1:
        strata_keys = strata_keys.reshape(-1, 1)

    unique_strata, inverse = np.unique(strata_keys, axis=0, return_inverse=True)

    perm = values.copy()
    for k in range(len(unique_strata)):
        idx = np.where(inverse == k)[0]
        if len(idx) > 1:
            rng.shuffle(perm[idx])
    return perm


def compute_group_cpi_importance(
    fold_results,
    group_map,
    cpi_settings,
    random_state: int = 123,
):
    """
    Conditional Permutation Importance at group level.

    For each fold and each group g:
    - For categorical groups: permute the column values within specified strata.
    - For continuous groups: bin the values into quantile bins and permute within bins.
    - Evaluate performance drop relative to baseline (delta).
    """
    rng = np.random.RandomState(random_state)
    group_importances = {g: [] for g in group_map}

    for fr in fold_results:
        model = fr["model"]
        X_val = fr["X_val"].copy()  # keep original categorical information
        y_val = fr["y_val"].copy()
        baseline_score = fr["baseline_score"]
        strata_val = fr.get("strata_val", None)  # DataFrame or None

        for g, cols in group_map.items():
            setting = cpi_settings.get(g, {})
            g_type = setting.get("type", "categorical")

            X_perm = X_val.copy()  # fresh copy for this group

            for col in cols:
                if col not in X_perm.columns:
                    continue

                # Keep original categorical attributes if present
                orig_series = X_val[col]
                orig_cats_attr = getattr(orig_series, "cat", None)
                orig_cats = (
                    orig_series.cat.categories
                    if orig_cats_attr is not None
                    else None
                )
                orig_vals = orig_series.to_numpy()

                if g_type == "categorical":
                    # Stratification keys for CPI
                    strata_cols = setting.get("strata", [])
                    if strata_cols and strata_val is not None:
                        strata_keys = strata_val[strata_cols].to_numpy()
                    else:
                        # No strata => global permutation
                        strata_keys = np.arange(len(X_perm))

                    perm_vals = stratified_permutation(
                        orig_vals,
                        strata_keys,
                        random_state=rng.randint(1_000_000_000),
                    )

                    # Fallback: if nothing changed, do a global permutation
                    if np.array_equal(perm_vals, orig_vals):
                        perm_vals = rng.permutation(orig_vals)

                    # Preserve Categorical categories if any
                    if orig_cats is not None:
                        X_perm[col] = pd.Categorical(perm_vals, categories=orig_cats)
                    else:
                        X_perm[col] = perm_vals

                elif g_type == "continuous":
                    # Quantile binning, then permutation within bins
                    n_bins = setting.get("n_bins", 5)
                    values = X_perm[col].to_numpy()
                    try:
                        bins = pd.qcut(values, q=n_bins, duplicates="drop")
                    except ValueError:
                        # Extreme case: fall back to a single bin
                        bins = pd.Series(np.zeros_like(values), index=X_perm.index)

                    strata_keys = np.asarray(bins).astype(str)

                    perm_vals = stratified_permutation(
                        values,
                        strata_keys,
                        random_state=rng.randint(1_000_000_000),
                    )

                    # Fallback: if nothing changed, do a global permutation
                    if np.array_equal(perm_vals, values):
                        perm_vals = rng.permutation(values)

                    X_perm[col] = perm_vals

            # Evaluate performance under permuted data
            y_pred_perm = model.predict(X_perm)
            score_perm = metric_func(y_val, y_pred_perm)
            delta = baseline_score - score_perm  # larger delta => more important

            group_importances[g].append(delta)

    rows = []
    for g, deltas in group_importances.items():
        deltas = np.array(deltas)
        rows.append(
            {
                "group": g,
                "delta_mean": deltas.mean(),
                "delta_std": deltas.std(ddof=1) if len(deltas) > 1 else 0.0,
            }
        )

    df_imp = pd.DataFrame(rows).set_index("group")
    df_imp["importance_mean"] = df_imp["delta_mean"]
    df_imp["rank"] = df_imp["importance_mean"].rank(
        ascending=False, method="average"
    ).astype(int)
    df_imp = df_imp.sort_values("importance_mean", ascending=False)
    return df_imp


# ----------------------------------------------------------------------
# Step 5: Group Gain importance
# ----------------------------------------------------------------------
def compute_group_gain_importance(fold_results, group_map):
    """
    Group importance based on LightGBM feature gain.

    For each fold:
    - Read feature gain from the underlying booster.
    - Sum gains over all features belonging to each group.
    - Average across folds.
    """
    group_scores = {g: [] for g in group_map}

    for fr in fold_results:
        model = fr["model"]
        booster = model.booster_

        feat_names = booster.feature_name()
        gains = booster.feature_importance(importance_type="gain")

        gain_series = pd.Series(gains, index=feat_names)

        for g, cols in group_map.items():
            # Only sum columns that exist in this booster
            group_gain = gain_series.reindex(cols).fillna(0.0).sum()
            group_scores[g].append(group_gain)

    rows = []
    for g, vals in group_scores.items():
        vals = np.array(vals)
        rows.append(
            {
                "group": g,
                "gain_mean": vals.mean(),
                "gain_std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
            }
        )

    df_gain = pd.DataFrame(rows).set_index("group")
    df_gain["importance_mean"] = df_gain["gain_mean"]
    df_gain["rank"] = df_gain["importance_mean"].rank(
        ascending=False, method="average"
    ).astype(int)
    df_gain = df_gain.sort_values("importance_mean", ascending=False)
    return df_gain


# ----------------------------------------------------------------------
# Step 6: Rank correlation between methods
# ----------------------------------------------------------------------
def compute_rank_correlation(df_drop, df_gain, df_shap):
    """
    Compute Spearman and Kendall correlation between rank vectors
    produced by Drop-Column, Gain and SHAP.
    """
    # Keep only groups that appear in all three
    groups = sorted(set(df_drop.index) & set(df_gain.index) & set(df_shap.index))
    data = pd.DataFrame(
        {
            "drop_rank": df_drop.loc[groups, "rank"].astype(float),
            "gain_rank": df_gain.loc[groups, "rank"].astype(float),
            "shap_rank": df_shap.loc[groups, "rank"].astype(float),
        },
        index=groups,
    )

    methods = ["drop", "gain", "shap"]
    cols = {
        "drop": "drop_rank",
        "gain": "gain_rank",
        "shap": "shap_rank",
    }

    rows = []
    for i, m1 in enumerate(methods):
        for j in range(i + 1, len(methods)):
            m2 = methods[j]
            r_spear, _ = spearmanr(data[cols[m1]], data[cols[m2]])
            r_kend, _ = kendalltau(data[cols[m1]], data[cols[m2]])
            rows.append(
                {
                    "method1": m1,
                    "method2": m2,
                    "spearman_r": r_spear,
                    "kendall_tau": r_kend,
                }
            )

    df_corr = pd.DataFrame(rows)
    return df_corr


# ----------------------------------------------------------------------
# CLI and main
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute group-level feature importance on raw data using "
            "SHAP, Drop-Column, Gain and optional CPI under LOGO CV."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/mnp_cv_dataset_raw.csv",
        help="Path to the raw CSV dataset.",
    )
    parser.add_argument(
        "--sep",
        type=str,
        default=";",
        help="CSV separator. Default=';'.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET_COL,
        help=f"Target column name. Default='{DEFAULT_TARGET_COL}'.",
    )
    parser.add_argument(
        "--logo-col",
        type=str,
        default=DEFAULT_GROUP_LOGO_COL,
        help=f"Group column used for LOGO CV. Default='{DEFAULT_GROUP_LOGO_COL}'.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results_group_importance",
        help="Output directory for CSV files.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help=f"Random seed for model training. Default={DEFAULT_RANDOM_STATE}.",
    )
    parser.add_argument(
        "--run-cpi",
        action="store_true",
        help="If set, also compute group-level CPI (may be slow).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, sep=args.sep)

    target_col = args.target
    group_col_logo = args.logo_col

    feature_cols = DEFAULT_FEATURE_COLS
    cat_cols = DEFAULT_CAT_COLS
    group_map = DEFAULT_GROUP_MAP
    cpi_settings = DEFAULT_CPI_SETTINGS
    strata_extra_cols = DEFAULT_STRATA_EXTRA_COLS

    # Sanity checks
    missing_cols = [c for c in feature_cols + [target_col, group_col_logo] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in data: {missing_cols}")

    print("[INFO] Data shape:", df.shape)
    print("[INFO] Target column:", target_col)
    print("[INFO] LOGO group column:", group_col_logo)
    print("[INFO] Feature columns:", feature_cols)
    print("[INFO] Categorical columns:", cat_cols)
    print("[INFO] Groups:", list(group_map.keys()))

    # 1) Baseline LOGO models
    fold_results = fit_baseline_models(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_cols=cat_cols,
        group_col_logo=group_col_logo,
        random_state=args.random_state,
        strata_extra_cols=strata_extra_cols,
    )

    # 2) Group SHAP
    df_shap = compute_group_shap_importance(
        fold_results=fold_results,
        group_map=group_map,
    )
    shap_path = out_dir / "group_importance_SHAP.csv"
    df_shap.to_csv(shap_path, sep=";", float_format="%.6f")
    print(f"[INFO] Saved SHAP group importance to {shap_path}")

    # 3) Group Drop-Column
    df_drop = compute_group_drop_column_importance(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        cat_cols=cat_cols,
        group_col_logo=group_col_logo,
        group_map=group_map,
        fold_results=fold_results,
        random_state=args.random_state,
    )
    drop_path = out_dir / "group_importance_DropColumn.csv"
    df_drop.to_csv(drop_path, sep=";", float_format="%.6f")
    print(f"[INFO] Saved Drop-Column group importance to {drop_path}")

    # 4) Optional group CPI
    if args.run_cpi:
        df_cpi = compute_group_cpi_importance(
            fold_results=fold_results,
            group_map=group_map,
            cpi_settings=cpi_settings,
        )
        cpi_path = out_dir / "group_importance_CPI.csv"
        df_cpi.to_csv(cpi_path, sep=";", float_format="%.6f")
        print(f"[INFO] Saved CPI group importance to {cpi_path}")

    # 5) Group Gain
    df_gain = compute_group_gain_importance(
        fold_results=fold_results,
        group_map=group_map,
    )
    gain_path = out_dir / "group_importance_gain.csv"
    df_gain.to_csv(gain_path, sep=";", float_format="%.6f")
    print(f"[INFO] Saved Gain group importance to {gain_path}")

    # 6) Rank correlation between Drop / Gain / SHAP
    df_corr = compute_rank_correlation(df_drop, df_gain, df_shap)
    corr_path = out_dir / "group_importance_rank_correlation.csv"
    df_corr.to_csv(corr_path, sep=";", index=False, float_format="%.3f")
    print(f"[INFO] Saved rank correlation to {corr_path}")


if __name__ == "__main__":
    main()
