#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
onehot_importance_raw.py

Compute per-feature importance on one-hot encoded data using
TreeSHAP + cross-validation on a RandomForest regressor.

Differences from older versions:
- Only SHAP-based per-feature importance is implemented.
- No group-level importance (no CT/TYP grouping, no ElasticNet, no permutation/drop).
- Clean CLI (argparse) and English comments.

Typical use:
    python onehot_importance_raw.py \
        --csv data/mnp_cv_dataset_onehot.csv \
        --target CV \
        --group-col Study_ID \
        --splits 5 \
        --seed 42 \
        --out-csv results/feature_SHAP_importance_onehot.csv
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

import shap
import logging
import warnings

logging.getLogger("shap").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)


RANDOM_STATE_DEFAULT = 42


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def get_cv_splitter(groups: np.ndarray, n_splits: int, random_state: int):
    """
    Return a scikit-learn CV splitter.

    If groups is not None -> GroupKFold (LOGO-style across studies).
    Otherwise -> shuffled KFold.
    """
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    else:
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def make_estimator_rf(random_state: int) -> Pipeline:
    """
    Build a simple, robust RandomForest pipeline with median imputation.

    This is the base model for TreeSHAP.
    """
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = Pipeline(
        steps=[
            ("imp", SimpleImputer(strategy="median")),
            ("rf", rf),
        ]
    )
    return pipe


# ----------------------------------------------------------------------
# SHAP per-feature importance with CV
# ----------------------------------------------------------------------
def shap_feature_importance_cv(
    estimator: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Compute per-feature SHAP importance via cross-validation.

    For each fold:
        - fit RF pipeline on training fold
        - transform validation fold with the fitted imputer
        - compute TreeSHAP values on the imputed matrix
        - aggregate mean(|SHAP|) per feature on that fold

    Then average mean(|SHAP|) across all folds.

    Returns
    -------
    df : pd.DataFrame
        Columns:
            feature         feature name
            mean_abs_shap   mean(|SHAP|) over all folds
    """
    splitter = get_cv_splitter(groups, n_splits, random_state)
    feat_names = X.columns.tolist()
    per_feat_scores = []

    for fold_id, (tr, va) in enumerate(splitter.split(X, y, groups=groups), start=1):
        X_tr, X_va = X.values[tr], X.values[va]
        y_tr = y[tr]

        # Clone pipeline manually (shallow copy of steps list)
        est = Pipeline(estimator.steps)
        est.fit(X_tr, y_tr)

        # Use the imputed matrix actually seen by the RF
        imp = est.named_steps["imp"]
        rf = est.named_steps["rf"]
        X_va_imp = imp.transform(X_va)

        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(X_va_imp)  # shape: (n_val, n_features)
        shap_abs_mean = np.mean(np.abs(sv), axis=0)
        per_feat_scores.append(shap_abs_mean)

        print(
            f"[INFO] Fold {fold_id}: "
            f"mean(|SHAP|) in [{shap_abs_mean.min():.4e}, {shap_abs_mean.max():.4e}]"
        )

    mean_abs = np.mean(np.vstack(per_feat_scores), axis=0)
    df = pd.DataFrame(
        {
            "feature": feat_names,
            "mean_abs_shap": mean_abs,
        }
    ).sort_values("mean_abs_shap", ascending=False)
    df = df.reset_index(drop=True)
    return df


# ----------------------------------------------------------------------
# Main workflow
# ----------------------------------------------------------------------
def run(
    csv_path: str,
    target_col: str,
    id_cols: list,
    group_col: str,
    sep: str,
    n_splits: int,
    random_state: int,
    out_csv: str,
):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Safer CSV reading for ';'-separated, UTF-8 with BOM
    try:
        df = pd.read_csv(csv_path, sep=sep, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, sep=sep)

    # Drop Excel-generated unnamed index columns if any
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    # Optional group column (for LOGO)
    groups_arr = None
    if group_col and group_col in df.columns:
        groups_arr = df[group_col].values
        if group_col not in id_cols:
            id_cols = list(id_cols) + [group_col]

    # Define feature columns: all except target and id_cols
    exclude = set([target_col] + list(id_cols))
    feature_cols = [c for c in df.columns if c not in exclude]

    if not feature_cols:
        raise ValueError(
            "No feature columns remain after excluding target and id_cols. "
            f"id_cols={id_cols}"
        )

    X = df[feature_cols].copy()
    y = df[target_col].values

    print(f"[INFO] Loaded data from: {csv_path}")
    print(f"[INFO] Shape: X={X.shape}, y={y.shape}")
    print(f"[INFO] Target column: {target_col}")
    if groups_arr is not None:
        print(f"[INFO] Using GroupKFold on '{group_col}' with n_splits={n_splits}")
    else:
        print(f"[INFO] Using shuffled KFold with n_splits={n_splits}")
    print(f"[INFO] Number of features: {len(feature_cols)}")

    # Build RF estimator
    rf_estimator = make_estimator_rf(random_state)

    # Compute SHAP importance
    df_shap = shap_feature_importance_cv(
        estimator=rf_estimator,
        X=X,
        y=y,
        groups=groups_arr,
        n_splits=n_splits,
        random_state=random_state,
    )

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_shap.to_csv(out_path, index=False, encoding="utf-8-sig", sep=";")
    print(f"[INFO] Per-feature SHAP importance saved to: {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-feature TreeSHAP importance on one-hot encoded data "
            "using RandomForest + cross-validation."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/mnp_cv_dataset_onehot.csv",
        help="Path to the one-hot encoded CSV dataset.",
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
        default="CV",
        help="Target column name. Default='CV'.",
    )
    parser.add_argument(
        "--id-cols",
        type=str,
        nargs="*",
        default=[],
        help="Columns to exclude from features (e.g. IDs, Study_ID if not used in X).",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="Study_ID",
        help=(
            "Optional group column for LOGO-style CV. "
            "If present, GroupKFold is used; otherwise shuffled KFold."
        ),
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of CV folds (KFold/GroupKFold). Default=5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_STATE_DEFAULT,
        help=f"Random seed. Default={RANDOM_STATE_DEFAULT}.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default="importance_onehot_feature_SHAP.csv",
        help="Output CSV for per-feature SHAP importance.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run(
        csv_path=args.csv,
        target_col=args.target,
        id_cols=args.id_cols,
        group_col=args.group_col,
        sep=args.sep,
        n_splits=args.splits,
        random_state=args.seed,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
