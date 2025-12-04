#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Horizontal comparison of 3 regressors on the raw dataset with native categorical
support (no one-hot encoding).

- Models: CatBoostRegressor, LGBMRegressor, HistGradientBoostingRegressor
- Features (by default): ["Z-Ave", "CO", "CT", "SHP", "TYP", "Time", "Study_ID"]
- Target: "CV"
- Metrics: RMSE, MAE, R2, CCC
- CV: RepeatedKFold (n_splits=5, n_repeats=3 by default)

This script expects that the raw CSV already contains the above columns.
"""

import os
os.environ["MPLBACKEND"] = "Agg"

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import clone

# CatBoost
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

# LightGBM
try:
    from lightgbm import LGBMRegressor
    import lightgbm as lgb
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

# Matplotlib global style
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
    }
)

RANDOM_STATE = 123

# Default configuration (can be overridden by CLI)
DEFAULT_INPUT_COLS = ["Z-Ave", "CO", "CT", "SHP", "TYP", "Time", "Study_ID"]
DEFAULT_TARGET_COL = "CV"
DEFAULT_CATEGORICAL_COLS = ["CT", "TYP", "Study_ID"]


# ----------------------------------------------------------------------
# Metrics & conformal prediction
# ----------------------------------------------------------------------
def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concordance Correlation Coefficient (CCC).
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mu_y = np.mean(y_true)
    mu_x = np.mean(y_pred)
    s_y2 = np.var(y_true, ddof=1)
    s_x2 = np.var(y_pred, ddof=1)
    s_xy = np.cov(y_true, y_pred, ddof=1)[0, 1]
    return float((2 * s_xy) / (s_x2 + s_y2 + (mu_x - mu_y) ** 2))


def compute_conformal_radius(y_true, y_pred, alpha=0.1) -> float:
    """
    Compute a (1 - alpha) conformal prediction radius using OOF residuals.

    Nonconformity scores are |y_true - y_pred|.
    We take the (1 - alpha) empirical quantile as the common radius.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = np.abs(y_true - y_pred)
    radius = np.quantile(residuals, 1.0 - alpha)
    return float(radius)


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute regression metrics on given predictions.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "CCC": ccc(y_true, y_pred),
    }


# ----------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def plot_true_vs_pred(
    y_true,
    y_pred,
    model_name: str,
    outdir: str = ".",
    dpi: int = 200,
    conformal_radius: float = None,
):
    """
    Plot true vs predicted scatter and save as PNG.

    If conformal_radius is not None, also plot a band around y = x:
        y = x ± conformal_radius
    as a global (1 - alpha) conformal prediction interval based on OOF residuals.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    metrics = evaluate_metrics(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(y_true, y_pred, alpha=0.7, s=18)

    vmin = float(np.nanmin(np.r_[y_true, y_pred]))
    vmax = float(np.nanmax(np.r_[y_true, y_pred]))

    # 1:1 line
    ax.plot([vmin, vmax], [vmin, vmax], ls="--")

    # Conformal prediction band
    if conformal_radius is not None:
        xs = np.linspace(vmin, vmax, 200)
        upper = xs + conformal_radius
        lower = xs - conformal_radius

        ax.plot(xs, upper, lw=1.0)
        ax.plot(xs, lower, lw=1.0)

        ax.fill_between(xs, lower, upper, alpha=0.10)

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name}")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.grid(False)

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    fname = outdir_path / f"true_vs_pred_{_safe_name(model_name)}.png"
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"[Plot] saved: {fname}")
    print(
        f"    RMSE={metrics['RMSE']:.4f}, "
        f"MAE={metrics['MAE']:.4f}, "
        f"R2={metrics['R2']:.4f}, "
        f"CCC={metrics['CCC']:.4f}"
    )


# ----------------------------------------------------------------------
# Data preparation
# ----------------------------------------------------------------------
def prepare_data(
    csv_path: str,
    sep: str,
    input_cols: list,
    target_col: str,
    categorical_cols: list,
):
    """
    Load the raw dataset and build 3 different "views":
    - X          : original DataFrame (for CatBoost)
    - X_lgb      : with categorical columns set to dtype 'category' (for LightGBM)
    - ct, mask   : OrdinalEncoder + categorical_features mask (for HGBR)
    """
    df = pd.read_csv(csv_path, sep=sep)

    missing_cols = [c for c in input_cols + [target_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in data: {missing_cols}")

    X = df[input_cols].copy()
    y = df[target_col].values

    for col in categorical_cols:
        if col not in X.columns:
            raise ValueError(f"CATEGORICAL_COLS includes non-existing column: {col}")

    # CatBoost: indices of categorical columns
    cat_indices = [X.columns.get_loc(c) for c in categorical_cols]

    # LightGBM: need pandas 'category' dtype
    X_lgb = X.copy()
    for c in categorical_cols:
        cats = X[c].astype("category").cat.categories
        X_lgb[c] = pd.Categorical(X[c], categories=cats)

    # HGBR: use OrdinalEncoder + categorical_features mask
    num_cols = [c for c in input_cols if c not in categorical_cols]
    ct = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
                categorical_cols,
            ),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    hgb_categorical_mask = np.array(
        [True] * len(categorical_cols) + [False] * len(num_cols)
    )

    return X, y, X_lgb, cat_indices, ct, hgb_categorical_mask


# ----------------------------------------------------------------------
# Build models (fixed hyperparameters)
# ----------------------------------------------------------------------
def build_models(hgb_categorical_mask):
    """
    Build a dict of models with manually specified hyperparameters.
    """
    models: dict = {}

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            loss_function="RMSE",
            random_state=RANDOM_STATE,
            depth=6,
            learning_rate=0.05,
            verbose=False,
            iterations=5000,
            l2_leaf_reg=5,
            allow_writing_files=False,
        )
    else:
        print("[WARN] catboost is not installed. CatBoost will be skipped.")

    if HAS_LGBM:
        models["LightGBM"] = LGBMRegressor(
            random_state=RANDOM_STATE,
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            min_data_in_leaf=3,
            min_data_per_group=3,
            max_cat_threshold=256,
            max_bin=511,
            subsample=0.8,
            colsample_bytree=0.8,
        )
    else:
        print("[WARN] lightgbm is not installed. LightGBM will be skipped.")

    # HistGradientBoostingRegressor
    models["HGBR"] = HistGradientBoostingRegressor(
        random_state=RANDOM_STATE,
        max_iter=2000,
        learning_rate=0.05,
        max_leaf_nodes=None,
        min_samples_leaf=10,
        max_bins=255,
        l2_regularization=1e-3,
        categorical_features=hgb_categorical_mask,
    )

    return models


# ----------------------------------------------------------------------
# RepeatedKFold OOF evaluation
# ----------------------------------------------------------------------
def evaluate_oof(models, X, y, X_lgb, ct, hgb_categorical_mask, cat_indices, cv, verbose=True):
    """
    Run RepeatedKFold OOF evaluation for CatBoost, LightGBM and HGBR.

    Parameters
    ----------
    models : dict
        Mapping from model name to model instance (not yet fitted).
    X : pd.DataFrame
        Original feature matrix (for CatBoost/HGBR).
    y : np.ndarray
        Target array.
    X_lgb : pd.DataFrame
        Features with categorical dtype for LightGBM.
    ct : ColumnTransformer
        OrdinalEncoder + passthrough for HGBR.
    hgb_categorical_mask : np.ndarray of bool
        Mask indicating which transformed columns are categorical (for HGBR).
    cat_indices : list of int
        Indices of categorical features (for CatBoost).
    cv : RepeatedKFold
        Cross-validation splitter.

    Returns
    -------
    df_metrics : pd.DataFrame
        Metrics table.
    oof_preds : dict
        Model name -> OOF predictions (length n_samples).
    """
    n = len(y)
    results = {}
    oof_preds = {}

    for name, model in models.items():
        print(f"\n=== Evaluating {name} (RepeatedKFold OOF) ===")

        y_pred_sum = np.zeros(n, dtype=float)
        y_pred_cnt = np.zeros(n, dtype=float)

        for fold_id, (trn_idx, val_idx) in enumerate(cv.split(X, y), start=1):
            X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
            y_trn, y_val = y[trn_idx], y[val_idx]

            if name == "CatBoost":
                if not HAS_CATBOOST:
                    continue
                m = clone(model)
                m.set_params(random_state=RANDOM_STATE)
                m.fit(
                    X_trn,
                    y_trn,
                    cat_features=cat_indices,
                    eval_set=(X_val, y_val),
                    verbose=False,
                )
                y_pred = m.predict(X_val)

            elif name == "LightGBM":
                if not HAS_LGBM:
                    continue
                m = clone(model)
                m.set_params(random_state=RANDOM_STATE)
                m.fit(
                    X_lgb.iloc[trn_idx],
                    y_trn,
                    eval_set=[(X_lgb.iloc[val_idx], y_val)],
                    eval_metric="l2",
                    callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)],
                )
                y_pred = m.predict(X_lgb.iloc[val_idx])

            else:  # HGBR
                # Fit encoder on train fold, then transform both train and val
                X_trn_hgb = ct.fit_transform(X_trn)
                X_val_hgb = ct.transform(X_val)
                m = clone(model)
                m.set_params(random_state=RANDOM_STATE)
                m.fit(X_trn_hgb, y_trn)
                y_pred = m.predict(X_val_hgb)

            y_pred_sum[val_idx] += y_pred
            y_pred_cnt[val_idx] += 1

            fold_metrics = evaluate_metrics(y_val, y_pred)
            if verbose:
                print(
                    f"  Fold {fold_id:02d}: "
                    f"RMSE={fold_metrics['RMSE']:.4f} | "
                    f"MAE={fold_metrics['MAE']:.4f} | "
                    f"R2={fold_metrics['R2']:.4f} | "
                    f"CCC={fold_metrics['CCC']:.4f}"
                )

        # Average predictions for each sample
        y_pred_oof = y_pred_sum / np.maximum(y_pred_cnt, 1.0)
        oof_preds[name] = y_pred_oof.copy()
        results[name] = evaluate_metrics(y, y_pred_oof)

    # Build metrics DataFrame
    df_metrics = (
        pd.DataFrame(results).T[["RMSE", "MAE", "R2", "CCC"]]
        .sort_values(by="RMSE", ascending=True)
        .reset_index()
        .rename(columns={"index": "Model"})
    )
    return df_metrics, oof_preds


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Horizontal comparison of 3 regressors (CatBoost, LightGBM, HGBR) "
            "on raw data with native categorical support, using OOF metrics "
            "and conformal prediction intervals."
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
        "--input-cols",
        nargs="+",
        default=DEFAULT_INPUT_COLS,
        help=(
            "Input feature columns. Default: "
            + ", ".join(DEFAULT_INPUT_COLS)
        ),
    )
    parser.add_argument(
        "--cat-cols",
        nargs="+",
        default=DEFAULT_CATEGORICAL_COLS,
        help=(
            "Categorical feature columns. Default: "
            + ", ".join(DEFAULT_CATEGORICAL_COLS)
        ),
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of KFold splits. Default=5.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats for RepeatedKFold. Default=3.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Miscoverage level for conformal band (alpha=0.1 -> 90%% interval).",
    )
    parser.add_argument(
        "--save-metrics",
        type=str,
        default="model_comparison_3models_nativecat.csv",
        help="Path to save metrics CSV.",
    )
    parser.add_argument(
        "--save-oof",
        action="store_true",
        help="If set, save OOF predictions for each model.",
    )
    parser.add_argument(
        "--plot-dir",
        type=str,
        default="figs_3models_nativecat",
        help="Directory to save true-vs-pred plots.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Prepare data
    X, y, X_lgb, cat_indices, ct, hgb_categorical_mask = prepare_data(
        csv_path=args.csv,
        sep=args.sep,
        input_cols=args.input_cols,
        target_col=args.target,
        categorical_cols=args.cat_cols,
    )

    print("Input columns:", args.input_cols)
    print("Categorical cols:", args.cat_cols)
    print("Target column:", args.target)
    print("X shape:", X.shape)

    # Build models
    models = build_models(hgb_categorical_mask)

    # RepeatedKFold CV
    cv = RepeatedKFold(
        n_splits=args.splits,
        n_repeats=args.repeats,
        random_state=RANDOM_STATE,
    )

    # OOF evaluation
    df_metrics, oof = evaluate_oof(
        models=models,
        X=X,
        y=y,
        X_lgb=X_lgb,
        ct=ct,
        hgb_categorical_mask=hgb_categorical_mask,
        cat_indices=cat_indices,
        cv=cv,
        verbose=True,
    )

    print("\n=== OOF summary (sorted by RMSE) ===")
    print(df_metrics.to_string(index=False))

    # Save metrics
    out_metrics_path = Path(args.save_metrics)
    out_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    df_metrics.to_csv(out_metrics_path, index=False, encoding="utf-8-sig", sep=";")
    print(f"[INFO] Metrics saved to: {out_metrics_path}")

    # Optionally save OOF predictions
    if args.save_oof:
        oof_df = pd.DataFrame({"y_true": y})
        for name, pred in oof.items():
            oof_df[f"y_pred_{_safe_name(name)}"] = pred
        oof_path = out_metrics_path.with_suffix(".oof.csv")
        oof_df.to_csv(oof_path, index=False, encoding="utf-8-sig", sep=";")
        print(f"[INFO] OOF predictions saved to: {oof_path}")

    # Plot scatter + conformal band for each model
    for name, y_pred in oof.items():
        radius = compute_conformal_radius(y_true=y, y_pred=y_pred, alpha=args.alpha)
        print(
            f"[Conformal] {name}: radius={radius:.4f} "
            f"(alpha={args.alpha}, coverage ≈ {1-args.alpha:.2%})"
        )
        plot_true_vs_pred(
            y_true=y,
            y_pred=y_pred,
            model_name=name,
            outdir=args.plot_dir,
            conformal_radius=radius,
        )


if __name__ == "__main__":
    main()
