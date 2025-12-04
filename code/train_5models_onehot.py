#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model comparison on the same one-hot encoded dataset with OOF metrics.

- Models: XGBoost, RandomForest, ElasticNet, SVR, Tabular MLP
- Metrics: RMSE, MAE, CCC, R2
- CV: RepeatedKFold (default n_splits=5, n_repeats=3, random_state=42)
- Assumes CT/TYP are already one-hot encoded (e.g., CT_*, TYP_*); no encoding here.
"""

import os
os.environ["MPLBACKEND"] = "Agg"  # Use non-interactive backend for safe plotting

import argparse
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.base import clone

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# Global plotting style
mpl.rcParams.update(
    {
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
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

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# Metrics & conformal prediction
# ----------------------------------------------------------------------
def ccc(y_true, y_pred):
    """
    Concordance Correlation Coefficient (Lin, 1989).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    mu_y = np.mean(y_true)
    mu_x = np.mean(y_pred)
    s_y2 = np.var(y_true, ddof=1)
    s_x2 = np.var(y_pred, ddof=1)
    s_xy = np.cov(y_true, y_pred, ddof=1)[0, 1]
    return (2 * s_xy) / (s_x2 + s_y2 + (mu_x - mu_y) ** 2)


def compute_conformal_radius(y_true, y_pred, alpha=0.1):
    """
    Compute a (1 - alpha) conformal prediction radius using OOF residuals.

    Nonconformity scores are |y_true - y_pred|.
    We take the (1 - alpha) empirical quantile as the common radius.

    Parameters
    ----------
    y_true : array-like
        True targets (from OOF).
    y_pred : array-like
        OOF predictions.
    alpha : float, default 0.1
        Miscoverage level. alpha = 0.1 -> 90% prediction interval.

    Returns
    -------
    radius : float
        The conformal radius (same unit as the target).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    residuals = np.abs(y_true - y_pred)
    radius = np.quantile(residuals, 1.0 - alpha)
    return float(radius)


# ----------------------------------------------------------------------
# Feature selection helper
# ----------------------------------------------------------------------
def infer_feature_columns(
    df,
    extra_prefixes=("CT_", "TYP_"),
    base_cols=("Z-Ave", "CO", "SHP"),
    also_allow=("CT", "TYP"),
):
    """
    Select base numeric columns plus any columns starting with CT_ or TYP_.
    If raw 'CT'/'TYP' (already numeric) exist, include them too.
    """
    cols = []
    # base numeric columns
    for c in base_cols:
        if c in df.columns:
            cols.append(c)

    # one-hot groups by prefix
    for pfx in extra_prefixes:
        cols.extend([c for c in df.columns if c.startswith(pfx)])

    # allow single-column CT/TYP if present (and numeric)
    for c in also_allow:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    # de-duplicate but keep order
    seen, ordered = set(), []
    for c in cols:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    if not ordered:
        raise ValueError(
            "No feature columns were found. "
            "Please check column names or adjust prefixes/base columns."
        )
    return ordered


# ----------------------------------------------------------------------
# Pipelines with manually specified hyperparameters
# ----------------------------------------------------------------------
def build_pipelines(numeric_features):
    """
    Build model pipelines with manually specified hyperparameters.

    For linear/SVR/MLP:
        - Median imputation + StandardScaler
    For tree-based/XGB:
        - Median imputation only
    """
    # Preprocessors
    scaler_block = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    tree_block = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Column transformer: all features are numeric already
    pre_scaled = ColumnTransformer(
        [("num", scaler_block, numeric_features)],
        remainder="drop",
    )

    pre_tree = ColumnTransformer(
        [("num", tree_block, numeric_features)],
        remainder="drop",
    )

    models = {}

    # XGBoost
    if HAS_XGB:
        models["XGBoost"] = Pipeline(
            steps=[
                ("prep", pre_tree),
                (
                    "model",
                    XGBRegressor(
                        # -------- MANUAL HYPERPARAMETERS (edit as needed) --------
                        n_estimators=800,
                        learning_rate=0.02,
                        max_depth=5,
                        min_child_weight=1,
                        subsample=0.85,
                        colsample_bytree=0.7,
                        reg_lambda=1.0,
                        reg_alpha=0.5,
                        gamma=2.0,
                        # ----------------------------------------------------------
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    # RandomForest
    models["RandomForest"] = Pipeline(
        steps=[
            ("prep", pre_tree),
            (
                "model",
                RandomForestRegressor(
                    # -------- MANUAL HYPERPARAMETERS (edit as needed) --------
                    n_estimators=600,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    bootstrap=True,
                    # ----------------------------------------------------------
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    # ElasticNet
    models["ElasticNet"] = Pipeline(
        steps=[
            ("prep", pre_scaled),
            (
                "model",
                ElasticNet(
                    # -------- MANUAL HYPERPARAMETERS (edit as needed) --------
                    alpha=0.1,
                    l1_ratio=0.8,
                    max_iter=20000,
                    # ----------------------------------------------------------
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    # SVR
    models["SVR"] = Pipeline(
        steps=[
            ("prep", pre_scaled),
            (
                "model",
                SVR(
                    # -------- MANUAL HYPERPARAMETERS (edit as needed) --------
                    kernel="rbf",
                    C=30.0,
                    epsilon=0.3,
                    gamma="scale",
                    # ----------------------------------------------------------
                ),
            ),
        ]
    )

    # Tabular MLP
    models["TabularMLP"] = Pipeline(
        steps=[
            ("prep", pre_scaled),
            (
                "model",
                MLPRegressor(
                    # -------- MANUAL HYPERPARAMETERS (edit as needed) --------
                    hidden_layer_sizes=(64,),
                    activation="relu",
                    alpha=1e-1,
                    batch_size=16,
                    learning_rate="adaptive",
                    learning_rate_init=1e-3,
                    early_stopping=True,
                    max_iter=1200,
                    # ----------------------------------------------------------
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return models


# ----------------------------------------------------------------------
# Blend (RF + XGB) based on OOF predictions
# ----------------------------------------------------------------------
def evaluate_blend(y_true, pred_a, pred_b, metric="rmse"):
    """
    Scan w in [0,1] and compute blend = w * pred_a + (1-w) * pred_b,
    then choose the best w.
    """
    ws = np.linspace(0, 1, 101)
    best = {
        "w": None,
        "rmse": np.inf,
        "mae": np.inf,
        "r2": -np.inf,
        "ccc": -np.inf,
    }
    for w in ws:
        yb = w * pred_a + (1 - w) * pred_b
        rmse = root_mean_squared_error(y_true, yb)
        mae = mean_absolute_error(y_true, yb)
        r2 = r2_score(y_true, yb)
        c = ccc(y_true, yb)
        if metric == "rmse":
            better = rmse < best["rmse"]
        else:
            better = c > best["ccc"]
        if better:
            best.update({"w": w, "rmse": rmse, "mae": mae, "r2": r2, "ccc": c})
    return best


def append_blend_row(metrics_df, y, oof, save_oof=False, prefer_metric="rmse"):
    """
    Compute RF+XGB weighted blend based on OOF predictions and
    append it as an extra row to metrics_df.
    """
    if "RandomForest" not in oof or "XGBoost" not in oof:
        print("[Blend] Skipped: missing OOF predictions for RandomForest or XGBoost.")
        return metrics_df

    y_true = np.asarray(y).ravel()
    pred_rf = np.asarray(oof["RandomForest"]).ravel()
    pred_xgb = np.asarray(oof["XGBoost"]).ravel()

    res = evaluate_blend(y_true, pred_rf, pred_xgb, metric=prefer_metric)

    row = {
        "Model": "Blend(RF,XGB)",
        "RMSE": res["rmse"],
        "MAE": res["mae"],
        "R2": res["r2"],
        "CCC": res["ccc"],
    }
    metrics_df = pd.concat(
        [metrics_df, pd.DataFrame([row])], ignore_index=True
    ).sort_values(by="RMSE", ascending=True)
    metrics_df = metrics_df.reset_index(drop=True)

    if save_oof:
        y_blend = res["w"] * pred_rf + (1 - res["w"]) * pred_xgb
        pd.DataFrame(
            {
                "y_true": y_true,
                "y_pred_Blend_RFXGB": y_blend,
            }
        ).to_csv("oof_blend_rfxgb.csv", index=False, encoding="utf-8-sig", sep=";")
        print("[Blend] OOF blend predictions saved to: oof_blend_rfxgb.csv")

    return metrics_df


# ----------------------------------------------------------------------
# RepeatedKFold OOF evaluation
# ----------------------------------------------------------------------
def evaluate_oof(models, X, y, cv, verbose=True):
    """
    RepeatedKFold OOF evaluation.

    For each model:
    - Train on each train split and predict on the corresponding test split.
    - Aggregate multiple predictions per sample by averaging.
    - Compute RMSE, MAE, R², CCC on the OOF predictions.

    Returns
    -------
    df_metrics : pd.DataFrame
        Metrics table.
    oof_preds : dict
        Model name -> OOF prediction array (length n_samples).
    """
    assert hasattr(X, "iloc"), "X must be a pandas DataFrame"
    assert hasattr(y, "iloc"), "y must be a pandas Series"

    n = len(y)
    rows = []
    oof_preds = {}

    for name, pipe in models.items():
        if verbose:
            print(f"\n=== {name} ===")

        # Aggregate predictions across repeats
        y_pred_sum = np.zeros(n, dtype=float)
        y_pred_cnt = np.zeros(n, dtype=int)

        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y)):
            if verbose:
                print(f"  Fold {fold_idx + 1}")

            m = clone(pipe)
            # Use iloc to keep column names aligned with ColumnTransformer
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            pred = m.predict(X.iloc[te_idx])

            y_pred_sum[te_idx] += pred
            y_pred_cnt[te_idx] += 1

        # Average predictions for each sample
        zero_mask = y_pred_cnt == 0
        if np.any(zero_mask):
            # Fallback: if some samples never got predicted, use mean(y)
            y_pred_cnt[zero_mask] = 1
            y_pred_sum[zero_mask] = y.mean()

        y_pred = y_pred_sum / y_pred_cnt
        oof_preds[name] = y_pred

        y_true = y.values
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        c = ccc(y_true, y_pred)

        if verbose:
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE : {mae:.4f}")
            print(f"R²  : {r2:.4f}")
            print(f"CCC : {c:.4f}")

        rows.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "CCC": c})

    df_metrics = (
        pd.DataFrame(rows)
        .sort_values(by="RMSE", ascending=True)
        .reset_index(drop=True)
    )
    return df_metrics, oof_preds


# ----------------------------------------------------------------------
# Plotting (with conformal prediction interval band)
# ----------------------------------------------------------------------
def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def plot_true_vs_pred(
    y_true,
    y_pred,
    model_name,
    outdir=".",
    dpi=200,
    conformal_radius=None,
):
    """
    Plot true vs predicted scatter and save as PNG.

    If conformal_radius is not None, also plot a band around y = x:
        y = x ± conformal_radius
    as a global (1 - alpha) conformal prediction interval based on OOF residuals.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    c = ccc(y_true, y_pred)

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

    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"true_vs_pred_{_safe_name(model_name)}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=dpi)
    plt.close(fig)
    print(f"[Plot] saved: {fname}")


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Horizontal comparison of 5 regressors on one-hot encoded data "
            "with OOF metrics and conformal prediction intervals."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/mnp_cv_dataset_onehot.csv",
        help="Path to the one-hot encoded CSV dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="CV",
        help="Target column name.",
    )
    parser.add_argument(
        "--ct_prefix",
        type=str,
        default="CT_",
        help="One-hot prefix for CT (e.g., CT_).",
    )
    parser.add_argument(
        "--typ_prefix",
        type=str,
        default="TYP_",
        help="One-hot prefix for TYP (e.g., TYP_).",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed.",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of KFold splits.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats for RepeatedKFold.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Miscoverage level for conformal prediction interval "
             "(alpha=0.1 -> 90%% band).",
    )
    parser.add_argument(
        "--save_metrics",
        type=str,
        default="model_comparison_5models_onehot.csv",
        help="Path to save metrics CSV.",
    )
    parser.add_argument(
        "--save_oof",
        action="store_true",
        help="If set, saves OOF predictions per model.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default="figs_5models",
        help="Output directory for true-vs-pred plots.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv(args.csv, sep=";")

    if args.target not in df.columns:   
        raise ValueError(
            f"Target column `{args.target}` not found in the CSV. "
            f"Available columns (first 20): {list(df.columns)[:20]} ..."
        )

    feature_cols = infer_feature_columns(
        df,
        extra_prefixes=(args.ct_prefix, args.typ_prefix),
        base_cols=("Z-Ave", "CO", "SHP"),
        also_allow=("CT", "TYP"),
    )

    X = df[feature_cols].copy()
    y = df[args.target].copy()

    print("Feature columns used:", feature_cols)
    print("Sample size / feature dimension:", X.shape)

    # ------------------------------------------------------------------
    # CV, models (fixed hyperparameters), OOF evaluation
    # ------------------------------------------------------------------
    cv = RepeatedKFold(
        n_splits=args.splits,
        n_repeats=args.repeats,
        random_state=args.random_state,
    )

    models = build_pipelines(feature_cols)

    # 1) OOF evaluation for all models
    metrics_df, oof = evaluate_oof(models, X, y, cv, verbose=True)

    # 2) RF + XGB blend row
    metrics_df = append_blend_row(
        metrics_df,
        y,
        oof,
        save_oof=args.save_oof,
        prefer_metric="rmse",
    )

    print("\n=== Metrics sorted by RMSE (ascending) ===")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(
        args.save_metrics,
        index=False,
        encoding="utf-8-sig",
        sep=";",
    )
    print(f"\n[INFO] Metrics saved to: {args.save_metrics}")

    # ------------------------------------------------------------------
    # Plot true vs predicted for each model with conformal band
    # ------------------------------------------------------------------
    os.makedirs(args.plot_dir, exist_ok=True)

    for name, preds in oof.items():
        radius = compute_conformal_radius(
            y_true=y,
            y_pred=preds,
            alpha=args.alpha,
        )
        print(f"[Conformal] {name}: radius={radius:.4f} (alpha={args.alpha})")
        plot_true_vs_pred(
            y,
            preds,
            model_name=name,
            outdir=args.plot_dir,
            conformal_radius=radius,
        )


if __name__ == "__main__":
    main()
