# MNPs-cell-viability-ml

# MNP Cell Viability – Machine Learning Benchmark

This repository contains code and data processing pipelines for a machine-learning
study on **micro-/nanoplastics (MNPs) cell viability (CV)**. The project builds
regression models to predict cell viability from physicochemical properties,
exposure conditions and experimental context, and compares different modelling
strategies and feature importance methods.

> **Note:** This is the companion codebase for a research manuscript on
> micro-/nanoplastic cytotoxicity and machine learning modelling.

---

## 1. Project overview

- **Goal:**  
  Build and compare machine-learning models that predict cell viability (CV)
  from MNP properties and experimental conditions, and analyse feature
  importance at both the **single-variable** and **group** level.

- **Key ideas:**
  - Combine data from multiple published in vitro studies into a single dataset.
  - Evaluate two modelling strategies:
    - **One-hot encoded** categorical variables (cell type, particle type, etc.).
    - **Native categorical** modelling using algorithms that directly support
      categorical features.
  - Use **RepeatedKFold** and **Leave-One-Group-Out (LOGO)** to obtain
    robust out-of-fold (OOF) estimates.
  - Compute **group feature importance** using multiple methods
    (SHAP, Drop-Column, Gain, conditional permutation importance / CPI).
  - Visualise **true-vs-predicted** plots with **conformal prediction bands**
    as prediction intervals.

---

## 2. Repository structure

```text
mnp-cell-viability-ml/
├── code/
│   ├── preprocess_onehot.py          # raw → one-hot encoded dataset
│   ├── train_5models_onehot.py       # 5-model comparison on one-hot data
│   ├── train_3models_nativecat.py    # 3-model comparison on native categorical data
│   ├── onehot_importance_raw.py      # within-group (CT/TYP) importance on one-hot data
│   └── group_importance_raw.py       # group-level importance on raw data (LOGO + LGBM)
├── data/
│   ├── mnp_cv_dataset_raw.csv        # raw integrated dataset (from literature)
│   └── mnp_cv_dataset_onehot.csv     # one-hot encoded dataset (generated)
├── results/                          # metrics, plots and importance tables (generated)
├── notebooks/                        # optional demo / exploratory notebooks
└── README.md
