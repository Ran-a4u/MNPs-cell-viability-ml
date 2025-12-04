#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Preprocess raw MNP cytotoxicity dataset with one-hot encoding.

This script:
1) Loads a raw CSV file.
2) Applies one-hot encoding to selected categorical columns.
3) Saves the encoded dataset to a new CSV file.

Default behavior assumes:
- Input file:  data/mnp_cv_dataset_raw.csv
- Output file: data/mnp_cv_dataset_onehot.csv
- Separator:   ';'
- Target col:  'CV'
- Categorical cols: ['CT', 'TYP']
"""

import argparse
from typing import List

import pandas as pd


def encode_one_hot(
    input_path: str,
    output_path: str,
    categorical_cols: List[str],
    target_col: str = "CV",
    sep: str = ";",
) -> pd.DataFrame:
    """
    Load a CSV file, apply one-hot encoding to selected categorical columns,
    and save the encoded dataset.

    Parameters
    ----------
    input_path : str
        Path to the raw input CSV file.
    output_path : str
        Path to the output CSV file (encoded).
    categorical_cols : list of str
        Column names to be one-hot encoded.
    target_col : str, default 'CV'
        Name of the target column (kept as a normal column, not encoded).
    sep : str, default ';'
        Column separator used in the CSV files.

    Returns
    -------
    df_encoded : pd.DataFrame
        The encoded DataFrame that has been saved to disk.
    """
    print(f"[INFO] Loading data from: {input_path}")
    df = pd.read_csv(input_path, sep=sep)

    # Basic sanity checks for required columns
    missing_cats = [c for c in categorical_cols if c not in df.columns]
    if missing_cats:
        raise ValueError(
            f"The following categorical columns are missing in the input file: "
            f"{missing_cats}"
        )

    if target_col not in df.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in the input file. "
            f"Available columns: {list(df.columns)}"
        )

    print(f"[INFO] Input shape: {df.shape}")
    print(f"[INFO] Categorical columns to encode: {categorical_cols}")
    print(f"[INFO] Target column (not encoded): {target_col}")

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols)

    print(f"[INFO] Encoded shape: {df_encoded.shape}")

    # Save encoded dataframe
    df_encoded.to_csv(output_path, index=False, sep=sep)
    print(f"[INFO] Encoded dataset saved to: {output_path}")

    # Show a small preview
    print("[INFO] Preview of the encoded dataset (first 5 rows):")
    print(df_encoded.head().to_string(index=False))

    return df_encoded


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments using argparse.
    """
    parser = argparse.ArgumentParser(
        description="One-hot encode selected categorical columns "
                    "in the MNP cytotoxicity dataset."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/mnp_cv_dataset_raw.csv",
        help="Path to the raw input CSV file. "
             "Default: data/mnp_cv_dataset_raw.csv",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/mnp_cv_dataset_onehot.csv",
        help="Path to the output CSV file (encoded). "
             "Default: data/mnp_cv_dataset_onehot.csv",
    )

    parser.add_argument(
        "--sep",
        type=str,
        default=";",
        help="CSV separator used in the input/output files. "
             "Default: ';'",
    )

    parser.add_argument(
        "--target-col",
        type=str,
        default="CV",
        help="Name of the target column (kept as a normal column). "
             "Default: 'CV'",
    )

    parser.add_argument(
        "--cat-cols",
        nargs="+",
        default=["CT", "TYP", "Study_ID"],
        help=(
            "Categorical columns to one-hot encode. "
            "Provide one or more column names separated by spaces. "
            "Default: CT TYP"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    Main entry point of the script.
    """
    args = parse_args()

    encode_one_hot(
        input_path=args.input,
        output_path=args.output,
        categorical_cols=args.cat_cols,
        target_col=args.target_col,
        sep=args.sep,
    )


if __name__ == "__main__":
    main()
