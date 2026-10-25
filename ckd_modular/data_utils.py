import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from config import (
    DATA_PATH,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    TARGET,
)


def load_data():
    """
    Load the CKD dataset and prepare its features and target labels.

    Numeric features are converted to numeric values.
    Categorical features are converted to strings.
    Missing feature values are preserved.
    Rows with missing target labels are removed.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Remove rows whose target label is missing.
    df = df.dropna(subset=[TARGET]).copy()

    # Convert numeric columns to numeric values.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Normalize categorical values and preserve missing values.
    for column in CATEGORICAL_COLUMNS:
        df[column] = df[column].apply(
            lambda value: (
                np.nan
                if pd.isna(value)
                   or str(value).strip() in ("", "?")
                else str(value).strip()
            )
        )

    # Separate features and target.
    X = df[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    y_raw = df[TARGET].astype(str).str.strip()

    # Encode target labels.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    print(f"Dataset shape: {X.shape}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Natural missing feature cells: {int(X.isna().sum().sum())}")

    return X, y, label_encoder


def apply_missing_mask(X, rate, seed):
    """
    Add MCAR missing values to originally observed feature cells.

    The requested rate is applied to the number of originally
    observed cells. Naturally missing values remain missing.

    The input DataFrame is not modified.
    """

    if not 0 <= rate <= 1:
        raise ValueError("Missing rate must be between 0 and 1.")

    X_masked = X.copy()

    # Find cells that originally contain observed values.
    observed_mask = X.notna().to_numpy()
    observed_positions = np.flatnonzero(observed_mask)

    # Number of observed cells to mask.
    n_to_mask = int(np.floor(rate * len(observed_positions)))

    if n_to_mask == 0:
        return X_masked

    # Select cells randomly without replacement.
    rng = np.random.default_rng(seed)
    selected_positions = rng.choice(
        observed_positions,
        size=n_to_mask,
        replace=False,
    )

    # Convert flat positions into row and column indices.
    row_indices, column_indices = np.unravel_index(
        selected_positions,
        X.shape,
    )

    # Apply the additional missing values.
    X_masked_values = X_masked.to_numpy(dtype=object, copy=True)
    X_masked_values[row_indices, column_indices] = np.nan

    X_masked = pd.DataFrame(
        X_masked_values,
        index=X.index,
        columns=X.columns,
    )

    return X_masked
