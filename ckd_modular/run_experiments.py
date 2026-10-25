
import sys
from pathlib import Path

# Add the repository root to Python's module search path.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import (
    SEED_LIST,
    MISSING_RATES,
    OUTPUT_DIR,
    TARGET,
)
from data_utils import load_data, apply_missing_mask
from baseline_models import build_classical_models
from evaluation import calculate_metrics, save_summary_and_plots
from naim_runner import train_and_predict_naim


def split_dataset(X, y, seed):
    """
    Create stratified 70% train, 15% validation, 15% test splits.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=seed,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=seed,
        stratify=y_temp,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def record_result(
    results,
    seed,
    missing_rate,
    model_name,
    y_true,
    y_pred,
    train_macro_f1=np.nan,
    val_macro_f1=np.nan,
    best_epoch=np.nan,
):
    """
    Store metrics and experiment metadata.
    """

    metrics = calculate_metrics(y_true, y_pred)

    results.append({
        "seed": seed,
        "missing_rate": missing_rate,
        "model": model_name,
        **metrics,
        "train_macro_f1": train_macro_f1,
        "val_macro_f1": val_macro_f1,
        "best_epoch": best_epoch,
    })


def run_experiments():
    """
    Run the full CKD comparison experiment.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, label_encoder = load_data()

    print("\nTarget label mapping:")
    for index, label in enumerate(label_encoder.classes_):
        print(f"  {label} -> {index}")

    results = []
    mask_audit = []

    for seed in SEED_LIST:
        print("\n" + "=" * 65)
        print(f"Seed: {seed}")
        print("=" * 65)

        (
            X_train,
            X_val,
            X_test,
            y_train,
            y_val,
            y_test,
        ) = split_dataset(X, y, seed)

        for missing_rate in MISSING_RATES:
            print(
                f"\n--- Seed={seed}, "
                f"additional missing={missing_rate:.0%} ---"
            )

            # Apply masks once per split.
            # All four models receive these same masked datasets.
            X_train_masked = apply_missing_mask(
                X_train,
                missing_rate,
                seed + 1000,
            )
            X_val_masked = apply_missing_mask(
                X_val,
                missing_rate,
                seed + 2000,
            )
            X_test_masked = apply_missing_mask(
                X_test,
                missing_rate,
                seed + 3000,
            )

            mask_audit.append({
                "seed": seed,
                "missing_rate": missing_rate,
                "train_cells_masked": int(
                    X_train_masked.isna().sum().sum()
                    - X_train.isna().sum().sum()
                ),
                "val_cells_masked": int(
                    X_val_masked.isna().sum().sum()
                    - X_val.isna().sum().sum()
                ),
                "test_cells_masked": int(
                    X_test_masked.isna().sum().sum()
                    - X_test.isna().sum().sum()
                ),
                "train_rows": len(X_train),
                "val_rows": len(X_val),
                "test_rows": len(X_test),
            })

            # --------------------------------------------------
            # Classical models
            # --------------------------------------------------

            classical_models = build_classical_models(seed)

            for model_name, model in classical_models.items():
                print(f"\nTraining {model_name}...")

                model.fit(X_train_masked, y_train)

                train_predictions = model.predict(X_train_masked)
                val_predictions = model.predict(X_val_masked)
                test_predictions = model.predict(X_test_masked)

                from sklearn.metrics import f1_score

                train_f1 = f1_score(
                    y_train,
                    train_predictions,
                    average="macro",
                    zero_division=0,
                )
                val_f1 = f1_score(
                    y_val,
                    val_predictions,
                    average="macro",
                    zero_division=0,
                )

                record_result(
                    results,
                    seed,
                    missing_rate,
                    model_name,
                    y_test,
                    test_predictions,
                    train_macro_f1=train_f1,
                    val_macro_f1=val_f1,
                )

                print(
                    f"{model_name} test accuracy: "
                    f"{results[-1]['accuracy']:.4f}"
                )

            # --------------------------------------------------
            # NAIM
            # --------------------------------------------------

            print("\nTraining NAIM...")

            (
                naim_predictions,
                naim_train_f1,
                naim_val_f1,
                best_epoch,
            ) = train_and_predict_naim(
                X_train_masked,
                y_train,
                X_val_masked,
                y_val,
                X_test_masked,
                y_test,
                seed,
            )

            record_result(
                results,
                seed,
                missing_rate,
                "NAIM",
                y_test,
                naim_predictions,
                train_macro_f1=naim_train_f1,
                val_macro_f1=naim_val_f1,
                best_epoch=best_epoch,
            )

            print(
                f"NAIM test accuracy: "
                f"{results[-1]['accuracy']:.4f}; "
                f"best epoch: {best_epoch}"
            )

            # Save progress after each rate in case a later run fails.
            pd.DataFrame(results).to_csv(
                OUTPUT_DIR / "results_by_seed.csv",
                index=False,
            )
            pd.DataFrame(mask_audit).to_csv(
                OUTPUT_DIR / "mask_audit.csv",
                index=False,
            )

    # Save final results, summary, and plots.
    save_summary_and_plots(results)

    print("\nExperiment completed.")
    print(f"Results directory: {OUTPUT_DIR}")

    return pd.DataFrame(results)


if __name__ == "__main__":
    run_experiments()
