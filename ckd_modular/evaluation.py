
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
)

from config import OUTPUT_DIR


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.

    Label 0 = ckd
    Label 1 = notckd
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_ckd": recall_score(
            y_true,
            y_pred,
            labels=[0],
            average=None,
            zero_division=0,
        )[0],
        "recall_notckd": recall_score(
            y_true,
            y_pred,
            labels=[1],
            average=None,
            zero_division=0,
        )[0],
    }


def save_summary_and_plots(results):
    """
    Save per-experiment results, summary statistics,
    and plots.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    # Save all individual experiment results.
    results_path = OUTPUT_DIR / "results_by_seed.csv"
    results_df.to_csv(results_path, index=False)

    # Calculate mean and sample standard deviation across seeds.
    metric_columns = [
        "accuracy",
        "macro_f1",
        "recall_ckd",
        "recall_notckd",
    ]

    summary = (
        results_df
        .groupby(["model", "missing_rate"])[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten the multi-level column names.
    summary.columns = [
        "_".join(str(part) for part in column if part != "")
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    summary_path = OUTPUT_DIR / "summary_mean_std.csv"
    summary.to_csv(summary_path, index=False)

    # Generate one plot per metric.
    plot_metrics = [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro-F1"),
        ("recall_ckd", "Recall - CKD"),
        ("recall_notckd", "Recall - Not CKD"),
    ]

    for metric, title in plot_metrics:
        plt.figure(figsize=(9, 6))

        for model_name in results_df["model"].unique():
            model_data = (
                summary[summary["model"] == model_name]
                .sort_values("missing_rate")
            )

            mean_column = f"{metric}_mean"
            std_column = f"{metric}_std"

            plt.errorbar(
                model_data["missing_rate"],
                model_data[mean_column],
                yerr=model_data[std_column].fillna(0),
                marker="o",
                capsize=4,
                label=model_name,
            )

        plt.xlabel("Additional Missing Rate")
        plt.ylabel(title)
        plt.title(f"{title} vs. Additional Missing Rate")
        plt.xticks([0.10, 0.30, 0.50], ["10%", "30%", "50%"])
        plt.ylim(0, 1.05)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_path = OUTPUT_DIR / f"{metric}.png"
        plt.savefig(plot_path, dpi=200)
        plt.close()

    print(f"Individual results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Plots saved to: {OUTPUT_DIR}")
