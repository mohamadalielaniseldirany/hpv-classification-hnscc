import argparse
import html
import os
from pathlib import Path
import subprocess
import sys
import webbrowser

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


# This script cleans only the three original/base input CSV files.
# It does not change train.csv, validation.csv, or test.csv.

ROOT = Path(__file__).resolve().parent
TARGET_COLUMN = "HPV Status"
METADATA_COLUMNS = {"PatientID", "Split"}


DATASETS = [
    {
        "name": "Clinical_HECKTOR",
        "input_csv": ROOT / "Clinical_DLonly" / "HECKTOR_2025_Training_Task_3.csv",
        "output_csv": ROOT / "Clinical_DLonly" / "HECKTOR_2025_Training_Task_3_imputed.csv",
        "plot_dir": ROOT / "Clinical_DLonly" / "CLINICAL_ONLY_PLOTS" / "imputation_preprocessing",
    },
    {
        "name": "PET_features",
        "input_csv": ROOT / "PET_DLonly" / "pet_features.csv",
        "output_csv": ROOT / "PET_DLonly" / "pet_features_imputed.csv",
        "plot_dir": ROOT / "PET_DLonly" / "PET_ONLY_PLOTS" / "imputation_preprocessing",
    },
    {
        "name": "Clinical_PET_merged",
        "input_csv": ROOT / "Clinical_PET_DLonly" / "merged_clinical_pet.csv",
        "output_csv": ROOT / "Clinical_PET_DLonly" / "merged_clinical_pet_imputed.csv",
        "plot_dir": ROOT / "Clinical_PET_DLonly" / "CLINICALPET_COMBINED_PLOTS" / "imputation_preprocessing",
    },
]


def get_feature_columns(df):
    """Return columns that are real model features, not IDs, split labels, or target."""
    ignored = set(METADATA_COLUMNS)
    if TARGET_COLUMN in df.columns:
        ignored.add(TARGET_COLUMN)
    return [column for column in df.columns if column not in ignored]


def first_mode(series):
    """Return the most frequent observed value in a column."""
    modes = series.dropna().mode()
    if modes.empty:
        return "Unknown" if series.dtype == "object" else 0
    return modes.iloc[0]


def restore_integer_like_columns(df):
    """Convert float columns back to integer when every value is integer-like."""
    for column in df.columns:
        if pd.api.types.is_float_dtype(df[column]) and df[column].notna().all():
            integer_version = df[column].astype("int64")
            if (df[column] == integer_version).all():
                df[column] = integer_version
    return df


def save_missingness_plot(before_missing, after_missing, output_path, dataset_name):
    """Create a simple before/after bar chart of missing values by column."""
    plot_data = pd.DataFrame(
        {
            "Before handling": before_missing,
            "After handling": after_missing,
        }
    )

    # Show columns that had missing values first. If there were none, show all columns.
    important_columns = plot_data[plot_data["Before handling"] > 0].index.tolist()
    if not important_columns:
        important_columns = plot_data.index.tolist()

    plot_data = plot_data.loc[important_columns]

    height = max(4, min(12, 0.45 * len(plot_data) + 2))
    ax = plot_data.plot(kind="bar", figsize=(12, height), color=["#C44E52", "#4C72B0"])
    ax.set_title(f"{dataset_name}: Missing Values Before and After Handling")
    ax.set_xlabel("Column")
    ax.set_ylabel("Number of missing values")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_non_numeric_plot(before_count, after_count, output_path, dataset_name):
    """Create a small plot showing non-numeric feature columns before/after."""
    plot_data = pd.Series(
        {
            "Before handling": before_count,
            "After handling": after_count,
        }
    )
    ax = plot_data.plot(kind="bar", figsize=(6, 4), color=["#C44E52", "#55A868"])
    ax.set_title(f"{dataset_name}: Non-Numeric Feature Columns")
    ax.set_ylabel("Column count")
    ax.tick_params(axis="x", rotation=0)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_popup_viewer(result):
    """Create one browser-friendly page that shows the explanation and plots."""
    viewer_path = Path(result["plot_dir"]) / f"{result['dataset']}_popup_viewer.html"
    report_text = Path(result["report_txt"]).read_text(encoding="utf-8")
    missingness_plot = Path(result["missingness_plot"]).name
    non_numeric_plot = Path(result["non_numeric_plot"]).name
    method_csv = Path(result["method_csv"]).name

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(result["dataset"])} preprocessing results</title>
  <style>
    body {{
      margin: 32px;
      font-family: Arial, sans-serif;
      color: #1f2933;
      background: #f7f7f5;
    }}
    h1, h2 {{
      margin-bottom: 8px;
    }}
    .section {{
      margin: 0 0 28px;
      padding: 18px;
      background: #ffffff;
      border: 1px solid #d8d8d2;
      border-radius: 6px;
    }}
    img {{
      display: block;
      max-width: 100%;
      height: auto;
      border: 1px solid #c9c9c3;
      background: #ffffff;
    }}
    pre {{
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 14px;
    }}
    a {{
      color: #1b5e8c;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(result["dataset"])} preprocessing results</h1>
  <div class="section">
    <h2>Explanation</h2>
    <pre>{html.escape(report_text)}</pre>
    <p><a href="{html.escape(method_csv)}">Open method CSV</a></p>
  </div>
  <div class="section">
    <h2>Missing Values Before and After Handling</h2>
    <img src="{html.escape(missingness_plot)}" alt="Missing values before and after handling">
  </div>
  <div class="section">
    <h2>Non-Numeric Columns Before and After Handling</h2>
    <img src="{html.escape(non_numeric_plot)}" alt="Non-numeric columns before and after handling">
  </div>
</body>
</html>
"""
    viewer_path.write_text(html_text, encoding="utf-8")
    return viewer_path


def preprocess_dataset(config):
    """Load one base CSV, handle missing/categorical values, and save outputs."""
    input_csv = config["input_csv"]
    output_csv = config["output_csv"]
    plot_dir = config["plot_dir"]
    dataset_name = config["name"]
    plot_dir.mkdir(parents=True, exist_ok=True)

    original_df = pd.read_csv(input_csv)
    processed_df = original_df.copy()

    original_feature_columns = get_feature_columns(original_df)
    before_missing = original_df.isna().sum()
    before_non_numeric = [
        column
        for column in original_feature_columns
        if not pd.api.types.is_numeric_dtype(original_df[column])
    ]

    imputation_rows = []

    for column in original_feature_columns:
        missing_count = int(processed_df[column].isna().sum())
        if missing_count == 0:
            continue

        missing_indicator = f"{column}_missing"

        if column == "M-stage":
            processed_df[column] = processed_df[column].fillna("Unknown")
            imputation_rows.append(
                {
                    "column": column,
                    "missing_values_filled": missing_count,
                    "method": "Filled missing M-stage with 'Unknown'",
                    "reason": "M-stage is categorical, so missing values are kept as an explicit category before one-hot encoding.",
                }
            )
        else:
            processed_df[missing_indicator] = original_df[column].isna().astype(int)
            fill_value = first_mode(processed_df[column])
            processed_df[column] = processed_df[column].fillna(fill_value)
            imputation_rows.append(
                {
                    "column": column,
                    "missing_values_filled": missing_count,
                    "method": f"Mode imputation using value {fill_value}",
                    "reason": "The column is a coded clinical category/ordinal value, so the most frequent observed value preserves a valid category.",
                }
            )

    if "M-stage" in processed_df.columns:
        processed_df = pd.get_dummies(
            processed_df,
            columns=["M-stage"],
            prefix="M-stage",
            dummy_na=False,
            dtype=int,
        )
        imputation_rows.append(
            {
                "column": "M-stage",
                "missing_values_filled": int(original_df["M-stage"].isna().sum()),
                "method": "One-hot encoding after missing values were filled as 'Unknown'",
                "reason": "Keras MLP models need numeric inputs; one-hot encoding converts the categorical M-stage values into binary numeric columns.",
            }
        )

    processed_df = restore_integer_like_columns(processed_df)

    processed_feature_columns = get_feature_columns(processed_df)
    after_missing = processed_df.isna().sum()
    after_non_numeric = [
        column
        for column in processed_feature_columns
        if not pd.api.types.is_numeric_dtype(processed_df[column])
    ]

    processed_df.to_csv(output_csv, index=False)

    missing_summary = pd.DataFrame(
        {
            "column": before_missing.index,
            "missing_before": before_missing.values,
            "missing_after": [int(after_missing.get(column, 0)) for column in before_missing.index],
        }
    )
    missing_summary.to_csv(plot_dir / f"{dataset_name}_missingness_before_after.csv", index=False)

    non_numeric_summary = pd.DataFrame(
        {
            "stage": ["before_handling", "after_handling"],
            "non_numeric_feature_count": [len(before_non_numeric), len(after_non_numeric)],
            "non_numeric_feature_columns": [
                "; ".join(before_non_numeric) if before_non_numeric else "None",
                "; ".join(after_non_numeric) if after_non_numeric else "None",
            ],
        }
    )
    non_numeric_summary.to_csv(plot_dir / f"{dataset_name}_non_numeric_before_after.csv", index=False)

    imputation_summary = pd.DataFrame(imputation_rows)
    if imputation_summary.empty:
        imputation_summary = pd.DataFrame(
            [
                {
                    "column": "None",
                    "missing_values_filled": 0,
                    "method": "No missing-value imputation was needed",
                    "reason": "The base CSV already had no missing values in model feature columns.",
                }
            ]
        )
    imputation_summary.to_csv(plot_dir / f"{dataset_name}_imputation_methods.csv", index=False)

    save_missingness_plot(
        before_missing,
        after_missing,
        plot_dir / f"{dataset_name}_missing_values_before_after.png",
        dataset_name,
    )
    save_non_numeric_plot(
        len(before_non_numeric),
        len(after_non_numeric),
        plot_dir / f"{dataset_name}_non_numeric_before_after.png",
        dataset_name,
    )

    method_lines = []
    for _, row in imputation_summary.iterrows():
        method_lines.extend(
            [
                f"- Column: {row['column']}",
                f"  Missing values handled: {row['missing_values_filled']}",
                f"  Method used: {row['method']}",
                f"  How it works / why it was used: {row['reason']}",
            ]
        )

    if int(before_missing.sum()) == 0:
        missing_handling_lines = [
            "- No missing values were found in this base CSV, so no imputation was needed."
        ]
    else:
        missing_handling_lines = []
        mode_columns = [
            row["column"]
            for _, row in imputation_summary.iterrows()
            if str(row["method"]).startswith("Mode imputation")
        ]
        if mode_columns:
            missing_handling_lines.append(
                "- Missing coded clinical columns were filled with the mode, meaning the most frequent observed value in that column."
            )
            missing_handling_lines.append(
                "- A *_missing indicator column was added for those columns, so the model can still see where data was originally absent."
            )
        if "M-stage" in original_df.columns and int(original_df["M-stage"].isna().sum()) > 0:
            missing_handling_lines.append(
                "- Missing M-stage values were filled with 'Unknown', then M-stage was one-hot encoded into numeric binary columns."
            )

    if before_non_numeric:
        non_numeric_handling_lines = []
        if "M-stage" in before_non_numeric:
            non_numeric_handling_lines.append("- M-stage was converted into numeric one-hot columns.")
        remaining_non_numeric = [column for column in before_non_numeric if column != "M-stage"]
        if remaining_non_numeric:
            non_numeric_handling_lines.append(
                f"- These non-numeric model feature columns still need attention: {remaining_non_numeric}"
            )
    else:
        non_numeric_handling_lines = [
            "- No non-numeric model feature columns were found, so no encoding was needed."
        ]

    non_numeric_handling_lines.append(
        "- PatientID and Split were preserved as metadata columns and should be excluded from MLP features."
    )

    report_lines = [
        f"Dataset: {dataset_name}",
        f"Input CSV: {input_csv}",
        f"Output CSV: {output_csv}",
        "",
        "Important rule:",
        "- Original CSV files were not modified.",
        "- train.csv, validation.csv, and test.csv were not read or changed.",
        "- PatientID and Split were treated as metadata, not model features.",
        "",
        "How missing values were handled:",
        *missing_handling_lines,
        "",
        "How non-numeric values were handled:",
        *non_numeric_handling_lines,
        "",
        "Column-by-column methods used:",
        *method_lines,
        "",
        f"Missing values before: {int(before_missing.sum())}",
        f"Missing values after: {int(after_missing.sum())}",
        f"Non-numeric feature columns before: {before_non_numeric if before_non_numeric else 'None'}",
        f"Non-numeric feature columns after: {after_non_numeric if after_non_numeric else 'None'}",
    ]
    (plot_dir / f"{dataset_name}_imputation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    result = {
        "dataset": dataset_name,
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "missing_before": int(before_missing.sum()),
        "missing_after": int(after_missing.sum()),
        "non_numeric_features_before": before_non_numeric,
        "non_numeric_features_after": after_non_numeric,
        "plot_dir": str(plot_dir),
        "method_csv": str(plot_dir / f"{dataset_name}_imputation_methods.csv"),
        "report_txt": str(plot_dir / f"{dataset_name}_imputation_report.txt"),
        "missingness_plot": str(plot_dir / f"{dataset_name}_missing_values_before_after.png"),
        "non_numeric_plot": str(plot_dir / f"{dataset_name}_non_numeric_before_after.png"),
    }
    result["viewer_html"] = str(save_popup_viewer(result))
    return result


def print_preprocessing_result(result):
    """Print the important output locations after one dataset is processed."""
    print(f"{result['dataset']}")
    print(f"  New CSV: {result['output_csv']}")
    print(f"  Plot/report folder: {result['plot_dir']}")
    print(f"  Method table: {result['method_csv']}")
    print(f"  Explanation report: {result['report_txt']}")
    print(f"  Popup viewer: {result['viewer_html']}")
    print(f"  Missing-value plot: {result['missingness_plot']}")
    print(f"  Non-numeric plot: {result['non_numeric_plot']}")
    print(f"  Missing values: {result['missing_before']} -> {result['missing_after']}")
    print(f"  Non-numeric feature columns: {result['non_numeric_features_before']} -> {result['non_numeric_features_after']}")
    print()


def print_explanation_report(result):
    """Print the explanation report in the terminal/code-runner output."""
    report_path = Path(result["report_txt"])
    print("=" * 80)
    print(f"Explanation report: {report_path}")
    print("=" * 80)
    print(report_path.read_text(encoding="utf-8"))
    print("=" * 80)
    print()


def open_file(path):
    """Open one saved output file with the default app on this computer."""
    output_path = Path(path)
    if not output_path.exists():
        print(f"Could not open missing file: {output_path}")
        return

    try:
        if hasattr(os, "startfile"):
            os.startfile(str(output_path))
        else:
            webbrowser.open(output_path.resolve().as_uri())
    except OSError as error:
        print(f"Could not automatically open {output_path}: {error}")


def open_in_browser(path):
    """Open a local HTML file in the default browser."""
    output_path = Path(path)
    if not output_path.exists():
        print(f"Could not open missing file: {output_path}")
        return

    # On Windows, os.startfile usually gives the most reliable visible popup.
    if hasattr(os, "startfile"):
        open_file(output_path)
        return

    opened = webbrowser.open_new_tab(output_path.resolve().as_uri())
    if not opened:
        open_file(output_path)


def open_html_outputs(result):
    """Open the saved HTML viewer and output folder."""
    print("Opening the browser popup viewer with the explanation and plots...")
    open_in_browser(result["viewer_html"])
    open_file(result["plot_dir"])
    print(f"Popup viewer path: {result['viewer_html']}")
    print(f"Output folder path: {result['plot_dir']}")


def open_preprocessing_outputs(result):
    """Show the saved plots in a Matplotlib app/window after preprocessing."""
    viewer_script = ROOT / "show_imputation_plots_matplotlib.py"
    command = [
        sys.executable,
        str(viewer_script),
        "--dataset",
        result["dataset"],
        "--report",
        result["report_txt"],
        "--missingness-plot",
        result["missingness_plot"],
        "--non-numeric-plot",
        result["non_numeric_plot"],
    ]

    print("Opening the Matplotlib plot viewer...")
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode != 0:
        print("Matplotlib viewer could not open. Falling back to the HTML viewer.")
        open_html_outputs(result)


def add_popup_argument(parser):
    """Add a command-line option for people who want to save only, without popups."""
    parser.add_argument(
        "--no-popups",
        action="store_true",
        help="Save outputs but do not automatically open the plots/report.",
    )
    return parser


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess all three Week 9 base input CSV files."
    )
    add_popup_argument(parser)
    args = parser.parse_args()

    results = []
    for config in DATASETS:
        results.append(preprocess_dataset(config))

    print("Finished preprocessing the three base input CSV files.\n")
    for result in results:
        print_preprocessing_result(result)
        print_explanation_report(result)

    if not args.no_popups:
        for result in results:
            open_preprocessing_outputs(result)


if __name__ == "__main__":
    main()
