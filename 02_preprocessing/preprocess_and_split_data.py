# Import libraries for file paths, interpreter checks, and dataframes.
from pathlib import Path
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split


# Define project folders and the expected Python environment.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT.parent / "DATA_SET"
OUTPUT_DIR = Path(__file__).resolve().parent
EXPECTED_INTERPRETER = PROJECT_ROOT / "myenv" / "Scripts" / "python.exe"
MPL_CONFIG_DIR = OUTPUT_DIR / "matplotlib_cache"
MISSING_REPORTS_DIR = OUTPUT_DIR / "missing_data_reports"
AGE_REPORTS_DIR = OUTPUT_DIR / "age_distribution_reports"
SHOW_PLOTS = os.environ.get(
    "SHOW_PLOTS",
    os.environ.get("SHOW_MISSING_PLOTS", "1"),
) != "0"

# Put Matplotlib's cache inside the project so plotting works without trying to
# write into a locked Windows user folder.
MPL_CONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

import matplotlib

if SHOW_PLOTS:
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Define column names used repeatedly in the preprocessing pipeline.
ID_COLUMN = "PatientID"
TARGET_COLUMN = "HPV"
ORIGINAL_TARGET_COLUMN = "HPV Status"
AGE_COLUMN = "Age"
RANDOM_STATE = 42

# Define input and output CSV file names.
HECKTOR_FILE_NAME = "HECKTOR_2025_Training_Task_3.csv"
OUTPUT_FILE_NAME = "hecktor_priority_union.csv"
NO_RADIOMICS_OUTPUT_FILE_NAME = "hecktor_patients_id_with_no_radiomic_features.csv"
RADIOMICS_IDS_NOT_IN_HECKTOR_FILE_NAME = "radiomics_patient_ids_not_in_hecktor.csv"
TRAIN_FILE_NAME = "train.csv"
VALIDATION_FILE_NAME = "validation.csv"
TEST_FILE_NAME = "test.csv"


# Confirm that the script is running with the project virtual environment.
def check_interpreter() -> None:
    """Print the active Python interpreter and confirm the project venv."""
    current_interpreter = Path(sys.executable).resolve()
    expected_interpreter = EXPECTED_INTERPRETER.resolve()

    print("Current Python interpreter:")
    print(current_interpreter)

    if str(current_interpreter).lower() == str(expected_interpreter).lower():
        print("Venv check: OK, you are using the project virtual environment.\n")
    else:
        print("Venv check: WARNING, you are not using the project virtual environment.")
        print("Expected interpreter:")
        print(f"{expected_interpreter}\n")


# Load one CSV from the DATA_SET folder and standardize its headers.
def load_csv(file_name: str) -> pd.DataFrame:
    """Load one CSV file from the DATA_SET folder."""
    file_path = DATA_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {file_path}\n"
            "Put the CSV file in DATA_SET, or update DATA_DIR."
        )

    dataframe = pd.read_csv(file_path)
    dataframe.columns = dataframe.columns.str.strip()
    return dataframe


# Standardize PatientID values and keep one row per patient before merging.
def prepare_for_merge(dataframe: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Clean PatientID and keep one row per patient before merging."""
    if ID_COLUMN not in dataframe.columns:
        raise ValueError(f"{file_name} does not contain a {ID_COLUMN} column.")

    # Work on a copy so the original imported dataframe is not modified.
    prepared_dataframe = dataframe.copy()
    prepared_dataframe[ID_COLUMN] = (
        prepared_dataframe[ID_COLUMN].astype("string").str.strip()
    )

    # Remove rows without a usable PatientID.
    prepared_dataframe = prepared_dataframe[
        prepared_dataframe[ID_COLUMN].notna()
        & (prepared_dataframe[ID_COLUMN] != "")
    ]

    # If a patient appears more than once, keep the first row and report it.
    duplicated_ids = prepared_dataframe[ID_COLUMN].duplicated(keep=False)
    if duplicated_ids.any():
        duplicate_values = sorted(prepared_dataframe.loc[duplicated_ids, ID_COLUMN].unique())
        print(f"\nWarning: duplicate patient IDs found in {file_name}:")
        print(", ".join(duplicate_values))
        print("Keeping the first row for each duplicated patient ID.\n")
        prepared_dataframe = prepared_dataframe.drop_duplicates(
            subset=ID_COLUMN,
            keep="first",
        )

    return prepared_dataframe


# Extract clean unique patient IDs for overlap checks.
def get_patient_ids(dataframe: pd.DataFrame, file_name: str) -> pd.Series:
    """Return clean, unique patient IDs from one dataframe."""
    if ID_COLUMN not in dataframe.columns:
        raise ValueError(f"{file_name} does not contain a {ID_COLUMN} column.")

    patient_ids = dataframe[ID_COLUMN].astype("string").str.strip()
    patient_ids = patient_ids[patient_ids.notna() & (patient_ids != "")]
    return patient_ids.drop_duplicates()


# Prefix feature columns so CT, PET, and PET/CT names stay separate after merging.
def add_source_prefix(dataframe: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Prefix feature columns so features from each file stay clear."""
    source_name = Path(file_name).stem
    rename_map = {
        column_name: f"{source_name}__{column_name}"
        for column_name in dataframe.columns
        if column_name != ID_COLUMN
    }
    return dataframe.rename(columns=rename_map)


# Merge CT, PET, and PET/CT feature files onto the HECKTOR clinical table.
def build_hecktor_priority_union(
    clinical_dataframe: pd.DataFrame,
    feature_files: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Merge all feature files onto the HECKTOR clinical file by PatientID."""
    # Start from HECKTOR patients so the final table is HECKTOR-priority.
    combined_dataframe = prepare_for_merge(clinical_dataframe, HECKTOR_FILE_NAME)

    # Add each radiomics feature table by PatientID using a left merge.
    for file_name, feature_dataframe in feature_files:
        prepared_features = prepare_for_merge(feature_dataframe, file_name)
        prepared_features = add_source_prefix(prepared_features, file_name)

        combined_dataframe = combined_dataframe.merge(
            prepared_features,
            on=ID_COLUMN,
            how="left",
        )

    return combined_dataframe


# Separate patients with radiomics features from HECKTOR patients without them.
def split_by_radiomics_availability(
    combined_dataframe: pd.DataFrame,
    feature_files: list[tuple[str, pd.DataFrame]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate patients with radiomics from patients with no radiomics values."""
    radiomics_columns = []

    # Collect all radiomics feature columns from the prefixed feature tables.
    for file_name, _ in feature_files:
        source_prefix = f"{Path(file_name).stem}__"
        radiomics_columns.extend(
            column_name
            for column_name in combined_dataframe.columns
            if column_name.startswith(source_prefix)
        )

    # Keep patients with at least one radiomics value; save the rest as excluded IDs.
    has_radiomics_features = combined_dataframe[radiomics_columns].notna().any(axis=1)
    patients_with_radiomics = combined_dataframe.loc[has_radiomics_features].copy()
    patients_without_radiomics = combined_dataframe.loc[
        ~has_radiomics_features,
        [ID_COLUMN],
    ].copy()

    patients_without_radiomics = patients_without_radiomics.sort_values(
        ID_COLUMN,
        ignore_index=True,
    )

    return patients_with_radiomics, patients_without_radiomics


# List radiomics patients that are not present in the HECKTOR clinical CSV.
def build_radiomics_ids_not_in_hecktor(
    clinical_dataframe: pd.DataFrame,
    feature_files: list[tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """List radiomics patient IDs that are dropped because they are not in HECKTOR."""
    # HECKTOR IDs define the priority patient list for this task.
    hecktor_ids = set(get_patient_ids(clinical_dataframe, HECKTOR_FILE_NAME))
    dropped_ids_by_file = {}

    # For each radiomics file, report IDs that do not match any HECKTOR patient.
    for file_name, feature_dataframe in feature_files:
        feature_ids = set(get_patient_ids(feature_dataframe, file_name))
        dropped_ids_by_file[file_name] = pd.Series(
            sorted(feature_ids - hecktor_ids),
            dtype="string",
        )

    return pd.DataFrame(dropped_ids_by_file)


# Make sure the target column has the expected beginner-friendly name: HPV.
def standardize_hpv_target_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of the dataframe with the target column named HPV."""
    standardized_dataframe = dataframe.copy()

    if TARGET_COLUMN in standardized_dataframe.columns:
        return standardized_dataframe

    if ORIGINAL_TARGET_COLUMN in standardized_dataframe.columns:
        return standardized_dataframe.rename(
            columns={ORIGINAL_TARGET_COLUMN: TARGET_COLUMN}
        )

    raise ValueError(
        f"Target column not found. Expected {TARGET_COLUMN} or "
        f"{ORIGINAL_TARGET_COLUMN}."
    )


# Split the dataset into train, validation, and test while preserving HPV ratios.
def split_train_validation_test(
    df: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified 70% train, 15% validation, and 15% test datasets."""
    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    if df[target_column].isna().any():
        raise ValueError(
            f"{target_column} contains missing values. Fill or remove them "
            "before stratified splitting."
        )

    # X contains all input features; y contains only the HPV target labels.
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Stratified splitting is important because it keeps the HPV=0 and HPV=1
    # proportions nearly the same in every split.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        train_size=0.70,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # The temporary 30% is split equally into validation and test:
    # 15% validation and 15% test.
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # Add y back to each feature table so each saved CSV contains both the
    # input features and the HPV target column.
    train_df = X_train.copy()
    train_df[target_column] = y_train

    validation_df = X_validation.copy()
    validation_df[target_column] = y_validation

    test_df = X_test.copy()
    test_df[target_column] = y_test

    return train_df, validation_df, test_df


# Print an easy-to-read shape and class-distribution summary for one split.
def print_split_summary(
    split_name: str,
    split_dataframe: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> None:
    """Print the size and HPV class distribution for one dataset split."""
    print(f"\n{split_name} set shape:")
    print(split_dataframe.shape)

    print(f"\n{split_name} set {target_column} class distribution:")
    print(split_dataframe[target_column].value_counts(normalize=True).mul(100).round(2))
    print("\nClass counts:")
    print(split_dataframe[target_column].value_counts())


# Build a column-by-column missing-value summary for one dataset.
def build_missing_data_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return missing counts and percentages for every column."""
    missing_values = dataframe.isna().sum()
    non_missing_values = dataframe.notna().sum()

    if len(dataframe) == 0:
        missing_percent = missing_values * 0
    else:
        missing_percent = missing_values.div(len(dataframe)).mul(100)

    summary_dataframe = pd.DataFrame(
        {
            "column": dataframe.columns,
            "data_type": dataframe.dtypes.astype(str).values,
            "missing_values": missing_values.values,
            "missing_percent": missing_percent.round(2).values,
            "non_missing_values": non_missing_values.values,
        }
    )

    return summary_dataframe.sort_values(
        by=["missing_values", "column"],
        ascending=[False, True],
        ignore_index=True,
    )


# Save a visual missing-data report for one split.
def save_missing_data_plot(
    summary_dataframe: pd.DataFrame,
    split_name: str,
    output_path: Path,
    show_plot: bool = SHOW_PLOTS,
) -> bool:
    """Save and optionally display a plot for columns with missing values."""
    plot_dataframe = summary_dataframe[
        summary_dataframe["missing_values"] > 0
    ].sort_values(
        by="missing_percent",
        ascending=True,
    )

    if plot_dataframe.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.text(
            0.5,
            0.5,
            "No missing values found",
            ha="center",
            va="center",
            fontsize=14,
        )
        axis.set_axis_off()
    else:
        # Only plotting columns with missing values makes the chart readable.
        figure_height = max(4, len(plot_dataframe) * 0.8)
        figure, axis = plt.subplots(figsize=(12, figure_height))
        bars = axis.barh(
            plot_dataframe["column"],
            plot_dataframe["missing_percent"],
            color="#4C78A8",
        )

        for bar, missing_count, missing_percent in zip(
            bars,
            plot_dataframe["missing_values"],
            plot_dataframe["missing_percent"],
        ):
            axis.text(
                bar.get_width() + 1,
                bar.get_y() + bar.get_height() / 2,
                f"{missing_count} missing ({missing_percent:.2f}%)",
                va="center",
                fontsize=10,
            )

        axis.set_title(f"{split_name} set missing values")
        axis.set_xlabel("Missing values (%)")
        axis.set_ylabel("Columns with missing data")
        axis.set_xlim(0, max(30, plot_dataframe["missing_percent"].max() + 15))
        axis.grid(axis="x", linestyle="--", alpha=0.35)

    figure.canvas.manager.set_window_title(f"{split_name} missing-data plot")

    figure.tight_layout()

    try:
        figure.savefig(output_path, dpi=150)
    except PermissionError:
        print(f"\nCould not save because this plot is open or locked: {output_path}")
        print("Close the file and run the script again to overwrite it.")
        return False

    if not show_plot:
        plt.close(figure)

    return True


# Build simple statistics for the Age column in one split.
def build_age_summary(
    dataframe: pd.DataFrame,
    age_column: str = AGE_COLUMN,
) -> pd.DataFrame:
    """Return count, mean, median, min, max, and quartiles for Age."""
    if age_column not in dataframe.columns:
        raise ValueError(f"Age column not found: {age_column}")

    ages = pd.to_numeric(dataframe[age_column], errors="coerce")

    summary = {
        "count": ages.count(),
        "missing_values": ages.isna().sum(),
        "mean": ages.mean(),
        "median": ages.median(),
        "standard_deviation": ages.std(),
        "minimum": ages.min(),
        "25_percentile": ages.quantile(0.25),
        "75_percentile": ages.quantile(0.75),
        "maximum": ages.max(),
    }

    return pd.DataFrame([summary]).round(2)


# Save a histogram that shows how patient ages are distributed in one split.
def save_age_distribution_plot(
    split_dataframe: pd.DataFrame,
    split_name: str,
    output_path: Path,
    age_column: str = AGE_COLUMN,
    show_plot: bool = SHOW_PLOTS,
) -> bool:
    """Save and optionally display an age-distribution histogram."""
    if age_column not in split_dataframe.columns:
        raise ValueError(f"Age column not found: {age_column}")

    ages = pd.to_numeric(split_dataframe[age_column], errors="coerce").dropna()

    figure, axis = plt.subplots(figsize=(10, 6))
    figure.canvas.manager.set_window_title(f"{split_name} age distribution")

    if ages.empty:
        axis.text(
            0.5,
            0.5,
            "No valid age values found",
            ha="center",
            va="center",
            fontsize=14,
        )
        axis.set_axis_off()
    else:
        bins = min(10, max(5, ages.nunique()))
        counts, _, bars = axis.hist(
            ages,
            bins=bins,
            color="#59A14F",
            edgecolor="white",
        )

        # Add the number of patients above each bar so the plot can be read
        # without guessing from the y-axis.
        for count, bar in zip(counts, bars):
            if count == 0:
                continue

            axis.text(
                bar.get_x() + bar.get_width() / 2,
                count,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        mean_age = ages.mean()
        median_age = ages.median()
        axis.axvline(
            mean_age,
            color="#E15759",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_age:.2f}",
        )
        axis.axvline(
            median_age,
            color="#F28E2B",
            linestyle=":",
            linewidth=2,
            label=f"Median: {median_age:.2f}",
        )

        axis.set_title(f"{split_name} set age distribution")
        axis.set_xlabel("Age")
        axis.set_ylabel("Number of patients")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.legend()

    figure.tight_layout()

    try:
        figure.savefig(output_path, dpi=150)
    except PermissionError:
        print(f"\nCould not save because this plot is open or locked: {output_path}")
        print("Close the file and run the script again to overwrite it.")
        return False

    if not show_plot:
        plt.close(figure)

    return True


# Print and save age-distribution reports for one split.
def analyze_age_distribution(
    split_name: str,
    split_dataframe: pd.DataFrame,
    report_directory: Path = AGE_REPORTS_DIR,
) -> None:
    """Print age statistics and save a CSV plus histogram plot."""
    report_directory.mkdir(exist_ok=True)
    safe_split_name = split_name.lower().replace(" ", "_")
    summary_path = report_directory / f"{safe_split_name}_age_summary.csv"
    plot_path = report_directory / f"{safe_split_name}_age_distribution.png"

    age_summary = build_age_summary(split_dataframe)
    summary_saved = save_csv(age_summary, summary_path)
    plot_saved = save_age_distribution_plot(split_dataframe, split_name, plot_path)

    print(f"\nAge distribution analysis for {split_name} set:")
    print(age_summary.to_string(index=False))

    if summary_saved:
        print(f"\nAge summary saved to: {summary_path}")
    if plot_saved:
        print(f"Age distribution plot saved to: {plot_path}")


# Print and save missing-value reports for one split.
def analyze_missing_data(
    split_name: str,
    split_dataframe: pd.DataFrame,
    report_directory: Path = MISSING_REPORTS_DIR,
) -> None:
    """Print a missing-data summary and save a CSV plus plot."""
    report_directory.mkdir(exist_ok=True)
    safe_split_name = split_name.lower().replace(" ", "_")
    summary_path = report_directory / f"{safe_split_name}_missing_summary.csv"
    plot_path = report_directory / f"{safe_split_name}_missing_plot.png"

    summary_dataframe = build_missing_data_summary(split_dataframe)
    summary_saved = save_csv(summary_dataframe, summary_path)
    plot_saved = save_missing_data_plot(summary_dataframe, split_name, plot_path)

    total_cells = split_dataframe.shape[0] * split_dataframe.shape[1]
    total_missing_values = int(summary_dataframe["missing_values"].sum())
    columns_with_missing = summary_dataframe[
        summary_dataframe["missing_values"] > 0
    ]

    if total_cells == 0:
        total_missing_percent = 0
    else:
        total_missing_percent = (total_missing_values / total_cells) * 100

    print(f"\nMissing-data analysis for {split_name} set:")
    print(f"Shape: {split_dataframe.shape}")
    print(
        "Total missing cells: "
        f"{total_missing_values} / {total_cells} "
        f"({total_missing_percent:.2f}%)"
    )
    print(
        "Columns with missing values: "
        f"{columns_with_missing.shape[0]} / {split_dataframe.shape[1]}"
    )

    if columns_with_missing.empty:
        print("No missing values found.")
    else:
        print("\nColumns that contain missing values:")
        print(
            columns_with_missing[
                ["column", "missing_values", "missing_percent"]
            ].to_string(index=False)
        )

    if summary_saved:
        print(f"\nColumn-by-column missing summary saved to: {summary_path}")
    if plot_saved:
        print(f"Column-by-column missing plot saved to: {plot_path}")


# Open the plots in popup windows after all reports are ready.
def show_all_open_plots() -> None:
    """Display all currently open Matplotlib figures."""
    if not SHOW_PLOTS:
        return

    if plt.get_fignums():
        print("\nOpening plot windows.")
        print("Close the plot windows when you are finished viewing them.")
        plt.show()


# Save a CSV while giving a friendly message if the file is open in another app.
def save_csv(dataframe: pd.DataFrame, output_path: Path) -> bool:
    """Save a dataframe and keep the script running if a CSV is open elsewhere."""
    try:
        dataframe.to_csv(output_path, index=False)
    except PermissionError:
        print(f"\nCould not save because this file is open or locked: {output_path}")
        print("Close the file and run the script again to overwrite it.")
        return False

    return True


# Start the script by confirming the active Python interpreter.
check_interpreter()

# Step 1: Import the HECKTOR clinical CSV and the three radiomics feature CSVs.
df1 = load_csv(HECKTOR_FILE_NAME)
df2 = load_csv("ct_features.csv")
df3 = load_csv("pet_features.csv")
df4 = load_csv("petct_features.csv")

# Step 2: Keep feature files paired with their source filenames for merging.
feature_files = [
    ("ct_features.csv", df2),
    ("pet_features.csv", df3),
    ("petct_features.csv", df4),
]

# Step 3: Build the unified HECKTOR-priority dataset and remove no-radiomics patients.
combined_df = build_hecktor_priority_union(df1, feature_files)
combined_df, no_radiomics_patient_ids_df = split_by_radiomics_availability(
    combined_df,
    feature_files,
)
combined_df = standardize_hpv_target_column(combined_df)

# Step 4: Record radiomics IDs that are excluded because they are not in HECKTOR.
radiomics_ids_not_in_hecktor_df = build_radiomics_ids_not_in_hecktor(
    df1,
    feature_files,
)

# Step 5: Define where each output CSV will be saved.
combined_output_path = OUTPUT_DIR / OUTPUT_FILE_NAME
no_radiomics_output_path = OUTPUT_DIR / NO_RADIOMICS_OUTPUT_FILE_NAME
radiomics_ids_not_in_hecktor_output_path = (
    OUTPUT_DIR / RADIOMICS_IDS_NOT_IN_HECKTOR_FILE_NAME
)
train_output_path = OUTPUT_DIR / TRAIN_FILE_NAME
validation_output_path = OUTPUT_DIR / VALIDATION_FILE_NAME
test_output_path = OUTPUT_DIR / TEST_FILE_NAME

# Step 6: Split the final dataset into balanced train, validation, and test CSVs.
train_df, validation_df, test_df = split_train_validation_test(combined_df)

# Step 7: Save the excluded-ID reports, the unified dataset, and the split datasets.
save_csv(
    radiomics_ids_not_in_hecktor_df,
    radiomics_ids_not_in_hecktor_output_path,
)
save_csv(combined_df, combined_output_path)
save_csv(no_radiomics_patient_ids_df, no_radiomics_output_path)
train_saved = save_csv(train_df, train_output_path)
validation_saved = save_csv(validation_df, validation_output_path)
test_saved = save_csv(test_df, test_output_path)

# Step 8: Print dataset summaries and saved file locations.
print("\nCombined HECKTOR-priority CSV shape after dropping no-radiomics patients:")
print(combined_df.shape)

print("\nHECKTOR patients with no radiomics features:")
print(no_radiomics_patient_ids_df.shape[0])

print("\nRadiomics patients not found in HECKTOR:")
print(radiomics_ids_not_in_hecktor_df.count())

print_split_summary("Training", train_df)
print_split_summary("Validation", validation_df)
print_split_summary("Test", test_df)

# Before preprocessing, analyze missing values in each split. This gives us a
# baseline report that we can compare against after imputation/encoding later.
analyze_missing_data("Training", train_df)
analyze_missing_data("Validation", validation_df)
analyze_missing_data("Test", test_df)

# Also check whether the age distribution looks similar across the splits.
# This helps us see if one CSV accidentally got much younger or older patients.
analyze_age_distribution("Training", train_df)
analyze_age_distribution("Validation", validation_df)
analyze_age_distribution("Test", test_df)
show_all_open_plots()

# Balanced splitting helps avoid biased evaluation. If a validation or test set
# accidentally has too many samples from one HPV class, model performance can
# look better or worse than it really is.
if train_saved and validation_saved and test_saved:
    print("\nSaved split CSV files successfully:")
    print(train_output_path)
    print(validation_output_path)
    print(test_output_path)
