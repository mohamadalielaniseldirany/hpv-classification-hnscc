from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    roc_auc_score,
    roc_curve,
)


RANDOM_STATE = 42
THRESHOLD = 0.5
N_CALIBRATION_BINS = 10
MAX_SHAP_SAMPLES = 300
TOP_FEATURES_FOR_PLOTS = 5
OPEN_PLOTS_AFTER_SAVING = os.environ.get("OPEN_PLOTS", "1").strip().lower() in {"1", "true", "yes", "y"}

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Clinical_PET"
PLOTS_DIR = OUTPUT_DIR / "plots"

PATIENT_ID_COLUMN = "PatientID"
TARGET_COLUMN = "HPV Status"

CLINICAL_FEATURES = [
    "CenterID",
    "Age",
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "Treatment",
    "M-stage",
]

CLINICAL_CATEGORICAL_FEATURES = [
    "CenterID",
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "Treatment",
    "M-stage",
]

PET_EXCLUDE_NAMES = {
    "patientid",
    "patient_id",
    "subjectid",
    "subject_id",
    "caseid",
    "case_id",
    "id",
    "hpv",
    "hpv status",
    "hpv_status",
    "target",
    "label",
    "class",
    "outcome",
    "split",
}


def find_required_file(exact_names: list[str], patterns: list[str] | None = None) -> Path:
    files = [path for path in SCRIPT_DIR.iterdir() if path.is_file()]
    by_lower_name = {path.name.lower(): path for path in files}

    for name in exact_names:
        match = by_lower_name.get(name.lower())
        if match is not None:
            return match

    for pattern in patterns or []:
        matches = sorted(path for path in SCRIPT_DIR.glob(pattern) if path.is_file())
        if matches:
            return matches[0]

    expected = ", ".join(exact_names + (patterns or []))
    raise FileNotFoundError(f"Could not find required file in {SCRIPT_DIR}: {expected}")


def discover_files() -> dict[str, Path]:
    return {
        "train": find_required_file(["train.csv"]),
        "validation": find_required_file(["validation.csv"]),
        "test": find_required_file(["test.csv"]),
        "clinical": find_required_file(["HECKTOR_2025_Training_Task_3.csv"]),
        "pet": find_required_file(
            ["PET_features.csv", "pet_features.csv"],
            patterns=["*PET*features*.csv", "*pet*features*.csv"],
        ),
    }


def normalize_patient_ids(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if PATIENT_ID_COLUMN not in df.columns:
        raise ValueError(f"{source_name} is missing required column: {PATIENT_ID_COLUMN}")

    normalized = df[PATIENT_ID_COLUMN].astype("string").str.strip()
    missing_mask = normalized.isna() | normalized.eq("")
    if missing_mask.any():
        raise ValueError(
            f"{source_name} has {int(missing_mask.sum())} missing/blank PatientID values."
        )

    df = df.copy()
    df[PATIENT_ID_COLUMN] = normalized
    return df


def validate_unique_patient_ids(df: pd.DataFrame, source_name: str) -> None:
    duplicated = df[PATIENT_ID_COLUMN][df[PATIENT_ID_COLUMN].duplicated()].unique()
    if len(duplicated) > 0:
        preview = ", ".join(map(str, duplicated[:10]))
        raise ValueError(f"{source_name} has duplicate PatientID values, including: {preview}")


def read_split_patient_ids(path: Path, split_name: str) -> list[str]:
    split_df = pd.read_csv(path, usecols=[PATIENT_ID_COLUMN])
    split_df = normalize_patient_ids(split_df, split_name)
    validate_unique_patient_ids(split_df, split_name)
    return split_df[PATIENT_ID_COLUMN].tolist()


def validate_disjoint_splits(split_ids: dict[str, list[str]]) -> None:
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    split_names = list(split_sets)

    for index, left_name in enumerate(split_names):
        for right_name in split_names[index + 1 :]:
            overlap = split_sets[left_name] & split_sets[right_name]
            if overlap:
                preview = ", ".join(sorted(overlap)[:10])
                raise ValueError(
                    f"{left_name} and {right_name} splits overlap on PatientID values: {preview}"
                )


def load_clinical_source(path: Path) -> pd.DataFrame:
    clinical_df = pd.read_csv(path)
    clinical_df = normalize_patient_ids(clinical_df, path.name)

    required_columns = [PATIENT_ID_COLUMN, TARGET_COLUMN] + CLINICAL_FEATURES
    missing_columns = [col for col in required_columns if col not in clinical_df.columns]
    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required clinical columns: {', '.join(missing_columns)}"
        )

    clinical_df = clinical_df[required_columns].copy()
    validate_unique_patient_ids(clinical_df, path.name)
    return clinical_df


def is_pet_feature_column(column_name: str) -> bool:
    lower_name = column_name.strip().lower()
    if lower_name in PET_EXCLUDE_NAMES:
        return False
    if "hpv" in lower_name:
        return False
    return True


def load_pet_source(path: Path) -> tuple[pd.DataFrame, list[str]]:
    pet_df = pd.read_csv(path)
    pet_df = normalize_patient_ids(pet_df, path.name)

    pet_feature_columns = [
        column for column in pet_df.columns if column != PATIENT_ID_COLUMN and is_pet_feature_column(column)
    ]
    if not pet_feature_columns:
        raise ValueError(f"No PET feature columns were found in {path.name}.")

    duplicate_clinical_names = sorted(set(pet_feature_columns) & set(CLINICAL_FEATURES))
    if duplicate_clinical_names:
        raise ValueError(
            "PET feature columns overlap with clinical feature names: "
            + ", ".join(duplicate_clinical_names)
        )

    pet_df = pet_df[[PATIENT_ID_COLUMN] + pet_feature_columns].copy()
    validate_unique_patient_ids(pet_df, path.name)
    return pet_df, pet_feature_columns


def build_merged_source(
    clinical_df: pd.DataFrame,
    pet_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = clinical_df.merge(
        pet_df,
        on=PATIENT_ID_COLUMN,
        how="inner",
        validate="one_to_one",
    )

    if merged_df.empty:
        raise ValueError("Clinical and PET sources have no matching PatientID values.")

    return merged_df


def subset_by_split_ids(
    merged_df: pd.DataFrame,
    patient_ids: list[str],
    split_name: str,
) -> pd.DataFrame:
    split_df = pd.DataFrame({PATIENT_ID_COLUMN: patient_ids})
    split_merged = split_df.merge(
        merged_df,
        on=PATIENT_ID_COLUMN,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_ids = split_merged.loc[
        split_merged["_merge"].ne("both"), PATIENT_ID_COLUMN
    ].tolist()
    if missing_ids:
        preview = ", ".join(missing_ids[:10])
        raise ValueError(
            f"{split_name} has PatientID values missing from the Clinical + PET merge: {preview}"
        )

    split_merged = split_merged.drop(columns="_merge")
    split_merged.insert(1, "Split", split_name)
    return split_merged


def encode_hpv_target(target: pd.Series) -> pd.Series:
    numeric_target = pd.to_numeric(target, errors="coerce")
    numeric_values = set(numeric_target.dropna().unique().tolist())
    if not numeric_target.isna().any() and numeric_values.issubset({0, 1}):
        return numeric_target.astype(int)

    positive_values = {
        "1",
        "1.0",
        "positive",
        "pos",
        "yes",
        "true",
        "hpv+",
        "hpv positive",
        "p16+",
        "p16 positive",
    }
    negative_values = {
        "0",
        "0.0",
        "negative",
        "neg",
        "no",
        "false",
        "hpv-",
        "hpv negative",
        "p16-",
        "p16 negative",
    }

    normalized = target.astype("string").str.strip().str.lower()
    encoded = pd.Series(np.nan, index=target.index, dtype="float64")
    encoded.loc[normalized.isin(positive_values)] = 1
    encoded.loc[normalized.isin(negative_values)] = 0

    if encoded.isna().any():
        unknown_values = sorted(target[encoded.isna()].dropna().astype(str).unique().tolist())
        raise ValueError(
            "Could not encode HPV Status values as binary 0/1. Unknown values: "
            + ", ".join(unknown_values)
        )

    return encoded.astype(int)


def prepare_feature_matrices(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    x_train = train_df[feature_columns].copy()
    x_validation = validation_df[feature_columns].copy()
    x_test = test_df[feature_columns].copy()

    categorical_features = [
        column for column in CLINICAL_CATEGORICAL_FEATURES if column in feature_columns
    ]
    numeric_features = [column for column in feature_columns if column not in categorical_features]

    for column in numeric_features:
        x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
        x_validation[column] = pd.to_numeric(x_validation[column], errors="coerce")
        x_test[column] = pd.to_numeric(x_test[column], errors="coerce")

    all_feature_rows = pd.concat(
        [x_train[categorical_features], x_validation[categorical_features], x_test[categorical_features]],
        axis=0,
    )
    for column in categorical_features:
        categories = pd.Categorical(all_feature_rows[column]).categories
        x_train[column] = pd.Categorical(x_train[column], categories=categories)
        x_validation[column] = pd.Categorical(x_validation[column], categories=categories)
        x_test[column] = pd.Categorical(x_test[column], categories=categories)

    return x_train, x_validation, x_test, categorical_features


def safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return float("nan")
    return numerator / denominator


def build_calibration_table(
    y_true: np.ndarray,
    y_probability: np.ndarray,
    n_bins: int = N_CALIBRATION_BINS,
) -> tuple[pd.DataFrame, float]:
    bin_ids = np.minimum((y_probability * n_bins).astype(int), n_bins - 1)
    rows = []
    ece = 0.0

    for bin_index in range(n_bins):
        lower = bin_index / n_bins
        upper = (bin_index + 1) / n_bins
        mask = bin_ids == bin_index
        sample_count = int(mask.sum())

        if sample_count > 0:
            mean_probability = float(y_probability[mask].mean())
            observed_rate = float(y_true[mask].mean())
            calibration_gap = abs(mean_probability - observed_rate)
            ece += (sample_count / len(y_true)) * calibration_gap
        else:
            mean_probability = np.nan
            observed_rate = np.nan
            calibration_gap = np.nan

        right_bracket = "]" if bin_index == n_bins - 1 else ")"
        rows.append(
            {
                "Bin range": f"[{lower:.1f}, {upper:.1f}{right_bracket}",
                "Sample count": sample_count,
                "Mean predicted probability": mean_probability,
                "Observed HPV positive rate": observed_rate,
                "Absolute calibration gap": calibration_gap,
            }
        )

    return pd.DataFrame(rows), float(ece)


def evaluate_predictions(
    y_true: pd.Series,
    y_probability: np.ndarray,
    set_name: str,
) -> tuple[OrderedDict[str, float], pd.DataFrame, pd.DataFrame]:
    y_true_array = y_true.to_numpy(dtype=int)
    y_predicted = (y_probability >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_array, y_predicted, labels=[0, 1]).ravel()
    calibration_table, ece = build_calibration_table(y_true_array, y_probability)

    metrics = OrderedDict(
        [
            ("ROC-AUC", roc_auc_score(y_true_array, y_probability)),
            ("Accuracy", accuracy_score(y_true_array, y_predicted)),
            ("Balanced Accuracy", balanced_accuracy_score(y_true_array, y_predicted)),
            ("Sensitivity", safe_divide(tp, tp + fn)),
            ("Specificity", safe_divide(tn, tn + fp)),
            ("Precision", precision_score(y_true_array, y_predicted, zero_division=0)),
            ("F1-score", f1_score(y_true_array, y_predicted, zero_division=0)),
            ("Brier Score", brier_score_loss(y_true_array, y_probability)),
            ("PR-AUC", average_precision_score(y_true_array, y_probability)),
            ("Expected Calibration Error (ECE)", ece),
        ]
    )

    confusion_df = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual HPV negative", "Actual HPV positive"],
        columns=["Predicted HPV negative", "Predicted HPV positive"],
    )
    confusion_df.insert(0, "Set", set_name)

    return metrics, confusion_df, calibration_table


def print_results(
    set_name: str,
    metrics: OrderedDict[str, float],
    confusion_df: pd.DataFrame,
    calibration_table: pd.DataFrame,
) -> None:
    print(f"\n{set_name} Set Results")
    print("=" * (len(set_name) + 12))

    for metric, value in metrics.items():
        print(f"{metric}: {value:.6f}")

    print("\nConfusion Matrix (rows=true, columns=predicted)")
    print(confusion_df.drop(columns="Set").to_string())

    print("\nCalibration Table")
    print(calibration_table.to_string(index=False))


def build_feature_importance(model: LGBMClassifier, feature_columns: list[str]) -> pd.DataFrame:
    booster = model.booster_
    importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Gain importance": booster.feature_importance(importance_type="gain"),
            "Split importance": booster.feature_importance(importance_type="split"),
        }
    )
    importance_df = importance_df.sort_values(
        ["Gain importance", "Split importance"],
        ascending=False,
    ).reset_index(drop=True)
    importance_df.insert(0, "Rank", np.arange(1, len(importance_df) + 1))
    return importance_df


def save_roc_curve(y_true: pd.Series, y_probability: np.ndarray, set_name: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    auc_value = roc_auc_score(y_true, y_probability)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color="#1769aa", linewidth=2, label=f"ROC-AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{set_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{set_name.lower()}_roc_curve.png", dpi=300)
    plt.close()


def save_precision_recall_curve(y_true: pd.Series, y_probability: np.ndarray, set_name: str) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    pr_auc = average_precision_score(y_true, y_probability)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, color="#b64b39", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{set_name} Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{set_name.lower()}_precision_recall_curve.png", dpi=300)
    plt.close()


def save_calibration_curve(calibration_table: pd.DataFrame, set_name: str) -> None:
    non_empty = calibration_table[calibration_table["Sample count"].gt(0)]

    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1, label="Perfect")
    plt.plot(
        non_empty["Mean predicted probability"],
        non_empty["Observed HPV positive rate"],
        marker="o",
        color="#2c7a4b",
        linewidth=2,
        label="Observed",
    )
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed HPV Positive Rate")
    plt.title(f"{set_name} Calibration Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{set_name.lower()}_calibration_curve.png", dpi=300)
    plt.close()


def save_confusion_heatmap(confusion_df: pd.DataFrame, set_name: str) -> None:
    matrix = confusion_df.drop(columns="Set").to_numpy()

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar(label="Count")
    plt.xticks([0, 1], ["Pred HPV-", "Pred HPV+"])
    plt.yticks([0, 1], ["Actual HPV-", "Actual HPV+"])
    plt.title(f"{set_name} Confusion Matrix")

    max_value = matrix.max() if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            text_color = "white" if value > max_value / 2 else "black"
            plt.text(column, row, str(value), ha="center", va="center", color=text_color)

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{set_name.lower()}_confusion_matrix_heatmap.png", dpi=300)
    plt.close()


def save_probability_distribution(
    y_true: pd.Series,
    y_probability: np.ndarray,
    set_name: str,
) -> None:
    y_true_array = y_true.to_numpy(dtype=int)

    plt.figure(figsize=(8, 6))
    plt.hist(
        y_probability[y_true_array == 0],
        bins=20,
        alpha=0.70,
        color="#4b77be",
        label="Actual HPV negative",
        range=(0, 1),
    )
    plt.hist(
        y_probability[y_true_array == 1],
        bins=20,
        alpha=0.70,
        color="#d26a5c",
        label="Actual HPV positive",
        range=(0, 1),
    )
    plt.axvline(THRESHOLD, color="#222222", linestyle="--", linewidth=1, label="Threshold = 0.5")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Sample Count")
    plt.title(f"{set_name} Probability Distribution")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"{set_name.lower()}_probability_distribution.png", dpi=300)
    plt.close()


def save_feature_importance_plot(importance_df: pd.DataFrame) -> None:
    plot_df = importance_df.head(TOP_FEATURES_FOR_PLOTS).sort_values("Gain importance")

    plt.figure(figsize=(10, 9))
    plt.barh(plot_df["Feature"], plot_df["Gain importance"], color="#445e93")
    plt.xlabel("Gain Importance")
    plt.title(f"Top {len(plot_df)} LightGBM Feature Importances")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_plot.png", dpi=300)
    plt.close()


def convert_categoricals_for_shap_plot(x_data: pd.DataFrame) -> pd.DataFrame:
    x_plot = x_data.copy()
    for column in x_plot.select_dtypes(include=["category"]).columns:
        x_plot[column] = x_plot[column].cat.codes.replace(-1, np.nan).astype(float)
    for column in x_plot.columns:
        x_plot[column] = pd.to_numeric(x_plot[column], errors="coerce")
    return x_plot


def save_shap_plots(
    model: LGBMClassifier,
    x_data: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> None:
    if x_data.empty:
        raise ValueError("Cannot make SHAP plots because the SHAP input set is empty.")

    shap_sample = x_data
    if len(shap_sample) > MAX_SHAP_SAMPLES:
        shap_sample = shap_sample.sample(n=MAX_SHAP_SAMPLES, random_state=RANDOM_STATE)

    shap_contributions = model.booster_.predict(shap_sample, pred_contrib=True)
    shap_values = np.asarray(shap_contributions[:, :-1])
    base_values = np.asarray(shap_contributions[:, -1])
    x_plot = convert_categoricals_for_shap_plot(shap_sample)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_abs_order = np.argsort(mean_abs_shap)[::-1]

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        x_plot,
        max_display=TOP_FEATURES_FOR_PLOTS,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    most_important_feature = importance_df.loc[0, "Feature"]
    if most_important_feature not in x_plot.columns:
        most_important_feature = x_plot.columns[mean_abs_order[0]]

    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        most_important_feature,
        shap_values,
        x_plot,
        interaction_index=None,
        show=False,
    )
    plt.title(f"SHAP Dependence: {most_important_feature}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_dependence_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    heatmap_indices = mean_abs_order[: min(TOP_FEATURES_FOR_PLOTS, len(mean_abs_order))]
    heatmap_values = shap_values[:, heatmap_indices]
    heatmap_data = x_plot.iloc[:, heatmap_indices]
    heatmap_explanation = shap.Explanation(
        values=heatmap_values,
        base_values=base_values,
        data=heatmap_data.to_numpy(),
        feature_names=heatmap_data.columns.tolist(),
    )

    shap.plots.heatmap(
        heatmap_explanation,
        max_display=min(TOP_FEATURES_FOR_PLOTS, len(heatmap_indices)),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_heatmap_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def open_saved_plots() -> None:
    if not OPEN_PLOTS_AFTER_SAVING:
        return

    plot_paths = sorted(PLOTS_DIR.glob("*.png"))
    if not plot_paths:
        return

    if not hasattr(os, "startfile"):
        print(f"\nPlots were saved in {PLOTS_DIR}, but automatic popups are only enabled on Windows.")
        return

    print(f"\nOpening {len(plot_paths)} saved plot PNG files...")
    for plot_path in plot_paths:
        os.startfile(str(plot_path))  # type: ignore[attr-defined]


def save_evaluation_plots(
    y_validation: pd.Series,
    validation_probability: np.ndarray,
    validation_confusion: pd.DataFrame,
    validation_calibration: pd.DataFrame,
    y_test: pd.Series,
    test_probability: np.ndarray,
    test_confusion: pd.DataFrame,
    test_calibration: pd.DataFrame,
) -> None:
    for set_name, y_true, y_probability, confusion_df, calibration_table in [
        ("Validation", y_validation, validation_probability, validation_confusion, validation_calibration),
        ("Test", y_test, test_probability, test_confusion, test_calibration),
    ]:
        save_calibration_curve(calibration_table, set_name)
        save_roc_curve(y_true, y_probability, set_name)
        save_precision_recall_curve(y_true, y_probability, set_name)
        save_confusion_heatmap(confusion_df, set_name)
        save_probability_distribution(y_true, y_probability, set_name)


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    OUTPUT_DIR.mkdir(exist_ok=True)
    PLOTS_DIR.mkdir(exist_ok=True)

    files = discover_files()
    print("Discovered files")
    for name, path in files.items():
        print(f"  {name}: {path.name}")

    split_ids = {
        "train": read_split_patient_ids(files["train"], "train"),
        "validation": read_split_patient_ids(files["validation"], "validation"),
        "test": read_split_patient_ids(files["test"], "test"),
    }
    validate_disjoint_splits(split_ids)

    clinical_df = load_clinical_source(files["clinical"])
    pet_df, pet_feature_columns = load_pet_source(files["pet"])
    merged_source_df = build_merged_source(clinical_df, pet_df)

    train_df = subset_by_split_ids(merged_source_df, split_ids["train"], "train")
    validation_df = subset_by_split_ids(
        merged_source_df,
        split_ids["validation"],
        "validation",
    )
    test_df = subset_by_split_ids(merged_source_df, split_ids["test"], "test")

    merged_fixed_split_df = pd.concat([train_df, validation_df, test_df], ignore_index=True)
    merged_csv_path = OUTPUT_DIR / "merged_clinical_pet.csv"
    merged_fixed_split_df.to_csv(merged_csv_path, index=False)

    feature_columns = CLINICAL_FEATURES + pet_feature_columns
    y_train = encode_hpv_target(train_df[TARGET_COLUMN])
    y_validation = encode_hpv_target(validation_df[TARGET_COLUMN])
    y_test = encode_hpv_target(test_df[TARGET_COLUMN])

    x_train, x_validation, x_test, categorical_features = prepare_feature_matrices(
        train_df,
        validation_df,
        test_df,
        feature_columns,
    )

    model = LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(
        x_train,
        y_train,
        categorical_feature=categorical_features,
    )

    validation_probability = model.predict_proba(x_validation)[:, 1]
    test_probability = model.predict_proba(x_test)[:, 1]

    validation_metrics, validation_confusion, validation_calibration = evaluate_predictions(
        y_validation,
        validation_probability,
        "Validation",
    )
    test_metrics, test_confusion, test_calibration = evaluate_predictions(
        y_test,
        test_probability,
        "Test",
    )

    importance_df = build_feature_importance(model, feature_columns)
    save_evaluation_plots(
        y_validation,
        validation_probability,
        validation_confusion,
        validation_calibration,
        y_test,
        test_probability,
        test_confusion,
        test_calibration,
    )
    save_feature_importance_plot(importance_df)
    save_shap_plots(model, x_validation, importance_df)

    print("\nDataset summary")
    print("===============")
    print(f"Train patients: {len(train_df)}")
    print(f"Validation patients: {len(validation_df)}")
    print(f"Test patients: {len(test_df)}")
    print(f"Clinical features: {len(CLINICAL_FEATURES)}")
    print(f"PET features: {len(pet_feature_columns)}")
    print(f"Total input features: {len(feature_columns)}")
    print(f"Categorical clinical features: {', '.join(categorical_features)}")
    print(f"Merged CSV: {merged_csv_path}")

    print_results("Validation", validation_metrics, validation_confusion, validation_calibration)
    print_results("Test", test_metrics, test_confusion, test_calibration)

    print("\nSaved outputs")
    print("=============")
    print(f"Merged CSV: {merged_csv_path}")
    print(f"Plots: {PLOTS_DIR}")
    open_saved_plots()


if __name__ == "__main__":
    main()
