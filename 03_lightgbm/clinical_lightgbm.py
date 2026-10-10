from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PLOTS_DIR = BASE_DIR / "plots"
MPLCONFIG_DIR = BASE_DIR / ".matplotlib_cache"
PLOTS_DIR.mkdir(exist_ok=True)
MPLCONFIG_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

try:
    import shap
except ImportError:
    shap = None


RANDOM_STATE = 42
THRESHOLD = 0.5
N_CALIBRATION_BINS = 10
OPEN_PLOTS_AFTER_RUN = os.environ.get("OPEN_PLOTS", "1").strip().lower() in {"1", "true", "yes", "y"}

TRAIN_PATH = BASE_DIR / "train.csv"
VALIDATION_PATH = BASE_DIR / "validation.csv"
TEST_PATH = BASE_DIR / "test.csv"
CLINICAL_PATH = BASE_DIR / "HECKTOR_2025_Training_Task_3.csv"

ID_COLUMN = "PatientID"
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
CATEGORICAL_FEATURES = ["M-stage"]


def load_patient_ids(path: Path) -> list[str]:
    df = pd.read_csv(path, usecols=[ID_COLUMN])
    return df[ID_COLUMN].astype(str).tolist()


def load_clinical_data() -> pd.DataFrame:
    expected_columns = [ID_COLUMN, *CLINICAL_FEATURES, TARGET_COLUMN]
    clinical = pd.read_csv(CLINICAL_PATH)
    missing_columns = [column for column in expected_columns if column not in clinical.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {CLINICAL_PATH.name}: {missing_columns}")

    duplicated_ids = clinical.loc[clinical[ID_COLUMN].duplicated(), ID_COLUMN].tolist()
    if duplicated_ids:
        raise ValueError(f"Duplicate PatientID values found in clinical file: {duplicated_ids[:10]}")

    clinical[ID_COLUMN] = clinical[ID_COLUMN].astype(str)
    return clinical


def subset_by_ids(clinical: pd.DataFrame, patient_ids: list[str], split_name: str) -> pd.DataFrame:
    missing_ids = sorted(set(patient_ids) - set(clinical[ID_COLUMN]))
    if missing_ids:
        raise ValueError(f"{split_name} has PatientIDs missing from clinical file: {missing_ids[:10]}")

    split = clinical.set_index(ID_COLUMN).loc[patient_ids].reset_index()
    return split[[ID_COLUMN, *CLINICAL_FEATURES, TARGET_COLUMN]].copy()


def split_xy(split: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    x = split[CLINICAL_FEATURES].copy()
    y = split[TARGET_COLUMN].astype(int).copy()
    return x, y


def prepare_categorical_columns(
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for feature in CATEGORICAL_FEATURES:
        categories = x_train[feature].astype("category").cat.categories
        for split in [x_train, x_validation, x_test]:
            known_or_missing = split[feature].isna() | split[feature].isin(categories)
            split[feature] = pd.Categorical(split[feature].where(known_or_missing), categories=categories)
    return x_train, x_validation, x_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        categorical_feature=CATEGORICAL_FEATURES,
    )
    return model


def calculate_metrics(y_true: pd.Series, y_probability: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    y_predicted = (y_probability >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, y_predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    metrics = {
        "ROC-AUC": roc_auc_score(y_true, y_probability),
        "Accuracy": accuracy_score(y_true, y_predicted),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_predicted),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_predicted, zero_division=0),
        "F1-score": f1_score(y_true, y_predicted, zero_division=0),
        "Brier Score": brier_score_loss(y_true, y_probability),
        "PR-AUC": average_precision_score(y_true, y_probability),
        "Expected Calibration Error": expected_calibration_error(y_true, y_probability),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    return metrics, cm


def calibration_table(y_true: pd.Series, y_probability: np.ndarray, n_bins: int = N_CALIBRATION_BINS) -> pd.DataFrame:
    y_true_array = np.asarray(y_true, dtype=float)
    y_probability = np.asarray(y_probability, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    rows = []
    for bin_index in range(n_bins):
        left = bin_edges[bin_index]
        right = bin_edges[bin_index + 1]
        if bin_index == n_bins - 1:
            in_bin = (y_probability >= left) & (y_probability <= right)
            bin_range = f"[{left:.1f}, {right:.1f}]"
        else:
            in_bin = (y_probability >= left) & (y_probability < right)
            bin_range = f"[{left:.1f}, {right:.1f})"

        count = int(in_bin.sum())
        if count:
            mean_probability = float(y_probability[in_bin].mean())
            observed_rate = float(y_true_array[in_bin].mean())
            calibration_gap = abs(mean_probability - observed_rate)
        else:
            mean_probability = np.nan
            observed_rate = np.nan
            calibration_gap = np.nan

        rows.append(
            {
                "bin_range": bin_range,
                "sample_count": count,
                "mean_predicted_probability": mean_probability,
                "observed_hpv_positive_rate": observed_rate,
                "absolute_calibration_gap": calibration_gap,
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(y_true: pd.Series, y_probability: np.ndarray) -> float:
    table = calibration_table(y_true, y_probability)
    total_count = table["sample_count"].sum()
    if total_count == 0:
        return np.nan

    non_empty = table[table["sample_count"] > 0].copy()
    weights = non_empty["sample_count"] / total_count
    return float((weights * non_empty["absolute_calibration_gap"]).sum())


def save_current_figure(filename: str) -> Path:
    output_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_roc_curve(split_name: str, y_true: pd.Series, y_probability: np.ndarray) -> Path:
    fpr, tpr, _ = roc_curve(y_true, y_probability)
    auc_value = roc_auc_score(y_true, y_probability)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(fpr, tpr, color="#2166ac", linewidth=2, label=f"ROC-AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name} ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    return save_current_figure(f"{split_name.lower()}_roc_curve.png")


def plot_precision_recall_curve(split_name: str, y_true: pd.Series, y_probability: np.ndarray) -> Path:
    precision, recall, _ = precision_recall_curve(y_true, y_probability)
    pr_auc = average_precision_score(y_true, y_probability)

    plt.figure(figsize=(6.5, 5.5))
    plt.plot(recall, precision, color="#1b7837", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split_name} Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.25)
    return save_current_figure(f"{split_name.lower()}_precision_recall_curve.png")


def plot_confusion_matrix_heatmap(split_name: str, cm: np.ndarray) -> Path:
    plt.figure(figsize=(5.5, 4.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Predicted HPV 0", "Predicted HPV 1"],
        yticklabels=["Actual HPV 0", "Actual HPV 1"],
    )
    plt.title(f"{split_name} Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    return save_current_figure(f"{split_name.lower()}_confusion_matrix_heatmap.png")


def plot_calibration_curve(split_name: str, calibration: pd.DataFrame) -> Path:
    non_empty = calibration[calibration["sample_count"] > 0]

    plt.figure(figsize=(6.5, 5.5))
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1, label="Perfect calibration")
    plt.plot(
        non_empty["mean_predicted_probability"],
        non_empty["observed_hpv_positive_rate"],
        marker="o",
        color="#b2182b",
        linewidth=2,
        label="Model",
    )
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed HPV Positive Rate")
    plt.title(f"{split_name} Calibration Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="upper left")
    plt.grid(alpha=0.25)
    return save_current_figure(f"{split_name.lower()}_calibration_curve.png")


def plot_probability_distribution(split_name: str, y_true: pd.Series, y_probability: np.ndarray) -> Path:
    probability_df = pd.DataFrame(
        {
            "HPV": y_true.astype(int).map({0: "HPV 0", 1: "HPV 1"}).to_numpy(),
            "Predicted probability": y_probability,
        }
    )

    plt.figure(figsize=(7, 5.2))
    sns.histplot(
        data=probability_df,
        x="Predicted probability",
        hue="HPV",
        bins=20,
        stat="count",
        common_norm=False,
        element="step",
        alpha=0.45,
    )
    plt.axvline(THRESHOLD, color="#333333", linestyle="--", linewidth=1.2, label=f"Threshold {THRESHOLD:.1f}")
    plt.xlabel("Predicted HPV Probability")
    plt.ylabel("Count")
    plt.title(f"{split_name} Probability Distribution")
    plt.xlim(0, 1)
    plt.grid(alpha=0.2)
    return save_current_figure(f"{split_name.lower()}_probability_distribution.png")


def plot_feature_importance(model: LGBMClassifier) -> Path:
    importance = model.booster_.feature_importance(importance_type="gain")
    if np.allclose(importance, 0):
        importance = model.booster_.feature_importance(importance_type="split")

    importance_df = (
        pd.DataFrame({"feature": CLINICAL_FEATURES, "importance": importance})
        .sort_values("importance", ascending=True)
        .reset_index(drop=True)
    )

    plt.figure(figsize=(8, 5.5))
    plt.barh(importance_df["feature"], importance_df["importance"], color="#4d9221")
    plt.xlabel("LightGBM Importance")
    plt.title("Clinical Feature Importance")
    plt.grid(axis="x", alpha=0.25)
    return save_current_figure("feature_importance.png")


def is_categorical_feature(values: pd.Series) -> bool:
    return isinstance(values.dtype, pd.CategoricalDtype) or values.dtype == object


def numeric_display_data(data: pd.DataFrame) -> pd.DataFrame:
    display = data.copy()
    for column in display.columns:
        if isinstance(display[column].dtype, pd.CategoricalDtype):
            codes = display[column].cat.codes.replace(-1, np.nan)
            display[column] = codes.astype(float)
        elif display[column].dtype == object:
            display[column] = pd.Categorical(display[column]).codes.astype(float)
            display.loc[display[column] < 0, column] = np.nan
    return display


def positive_class_shap_values(raw_shap_values: object) -> np.ndarray:
    if isinstance(raw_shap_values, list):
        return np.asarray(raw_shap_values[1])

    values = np.asarray(raw_shap_values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values


def positive_class_base_values(explainer: object, n_rows: int) -> np.ndarray:
    expected_value = getattr(explainer, "expected_value", 0.0)
    if isinstance(expected_value, list):
        base_value = expected_value[1]
    else:
        expected_array = np.asarray(expected_value)
        if expected_array.ndim > 0 and expected_array.size > 1:
            base_value = expected_array[1]
        else:
            base_value = float(expected_array)
    return np.repeat(base_value, n_rows)


def plot_categorical_shap_dependence(
    feature_name: str,
    feature_values: pd.Series,
    feature_shap_values: np.ndarray,
) -> Path:
    display_values = feature_values.astype(object).where(feature_values.notna(), "Missing").astype(str)
    labels = list(dict.fromkeys(display_values.tolist()))
    positions = {label: index for index, label in enumerate(labels)}
    rng = np.random.default_rng(RANDOM_STATE)
    x_positions = display_values.map(positions).to_numpy(dtype=float)
    x_positions = x_positions + rng.normal(loc=0.0, scale=0.035, size=len(x_positions))

    plt.figure(figsize=(7, 5.2))
    plt.scatter(
        x_positions,
        feature_shap_values,
        c=feature_shap_values,
        cmap="coolwarm",
        alpha=0.8,
        edgecolor="none",
    )
    plt.axhline(0, color="#555555", linestyle="--", linewidth=1)
    plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
    plt.xlabel(feature_name)
    plt.ylabel(f"SHAP value for {feature_name}")
    plt.title(f"SHAP Dependence: {feature_name}")
    plt.colorbar(label="SHAP value")
    plt.grid(axis="y", alpha=0.25)
    return save_current_figure(f"shap_dependence_{safe_filename(feature_name)}.png")


def plot_numeric_shap_dependence(
    feature_name: str,
    shap_values: np.ndarray,
    x_explain: pd.DataFrame,
) -> Path:
    shap_dependence_path = PLOTS_DIR / f"shap_dependence_{safe_filename(feature_name)}.png"
    shap.dependence_plot(
        feature_name,
        shap_values,
        x_explain,
        show=False,
        interaction_index=None,
    )
    plt.tight_layout()
    plt.savefig(shap_dependence_path, dpi=300, bbox_inches="tight")
    plt.close()
    return shap_dependence_path


def plot_shap_outputs(model: LGBMClassifier, x_explain: pd.DataFrame) -> list[Path]:
    if shap is None:
        print("\nSHAP is not installed, so SHAP plots were skipped.")
        return []

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(x_explain)
    shap_values = positive_class_shap_values(raw_shap_values)
    base_values = positive_class_base_values(explainer, len(x_explain))

    shap_summary_path = PLOTS_DIR / "shap_summary_plot.png"
    shap.summary_plot(shap_values, x_explain, show=False, max_display=len(CLINICAL_FEATURES))
    plt.tight_layout()
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    dependence_index = int(np.argmax(mean_abs_shap))
    dependence_feature = x_explain.columns[dependence_index]
    if is_categorical_feature(x_explain[dependence_feature]):
        shap_dependence_path = plot_categorical_shap_dependence(
            dependence_feature,
            x_explain[dependence_feature],
            shap_values[:, dependence_index],
        )
    else:
        shap_dependence_path = plot_numeric_shap_dependence(dependence_feature, shap_values, x_explain)

    shap_heatmap_path = PLOTS_DIR / "shap_heatmap_plot.png"
    heatmap_data = numeric_display_data(x_explain)
    explanation = shap.Explanation(
        values=shap_values,
        base_values=base_values,
        data=heatmap_data.to_numpy(),
        feature_names=list(x_explain.columns),
    )
    shap.plots.heatmap(explanation, show=False, max_display=len(CLINICAL_FEATURES))
    plt.tight_layout()
    plt.savefig(shap_heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()

    return [shap_summary_path, shap_dependence_path, shap_heatmap_path]


def safe_filename(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
    )


def evaluate_split(
    split_name: str,
    model: LGBMClassifier,
    x: pd.DataFrame,
    y: pd.Series,
) -> tuple[dict[str, object], pd.DataFrame, list[Path]]:
    y_probability = model.predict_proba(x)[:, 1]
    metrics, cm = calculate_metrics(y, y_probability)
    calibration = calibration_table(y, y_probability)

    plot_paths = [
        plot_roc_curve(split_name, y, y_probability),
        plot_precision_recall_curve(split_name, y, y_probability),
        plot_confusion_matrix_heatmap(split_name, cm),
        plot_calibration_curve(split_name, calibration),
        plot_probability_distribution(split_name, y, y_probability),
    ]
    return metrics, calibration, plot_paths


def print_split_summary(name: str, split: pd.DataFrame) -> None:
    counts = split[TARGET_COLUMN].value_counts().sort_index()
    hpv_0 = int(counts.get(0, 0))
    hpv_1 = int(counts.get(1, 0))
    print(f"{name}: {len(split)} patients | HPV 0: {hpv_0} | HPV 1: {hpv_1}")


def print_metrics_table(results: dict[str, dict[str, object]]) -> None:
    metric_order = [
        "ROC-AUC",
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1-score",
        "Brier Score",
        "PR-AUC",
        "Expected Calibration Error",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    table = pd.DataFrame(results).loc[metric_order]
    print("\nValidation and Test Metrics")
    print(table.to_string(float_format=lambda value: f"{value:.4f}"))


def print_confusion_matrices(results: dict[str, dict[str, object]]) -> None:
    for split_name, metrics in results.items():
        matrix = pd.DataFrame(
            [
                [int(metrics["TN"]), int(metrics["FP"])],
                [int(metrics["FN"]), int(metrics["TP"])],
            ],
            index=["Actual HPV 0", "Actual HPV 1"],
            columns=["Predicted HPV 0", "Predicted HPV 1"],
        )
        print(f"\n{split_name} Confusion Matrix")
        print(matrix.to_string())


def print_calibration_tables(calibration_tables: dict[str, pd.DataFrame]) -> None:
    for split_name, table in calibration_tables.items():
        print(f"\n{split_name} Calibration Table")
        print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def open_saved_plots(plot_paths: list[Path]) -> None:
    if not OPEN_PLOTS_AFTER_RUN:
        return

    print("\nOpening saved PNG plots...")
    for path in plot_paths:
        try:
            os.startfile(path)
        except OSError as error:
            print(f"Could not open {path.name}: {error}")


def main() -> None:
    train_ids = load_patient_ids(TRAIN_PATH)
    validation_ids = load_patient_ids(VALIDATION_PATH)
    test_ids = load_patient_ids(TEST_PATH)
    clinical = load_clinical_data()

    train_split = subset_by_ids(clinical, train_ids, "Train")
    validation_split = subset_by_ids(clinical, validation_ids, "Validation")
    test_split = subset_by_ids(clinical, test_ids, "Test")

    print("Clinical-only LightGBM baseline")
    print(f"Clinical source file: {CLINICAL_PATH.name}")
    print("Split source files: train.csv, validation.csv, test.csv")
    print(f"Clinical features: {', '.join(CLINICAL_FEATURES)}")
    print(f"Classification threshold: {THRESHOLD}")
    print("\nSplit Summary")
    print_split_summary("Train", train_split)
    print_split_summary("Validation", validation_split)
    print_split_summary("Test", test_split)

    x_train, y_train = split_xy(train_split)
    x_validation, y_validation = split_xy(validation_split)
    x_test, y_test = split_xy(test_split)
    x_train, x_validation, x_test = prepare_categorical_columns(x_train, x_validation, x_test)

    model = train_model(x_train, y_train)

    validation_metrics, validation_calibration, validation_plots = evaluate_split(
        "Validation",
        model,
        x_validation,
        y_validation,
    )
    test_metrics, test_calibration, test_plots = evaluate_split("Test", model, x_test, y_test)
    feature_importance_plot = plot_feature_importance(model)
    shap_plots = plot_shap_outputs(model, x_test)

    metric_results = {"Validation": validation_metrics, "Test": test_metrics}
    print_metrics_table(metric_results)
    print_confusion_matrices(metric_results)
    print_calibration_tables({"Validation": validation_calibration, "Test": test_calibration})

    all_plot_paths = [
        *validation_plots,
        *test_plots,
        feature_importance_plot,
        *shap_plots,
    ]
    print("\nSaved PNG plots")
    for path in all_plot_paths:
        print(f"- {path.relative_to(BASE_DIR)}")
    open_saved_plots(all_plot_paths)


if __name__ == "__main__":
    main()
