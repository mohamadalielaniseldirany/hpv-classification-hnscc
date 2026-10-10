from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parents[1]
VENV_PYTHON = PROJECT_ROOT / "myenv" / "Scripts" / "python.exe"


def same_path(left: Path | str, right: Path | str) -> bool:
    return os.path.normcase(os.path.abspath(str(left))) == os.path.normcase(os.path.abspath(str(right)))


def ensure_project_virtualenv() -> None:
    if VENV_PYTHON.exists() and not same_path(sys.executable, VENV_PYTHON):
        print(f"Re-running inside project virtual environment: {VENV_PYTHON}")
        os.execv(str(VENV_PYTHON), [str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


ensure_project_virtualenv()

PLOTS_DIR = BASE_DIR / "plots"
MPLCONFIG_DIR = BASE_DIR / ".matplotlib_cache"
NUMBA_CACHE_DIR = BASE_DIR / ".numba_cache"
PLOTS_DIR.mkdir(exist_ok=True)
MPLCONFIG_DIR.mkdir(exist_ok=True)
NUMBA_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))
os.environ.setdefault("NUMBA_CACHE_DIR", str(NUMBA_CACHE_DIR))


def ensure_required_packages() -> None:
    required_packages = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "seaborn": "seaborn",
        "sklearn": "scikit-learn",
        "lightgbm": "lightgbm",
        "shap": "shap",
    }
    missing_packages = [
        package_name
        for module_name, package_name in required_packages.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_packages:
        print(f"Installing missing packages into {sys.executable}: {', '.join(missing_packages)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])


ensure_required_packages()

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
TOP_SHAP_FEATURES = 5
OPEN_PLOTS_AFTER_RUN = os.environ.get("OPEN_PLOTS", "1").strip().lower() in {"1", "true", "yes", "y"}

TRAIN_PATH = BASE_DIR / "train.csv"
VALIDATION_PATH = BASE_DIR / "validation.csv"
TEST_PATH = BASE_DIR / "test.csv"
PET_FEATURES_PATH = BASE_DIR / "pet_features.csv"

ID_COLUMN = "PatientID"
TARGET_COLUMN = "HPV"


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {split_name} split file: {path}")

    split = pd.read_csv(path)
    missing_columns = [column for column in [ID_COLUMN, TARGET_COLUMN] if column not in split.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

    duplicated_ids = split.loc[split[ID_COLUMN].astype(str).duplicated(), ID_COLUMN].astype(str).tolist()
    if duplicated_ids:
        raise ValueError(f"{path.name} contains duplicate PatientID values: {duplicated_ids[:10]}")

    split[ID_COLUMN] = split[ID_COLUMN].astype(str)
    return split[[ID_COLUMN, TARGET_COLUMN]].copy()


def load_pet_features() -> tuple[pd.DataFrame, list[str]]:
    if not PET_FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing PET feature file: {PET_FEATURES_PATH}")

    pet_features = pd.read_csv(PET_FEATURES_PATH)
    if ID_COLUMN not in pet_features.columns:
        raise ValueError(f"{PET_FEATURES_PATH.name} is missing required column: {ID_COLUMN}")

    duplicated_ids = pet_features.loc[pet_features[ID_COLUMN].astype(str).duplicated(), ID_COLUMN].astype(str).tolist()
    if duplicated_ids:
        raise ValueError(f"{PET_FEATURES_PATH.name} contains duplicate PatientID values: {duplicated_ids[:10]}")

    pet_features[ID_COLUMN] = pet_features[ID_COLUMN].astype(str)
    feature_columns = [column for column in pet_features.columns if column not in {ID_COLUMN, TARGET_COLUMN}]
    if not feature_columns:
        raise ValueError(f"No PET radiomics feature columns found in {PET_FEATURES_PATH.name}")

    return pet_features, feature_columns


def clean_target(target: pd.Series) -> pd.Series:
    numeric_target = pd.to_numeric(target, errors="coerce")
    if numeric_target.isna().any():
        bad_values = target[numeric_target.isna()].astype(str).unique().tolist()
        raise ValueError(f"HPV target contains non-numeric values that cannot be used: {bad_values[:10]}")

    unique_values = sorted(numeric_target.dropna().unique().tolist())
    invalid_values = [value for value in unique_values if value not in [0, 1]]
    if invalid_values:
        raise ValueError(f"HPV target must contain only 0 and 1. Found: {invalid_values}")

    return numeric_target.astype(int)


def build_xy(
    split: pd.DataFrame,
    split_name: str,
    pet_features: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    patient_ids = split[ID_COLUMN].astype(str).tolist()
    available_ids = set(pet_features[ID_COLUMN])
    missing_ids = sorted(set(patient_ids) - available_ids)
    if missing_ids:
        raise ValueError(f"{split_name} has PatientIDs missing from {PET_FEATURES_PATH.name}: {missing_ids[:10]}")

    pet_index = pet_features.set_index(ID_COLUMN)
    x = pet_index.loc[patient_ids, feature_columns].reset_index(drop=True)
    x = x.apply(pd.to_numeric, errors="coerce")
    y = clean_target(split[TARGET_COLUMN]).reset_index(drop=True)
    return x, y


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
    model = LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(x_train, y_train)
    return model


def calculate_roc_auc(y_true: pd.Series, y_probability: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_probability))


def calculate_pr_auc(y_true: pd.Series, y_probability: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return np.nan
    return float(average_precision_score(y_true, y_probability))


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

        sample_count = int(in_bin.sum())
        if sample_count:
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
                "sample_count": sample_count,
                "mean_predicted_probability": mean_probability,
                "observed_hpv_positive_rate": observed_rate,
                "absolute_calibration_gap": calibration_gap,
            }
        )

    return pd.DataFrame(rows)


def expected_calibration_error(y_true: pd.Series, y_probability: np.ndarray) -> float:
    table = calibration_table(y_true, y_probability)
    total_count = int(table["sample_count"].sum())
    if total_count == 0:
        return np.nan

    non_empty = table[table["sample_count"] > 0]
    weights = non_empty["sample_count"] / total_count
    return float((weights * non_empty["absolute_calibration_gap"]).sum())


def calculate_metrics(y_true: pd.Series, y_probability: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    y_predicted = (y_probability >= THRESHOLD).astype(int)
    cm = confusion_matrix(y_true, y_predicted, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    metrics = {
        "ROC-AUC": calculate_roc_auc(y_true, y_probability),
        "Accuracy": accuracy_score(y_true, y_predicted),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_predicted),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_predicted, zero_division=0),
        "F1-score": f1_score(y_true, y_predicted, zero_division=0),
        "Brier Score": brier_score_loss(y_true, y_probability),
        "PR-AUC": calculate_pr_auc(y_true, y_probability),
        "Expected Calibration Error": expected_calibration_error(y_true, y_probability),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }
    return metrics, cm


def save_current_figure(filename: str) -> Path:
    output_path = PLOTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


def plot_roc_curve(split_name: str, y_true: pd.Series, y_probability: np.ndarray) -> Path:
    plt.figure(figsize=(6.5, 5.5))
    if y_true.nunique() >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_probability)
        auc_value = roc_auc_score(y_true, y_probability)
        plt.plot(fpr, tpr, color="#2166ac", linewidth=2, label=f"ROC-AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{split_name} ROC Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    return save_current_figure(f"{split_name.lower()}_roc_curve.png")


def plot_precision_recall_curve(split_name: str, y_true: pd.Series, y_probability: np.ndarray) -> Path:
    plt.figure(figsize=(6.5, 5.5))
    if y_true.nunique() >= 2:
        precision, recall, _ = precision_recall_curve(y_true, y_probability)
        pr_auc = average_precision_score(y_true, y_probability)
        plt.plot(recall, precision, color="#1b7837", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{split_name} Precision-Recall Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
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


def model_feature_importance(model: LGBMClassifier, feature_columns: list[str]) -> pd.DataFrame:
    importance = model.booster_.feature_importance(importance_type="gain")
    importance_name = "gain"
    if np.allclose(importance, 0):
        importance = model.booster_.feature_importance(importance_type="split")
        importance_name = "split"

    importance_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": importance,
            "importance_type": importance_name,
        }
    ).sort_values("importance", ascending=False)
    return importance_df.reset_index(drop=True)


def plot_feature_importance(model: LGBMClassifier, feature_columns: list[str]) -> tuple[Path, pd.DataFrame]:
    importance_df = model_feature_importance(model, feature_columns)
    plot_df = importance_df.sort_values("importance", ascending=True)
    height = max(7, 0.24 * len(feature_columns))

    plt.figure(figsize=(10.5, height))
    plt.barh(plot_df["feature"], plot_df["importance"], color="#4d9221")
    plt.xlabel(f"LightGBM Importance ({importance_df['importance_type'].iloc[0]})")
    plt.title("PET Radiomics Feature Importance")
    plt.tick_params(axis="y", labelsize=6)
    plt.grid(axis="x", alpha=0.25)
    return save_current_figure("feature_importance.png"), importance_df


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


def safe_filename(value: str) -> str:
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("-", "_")
        .replace(":", "_")
    )


def plot_shap_outputs(
    model: LGBMClassifier,
    x_explain: pd.DataFrame,
    feature_columns: list[str],
    importance_df: pd.DataFrame,
) -> list[Path]:
    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(x_explain)
    shap_values = positive_class_shap_values(raw_shap_values)
    base_values = positive_class_base_values(explainer, len(x_explain))
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_feature_count = min(TOP_SHAP_FEATURES, len(feature_columns))
    top_feature_indices = np.argsort(mean_abs_shap)[::-1][:top_feature_count]
    top_x_explain = x_explain.iloc[:, top_feature_indices]
    top_shap_values = shap_values[:, top_feature_indices]

    summary_height = max(4, 0.55 * top_feature_count)
    shap_summary_path = PLOTS_DIR / "shap_summary_plot.png"
    shap.summary_plot(
        top_shap_values,
        top_x_explain,
        show=False,
        max_display=top_feature_count,
        plot_size=(10.5, summary_height),
    )
    plt.tight_layout()
    plt.savefig(shap_summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    most_important_feature = importance_df.iloc[0]["feature"]
    shap_dependence_path = PLOTS_DIR / f"shap_dependence_{safe_filename(most_important_feature)}.png"
    shap.dependence_plot(
        most_important_feature,
        shap_values,
        x_explain,
        show=False,
        interaction_index=None,
    )
    plt.tight_layout()
    plt.savefig(shap_dependence_path, dpi=300, bbox_inches="tight")
    plt.close()

    shap_heatmap_path = PLOTS_DIR / "shap_heatmap_plot.png"
    explanation = shap.Explanation(
        values=top_shap_values,
        base_values=base_values,
        data=top_x_explain.to_numpy(),
        feature_names=list(top_x_explain.columns),
    )
    shap.plots.heatmap(explanation, show=False, max_display=top_feature_count)
    plt.gcf().set_size_inches(11, max(5, 0.8 * top_feature_count))
    plt.tight_layout()
    plt.savefig(shap_heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nSHAP dependence feature: {most_important_feature}")
    return [shap_summary_path, shap_dependence_path, shap_heatmap_path]


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


def print_split_summary(split_name: str, y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    hpv_0 = int(counts.get(0, 0))
    hpv_1 = int(counts.get(1, 0))
    print(f"{split_name}: {len(y)} patients | HPV 0: {hpv_0} | HPV 1: {hpv_1}")


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
        print("\nAutomatic plot opening is disabled because OPEN_PLOTS=0.")
        return

    print("\nOpening saved PNG plots...")
    for path in plot_paths:
        try:
            if sys.platform.startswith("win"):
                os.startfile(path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as error:
            print(f"Could not open {path.name}: {error}")


def main() -> None:
    train_split = load_split(TRAIN_PATH, "Train")
    validation_split = load_split(VALIDATION_PATH, "Validation")
    test_split = load_split(TEST_PATH, "Test")
    pet_features, feature_columns = load_pet_features()

    x_train, y_train = build_xy(train_split, "Train", pet_features, feature_columns)
    x_validation, y_validation = build_xy(validation_split, "Validation", pet_features, feature_columns)
    x_test, y_test = build_xy(test_split, "Test", pet_features, feature_columns)

    print("PET-only LightGBM baseline")
    print(f"Working folder: {BASE_DIR}")
    print(f"Python executable: {sys.executable}")
    print("Split source files: train.csv, validation.csv, test.csv")
    print(f"PET feature source file: {PET_FEATURES_PATH.name}")
    print(f"Target column: {TARGET_COLUMN}")
    print(f"Input feature count: {len(feature_columns)} PET radiomics features")
    print("Excluded from input: PatientID, HPV")
    print("Preprocessing: no imputation, no scaling, no normalization, no feature selection")
    print(f"Classification threshold: {THRESHOLD}")
    print("\nSplit Summary")
    print_split_summary("Train", y_train)
    print_split_summary("Validation", y_validation)
    print_split_summary("Test", y_test)

    model = train_model(x_train, y_train)

    validation_metrics, validation_calibration, validation_plots = evaluate_split(
        "Validation",
        model,
        x_validation,
        y_validation,
    )
    test_metrics, test_calibration, test_plots = evaluate_split("Test", model, x_test, y_test)
    feature_importance_plot, importance_df = plot_feature_importance(model, feature_columns)
    shap_plots = plot_shap_outputs(model, x_test, feature_columns, importance_df)

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
