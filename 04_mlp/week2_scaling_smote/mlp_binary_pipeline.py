from pathlib import Path
import os
import textwrap
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
POPUP_REPORT = os.environ.get("POPUP_REPORT", "1") != "0"

import matplotlib

if POPUP_REPORT:
    try:
        matplotlib.use("TkAgg")
    except Exception:
        matplotlib.use("Agg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from imblearn.over_sampling import SMOTE, SMOTENC
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


PATIENT_ID_COLUMN = "PatientID"
TARGET_COLUMN = "HPV Status"
SPLIT_TARGET_CANDIDATES = ["HPV Status", "HPV"]
SEED = 42
EPOCHS = 60
BATCH_SIZE = 32
DECISION_THRESHOLD = 0.5
CALIBRATION_BINS = 10
SHAP_BACKGROUND_SIZE = 30
SHAP_EXPLAIN_SIZE = 40
SHAP_NSAMPLES = 100
PERMUTATION_REPEATS = 5


@dataclass(frozen=True)
class PipelineConfig:
    model_name: str
    folder: Path
    input_csv: str
    sampler: str
    categorical_columns: list[str]
    excluded_columns: list[str]
    scale_features: bool
    scale_feature_prefixes: list[str]
    compare_before_scaling: bool
    plot_top_n: int | None


def run_mlp_classification(config: PipelineConfig) -> None:
    """Run one binary MLP classification experiment using fixed train/val/test splits."""
    tf.keras.utils.set_random_seed(SEED)

    output_dir = config.folder / "plots"
    output_dir.mkdir(exist_ok=True)

    entry_dataframe = load_csv(config.folder / config.input_csv, "entry input")
    split_data = {
        "Validation": build_split_dataset(entry_dataframe, config.folder / "validation.csv"),
        "Test": build_split_dataset(entry_dataframe, config.folder / "test.csv"),
    }
    train_dataset = build_split_dataset(entry_dataframe, config.folder / "train.csv")

    excluded_input_columns = [PATIENT_ID_COLUMN, TARGET_COLUMN, *config.excluded_columns]
    feature_columns = [
        column
        for column in train_dataset.columns
        if column not in excluded_input_columns
    ]
    validate_dataset(train_dataset, feature_columns, "Training")
    for split_name, split_dataframe in split_data.items():
        validate_dataset(split_dataframe, feature_columns, split_name)

    x_train = train_dataset[feature_columns].astype("float32")
    y_train = train_dataset[TARGET_COLUMN].astype(int)
    x_validation = split_data["Validation"][feature_columns].astype("float32")
    y_validation = split_data["Validation"][TARGET_COLUMN].astype(int)
    x_test = split_data["Test"][feature_columns].astype("float32")
    y_test = split_data["Test"][TARGET_COLUMN].astype(int)

    comparison_metrics_table = run_before_scaling_comparison(
        x_train,
        y_train,
        x_validation,
        y_validation,
        x_test,
        y_test,
        config,
    )

    x_train, x_validation, x_test, scaling_report = apply_optional_scaling(
        x_train,
        x_validation,
        x_test,
        config,
    )

    x_train_resampled, y_train_resampled, smote_report = apply_training_resampling(
        x_train,
        y_train,
        config,
    )

    tf.keras.utils.set_random_seed(SEED)
    model = build_mlp_model(input_dim=x_train_resampled.shape[1])
    history = model.fit(
        x_train_resampled.to_numpy(dtype="float32"),
        y_train_resampled.to_numpy(dtype="float32"),
        validation_data=(
            x_validation.to_numpy(dtype="float32"),
            y_validation.to_numpy(dtype="float32"),
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    validation_probabilities = predict_probabilities(model, x_validation)
    test_probabilities = predict_probabilities(model, x_test)

    metrics_table = build_metrics_table(
        y_validation,
        validation_probabilities,
        y_test,
        test_probabilities,
    )
    metrics_table.to_csv(output_dir / "validation_test_metrics.csv", index=False)
    comparison_metrics_table = finalize_scaling_comparison(
        output_dir,
        comparison_metrics_table,
        metrics_table,
        scaling_report,
    )

    save_confusion_matrix_csv(y_validation, validation_probabilities, output_dir, "validation")
    save_confusion_matrix_csv(y_test, test_probabilities, output_dir, "test")

    validation_calibration = build_calibration_table(y_validation, validation_probabilities)
    test_calibration = build_calibration_table(y_test, test_probabilities)
    validation_calibration.to_csv(output_dir / "validation_calibration_table.csv", index=False)
    test_calibration.to_csv(output_dir / "test_calibration_table.csv", index=False)
    pd.concat(
        [
            validation_calibration.assign(Split="Validation"),
            test_calibration.assign(Split="Test"),
        ],
        ignore_index=True,
    ).to_csv(output_dir / "validation_test_calibration_table.csv", index=False)

    pd.DataFrame(history.history).to_csv(output_dir / "training_history.csv", index=False)
    write_scaling_report(output_dir, scaling_report)
    write_smote_report(output_dir, smote_report)

    plot_roc_curves(
        y_validation,
        validation_probabilities,
        y_test,
        test_probabilities,
        output_dir / "roc_curve.png",
    )
    plot_precision_recall_curves(
        y_validation,
        validation_probabilities,
        y_test,
        test_probabilities,
        output_dir / "precision_recall_curve.png",
    )
    plot_calibration_curves(
        validation_calibration,
        test_calibration,
        output_dir / "calibration_curve.png",
    )
    plot_confusion_matrices(
        y_validation,
        validation_probabilities,
        y_test,
        test_probabilities,
        output_dir / "confusion_matrix_heatmap.png",
    )
    plot_probability_distributions(
        y_validation,
        validation_probabilities,
        y_test,
        test_probabilities,
        output_dir / "probability_distribution_plot.png",
    )
    plot_training_history(history, output_dir / "training_loss_curve.png")

    importance_table = calculate_permutation_importance(
        model,
        x_validation,
        y_validation,
        feature_columns,
    )
    importance_table.to_csv(output_dir / "permutation_feature_importance.csv", index=False)
    plot_feature_importance(
        importance_table,
        output_dir / "feature_importance_plot.png",
        config.plot_top_n,
    )

    shap_summary = create_shap_plots(
        model,
        x_train_resampled,
        x_validation,
        output_dir,
        config.plot_top_n,
    )
    write_run_summary(
        config,
        output_dir,
        train_dataset,
        split_data,
        feature_columns,
        metrics_table,
        comparison_metrics_table,
        scaling_report,
        smote_report,
        shap_summary,
    )
    save_full_classification_report(
        config,
        output_dir,
        train_dataset,
        split_data,
        feature_columns,
        metrics_table,
        comparison_metrics_table,
        y_validation,
        validation_probabilities,
        y_test,
        test_probabilities,
        validation_calibration,
        test_calibration,
        scaling_report,
        smote_report,
        shap_summary,
    )

    print(f"\nFinished {config.model_name}")
    print(f"Outputs saved to: {output_dir}")
    print(metrics_table.to_string(index=False))


def load_csv(path: Path, label: str) -> pd.DataFrame:
    """Load a CSV and fail clearly if it is missing."""
    if not path.exists():
        raise FileNotFoundError(f"{label} file not found: {path}")

    return pd.read_csv(path)


def build_split_dataset(entry_dataframe: pd.DataFrame, split_path: Path) -> pd.DataFrame:
    """Use the existing split file to select entry rows and attach the target."""
    split_dataframe = load_csv(split_path, split_path.name)

    if PATIENT_ID_COLUMN not in entry_dataframe.columns:
        raise ValueError(f"Entry file is missing {PATIENT_ID_COLUMN}.")
    if PATIENT_ID_COLUMN not in split_dataframe.columns:
        raise ValueError(f"{split_path.name} is missing {PATIENT_ID_COLUMN}.")

    target_column = find_target_column(split_dataframe, split_path.name)
    split_labels = split_dataframe[[PATIENT_ID_COLUMN, target_column]].rename(
        columns={target_column: TARGET_COLUMN}
    )

    entry_features = entry_dataframe.drop(
        columns=[column for column in SPLIT_TARGET_CANDIDATES if column in entry_dataframe.columns]
    )
    merged_split = split_labels.merge(
        entry_features,
        on=PATIENT_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    missing_feature_rows = int(merged_split.drop(columns=[TARGET_COLUMN]).isna().any(axis=1).sum())
    if missing_feature_rows:
        raise ValueError(
            f"{split_path.name} has {missing_feature_rows} PatientID values that are "
            "not available in the entry input file."
        )

    feature_columns = [
        column for column in merged_split.columns if column not in [PATIENT_ID_COLUMN, TARGET_COLUMN]
    ]
    return merged_split[[PATIENT_ID_COLUMN, *feature_columns, TARGET_COLUMN]]


def find_target_column(dataframe: pd.DataFrame, label: str) -> str:
    """Find the target column in a split file and map it to HPV Status in memory."""
    for candidate in SPLIT_TARGET_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(f"{label} must contain one of these target columns: {SPLIT_TARGET_CANDIDATES}")


def validate_dataset(dataframe: pd.DataFrame, feature_columns: list[str], split_name: str) -> None:
    """Confirm data is numeric, complete, and ready for the MLP run."""
    if dataframe[PATIENT_ID_COLUMN].duplicated().any():
        raise ValueError(f"{split_name} split has duplicated PatientID values.")

    if dataframe[[*feature_columns, TARGET_COLUMN]].isna().any().any():
        raise ValueError(
            f"{split_name} split contains missing values. The rules say not to "
            "manually impute in the classification script."
        )

    for column in feature_columns:
        converted_column = pd.to_numeric(dataframe[column], errors="coerce")
        if converted_column.isna().any():
            raise ValueError(f"{split_name} feature is not numeric: {column}")

    unique_targets = sorted(dataframe[TARGET_COLUMN].astype(int).unique())
    if unique_targets != [0, 1]:
        raise ValueError(f"{split_name} target must be binary 0/1. Found: {unique_targets}")


def apply_optional_scaling(
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    x_test: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Optionally fit StandardScaler on training features only."""
    columns_to_scale = get_columns_to_scale(x_train.columns.tolist(), config)
    if not config.scale_features:
        return (
            x_train,
            x_validation,
            x_test,
            {
                "scaling_applied": False,
                "method": "none",
                "fit_split": "none",
                "scaled_feature_count": 0,
                "unscaled_feature_count": len(x_train.columns),
                "scale_scope": "none",
                "scale_feature_prefixes": config.scale_feature_prefixes,
            },
        )

    if not columns_to_scale:
        raise ValueError("Scaling was requested, but no matching feature columns were found.")

    scaler = StandardScaler()
    scaled_train = x_train.copy()
    scaled_validation = x_validation.copy()
    scaled_test = x_test.copy()
    scaled_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale]).astype(
        "float32"
    )
    scaled_validation[columns_to_scale] = scaler.transform(
        x_validation[columns_to_scale]
    ).astype("float32")
    scaled_test[columns_to_scale] = scaler.transform(x_test[columns_to_scale]).astype(
        "float32"
    )

    return (
        scaled_train.astype("float32"),
        scaled_validation.astype("float32"),
        scaled_test.astype("float32"),
        {
            "scaling_applied": True,
            "method": "StandardScaler",
            "fit_split": "Training only",
            "scaled_feature_count": len(columns_to_scale),
            "unscaled_feature_count": len(x_train.columns) - len(columns_to_scale),
            "scale_scope": build_scale_scope_text(config, columns_to_scale),
            "scale_feature_prefixes": config.scale_feature_prefixes,
        },
    )


def get_columns_to_scale(feature_columns: list[str], config: PipelineConfig) -> list[str]:
    """Return the feature columns selected for StandardScaler."""
    if not config.scale_features:
        return []
    if not config.scale_feature_prefixes:
        return feature_columns
    return [
        column
        for column in feature_columns
        if any(column.startswith(prefix) for prefix in config.scale_feature_prefixes)
    ]


def build_scale_scope_text(config: PipelineConfig, columns_to_scale: list[str]) -> str:
    """Describe which features were selected for scaling."""
    if not config.scale_feature_prefixes:
        return "all input features"
    return (
        f"features starting with {', '.join(config.scale_feature_prefixes)} "
        f"({len(columns_to_scale)} columns)"
    )


def run_before_scaling_comparison(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    config: PipelineConfig,
) -> pd.DataFrame | None:
    """Train a no-scaling comparison model before the official scaled model."""
    if not (config.scale_features and config.compare_before_scaling):
        return None

    tf.keras.utils.set_random_seed(SEED)
    x_train_resampled, y_train_resampled, _report = apply_training_resampling(
        x_train,
        y_train,
        config,
    )
    model = build_mlp_model(input_dim=x_train_resampled.shape[1])
    model.fit(
        x_train_resampled.to_numpy(dtype="float32"),
        y_train_resampled.to_numpy(dtype="float32"),
        validation_data=(
            x_validation.to_numpy(dtype="float32"),
            y_validation.to_numpy(dtype="float32"),
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    validation_probabilities = predict_probabilities(model, x_validation)
    test_probabilities = predict_probabilities(model, x_test)
    metrics_table = build_metrics_table(
        y_validation,
        validation_probabilities,
        y_test,
        test_probabilities,
    )
    metrics_table.insert(0, "Experiment", "Before scaling")
    return metrics_table


def finalize_scaling_comparison(
    output_dir: Path,
    before_metrics_table: pd.DataFrame | None,
    after_metrics_table: pd.DataFrame,
    scaling_report: dict[str, object],
) -> pd.DataFrame | None:
    """Save before/after scaling comparison metrics and plot."""
    if before_metrics_table is None:
        return None

    after_comparison = after_metrics_table.copy()
    after_label = f"After {scaling_report['method']}"
    after_comparison.insert(0, "Experiment", after_label)
    comparison_table = pd.concat(
        [before_metrics_table, after_comparison],
        ignore_index=True,
    )
    comparison_table.to_csv(output_dir / "scaling_comparison_metrics.csv", index=False)
    plot_scaling_comparison(
        comparison_table,
        output_dir / "scaling_comparison_plot.png",
    )
    return comparison_table


def apply_training_resampling(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """Apply SMOTE or SMOTENC to the training data only."""
    before_counts = y_train.value_counts().sort_index().to_dict()

    if config.sampler.upper() == "SMOTE":
        sampler = SMOTE(random_state=SEED)
        sampler_name = "SMOTE"
        categorical_columns = []
    elif config.sampler.upper() == "SMOTENC":
        categorical_columns = [
            column for column in config.categorical_columns if column in x_train.columns
        ]
        categorical_indices = [x_train.columns.get_loc(column) for column in categorical_columns]
        if not categorical_indices:
            raise ValueError("SMOTENC was requested but no categorical columns were found.")
        sampler = SMOTENC(categorical_features=categorical_indices, random_state=SEED)
        sampler_name = "SMOTENC"
    else:
        raise ValueError(f"Unknown sampler: {config.sampler}")

    x_resampled, y_resampled = sampler.fit_resample(x_train, y_train)
    x_resampled = pd.DataFrame(x_resampled, columns=x_train.columns).astype("float32")
    y_resampled = pd.Series(y_resampled, name=TARGET_COLUMN).astype(int)
    after_counts = y_resampled.value_counts().sort_index().to_dict()

    report = {
        "sampler": sampler_name,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "categorical_columns": categorical_columns,
        "original_train_rows": int(len(y_train)),
        "resampled_train_rows": int(len(y_resampled)),
    }

    return x_resampled, y_resampled, report


def build_mlp_model(input_dim: int) -> keras.Model:
    """Build one fixed MLP architecture for all three tasks."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.20),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.AUC(name="roc_auc"),
        ],
    )
    return model


def predict_probabilities(model: keras.Model, x_values: pd.DataFrame) -> np.ndarray:
    """Return positive-class probabilities from the trained MLP."""
    probabilities = model.predict(x_values.to_numpy(dtype="float32"), verbose=0)
    return probabilities.reshape(-1)


def build_metrics_table(
    y_validation: pd.Series,
    validation_probabilities: np.ndarray,
    y_test: pd.Series,
    test_probabilities: np.ndarray,
) -> pd.DataFrame:
    """Create one table containing all requested validation and test metrics."""
    rows = [
        build_metric_row("Validation", y_validation, validation_probabilities),
        build_metric_row("Test", y_test, test_probabilities),
    ]
    return pd.DataFrame(rows)


def build_metric_row(split_name: str, y_true: pd.Series, probabilities: np.ndarray) -> dict[str, object]:
    """Calculate all requested metrics for one split."""
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)

    return {
        "Split": split_name,
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_true, predictions),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "F1-score": f1_score(y_true, predictions, zero_division=0),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Confusion Matrix": f"[[{tn}, {fp}], [{fn}, {tp}]]",
        "Brier Score": brier_score_loss(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "Expected Calibration Error (ECE)": calculate_ece(y_true, probabilities),
    }


def safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division-by-zero in sensitivity and specificity."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def calculate_ece(y_true: pd.Series, probabilities: np.ndarray) -> float:
    """Calculate Expected Calibration Error using fixed 10 probability bins."""
    calibration_table = build_calibration_table(y_true, probabilities)
    total_count = calibration_table["Sample count"].sum()
    if total_count == 0:
        return 0.0

    weighted_gap = (
        calibration_table["Sample count"] * calibration_table["Absolute calibration gap"]
    ).sum()
    return float(weighted_gap / total_count)


def build_calibration_table(y_true: pd.Series, probabilities: np.ndarray) -> pd.DataFrame:
    """Build the requested 10-bin calibration matrix/table."""
    y_array = np.asarray(y_true, dtype=int)
    probability_array = np.asarray(probabilities, dtype=float)
    rows = []

    for bin_index in range(CALIBRATION_BINS):
        lower = bin_index / CALIBRATION_BINS
        upper = (bin_index + 1) / CALIBRATION_BINS
        if bin_index == CALIBRATION_BINS - 1:
            mask = (probability_array >= lower) & (probability_array <= upper)
            bin_range = f"[{lower:.1f}, {upper:.1f}]"
        else:
            mask = (probability_array >= lower) & (probability_array < upper)
            bin_range = f"[{lower:.1f}, {upper:.1f})"

        sample_count = int(mask.sum())
        if sample_count == 0:
            mean_probability = np.nan
            observed_rate = np.nan
            calibration_gap = np.nan
        else:
            mean_probability = float(probability_array[mask].mean())
            observed_rate = float(y_array[mask].mean())
            calibration_gap = abs(mean_probability - observed_rate)

        rows.append(
            {
                "Bin range": bin_range,
                "Sample count": sample_count,
                "Mean predicted probability": mean_probability,
                "Observed HPV positive rate": observed_rate,
                "Absolute calibration gap": calibration_gap,
            }
        )

    return pd.DataFrame(rows)


def save_confusion_matrix_csv(
    y_true: pd.Series,
    probabilities: np.ndarray,
    output_dir: Path,
    split_label: str,
) -> None:
    """Save one 2x2 confusion matrix CSV."""
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    matrix_dataframe = pd.DataFrame(
        matrix,
        index=["Actual HPV 0", "Actual HPV 1"],
        columns=["Predicted HPV 0", "Predicted HPV 1"],
    )
    matrix_dataframe.to_csv(output_dir / f"{split_label}_confusion_matrix.csv")


def plot_roc_curves(
    y_validation: pd.Series,
    validation_probabilities: np.ndarray,
    y_test: pd.Series,
    test_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save validation/test ROC curves in one PNG."""
    figure, axis = plt.subplots(figsize=(7, 6))
    for label, y_true, probabilities in [
        ("Validation", y_validation, validation_probabilities),
        ("Test", y_test, test_probabilities),
    ]:
        fpr, tpr, _ = roc_curve(y_true, probabilities)
        auc_value = roc_auc_score(y_true, probabilities)
        axis.plot(fpr, tpr, label=f"{label} ROC-AUC={auc_value:.3f}")

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_title("ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_precision_recall_curves(
    y_validation: pd.Series,
    validation_probabilities: np.ndarray,
    y_test: pd.Series,
    test_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save validation/test precision-recall curves in one PNG."""
    figure, axis = plt.subplots(figsize=(7, 6))
    for label, y_true, probabilities in [
        ("Validation", y_validation, validation_probabilities),
        ("Test", y_test, test_probabilities),
    ]:
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        pr_auc = average_precision_score(y_true, probabilities)
        axis.plot(recall, precision, label=f"{label} PR-AUC={pr_auc:.3f}")

    axis.set_title("Precision-Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_calibration_curves(
    validation_calibration: pd.DataFrame,
    test_calibration: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save validation/test calibration curves in one PNG."""
    figure, axis = plt.subplots(figsize=(7, 6))
    for label, table in [
        ("Validation", validation_calibration),
        ("Test", test_calibration),
    ]:
        non_empty_bins = table[table["Sample count"] > 0]
        axis.plot(
            non_empty_bins["Mean predicted probability"],
            non_empty_bins["Observed HPV positive rate"],
            marker="o",
            label=label,
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.set_title("Calibration Curve (10 Bins)")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed HPV positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_confusion_matrices(
    y_validation: pd.Series,
    validation_probabilities: np.ndarray,
    y_test: pd.Series,
    test_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save validation/test confusion matrices side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, label, y_true, probabilities in [
        (axes[0], "Validation", y_validation, validation_probabilities),
        (axes[1], "Test", y_test, test_probabilities),
    ]:
        predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_title(f"{label} Confusion Matrix")
        axis.set_xticks([0, 1])
        axis.set_yticks([0, 1])
        axis.set_xticklabels(["Pred 0", "Pred 1"])
        axis.set_yticklabels(["Actual 0", "Actual 1"])
        for row_index in range(2):
            for column_index in range(2):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=12,
                )
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_probability_distributions(
    y_validation: pd.Series,
    validation_probabilities: np.ndarray,
    y_test: pd.Series,
    test_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    """Save predicted probability distributions by true class."""
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, label, y_true, probabilities in [
        (axes[0], "Validation", y_validation, validation_probabilities),
        (axes[1], "Test", y_test, test_probabilities),
    ]:
        y_array = np.asarray(y_true, dtype=int)
        axis.hist(
            probabilities[y_array == 0],
            bins=10,
            alpha=0.65,
            label="Actual HPV 0",
            color="#4C78A8",
        )
        axis.hist(
            probabilities[y_array == 1],
            bins=10,
            alpha=0.65,
            label="Actual HPV 1",
            color="#E15759",
        )
        axis.axvline(DECISION_THRESHOLD, linestyle="--", color="black", label="0.5 threshold")
        axis.set_title(f"{label} Probability Distribution")
        axis.set_xlabel("Predicted HPV-positive probability")
        axis.set_ylabel("Number of patients")
        axis.set_xlim(0, 1)
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_training_history(history: keras.callbacks.History, output_path: Path) -> None:
    """Save training and validation loss curves."""
    history_dataframe = pd.DataFrame(history.history)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(history_dataframe["loss"], label="Training loss")
    axis.plot(history_dataframe["val_loss"], label="Validation loss")
    axis.set_title("Training Loss Curve")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Binary crossentropy")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_scaling_comparison(comparison_table: pd.DataFrame, output_path: Path) -> None:
    """Save a before/after scaling metric comparison plot."""
    metrics_to_plot = [
        "ROC-AUC",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "F1-score",
        "Brier Score",
        "Expected Calibration Error (ECE)",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for axis, split_name in zip(axes, ["Validation", "Test"]):
        split_table = comparison_table[comparison_table["Split"] == split_name].set_index(
            "Experiment"
        )
        x_positions = np.arange(len(metrics_to_plot))
        bar_width = 0.36
        experiments = list(split_table.index)
        for index, experiment in enumerate(experiments):
            offset = (index - 0.5) * bar_width
            axis.bar(
                x_positions + offset,
                split_table.loc[experiment, metrics_to_plot].astype(float),
                width=bar_width,
                label=experiment,
            )

        axis.set_title(f"{split_name}: Before vs After Scaling")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(metrics_to_plot, rotation=35, ha="right")
        axis.set_ylim(0, 1.05)
        axis.grid(axis="y", alpha=0.3)
        axis.legend()

    figure.suptitle("PET Scaling Comparison", fontsize=14, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def calculate_permutation_importance(
    model: keras.Model,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Estimate post-hoc feature importance by validation ROC-AUC drop."""
    baseline_probabilities = predict_probabilities(model, x_validation)
    baseline_auc = roc_auc_score(y_validation, baseline_probabilities)
    rng = np.random.default_rng(SEED)
    rows = []

    for feature in feature_columns:
        drops = []
        for _ in range(PERMUTATION_REPEATS):
            permuted_x = x_validation.copy()
            permuted_x[feature] = rng.permutation(permuted_x[feature].to_numpy())
            permuted_probabilities = predict_probabilities(model, permuted_x)
            permuted_auc = roc_auc_score(y_validation, permuted_probabilities)
            drops.append(baseline_auc - permuted_auc)

        rows.append(
            {
                "Feature": feature,
                "Mean ROC-AUC drop": float(np.mean(drops)),
                "Std ROC-AUC drop": float(np.std(drops)),
            }
        )

    importance_table = pd.DataFrame(rows)
    return importance_table.sort_values("Mean ROC-AUC drop", ascending=False).reset_index(drop=True)


def plot_feature_importance(
    importance_table: pd.DataFrame,
    output_path: Path,
    top_n: int | None,
) -> None:
    """Save a post-hoc feature importance plot."""
    plot_table = importance_table.copy()
    if top_n is not None:
        plot_table = plot_table.head(top_n)

    plot_table = plot_table.sort_values("Mean ROC-AUC drop", ascending=True)
    figure, axis = plt.subplots(figsize=(9, max(4, 0.38 * len(plot_table))))
    axis.barh(plot_table["Feature"], plot_table["Mean ROC-AUC drop"], color="#4C78A8")
    axis.set_title("Permutation Feature Importance")
    axis.set_xlabel("Mean validation ROC-AUC drop after permutation")
    axis.grid(axis="x", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def create_shap_plots(
    model: keras.Model,
    x_train_resampled: pd.DataFrame,
    x_validation: pd.DataFrame,
    output_dir: Path,
    top_n: int | None,
) -> dict[str, object]:
    """Create SHAP summary, dependence, and heatmap plots using the trained MLP."""
    background = x_train_resampled.sample(
        n=min(SHAP_BACKGROUND_SIZE, len(x_train_resampled)),
        random_state=SEED,
    )
    explained = x_validation.sample(
        n=min(SHAP_EXPLAIN_SIZE, len(x_validation)),
        random_state=SEED,
    )

    def predict_for_shap(values: np.ndarray) -> np.ndarray:
        return model.predict(np.asarray(values, dtype="float32"), verbose=0).reshape(-1)

    explainer = shap.KernelExplainer(predict_for_shap, background)
    shap_values = explainer.shap_values(explained, nsamples=SHAP_NSAMPLES)
    shap_array = normalize_shap_values(shap_values)

    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    top_feature_index = int(np.argmax(mean_abs_shap))
    top_feature = explained.columns[top_feature_index]
    max_display = top_n if top_n is not None else min(20, explained.shape[1])

    shap_importance = pd.DataFrame(
        {
            "Feature": explained.columns,
            "Mean absolute SHAP value": mean_abs_shap,
        }
    ).sort_values("Mean absolute SHAP value", ascending=False)
    shap_importance.to_csv(output_dir / "shap_mean_absolute_importance.csv", index=False)

    plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_array, explained, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_plot.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    shap.dependence_plot(
        top_feature,
        shap_array,
        explained,
        interaction_index=None,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_dependence_plot.png", dpi=160, bbox_inches="tight")
    plt.close()

    explanation = shap.Explanation(
        values=shap_array,
        data=explained.to_numpy(),
        feature_names=list(explained.columns),
    )
    shap.plots.heatmap(explanation, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_heatmap_plot.png", dpi=160, bbox_inches="tight")
    plt.close()

    (output_dir / "shap_dependence_feature.txt").write_text(top_feature, encoding="utf-8")
    return {
        "explained_rows": int(len(explained)),
        "background_rows": int(len(background)),
        "top_feature": top_feature,
    }


def normalize_shap_values(shap_values: object) -> np.ndarray:
    """Normalize SHAP output shape across SHAP versions."""
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3:
        shap_array = shap_array[:, :, 0]

    return shap_array


def write_smote_report(output_dir: Path, report: dict[str, object]) -> None:
    """Save a small report explaining how SMOTE/SMOTENC changed the training set."""
    categorical_columns = report["categorical_columns"]
    if categorical_columns:
        categorical_text = ", ".join(categorical_columns)
    else:
        categorical_text = "none; all features were treated as continuous"

    lines = [
        "SMOTE / SMOTENC training-only resampling report",
        "",
        f"Sampler used: {report['sampler']}",
        f"Original training rows: {report['original_train_rows']}",
        f"Resampled training rows: {report['resampled_train_rows']}",
        f"Class counts before resampling: {report['before_counts']}",
        f"Class counts after resampling: {report['after_counts']}",
        f"Categorical columns for SMOTENC: {categorical_text}",
        "",
        "Resampling was applied only to the training set. Validation and test sets "
        "were never resampled.",
    ]
    (output_dir / "smote_report.txt").write_text("\n".join(lines), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "Sampler": report["sampler"],
                "Original training rows": report["original_train_rows"],
                "Resampled training rows": report["resampled_train_rows"],
                "Class counts before": str(report["before_counts"]),
                "Class counts after": str(report["after_counts"]),
                "Categorical column count": len(categorical_columns),
            }
        ]
    ).to_csv(output_dir / "smote_report.csv", index=False)


def write_scaling_report(output_dir: Path, report: dict[str, object]) -> None:
    """Save a small report explaining whether scaling was applied."""
    if report["scaling_applied"]:
        lines = [
            "Scaling report",
            "",
            f"Scaling applied: yes",
            f"Method: {report['method']}",
            f"Scale scope: {report['scale_scope']}",
            f"Scaler fit split: {report['fit_split']}",
            f"Scaled feature count: {report['scaled_feature_count']}",
            f"Unscaled feature count: {report['unscaled_feature_count']}",
            "",
            "The scaler was fitted on the training feature matrix only. The same fitted "
            "scaler transformed validation and test features before prediction.",
        ]
    else:
        lines = [
            "Scaling report",
            "",
            "Scaling applied: no",
        ]
    (output_dir / "scaling_report.txt").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame([report]).to_csv(output_dir / "scaling_report.csv", index=False)


def write_run_summary(
    config: PipelineConfig,
    output_dir: Path,
    train_dataset: pd.DataFrame,
    split_data: dict[str, pd.DataFrame],
    feature_columns: list[str],
    metrics_table: pd.DataFrame,
    comparison_metrics_table: pd.DataFrame | None,
    scaling_report: dict[str, object],
    smote_report: dict[str, object],
    shap_summary: dict[str, object],
) -> None:
    """Save a readable summary for someone new to deep learning."""
    split_lines = [
        f"Training rows: {len(train_dataset)}",
        f"Validation rows: {len(split_data['Validation'])}",
        f"Test rows: {len(split_data['Test'])}",
    ]
    target_lines = [
        f"Training target counts: {train_dataset[TARGET_COLUMN].value_counts().sort_index().to_dict()}",
        f"Validation target counts: {split_data['Validation'][TARGET_COLUMN].value_counts().sort_index().to_dict()}",
        f"Test target counts: {split_data['Test'][TARGET_COLUMN].value_counts().sort_index().to_dict()}",
    ]
    rules = [
        "No new split was created.",
        "No hyperparameter tuning was performed.",
        "No automatic feature selection, filtering, or feature engineering was performed.",
        f"Explicitly ignored columns by request: {format_excluded_columns(config.excluded_columns)}.",
        build_scaling_rule_text(scaling_report),
        "No manual imputation was performed in this classification script.",
        "PatientID was not used as an input feature.",
        "HPV Status was used only as the target label, never as an input feature.",
    ]
    smote_text = (
        f"{smote_report['sampler']} was applied only on the training set. "
        f"Class counts changed from {smote_report['before_counts']} to "
        f"{smote_report['after_counts']}."
    )
    shap_text = (
        f"SHAP used {shap_summary['background_rows']} background rows and explained "
        f"{shap_summary['explained_rows']} validation rows. The automatically selected "
        f"dependence feature was {shap_summary['top_feature']}."
    )

    summary = [
        f"{config.model_name} MLP Binary Classification Report",
        "",
        "Data",
        *split_lines,
        f"Input feature count: {len(feature_columns)}",
        "",
        "Target balance",
        *target_lines,
        "",
        "Scaling",
        build_scaling_rule_text(scaling_report),
        "",
        "Rules followed",
        *[f"- {rule}" for rule in rules],
        "",
        "SMOTE / SMOTENC",
        textwrap.fill(smote_text, width=90),
        "",
        "SHAP",
        textwrap.fill(shap_text, width=90),
        "",
        "Metrics",
        metrics_table.to_string(index=False),
    ]
    if comparison_metrics_table is not None:
        summary.extend(
            [
                "",
                "Before/After Scaling Comparison",
                comparison_metrics_table.to_string(index=False),
            ]
        )
    (output_dir / "run_summary.txt").write_text("\n".join(summary), encoding="utf-8")


def save_full_classification_report(
    config: PipelineConfig,
    output_dir: Path,
    train_dataset: pd.DataFrame,
    split_data: dict[str, pd.DataFrame],
    feature_columns: list[str],
    metrics_table: pd.DataFrame,
    comparison_metrics_table: pd.DataFrame | None,
    y_validation: pd.Series,
    validation_probabilities: np.ndarray,
    y_test: pd.Series,
    test_probabilities: np.ndarray,
    validation_calibration: pd.DataFrame,
    test_calibration: pd.DataFrame,
    scaling_report: dict[str, object],
    smote_report: dict[str, object],
    shap_summary: dict[str, object],
) -> None:
    """Save one readable, vertically scrollable Matplotlib report."""
    figure = Figure(figsize=(16, 80), dpi=110)
    grid = figure.add_gridspec(
        11,
        2,
        height_ratios=[2.1, 4.2, 3.0, 5.2, 4.8, 4.8, 4.8, 5.0, 5.0, 5.0, 7.0],
        hspace=0.70,
        wspace=0.28,
    )

    title_axis = figure.add_subplot(grid[0, :])
    comparison_table_axis = figure.add_subplot(grid[1, 0])
    comparison_plot_axis = figure.add_subplot(grid[1, 1])
    validation_metrics_axis = figure.add_subplot(grid[2, 0])
    test_metrics_axis = figure.add_subplot(grid[2, 1])
    validation_calibration_axis = figure.add_subplot(grid[3, 0])
    test_calibration_axis = figure.add_subplot(grid[3, 1])

    validation_roc_axis = figure.add_subplot(grid[4, 0])
    test_roc_axis = figure.add_subplot(grid[4, 1])
    validation_pr_axis = figure.add_subplot(grid[5, 0])
    test_pr_axis = figure.add_subplot(grid[5, 1])
    validation_curve_axis = figure.add_subplot(grid[6, 0])
    test_curve_axis = figure.add_subplot(grid[6, 1])
    confusion_axis = figure.add_subplot(grid[7, 0])
    probability_axis = figure.add_subplot(grid[7, 1])
    feature_axis = figure.add_subplot(grid[8, 0])
    shap_summary_axis = figure.add_subplot(grid[8, 1])
    shap_dependence_axis = figure.add_subplot(grid[9, 0])
    training_axis = figure.add_subplot(grid[9, 1])
    shap_heatmap_axis = figure.add_subplot(grid[10, :])

    draw_full_report_header(
        title_axis,
        config,
        train_dataset,
        split_data,
        feature_columns,
        scaling_report,
        smote_report,
        shap_summary,
    )
    draw_scaling_comparison_table(comparison_table_axis, comparison_metrics_table)
    if comparison_metrics_table is None:
        draw_no_comparison_note(comparison_plot_axis)
    else:
        draw_saved_png(
            comparison_plot_axis,
            output_dir / "scaling_comparison_plot.png",
            "Before vs After Scaling Plot",
        )
    draw_metrics_table(validation_metrics_axis, metrics_table, "Validation")
    draw_metrics_table(test_metrics_axis, metrics_table, "Test")
    draw_calibration_table(validation_calibration_axis, validation_calibration, "Validation")
    draw_calibration_table(test_calibration_axis, test_calibration, "Test")
    draw_single_roc_curve(validation_roc_axis, y_validation, validation_probabilities, "Validation")
    draw_single_roc_curve(test_roc_axis, y_test, test_probabilities, "Test")
    draw_single_precision_recall_curve(
        validation_pr_axis,
        y_validation,
        validation_probabilities,
        "Validation",
    )
    draw_single_precision_recall_curve(test_pr_axis, y_test, test_probabilities, "Test")
    draw_single_calibration_curve(validation_curve_axis, validation_calibration, "Validation")
    draw_single_calibration_curve(test_curve_axis, test_calibration, "Test")
    draw_saved_png(confusion_axis, output_dir / "confusion_matrix_heatmap.png", "Confusion Matrix Heatmap")
    draw_saved_png(probability_axis, output_dir / "probability_distribution_plot.png", "Probability Distribution")
    draw_saved_png(feature_axis, output_dir / "feature_importance_plot.png", "Feature Importance")
    draw_saved_png(shap_summary_axis, output_dir / "shap_summary_plot.png", "SHAP Summary Plot")
    draw_saved_png(shap_dependence_axis, output_dir / "shap_dependence_plot.png", "SHAP Dependence Plot")
    draw_saved_png(training_axis, output_dir / "training_loss_curve.png", "Training Loss Curve")
    draw_saved_png(shap_heatmap_axis, output_dir / "shap_heatmap_plot.png", "SHAP Heatmap Plot")

    figure.subplots_adjust(top=0.985, bottom=0.015, left=0.035, right=0.985)
    report_path = output_dir / "classification_full_report.png"
    figure.savefig(report_path, dpi=150, bbox_inches="tight")
    show_report_popup(figure, report_path)
    plt.close(figure)


def show_report_popup(figure: Figure, report_path: Path) -> None:
    """Show the full report in a Matplotlib window with scrollbars."""
    if not POPUP_REPORT:
        return

    try:
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        window = tk.Tk()
        window.title("MLP Classification Full Report")
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        window_width = min(1550, max(1000, screen_width - 100))
        window_height = min(900, max(700, screen_height - 140))
        window.geometry(f"{window_width}x{window_height}+40+40")

        outer_frame = ttk.Frame(window)
        outer_frame.pack(fill="both", expand=True)

        toolbar_frame = ttk.Frame(outer_frame)
        toolbar_frame.pack(fill="x")

        scroll_frame = ttk.Frame(outer_frame)
        scroll_frame.pack(fill="both", expand=True)

        y_scrollbar = ttk.Scrollbar(scroll_frame, orient="vertical")
        y_scrollbar.pack(side="right", fill="y")

        x_scrollbar = ttk.Scrollbar(scroll_frame, orient="horizontal")
        x_scrollbar.pack(side="bottom", fill="x")

        scroll_canvas = tk.Canvas(
            scroll_frame,
            bg="white",
            xscrollcommand=x_scrollbar.set,
            yscrollcommand=y_scrollbar.set,
        )
        scroll_canvas.pack(side="left", fill="both", expand=True)

        y_scrollbar.config(command=scroll_canvas.yview)
        x_scrollbar.config(command=scroll_canvas.xview)

        figure_canvas = FigureCanvasTkAgg(figure, master=scroll_canvas)
        figure_widget = figure_canvas.get_tk_widget()
        scroll_canvas.create_window((0, 0), window=figure_widget, anchor="nw")

        toolbar = NavigationToolbar2Tk(figure_canvas, toolbar_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")

        def update_scroll_region(_event: object | None = None) -> None:
            scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all"))

        def mousewheel_scroll(event: tk.Event) -> None:
            scroll_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        figure_widget.bind("<Configure>", update_scroll_region)
        scroll_canvas.bind_all("<MouseWheel>", mousewheel_scroll)
        figure_canvas.draw()
        update_scroll_region()

        print(f"Opened scrollable Matplotlib full report: {report_path}")
        window.mainloop()
    except Exception as error:
        print(f"Full report saved here: {report_path}")
        print(f"Could not show it in a Matplotlib window: {error}")


def draw_full_report_header(
    axis: Axes,
    config: PipelineConfig,
    train_dataset: pd.DataFrame,
    split_data: dict[str, pd.DataFrame],
    feature_columns: list[str],
    scaling_report: dict[str, object],
    smote_report: dict[str, object],
    shap_summary: dict[str, object],
) -> None:
    """Draw the top text block of the one-page report."""
    axis.axis("off")

    target_balance_text = (
        f"Train {train_dataset[TARGET_COLUMN].value_counts().sort_index().to_dict()} | "
        f"Validation {split_data['Validation'][TARGET_COLUMN].value_counts().sort_index().to_dict()} | "
        f"Test {split_data['Test'][TARGET_COLUMN].value_counts().sort_index().to_dict()}"
    )
    smote_text = (
        f"{smote_report['sampler']} applied to training only: "
        f"{smote_report['before_counts']} -> {smote_report['after_counts']}"
    )
    if smote_report["categorical_columns"]:
        smote_text += f"; categorical columns={len(smote_report['categorical_columns'])}"
    else:
        smote_text += "; categorical columns=none"

    shap_text = (
        f"SHAP explained {shap_summary['explained_rows']} validation rows using "
        f"{shap_summary['background_rows']} background rows; dependence feature: "
        f"{shap_summary['top_feature']}"
    )
    rule_text = (
        "Rules followed: train first, then validation, then test; no new split; "
        "no hyperparameter tuning; no automatic feature selection/filtering/engineering; "
        "no manual imputation in this script; PatientID excluded from input features; "
        "HPV Status used only as the target label."
    )
    scaling_text = build_scaling_rule_text(scaling_report)
    center_text = (
        "CenterID was explicitly ignored and was not passed into the MLP."
        if "CenterID" in config.excluded_columns
        else "CenterID was not explicitly ignored in this run."
    )

    summary_lines = [
        f"Input file: {config.input_csv}",
        f"Rows: train={len(train_dataset)}, validation={len(split_data['Validation'])}, test={len(split_data['Test'])}",
        f"Feature count used by MLP: {len(feature_columns)}",
        f"Ignored input columns: {format_excluded_columns(config.excluded_columns)}",
        scaling_text,
        center_text,
        f"Target balance: {target_balance_text}",
        smote_text,
        shap_text,
        rule_text,
    ]

    axis.text(
        0,
        0.98,
        f"{config.model_name} MLP Binary Classification Full Report",
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        transform=axis.transAxes,
    )
    axis.text(
        0,
        0.72,
        "\n".join(textwrap.fill(line, width=135) for line in summary_lines),
        ha="left",
        va="top",
        fontsize=12,
        linespacing=1.35,
        transform=axis.transAxes,
    )


def draw_metrics_table(axis: Axes, metrics_table: pd.DataFrame, split_name: str) -> None:
    """Draw one readable numerical metrics table for one split."""
    axis.axis("off")
    axis.text(
        0.5,
        0.98,
        f"{split_name} Numerical Metrics",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        transform=axis.transAxes,
    )

    row = metrics_table.loc[metrics_table["Split"] == split_name].iloc[0]
    metric_names = [
        "ROC-AUC",
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "Precision",
        "F1-score",
        "Confusion Matrix",
        "Brier Score",
        "PR-AUC",
        "Expected Calibration Error (ECE)",
    ]
    display_table = pd.DataFrame(
        {
            "Metric": metric_names,
            split_name: [format_report_value(row[metric_name]) for metric_name in metric_names],
        }
    )

    table = axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        bbox=[0.0, 0.0, 1.0, 0.84],
        cellLoc="left",
        colLoc="left",
    )
    style_table(table, font_size=10.0, y_scale=1.45)


def draw_scaling_comparison_table(
    axis: Axes,
    comparison_metrics_table: pd.DataFrame | None,
) -> None:
    """Draw before/after scaling metrics in the full report."""
    axis.axis("off")
    axis.text(
        0.5,
        0.98,
        "Before vs After Scaling Metrics",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        transform=axis.transAxes,
    )

    if comparison_metrics_table is None:
        axis.text(
            0.5,
            0.45,
            "No before/after scaling comparison was requested for this run.",
            ha="center",
            va="center",
            fontsize=11,
            transform=axis.transAxes,
        )
        return

    display_columns = [
        "Experiment",
        "Split",
        "ROC-AUC",
        "Balanced Accuracy",
        "Sensitivity",
        "Specificity",
        "F1-score",
        "Brier Score",
        "Expected Calibration Error (ECE)",
    ]
    display_table = comparison_metrics_table[display_columns].copy()
    rename_columns = {
        "Balanced Accuracy": "Bal Acc",
        "Sensitivity": "Sens",
        "Specificity": "Spec",
        "F1-score": "F1",
        "Brier Score": "Brier",
        "Expected Calibration Error (ECE)": "ECE",
    }
    display_table = display_table.rename(columns=rename_columns)
    for column in display_table.columns:
        if column not in ["Experiment", "Split"]:
            display_table[column] = display_table[column].map(format_report_value)

    table = axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        bbox=[0.0, 0.0, 1.0, 0.82],
        cellLoc="center",
    )
    style_table(table, font_size=7.6, y_scale=1.35)


def draw_no_comparison_note(axis: Axes) -> None:
    """Draw a placeholder when no scaling comparison exists."""
    axis.axis("off")
    axis.text(
        0.5,
        0.5,
        "No before/after scaling plot for this run.",
        ha="center",
        va="center",
        fontsize=11,
        transform=axis.transAxes,
    )


def draw_calibration_table(
    axis: Axes,
    calibration_table: pd.DataFrame,
    split_name: str,
) -> None:
    """Draw one readable 10-bin calibration table for one split."""
    axis.axis("off")
    axis.text(
        0.5,
        0.98,
        f"{split_name} Calibration Table (10 Bins)",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        transform=axis.transAxes,
    )

    display_table = calibration_table.rename(
        columns={
            "Bin range": "Bin",
            "Sample count": "N",
            "Mean predicted probability": "Mean pred",
            "Observed HPV positive rate": "Observed +",
            "Absolute calibration gap": "Gap",
        }
    )[["Bin", "N", "Mean pred", "Observed +", "Gap"]].copy()
    for column in [
        "Mean pred",
        "Observed +",
        "Gap",
    ]:
        display_table[column] = display_table[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )

    table = axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        bbox=[0.0, 0.0, 1.0, 0.84],
        cellLoc="center",
    )
    style_table(table, font_size=9.0, y_scale=1.45)


def draw_single_roc_curve(
    axis: Axes,
    y_true: pd.Series,
    probabilities: np.ndarray,
    split_name: str,
) -> None:
    """Draw one ROC curve directly in the full report."""
    fpr, tpr, _thresholds = roc_curve(y_true, probabilities)
    auc_value = roc_auc_score(y_true, probabilities)
    axis.plot(fpr, tpr, color="#4C78A8", linewidth=2.0, label=f"ROC-AUC={auc_value:.3f}")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_title(f"{split_name} ROC Curve", fontsize=14, fontweight="bold")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right")


def draw_single_precision_recall_curve(
    axis: Axes,
    y_true: pd.Series,
    probabilities: np.ndarray,
    split_name: str,
) -> None:
    """Draw one precision-recall curve directly in the full report."""
    precision, recall, _thresholds = precision_recall_curve(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)
    positive_rate = float(np.mean(np.asarray(y_true, dtype=int)))
    axis.plot(recall, precision, color="#F58518", linewidth=2.0, label=f"PR-AUC={pr_auc:.3f}")
    axis.axhline(
        positive_rate,
        linestyle="--",
        color="gray",
        label=f"Baseline={positive_rate:.3f}",
    )
    axis.set_title(f"{split_name} Precision-Recall Curve", fontsize=14, fontweight="bold")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.3)
    axis.legend(loc="lower left")


def draw_single_calibration_curve(
    axis: Axes,
    calibration_table: pd.DataFrame,
    split_name: str,
) -> None:
    """Draw one 10-bin calibration curve directly in the full report."""
    non_empty_bins = calibration_table[calibration_table["Sample count"] > 0]
    axis.plot(
        non_empty_bins["Mean predicted probability"],
        non_empty_bins["Observed HPV positive rate"],
        color="#54A24B",
        marker="o",
        linewidth=2.0,
        label=split_name,
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    axis.set_title(f"{split_name} Calibration Curve (10 Bins)", fontsize=14, fontweight="bold")
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed HPV positive rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.grid(alpha=0.3)
    axis.legend(loc="upper left")


def format_report_value(value: object) -> str:
    """Format table values so the report stays readable."""
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def build_scaling_rule_text(scaling_report: dict[str, object]) -> str:
    """Build a readable one-line scaling description."""
    if not scaling_report["scaling_applied"]:
        return "No scaling or normalization was performed in this classification script."

    return (
        f"{scaling_report['method']} scaling was fitted on training only for "
        f"{scaling_report['scale_scope']} and then applied to validation/test before "
        "SMOTE and prediction."
    )


def format_excluded_columns(excluded_columns: list[str]) -> str:
    """Format explicitly ignored input columns for summaries and plots."""
    if not excluded_columns:
        return "none"
    return ", ".join(excluded_columns)


def style_table(table: object, font_size: float, y_scale: float) -> None:
    """Apply a consistent compact style to Matplotlib tables."""
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, y_scale)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#EAEAEA")


def draw_saved_png(axis: Axes, image_path: Path, title: str) -> None:
    """Embed an already-saved PNG into the full report."""
    axis.axis("off")
    axis.set_title(title, fontsize=13, fontweight="bold")

    if not image_path.exists():
        axis.text(
            0.5,
            0.5,
            f"Missing image:\n{image_path.name}",
            ha="center",
            va="center",
            fontsize=11,
            transform=axis.transAxes,
        )
        return

    image = plt.imread(image_path)
    axis.imshow(image)
