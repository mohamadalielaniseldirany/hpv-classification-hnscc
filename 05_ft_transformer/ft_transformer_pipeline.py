from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import textwrap

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
import seaborn as sns
import tensorflow as tf
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
from tensorflow import keras
from tensorflow.keras import layers


PATIENT_ID_COLUMN = "PatientID"
TARGET_COLUMN = "HPV Status"
SPLIT_TARGET_CANDIDATES = ["HPV Status", "HPV"]
SEED = 42

# Fixed model settings. These are not tuned per experiment.
EPOCHS = 60
BATCH_SIZE = 32
TOKEN_DIM = 32
TRANSFORMER_BLOCKS = 3
ATTENTION_HEADS = 4
FFN_MULTIPLIER = 2
DROPOUT_RATE = 0.10
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

DECISION_THRESHOLD = 0.5
CALIBRATION_BINS = 10
PERMUTATION_REPEATS = 3
SHAP_BACKGROUND_SIZE = 25
SHAP_EXPLAIN_SIZE = 25
SHAP_NSAMPLES = 80


@dataclass(frozen=True)
class FTTransformerConfig:
    """Configuration that changes only the experiment name and input file."""

    experiment_name: str
    folder: Path
    input_csv: str
    identifier_columns: tuple[str, ...] = ("CenterID",)
    plot_top_n: int | None = None
    shap_top_n: int | None = None


class NumericalFeatureTokenizer(layers.Layer):
    """Turn each numeric column into one trainable feature token."""

    def __init__(self, token_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.token_dim = token_dim

    def build(self, input_shape):
        feature_count = int(input_shape[-1])
        self.weight = self.add_weight(
            name="feature_weight",
            shape=(feature_count, self.token_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="feature_bias",
            shape=(feature_count, self.token_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        expanded = tf.expand_dims(inputs, axis=-1)
        return expanded * tf.expand_dims(self.weight, axis=0) + tf.expand_dims(self.bias, axis=0)

    def get_config(self):
        config = super().get_config()
        config.update({"token_dim": self.token_dim})
        return config


class AddClassToken(layers.Layer):
    """Add a learned [CLS] token that becomes the final table representation."""

    def __init__(self, token_dim: int, **kwargs):
        super().__init__(**kwargs)
        self.token_dim = token_dim

    def build(self, input_shape):
        self.class_token = self.add_weight(
            name="class_token",
            shape=(1, 1, self.token_dim),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        repeated_token = tf.tile(self.class_token, [batch_size, 1, 1])
        return tf.concat([repeated_token, inputs], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"token_dim": self.token_dim})
        return config


def run_ft_transformer_classification(config: FTTransformerConfig) -> None:
    """Run one FT-Transformer binary classification experiment."""
    tf.keras.utils.set_random_seed(SEED)

    output_dir = config.folder / "plots"
    output_dir.mkdir(exist_ok=True)

    split_data, feature_columns = load_experiment_splits(config)
    x_train = split_data["Training"][feature_columns].to_numpy(dtype="float32")
    y_train = split_data["Training"][TARGET_COLUMN].to_numpy(dtype="int32")
    x_validation = split_data["Validation"][feature_columns].to_numpy(dtype="float32")
    y_validation = split_data["Validation"][TARGET_COLUMN].to_numpy(dtype="int32")
    x_test = split_data["Test"][feature_columns].to_numpy(dtype="float32")
    y_test = split_data["Test"][TARGET_COLUMN].to_numpy(dtype="int32")

    model = build_ft_transformer_model(input_dim=len(feature_columns))
    write_model_summary(model, output_dir)

    history = model.fit(
        x_train,
        y_train.astype("float32"),
        validation_data=(x_validation, y_validation.astype("float32")),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
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
        x_train,
        x_validation,
        feature_columns,
        importance_table,
        output_dir,
        config.shap_top_n,
    )

    write_run_summary(
        config,
        output_dir,
        split_data,
        feature_columns,
        metrics_table,
        shap_summary,
    )
    save_full_classification_report(
        config,
        output_dir,
        split_data,
        feature_columns,
        metrics_table,
        y_validation,
        validation_probabilities,
        y_test,
        test_probabilities,
        validation_calibration,
        test_calibration,
        shap_summary,
    )

    print(f"\nFinished {config.experiment_name} FT-Transformer classification.")
    print(f"Outputs saved to: {output_dir}")
    print(metrics_table.to_string(index=False))


def load_experiment_splits(
    config: FTTransformerConfig,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Load the fixed train/validation/test split without creating a new split."""
    input_dataframe = load_csv(config.folder / config.input_csv, "input feature file")
    if PATIENT_ID_COLUMN not in input_dataframe.columns:
        raise ValueError(f"{config.input_csv} is missing {PATIENT_ID_COLUMN}.")

    feature_dataframe = input_dataframe.drop(
        columns=[column for column in SPLIT_TARGET_CANDIDATES if column in input_dataframe.columns]
    )
    excluded_columns = {PATIENT_ID_COLUMN, *config.identifier_columns}
    feature_columns = [column for column in feature_dataframe.columns if column not in excluded_columns]

    if not feature_columns:
        raise ValueError("No feature columns were found after removing identifiers and the target.")

    split_data = {
        "Training": build_split_dataset(feature_dataframe, config.folder / "train.csv", feature_columns),
        "Validation": build_split_dataset(
            feature_dataframe,
            config.folder / "validation.csv",
            feature_columns,
        ),
        "Test": build_split_dataset(feature_dataframe, config.folder / "test.csv", feature_columns),
    }

    for split_name, split_dataframe in split_data.items():
        validate_split_dataframe(split_dataframe, feature_columns, split_name)

    return split_data, feature_columns


def load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def build_split_dataset(
    feature_dataframe: pd.DataFrame,
    split_path: Path,
    feature_columns: list[str],
) -> pd.DataFrame:
    split_dataframe = load_csv(split_path, split_path.name)
    if PATIENT_ID_COLUMN not in split_dataframe.columns:
        raise ValueError(f"{split_path.name} is missing {PATIENT_ID_COLUMN}.")

    target_column = find_target_column(split_dataframe, split_path.name)
    split_labels = split_dataframe[[PATIENT_ID_COLUMN, target_column]].rename(
        columns={target_column: TARGET_COLUMN}
    )
    selected_features = feature_dataframe[[PATIENT_ID_COLUMN, *feature_columns]]
    merged = split_labels.merge(
        selected_features,
        on=PATIENT_ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    missing_rows = int(merged[feature_columns].isna().any(axis=1).sum())
    if missing_rows:
        raise ValueError(
            f"{split_path.name} has {missing_rows} patients that are not available "
            "in the input feature file. No manual imputation is performed here."
        )

    return merged[[PATIENT_ID_COLUMN, *feature_columns, TARGET_COLUMN]]


def find_target_column(dataframe: pd.DataFrame, label: str) -> str:
    for candidate in SPLIT_TARGET_CANDIDATES:
        if candidate in dataframe.columns:
            return candidate
    raise ValueError(f"{label} must contain one of these target columns: {SPLIT_TARGET_CANDIDATES}")


def validate_split_dataframe(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> None:
    if dataframe[PATIENT_ID_COLUMN].duplicated().any():
        raise ValueError(f"{split_name} split has duplicated {PATIENT_ID_COLUMN} values.")

    if dataframe[[*feature_columns, TARGET_COLUMN]].isna().any().any():
        raise ValueError(
            f"{split_name} split contains missing values. This script does not manually impute."
        )

    converted_features = dataframe[feature_columns].apply(pd.to_numeric, errors="coerce")
    if converted_features.isna().any().any():
        bad_columns = [
            column
            for column in feature_columns
            if converted_features[column].isna().any()
        ]
        raise ValueError(f"{split_name} has non-numeric feature columns: {bad_columns}")

    if not np.isfinite(converted_features.to_numpy(dtype="float32")).all():
        raise ValueError(f"{split_name} contains infinite feature values.")

    targets = sorted(dataframe[TARGET_COLUMN].astype(int).unique())
    if targets != [0, 1]:
        raise ValueError(f"{split_name} target must be binary 0/1. Found: {targets}")

    dataframe[feature_columns] = converted_features.astype("float32")
    dataframe[TARGET_COLUMN] = dataframe[TARGET_COLUMN].astype(int)


def build_ft_transformer_model(input_dim: int) -> keras.Model:
    """Build a compact FT-Transformer for tabular binary classification."""
    inputs = keras.Input(shape=(input_dim,), name="tabular_features")
    tokens = NumericalFeatureTokenizer(TOKEN_DIM, name="feature_tokenizer")(inputs)
    tokens = AddClassToken(TOKEN_DIM, name="add_class_token")(tokens)

    key_dim = TOKEN_DIM // ATTENTION_HEADS
    hidden_dim = TOKEN_DIM * FFN_MULTIPLIER

    for block_index in range(TRANSFORMER_BLOCKS):
        attention_input = layers.LayerNormalization(
            name=f"block_{block_index + 1}_attention_norm"
        )(tokens)
        attention_output = layers.MultiHeadAttention(
            num_heads=ATTENTION_HEADS,
            key_dim=key_dim,
            dropout=DROPOUT_RATE,
            name=f"block_{block_index + 1}_attention",
        )(attention_input, attention_input)
        attention_output = layers.Dropout(
            DROPOUT_RATE,
            name=f"block_{block_index + 1}_attention_dropout",
        )(attention_output)
        tokens = layers.Add(name=f"block_{block_index + 1}_attention_residual")(
            [tokens, attention_output]
        )

        ffn_input = layers.LayerNormalization(name=f"block_{block_index + 1}_ffn_norm")(tokens)
        ffn_output = layers.Dense(
            hidden_dim,
            activation="gelu",
            name=f"block_{block_index + 1}_ffn_dense_1",
        )(ffn_input)
        ffn_output = layers.Dropout(DROPOUT_RATE, name=f"block_{block_index + 1}_ffn_dropout_1")(
            ffn_output
        )
        ffn_output = layers.Dense(TOKEN_DIM, name=f"block_{block_index + 1}_ffn_dense_2")(
            ffn_output
        )
        ffn_output = layers.Dropout(DROPOUT_RATE, name=f"block_{block_index + 1}_ffn_dropout_2")(
            ffn_output
        )
        tokens = layers.Add(name=f"block_{block_index + 1}_ffn_residual")([tokens, ffn_output])

    class_token = layers.Lambda(lambda tensor: tensor[:, 0, :], name="class_token_output")(tokens)
    class_token = layers.LayerNormalization(name="head_norm")(class_token)
    class_token = layers.Dense(TOKEN_DIM, activation="gelu", name="head_dense")(class_token)
    class_token = layers.Dropout(DROPOUT_RATE, name="head_dropout")(class_token)
    outputs = layers.Dense(1, activation="sigmoid", name="hpv_probability")(class_token)

    model = keras.Model(inputs=inputs, outputs=outputs, name="FT_Transformer")
    optimizer_class = getattr(keras.optimizers, "AdamW", keras.optimizers.Adam)
    try:
        optimizer = optimizer_class(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    except TypeError:
        optimizer = optimizer_class(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.BinaryAccuracy(name="accuracy", threshold=DECISION_THRESHOLD),
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def write_model_summary(model: keras.Model, output_dir: Path) -> None:
    lines: list[str] = []
    model.summary(print_fn=lines.append)
    (output_dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def predict_probabilities(model: keras.Model, features: np.ndarray) -> np.ndarray:
    probabilities = model.predict(features.astype("float32"), verbose=0).reshape(-1)
    return np.clip(probabilities, 0.0, 1.0)


def build_metrics_table(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            calculate_metrics(y_validation, validation_probabilities, "Validation"),
            calculate_metrics(y_test, test_probabilities, "Test"),
        ]
    )


def calculate_metrics(y_true: np.ndarray, probabilities: np.ndarray, split_name: str) -> dict[str, object]:
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    sensitivity = safe_divide(tp, tp + fn)
    specificity = safe_divide(tn, tn + fp)
    return {
        "Split": split_name,
        "ROC-AUC": safe_roc_auc(y_true, probabilities),
        "Accuracy": accuracy_score(y_true, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_true, predictions),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "F1-score": f1_score(y_true, predictions, zero_division=0),
        "Confusion Matrix": f"TN={tn}, FP={fp}, FN={fn}, TP={tp}",
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Brier Score": brier_score_loss(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "Expected Calibration Error (ECE)": calculate_ece(y_true, probabilities),
    }


def safe_roc_auc(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, probabilities))
    except ValueError:
        return float("nan")


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def calculate_ece(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    table = build_calibration_table(y_true, probabilities)
    total = table["Sample count"].sum()
    if total == 0:
        return float("nan")
    weighted_gap = table["Sample count"] * table["Absolute calibration gap"].fillna(0.0)
    return float(weighted_gap.sum() / total)


def build_calibration_table(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bin_edges = np.linspace(0.0, 1.0, CALIBRATION_BINS + 1)

    for index in range(CALIBRATION_BINS):
        lower = bin_edges[index]
        upper = bin_edges[index + 1]
        if index == CALIBRATION_BINS - 1:
            mask = (probabilities >= lower) & (probabilities <= upper)
            label = f"[{lower:.1f}, {upper:.1f}]"
        else:
            mask = (probabilities >= lower) & (probabilities < upper)
            label = f"[{lower:.1f}, {upper:.1f})"

        count = int(mask.sum())
        if count:
            mean_probability = float(probabilities[mask].mean())
            observed_rate = float(y_true[mask].mean())
            gap = abs(mean_probability - observed_rate)
        else:
            mean_probability = float("nan")
            observed_rate = float("nan")
            gap = float("nan")

        rows.append(
            {
                "Bin range": label,
                "Sample count": count,
                "Mean predicted probability": mean_probability,
                "Observed HPV positive rate": observed_rate,
                "Absolute calibration gap": gap,
            }
        )

    return pd.DataFrame(rows)


def save_confusion_matrix_csv(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    split_name: str,
) -> None:
    predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    matrix_dataframe = pd.DataFrame(
        matrix,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )
    matrix_dataframe.to_csv(output_dir / f"{split_name}_confusion_matrix.csv")


def plot_roc_curves(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    for label, y_true, probabilities in [
        ("Validation", y_validation, validation_probabilities),
        ("Test", y_test, test_probabilities),
    ]:
        fpr, tpr, _thresholds = roc_curve(y_true, probabilities)
        auc_value = safe_roc_auc(y_true, probabilities)
        plt.plot(fpr, tpr, linewidth=2, label=f"{label} ROC-AUC={auc_value:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    save_current_figure(output_path)


def plot_precision_recall_curves(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    for label, y_true, probabilities in [
        ("Validation", y_validation, validation_probabilities),
        ("Test", y_test, test_probabilities),
    ]:
        precision, recall, _thresholds = precision_recall_curve(y_true, probabilities)
        pr_auc = average_precision_score(y_true, probabilities)
        plt.plot(recall, precision, linewidth=2, label=f"{label} PR-AUC={pr_auc:.3f}")

    baseline = float(np.mean(np.concatenate([y_validation, y_test])))
    plt.axhline(baseline, linestyle="--", color="gray", label=f"Baseline={baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.3)
    plt.legend(loc="lower left")
    save_current_figure(output_path)


def plot_calibration_curves(
    validation_table: pd.DataFrame,
    test_table: pd.DataFrame,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 6))
    for label, table in [("Validation", validation_table), ("Test", test_table)]:
        non_empty = table[table["Sample count"] > 0]
        plt.plot(
            non_empty["Mean predicted probability"],
            non_empty["Observed HPV positive rate"],
            marker="o",
            linewidth=2,
            label=label,
        )

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed HPV positive rate")
    plt.title("Calibration Curve (10 Bins)")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend(loc="upper left")
    save_current_figure(output_path)


def plot_confusion_matrices(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, label, y_true, probabilities in [
        (axes[0], "Validation", y_validation, validation_probabilities),
        (axes[1], "Test", y_test, test_probabilities),
    ]:
        predictions = (probabilities >= DECISION_THRESHOLD).astype(int)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["Actual 0", "Actual 1"],
            ax=axis,
        )
        axis.set_title(f"{label} Confusion Matrix")
    figure.tight_layout()
    save_current_figure(output_path)


def plot_probability_distributions(
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)
    for axis, split_name, y_true, probabilities in [
        (axes[0], "Validation", y_validation, validation_probabilities),
        (axes[1], "Test", y_test, test_probabilities),
    ]:
        plot_dataframe = pd.DataFrame(
            {
                "Predicted HPV-positive probability": probabilities,
                "Actual HPV Status": y_true.astype(str),
            }
        )
        sns.histplot(
            data=plot_dataframe,
            x="Predicted HPV-positive probability",
            hue="Actual HPV Status",
            bins=np.linspace(0, 1, 21),
            element="step",
            stat="count",
            common_norm=False,
            ax=axis,
        )
        axis.axvline(DECISION_THRESHOLD, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{split_name} Probability Distribution")
        axis.set_xlim(0, 1)
    figure.tight_layout()
    save_current_figure(output_path)


def plot_training_history(history: keras.callbacks.History, output_path: Path) -> None:
    history_dataframe = pd.DataFrame(history.history)
    plt.figure(figsize=(8, 5))
    plt.plot(history_dataframe["loss"], label="Training loss")
    plt.plot(history_dataframe["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Binary crossentropy")
    plt.title("Training Loss Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    save_current_figure(output_path)


def save_current_figure(output_path: Path) -> None:
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def calculate_permutation_importance(
    model: keras.Model,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    baseline_auc = safe_roc_auc(y_validation, predict_probabilities(model, x_validation))
    rows: list[dict[str, object]] = []

    for feature_index, feature_name in enumerate(feature_columns):
        drops = []
        for _repeat in range(PERMUTATION_REPEATS):
            permuted = x_validation.copy()
            shuffled_values = permuted[:, feature_index].copy()
            rng.shuffle(shuffled_values)
            permuted[:, feature_index] = shuffled_values
            permuted_auc = safe_roc_auc(y_validation, predict_probabilities(model, permuted))
            drops.append(baseline_auc - permuted_auc)

        rows.append(
            {
                "feature": feature_name,
                "mean_auc_drop": float(np.nanmean(drops)),
                "std_auc_drop": float(np.nanstd(drops)),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_auc_drop", ascending=False).reset_index(drop=True)


def plot_feature_importance(
    importance_table: pd.DataFrame,
    output_path: Path,
    top_n: int | None,
) -> None:
    display_table = importance_table.copy()
    if top_n is not None:
        display_table = display_table.head(top_n)
    display_table = display_table.sort_values("mean_auc_drop")

    height = max(5, 0.34 * len(display_table) + 1.5)
    plt.figure(figsize=(9, height))
    plt.barh(display_table["feature"], display_table["mean_auc_drop"], color="#4C78A8")
    plt.xlabel("Mean ROC-AUC drop after permutation")
    plt.title("Permutation Feature Importance")
    plt.grid(axis="x", alpha=0.3)
    save_current_figure(output_path)


def create_shap_plots(
    model: keras.Model,
    x_train: np.ndarray,
    x_validation: np.ndarray,
    feature_columns: list[str],
    importance_table: pd.DataFrame,
    output_dir: Path,
    top_n: int | None,
) -> dict[str, object]:
    try:
        import shap

        selected_features = choose_shap_features(feature_columns, importance_table, top_n)
        selected_indices = [feature_columns.index(feature) for feature in selected_features]

        background_full = x_train[: min(SHAP_BACKGROUND_SIZE, len(x_train))]
        explain_full = x_validation[: min(SHAP_EXPLAIN_SIZE, len(x_validation))]
        background_selected = background_full[:, selected_indices]
        explain_selected = explain_full[:, selected_indices]
        base_values = background_full.mean(axis=0)

        def predict_from_selected(selected_values: np.ndarray) -> np.ndarray:
            selected_values = np.asarray(selected_values, dtype="float32")
            if selected_values.ndim == 1:
                selected_values = selected_values.reshape(1, -1)
            full_values = np.tile(base_values, (selected_values.shape[0], 1)).astype("float32")
            full_values[:, selected_indices] = selected_values
            return predict_probabilities(model, full_values)

        explainer = shap.KernelExplainer(predict_from_selected, background_selected)
        shap_values = explainer.shap_values(explain_selected, nsamples=SHAP_NSAMPLES)
        shap_values = normalize_shap_values(shap_values)
        explain_dataframe = pd.DataFrame(explain_selected, columns=selected_features)

        mean_abs = np.abs(shap_values).mean(axis=0)
        dependence_feature = selected_features[int(np.argmax(mean_abs))]

        pd.DataFrame(
            {"feature": selected_features, "mean_abs_shap": mean_abs}
        ).sort_values("mean_abs_shap", ascending=False).to_csv(
            output_dir / "shap_mean_absolute_values.csv",
            index=False,
        )
        pd.DataFrame({"selected_feature": selected_features}).to_csv(
            output_dir / "shap_selected_features.csv",
            index=False,
        )

        shap.summary_plot(
            shap_values,
            explain_dataframe,
            show=False,
            max_display=len(selected_features),
        )
        plt.title("SHAP Summary Plot")
        save_current_figure(output_dir / "shap_summary_plot.png")

        shap.dependence_plot(
            dependence_feature,
            shap_values,
            explain_dataframe,
            show=False,
        )
        plt.title(f"SHAP Dependence Plot: {dependence_feature}")
        save_current_figure(output_dir / "shap_dependence_plot.png")

        save_shap_heatmap(
            shap_values,
            explain_dataframe,
            selected_features,
            output_dir / "shap_heatmap_plot.png",
        )

        return {
            "status": "created",
            "selected_features": selected_features,
            "dependence_feature": dependence_feature,
            "message": "SHAP plots were created with the trained FT-Transformer.",
        }
    except Exception as error:
        message = f"SHAP plots could not be created: {error}"
        for filename, title in [
            ("shap_summary_plot.png", "SHAP Summary Plot"),
            ("shap_dependence_plot.png", "SHAP Dependence Plot"),
            ("shap_heatmap_plot.png", "SHAP Heatmap Plot"),
        ]:
            save_placeholder_plot(output_dir / filename, title, message)
        (output_dir / "shap_status.txt").write_text(message, encoding="utf-8")
        return {
            "status": "failed",
            "selected_features": [],
            "dependence_feature": "not available",
            "message": message,
        }


def choose_shap_features(
    feature_columns: list[str],
    importance_table: pd.DataFrame,
    top_n: int | None,
) -> list[str]:
    if top_n is None:
        return feature_columns

    selected = importance_table.head(top_n)["feature"].tolist()
    return [feature for feature in selected if feature in feature_columns]


def normalize_shap_values(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    array = np.asarray(shap_values)
    if array.ndim == 3:
        array = array[:, :, 0]
    return array


def save_shap_heatmap(
    shap_values: np.ndarray,
    explain_dataframe: pd.DataFrame,
    selected_features: list[str],
    output_path: Path,
) -> None:
    try:
        import shap

        explanation = shap.Explanation(
            values=shap_values,
            data=explain_dataframe.to_numpy(),
            feature_names=selected_features,
        )
        shap.plots.heatmap(explanation, max_display=len(selected_features), show=False)
        plt.title("SHAP Heatmap Plot")
        save_current_figure(output_path)
    except Exception:
        plt.figure(figsize=(10, 6))
        sns.heatmap(
            shap_values,
            cmap="coolwarm",
            center=0,
            xticklabels=selected_features,
            yticklabels=False,
        )
        plt.title("SHAP Heatmap Plot")
        plt.xlabel("Feature")
        plt.ylabel("Validation sample")
        save_current_figure(output_path)


def save_placeholder_plot(output_path: Path, title: str, message: str) -> None:
    plt.figure(figsize=(8, 4.5))
    plt.axis("off")
    plt.title(title)
    plt.text(
        0.5,
        0.5,
        textwrap.fill(message, width=82),
        ha="center",
        va="center",
        transform=plt.gca().transAxes,
    )
    save_current_figure(output_path)


def write_run_summary(
    config: FTTransformerConfig,
    output_dir: Path,
    split_data: dict[str, pd.DataFrame],
    feature_columns: list[str],
    metrics_table: pd.DataFrame,
    shap_summary: dict[str, object],
) -> None:
    lines = [
        f"Experiment: {config.experiment_name}",
        f"Input CSV: {config.folder / config.input_csv}",
        f"Results folder: {output_dir}",
        f"Model: Keras FT-Transformer",
        f"Fixed epochs: {EPOCHS}",
        f"Batch size: {BATCH_SIZE}",
        f"Feature count used by model: {len(feature_columns)}",
        f"Identifier columns excluded from model input: {', '.join((PATIENT_ID_COLUMN, *config.identifier_columns))}",
        f"Target column: {TARGET_COLUMN}",
        f"Training target counts: {split_data['Training'][TARGET_COLUMN].value_counts().sort_index().to_dict()}",
        f"Validation target counts: {split_data['Validation'][TARGET_COLUMN].value_counts().sort_index().to_dict()}",
        f"Test target counts: {split_data['Test'][TARGET_COLUMN].value_counts().sort_index().to_dict()}",
        "",
        "Rules followed:",
        "- HPV Status was used only as the target label, never as an input feature.",
        "- Existing train/validation/test split files were reused.",
        "- No new split was created.",
        "- No hyperparameter tuning was performed.",
        "- No feature selection, feature filtering, or feature engineering was performed.",
        "- No scaling or normalization was performed.",
        "- No manual imputation was performed.",
        "- Fixed decision threshold for class labels: 0.5.",
        "",
        "Validation/Test metrics:",
        metrics_table.to_string(index=False),
        "",
        "SHAP:",
        str(shap_summary.get("message", "")),
        f"SHAP dependence feature: {shap_summary.get('dependence_feature', 'not available')}",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def save_full_classification_report(
    config: FTTransformerConfig,
    output_dir: Path,
    split_data: dict[str, pd.DataFrame],
    feature_columns: list[str],
    metrics_table: pd.DataFrame,
    y_validation: np.ndarray,
    validation_probabilities: np.ndarray,
    y_test: np.ndarray,
    test_probabilities: np.ndarray,
    validation_calibration: pd.DataFrame,
    test_calibration: pd.DataFrame,
    shap_summary: dict[str, object],
) -> None:
    figure = Figure(figsize=(13, 68), dpi=100)
    grid = figure.add_gridspec(
        10,
        2,
        height_ratios=[2.0, 3.0, 4.8, 4.8, 4.8, 4.8, 5.0, 5.0, 5.0, 7.0],
        hspace=0.72,
        wspace=0.28,
    )

    header_axis = figure.add_subplot(grid[0, :])
    metrics_axis = figure.add_subplot(grid[1, :])
    validation_calibration_axis = figure.add_subplot(grid[2, 0])
    test_calibration_axis = figure.add_subplot(grid[2, 1])
    validation_roc_axis = figure.add_subplot(grid[3, 0])
    test_roc_axis = figure.add_subplot(grid[3, 1])
    validation_pr_axis = figure.add_subplot(grid[4, 0])
    test_pr_axis = figure.add_subplot(grid[4, 1])
    validation_curve_axis = figure.add_subplot(grid[5, 0])
    test_curve_axis = figure.add_subplot(grid[5, 1])
    confusion_axis = figure.add_subplot(grid[6, 0])
    probability_axis = figure.add_subplot(grid[6, 1])
    feature_axis = figure.add_subplot(grid[7, 0])
    shap_summary_axis = figure.add_subplot(grid[7, 1])
    shap_dependence_axis = figure.add_subplot(grid[8, 0])
    training_axis = figure.add_subplot(grid[8, 1])
    shap_heatmap_axis = figure.add_subplot(grid[9, :])

    draw_report_header(header_axis, config, split_data, feature_columns, shap_summary)
    draw_metrics_table(metrics_axis, metrics_table)
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


def draw_report_header(
    axis: Axes,
    config: FTTransformerConfig,
    split_data: dict[str, pd.DataFrame],
    feature_columns: list[str],
    shap_summary: dict[str, object],
) -> None:
    axis.axis("off")
    text = "\n".join(
        [
            f"{config.experiment_name} FT-Transformer Classification",
            f"Input: {config.input_csv}",
            f"Features used: {len(feature_columns)} | Train/Validation/Test rows: "
            f"{len(split_data['Training'])}/{len(split_data['Validation'])}/{len(split_data['Test'])}",
            "No new split, no tuning, no feature selection/filtering/engineering, "
            "no scaling/normalization, no manual imputation.",
            f"SHAP status: {shap_summary.get('status', 'unknown')} | "
            f"Dependence feature: {shap_summary.get('dependence_feature', 'not available')}",
        ]
    )
    axis.text(
        0.01,
        0.96,
        text,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
        transform=axis.transAxes,
    )


def draw_metrics_table(axis: Axes, metrics_table: pd.DataFrame) -> None:
    axis.axis("off")
    display_columns = [
        "Split",
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
    display_table = metrics_table[display_columns].copy()
    for column in display_table.columns:
        if column not in ["Split", "Confusion Matrix"]:
            display_table[column] = display_table[column].map(format_report_value)

    table = axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        bbox=[0.0, 0.0, 1.0, 0.86],
        cellLoc="center",
    )
    style_table(table, font_size=6.2, y_scale=1.35)
    axis.set_title("Validation and Test Metrics", fontsize=15, fontweight="bold")


def draw_calibration_table(
    axis: Axes,
    calibration_table: pd.DataFrame,
    split_name: str,
) -> None:
    axis.axis("off")
    display_table = calibration_table.rename(
        columns={
            "Bin range": "Bin",
            "Sample count": "N",
            "Mean predicted probability": "Mean pred",
            "Observed HPV positive rate": "Observed +",
            "Absolute calibration gap": "Gap",
        }
    )[["Bin", "N", "Mean pred", "Observed +", "Gap"]].copy()
    for column in ["Mean pred", "Observed +", "Gap"]:
        display_table[column] = display_table[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )

    table = axis.table(
        cellText=display_table.values,
        colLabels=display_table.columns,
        bbox=[0.0, 0.0, 1.0, 0.84],
        cellLoc="center",
    )
    style_table(table, font_size=8.0, y_scale=1.35)
    axis.set_title(f"{split_name} Calibration Table (10 Bins)", fontsize=15, fontweight="bold")


def draw_single_roc_curve(
    axis: Axes,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    split_name: str,
) -> None:
    fpr, tpr, _thresholds = roc_curve(y_true, probabilities)
    auc_value = safe_roc_auc(y_true, probabilities)
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
    y_true: np.ndarray,
    probabilities: np.ndarray,
    split_name: str,
) -> None:
    precision, recall, _thresholds = precision_recall_curve(y_true, probabilities)
    pr_auc = average_precision_score(y_true, probabilities)
    positive_rate = float(np.mean(y_true))
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
    non_empty = calibration_table[calibration_table["Sample count"] > 0]
    axis.plot(
        non_empty["Mean predicted probability"],
        non_empty["Observed HPV positive rate"],
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


def draw_saved_png(axis: Axes, image_path: Path, title: str) -> None:
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


def format_report_value(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return "" if np.isnan(value) else f"{value:.3f}"
    return str(value)


def style_table(table: object, font_size: float, y_scale: float) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, y_scale)
    for (row, _column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#EAEAEA")


def show_report_popup(figure: Figure, report_path: Path) -> None:
    """Show the full report in a Matplotlib window with vertical/horizontal scrolling."""
    if not POPUP_REPORT:
        return

    try:
        import tkinter as tk
        from tkinter import ttk

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        window = tk.Tk()
        window.title("FT-Transformer Classification Full Report")
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        window_width = min(1350, max(1000, screen_width - 140))
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
        print(f"Popup window could not be opened: {error}")
