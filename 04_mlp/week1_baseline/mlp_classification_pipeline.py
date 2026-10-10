import argparse
import html
import importlib
import math
import os
from pathlib import Path
import subprocess
import sys
import webbrowser
import warnings


# Third-party libraries are imported after the environment check and auto-install
# step. This lets the script explain missing packages instead of crashing first.
np = None
pd = None
plt = None
sns = None


TARGET_COLUMN = "HPV Status"
PATIENT_ID_COLUMN = "PatientID"
SPLIT_COLUMN = "Split"
METADATA_COLUMNS = {PATIENT_ID_COLUMN, SPLIT_COLUMN}
RANDOM_STATE = 42
MPL_CACHE_DIR = Path(__file__).resolve().parent / ".matplotlib_cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def check_virtual_environment(expected_names=("myenv", "tfenv")):
    """Make sure packages are installed into the intended virtual environment."""
    executable = Path(sys.executable)
    prefix = Path(sys.prefix)
    expected_names = tuple(name.lower() for name in expected_names)
    executable_text = str(executable).lower()
    prefix_text = str(prefix).lower()
    if not any(name in executable_text or name in prefix_text for name in expected_names):
        raise RuntimeError(
            "This script should be run with the project virtual environment.\n"
            f"Expected one of these environment names: {', '.join(expected_names)}\n"
            f"Current Python executable: {sys.executable}\n\n"
            "Create one from the repository root with: python -m venv tfenv\n"
            "then run: tfenv/Scripts/python path/to/the_script.py (Windows) "
            "or tfenv/bin/python path/to/the_script.py (Linux/macOS)"
        )


def import_or_install(module_name, pip_name=None, install_missing=True):
    """Import a package; if missing, install it into the current Python env."""
    pip_name = pip_name or module_name
    try:
        return importlib.import_module(module_name)
    except ImportError:
        if not install_missing:
            raise

        print(f"Missing library '{module_name}'. Installing '{pip_name}' now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
        return importlib.import_module(module_name)


def prepare_libraries(backend, install_missing=True):
    """Load required libraries and choose the available MLP backend."""
    global np, pd, plt, sns

    np = import_or_install("numpy", install_missing=install_missing)
    pd = import_or_install("pandas", install_missing=install_missing)

    matplotlib = import_or_install("matplotlib", install_missing=install_missing)
    matplotlib.use("Agg")
    plt = importlib.import_module("matplotlib.pyplot")

    sns = import_or_install("seaborn", install_missing=install_missing)
    import_or_install("sklearn", "scikit-learn", install_missing=install_missing)
    import_or_install("scipy", install_missing=install_missing)

    tensorflow = None
    if backend in {"auto", "keras"}:
        if sys.version_info >= (3, 13):
            message = (
                "\nEnvironment note: TensorFlow/Keras was requested first, but this "
                f"virtual environment uses Python {sys.version.split()[0]}.\n"
                "TensorFlow is normally not available for Python 3.13+ yet, so this "
                "run will use sklearn's MLPClassifier fallback.\n"
                "This is still an MLP classifier. To force real TensorFlow/Keras, "
                "create a Python 3.10-3.12 virtual environment and rerun with "
                "--backend keras.\n"
            )
            if backend == "keras":
                raise RuntimeError(message)
            print(message)
        else:
            try:
                tensorflow = import_or_install(
                    "tensorflow",
                    "tensorflow",
                    install_missing=install_missing,
                )
            except Exception as error:
                if backend == "keras":
                    raise RuntimeError(
                        "TensorFlow/Keras was requested but could not be imported or installed."
                    ) from error
                print(
                    "Environment note: TensorFlow/Keras could not be imported or "
                    f"installed, so this run will use sklearn's MLPClassifier fallback.\n"
                    f"Reason: {error}\n"
                )

    if tensorflow is not None:
        tensorflow.get_logger().setLevel("ERROR")
        return {"backend": "keras", "tensorflow": tensorflow}

    if backend == "keras":
        raise RuntimeError("TensorFlow/Keras backend was requested but is unavailable.")

    return {"backend": "sklearn", "tensorflow": None}


def ensure_shap(install_missing=True):
    """Import SHAP only when SHAP plots are about to be created."""
    return import_or_install("shap", install_missing=install_missing)


def load_experiment_dataframe(config):
    """Load one imputed input and attach official split/target columns if needed."""
    input_csv = Path(config["input_csv"])
    data = pd.read_csv(input_csv)

    if config.get("split_target_csv"):
        split_target = pd.read_csv(config["split_target_csv"])
        needed = [PATIENT_ID_COLUMN, SPLIT_COLUMN]
        if TARGET_COLUMN not in data.columns:
            needed.append(TARGET_COLUMN)

        split_target = split_target[needed].drop_duplicates(subset=[PATIENT_ID_COLUMN])
        before_rows = len(data)
        data = data.merge(split_target, on=PATIENT_ID_COLUMN, how="inner")
        after_rows = len(data)
        omitted_rows = before_rows - after_rows
    else:
        omitted_rows = 0

    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing after loading data.")
    if SPLIT_COLUMN not in data.columns:
        raise ValueError(
            f"Split column '{SPLIT_COLUMN}' is missing. The script will not create a new split."
        )

    data[SPLIT_COLUMN] = data[SPLIT_COLUMN].astype(str).str.lower()
    expected_splits = {"train", "validation", "test"}
    found_splits = set(data[SPLIT_COLUMN].unique())
    missing_splits = expected_splits - found_splits
    if missing_splits:
        raise ValueError(f"Missing required split labels: {sorted(missing_splits)}")

    return data, omitted_rows


def get_feature_columns(data):
    """Use every numeric model feature except ID, split, and target columns."""
    ignored = set(METADATA_COLUMNS)
    ignored.update({TARGET_COLUMN, "HPV"})
    feature_columns = [column for column in data.columns if column not in ignored]

    non_numeric = [
        column for column in feature_columns if not pd.api.types.is_numeric_dtype(data[column])
    ]
    if non_numeric:
        raise ValueError(
            "These feature columns are not numeric and cannot be passed to an MLP:\n"
            + "\n".join(non_numeric)
        )

    missing_counts = data[feature_columns + [TARGET_COLUMN]].isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if not missing_counts.empty:
        raise ValueError(
            "Missing values remain in the model input. Re-run imputation first:\n"
            + missing_counts.to_string()
        )

    values = data[feature_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Infinite values were found in the feature matrix.")

    return feature_columns


def split_xy(data, feature_columns):
    """Create train, validation, and test arrays from existing split labels."""
    split_data = {}
    for split_name in ["train", "validation", "test"]:
        subset = data[data[SPLIT_COLUMN] == split_name].copy()
        x = subset[feature_columns].to_numpy(dtype=np.float32)
        y = subset[TARGET_COLUMN].astype(int).to_numpy()
        ids = subset[PATIENT_ID_COLUMN].astype(str).to_numpy()
        split_data[split_name] = {"x": x, "y": y, "ids": ids, "frame": subset}
    return split_data


def build_keras_model(input_dim, tensorflow):
    """Build a fixed, beginner-friendly Keras MLP."""
    model = tensorflow.keras.Sequential(
        [
            tensorflow.keras.layers.Input(shape=(input_dim,)),
            tensorflow.keras.layers.Dense(64, activation="relu"),
            tensorflow.keras.layers.Dense(32, activation="relu"),
            tensorflow.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(split_data, feature_count, backend_info):
    """Train a fixed MLP model without hyperparameter tuning."""
    backend = backend_info["backend"]
    x_train = split_data["train"]["x"]
    y_train = split_data["train"]["y"]

    if backend == "keras":
        tensorflow = backend_info["tensorflow"]
        tensorflow.random.set_seed(RANDOM_STATE)
        model = build_keras_model(feature_count, tensorflow)
        history = model.fit(
            x_train,
            y_train,
            epochs=80,
            batch_size=32,
            verbose=0,
        )
        return {"backend": "keras", "model": model, "history": history.history}

    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=1e-5,
        max_iter=500,
        batch_size=32,
        random_state=RANDOM_STATE,
        early_stopping=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(x_train, y_train)

    history = {}
    if hasattr(model, "loss_curve_"):
        history["loss"] = list(model.loss_curve_)

    return {"backend": "sklearn", "model": model, "history": history}


def predict_positive_probability(model_bundle, x):
    """Return the predicted HPV-positive probability from either backend."""
    model = model_bundle["model"]
    if model_bundle["backend"] == "keras":
        return model.predict(x, verbose=0).reshape(-1)

    probabilities = model.predict_proba(x)
    return probabilities[:, 1]


def save_training_history(model_bundle, output_dir):
    """Save the training loss curve when the backend exposes one."""
    history = model_bundle.get("history") or {}
    if not history:
        return

    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    if "loss" in history_df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(history_df["loss"], label="Training loss")
        plt.xlabel("Epoch / iteration")
        plt.ylabel("Loss")
        plt.title("MLP Training Loss")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss_curve.png", dpi=300, bbox_inches="tight")
        plt.close()


def safe_auc(y_true, y_score):
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return np.nan
    return roc_auc_score(y_true, y_score)


def calculate_metrics(y_true, y_prob):
    """Calculate all required numerical metrics for one split."""
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "ROC-AUC": safe_auc(y_true, y_prob),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "Sensitivity / Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "Brier Score": brier_score_loss(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def calibration_table(y_true, y_prob, bins=10):
    """Create the 10-bin calibration table and ECE."""
    rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)

    for index in range(bins):
        low = edges[index]
        high = edges[index + 1]
        if index == bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
            label = f"[{low:.1f}, {high:.1f}]"
        else:
            mask = (y_prob >= low) & (y_prob < high)
            label = f"[{low:.1f}, {high:.1f})"

        count = int(mask.sum())
        if count:
            mean_prob = float(np.mean(y_prob[mask]))
            observed_rate = float(np.mean(y_true[mask]))
            gap = abs(mean_prob - observed_rate)
            ece += (count / n) * gap
        else:
            mean_prob = np.nan
            observed_rate = np.nan
            gap = np.nan

        rows.append(
            {
                "Bin range": label,
                "Sample count": count,
                "Mean predicted probability": mean_prob,
                "Observed HPV positive rate": observed_rate,
                "Absolute calibration gap": gap,
            }
        )

    return pd.DataFrame(rows), ece


def save_calibration_curve(table, ece, split_name, output_dir):
    plot_data = table[table["Sample count"] > 0]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.plot(
        plot_data["Mean predicted probability"],
        plot_data["Observed HPV positive rate"],
        marker="o",
        label=f"{split_name} (ECE={ece:.3f})",
    )
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed HPV positive rate")
    plt.title(f"Calibration Curve - {split_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_roc_curve(y_true, y_prob, split_name, output_dir):
    from sklearn.metrics import auc, roc_curve

    plt.figure(figsize=(6, 6))
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"ROC-AUC={roc_auc:.3f}")
    else:
        plt.text(0.5, 0.5, "ROC unavailable: only one class present", ha="center")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {split_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_precision_recall_curve(y_true, y_prob, split_name, output_dir):
    from sklearn.metrics import average_precision_score, precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(6, 6))
    plt.plot(recall, precision, label=f"PR-AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {split_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_precision_recall_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_heatmap(y_true, y_prob, split_name, output_dir):
    from sklearn.metrics import confusion_matrix

    y_pred = (y_prob >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    plt.title(f"Confusion Matrix - {split_name}")
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_confusion_matrix_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_probability_distribution(y_true, y_prob, split_name, output_dir):
    plot_df = pd.DataFrame({"HPV Status": y_true, "Predicted probability": y_prob})
    plt.figure(figsize=(8, 5))
    sns.histplot(
        data=plot_df,
        x="Predicted probability",
        hue="HPV Status",
        bins=20,
        kde=False,
        stat="density",
        common_norm=False,
    )
    plt.title(f"Predicted Probability Distribution - {split_name}")
    plt.xlabel("Predicted HPV-positive probability")
    plt.xlim(0, 1)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / f"{split_name}_probability_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()


def evaluate_split(split_name, y_true, y_prob, ids, output_dir):
    """Save metrics, predictions, calibration table, and evaluation plots."""
    metrics = calculate_metrics(y_true, y_prob)
    calibration, ece = calibration_table(y_true, y_prob, bins=10)
    metrics["Expected Calibration Error / ECE"] = ece

    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "Split", split_name)
    metrics_df.to_csv(output_dir / f"{split_name}_metrics.csv", index=False)

    confusion_df = pd.DataFrame(
        [[metrics["TN"], metrics["FP"]], [metrics["FN"], metrics["TP"]]],
        index=["Actual HPV negative", "Actual HPV positive"],
        columns=["Predicted HPV negative", "Predicted HPV positive"],
    )
    confusion_df.to_csv(output_dir / f"{split_name}_confusion_matrix.csv")

    calibration.to_csv(output_dir / f"{split_name}_calibration_table.csv", index=False)

    prediction_df = pd.DataFrame(
        {
            PATIENT_ID_COLUMN: ids,
            "True HPV Status": y_true,
            "Predicted HPV Probability": y_prob,
            "Predicted HPV Status": (y_prob >= 0.5).astype(int),
        }
    )
    prediction_df.to_csv(output_dir / f"{split_name}_predictions.csv", index=False)

    save_calibration_curve(calibration, ece, split_name, output_dir)
    save_roc_curve(y_true, y_prob, split_name, output_dir)
    save_precision_recall_curve(y_true, y_prob, split_name, output_dir)
    save_confusion_matrix_heatmap(y_true, y_prob, split_name, output_dir)
    save_probability_distribution(y_true, y_prob, split_name, output_dir)

    return metrics_df


def permutation_importance(model_bundle, x, y, feature_columns, output_dir, max_display):
    """Calculate model-agnostic permutation importance using validation ROC-AUC."""
    baseline_prob = predict_positive_probability(model_bundle, x)
    baseline_auc = safe_auc(y, baseline_prob)

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for column_index, feature_name in enumerate(feature_columns):
        drops = []
        for repeat in range(5):
            x_permuted = x.copy()
            x_permuted[:, column_index] = rng.permutation(x_permuted[:, column_index])
            permuted_prob = predict_positive_probability(model_bundle, x_permuted)
            permuted_auc = safe_auc(y, permuted_prob)
            drops.append(baseline_auc - permuted_auc)

        rows.append(
            {
                "feature": feature_name,
                "mean_auc_drop": float(np.nanmean(drops)),
                "std_auc_drop": float(np.nanstd(drops)),
            }
        )

    importance_df = pd.DataFrame(rows).sort_values("mean_auc_drop", ascending=False)
    importance_df.to_csv(output_dir / "permutation_feature_importance.csv", index=False)

    display_df = importance_df.head(max_display).sort_values("mean_auc_drop")
    plt.figure(figsize=(9, max(4, 0.42 * len(display_df) + 1.5)))
    plt.barh(display_df["feature"], display_df["mean_auc_drop"], color="#4C72B0")
    plt.xlabel("Mean ROC-AUC drop after permutation")
    plt.title("Permutation Feature Importance")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    return importance_df


def save_shap_heatmap_fallback(shap_values, x_explain, feature_names, max_display, output_dir):
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs)[::-1][:max_display]
    heatmap_df = pd.DataFrame(
        shap_values[:, top_indices],
        columns=[feature_names[index] for index in top_indices],
    )
    plt.figure(figsize=(10, 7))
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0)
    plt.title("SHAP Heatmap Plot")
    plt.xlabel("Feature")
    plt.ylabel("Explained sample")
    plt.tight_layout()
    plt.savefig(output_dir / "shap_heatmap_plot.png", dpi=300, bbox_inches="tight")
    plt.close()


def save_shap_plots(model_bundle, split_data, feature_columns, output_dir, max_display, install_missing):
    """Create SHAP summary, dependence, and heatmap plots from the trained MLP."""
    shap = ensure_shap(install_missing=install_missing)

    validation_x = split_data["validation"]["x"]
    if len(validation_x) == 0:
        raise ValueError("Validation set is empty; SHAP plots need validation samples.")

    rng = np.random.default_rng(RANDOM_STATE)
    background_size = min(50, len(split_data["train"]["x"]))
    explain_size = min(60, len(validation_x))

    background_indices = rng.choice(len(split_data["train"]["x"]), size=background_size, replace=False)
    explain_indices = rng.choice(len(validation_x), size=explain_size, replace=False)

    background = split_data["train"]["x"][background_indices]
    x_explain = validation_x[explain_indices]
    x_explain_df = pd.DataFrame(x_explain, columns=feature_columns)

    def predict_for_shap(array):
        array = np.asarray(array, dtype=np.float32)
        return predict_positive_probability(model_bundle, array)

    explainer = shap.KernelExplainer(predict_for_shap, background)
    nsamples = min(100, 2 * len(feature_columns) + 1)
    shap_values = explainer.shap_values(x_explain, nsamples=nsamples)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]

    shap_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "mean_absolute_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    shap_importance.to_csv(output_dir / "shap_mean_absolute_importance.csv", index=False)

    top_feature = shap_importance.iloc[0]["feature"]
    (output_dir / "shap_dependence_feature.txt").write_text(
        f"Automatically selected dependence feature: {top_feature}\n",
        encoding="utf-8",
    )

    plt.figure()
    shap.summary_plot(
        shap_values,
        x_explain_df,
        feature_names=feature_columns,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    shap.dependence_plot(
        top_feature,
        shap_values,
        x_explain_df,
        feature_names=feature_columns,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_dependence_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    try:
        explanation = shap.Explanation(
            values=shap_values,
            data=x_explain_df.to_numpy(),
            feature_names=feature_columns,
        )
        shap.plots.heatmap(explanation, max_display=max_display, show=False)
        plt.tight_layout()
        plt.savefig(output_dir / "shap_heatmap_plot.png", dpi=300, bbox_inches="tight")
        plt.close()
    except Exception:
        save_shap_heatmap_fallback(
            shap_values,
            x_explain,
            feature_columns,
            max_display,
            output_dir,
        )


def save_run_summary(config, data, omitted_rows, feature_columns, backend_info, output_dir):
    """Save a beginner-readable summary of exactly what was trained."""
    split_counts = data[SPLIT_COLUMN].value_counts().to_dict()
    class_counts = data[TARGET_COLUMN].value_counts().to_dict()
    summary_lines = [
        f"Experiment: {config['experiment_name']}",
        f"Input CSV: {config['input_csv']}",
        f"Results folder: {output_dir}",
        f"Backend used: {backend_info['backend']}",
        "",
        "Important rules followed:",
        "- Original CSV files were not modified.",
        "- Existing train/validation/test split labels were reused.",
        "- No new split was created.",
        "- No hyperparameter tuning was performed.",
        "- No feature selection/filtering was used for model training.",
        "- No scaling, normalization, or standardization was applied.",
        "- HPV Status, PatientID, and Split were excluded from model features.",
        "",
        f"Rows available after matching to official split labels: {len(data)}",
        f"Rows omitted because they had no official split/target match: {omitted_rows}",
        f"Split counts: {split_counts}",
        f"Target counts: {class_counts}",
        f"Number of model input features: {len(feature_columns)}",
        "",
        "Model:",
        "- Fixed MLP architecture: 64 hidden units, then 32 hidden units, then one sigmoid output.",
        "- Fixed decision threshold for class labels: 0.5.",
        "- Feature importance is model-agnostic permutation importance on the validation set.",
        "- SHAP uses the trained MLP prediction function and a model-agnostic KernelExplainer.",
    ]
    (output_dir / "run_summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")
    pd.DataFrame({"feature": feature_columns}).to_csv(output_dir / "model_features.csv", index=False)


def print_metrics_to_terminal(metrics_df):
    """Print the main numerical metrics directly in the VS Code output panel."""
    display_columns = [
        "Split",
        "ROC-AUC",
        "Accuracy",
        "Balanced Accuracy",
        "Sensitivity / Recall",
        "Specificity",
        "Precision",
        "F1-score",
        "Brier Score",
        "PR-AUC",
        "Expected Calibration Error / ECE",
        "TN",
        "FP",
        "FN",
        "TP",
    ]
    display_df = metrics_df[display_columns].copy()
    for column in display_df.columns:
        if column != "Split":
            display_df[column] = display_df[column].map(
                lambda value: f"{value:.4f}" if isinstance(value, float) else value
            )

    print("\nNumerical metrics for validation and test")
    print("=" * 100)
    print(display_df.to_string(index=False))
    print("=" * 100)
    print()


def relative_image(output_dir, file_name):
    """Return a relative image path if the image exists, otherwise None."""
    path = output_dir / file_name
    if path.exists():
        return path.name
    return None


def image_block(title, image_name):
    """Build one HTML image block for the popup results viewer."""
    if image_name is None:
        return (
            f"<section><h3>{html.escape(title)}</h3>"
            "<p class=\"missing\">This plot was not generated.</p></section>"
        )

    return (
        f"<section><h3>{html.escape(title)}</h3>"
        f"<img src=\"{html.escape(image_name)}\" alt=\"{html.escape(title)}\"></section>"
    )


def save_classification_viewer(config, output_dir, metrics_df, shap_was_skipped):
    """Create one browser page containing metrics and all saved plots."""
    summary_path = output_dir / "run_summary.txt"
    summary_text = summary_path.read_text(encoding="utf-8") if summary_path.exists() else ""

    metrics_html = metrics_df.to_html(
        index=False,
        float_format=lambda value: f"{value:.4f}",
        classes="metrics",
        border=0,
    )

    split_sections = []
    for split_name in ["validation", "test"]:
        split_sections.append(
            f"""
            <h2>{split_name.title()} Plots</h2>
            <div class="grid">
              {image_block("Calibration Curve", relative_image(output_dir, f"{split_name}_calibration_curve.png"))}
              {image_block("ROC Curve", relative_image(output_dir, f"{split_name}_roc_curve.png"))}
              {image_block("Precision-Recall Curve", relative_image(output_dir, f"{split_name}_precision_recall_curve.png"))}
              {image_block("Confusion Matrix Heatmap", relative_image(output_dir, f"{split_name}_confusion_matrix_heatmap.png"))}
              {image_block("Probability Distribution", relative_image(output_dir, f"{split_name}_probability_distribution.png"))}
            </div>
            """
        )

    shap_note = ""
    if shap_was_skipped:
        shap_note = "<p class=\"missing\">SHAP was skipped for this run because --skip-shap was used.</p>"

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html.escape(config["experiment_name"])} MLP results</title>
  <style>
    body {{
      margin: 28px;
      font-family: Arial, sans-serif;
      color: #1f2933;
      background: #f7f7f5;
    }}
    h1, h2, h3 {{
      margin-bottom: 10px;
    }}
    .panel, section {{
      background: #ffffff;
      border: 1px solid #d7d7d0;
      border-radius: 6px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid #c8c8c0;
      background: #ffffff;
    }}
    table.metrics {{
      border-collapse: collapse;
      width: 100%;
      font-size: 13px;
      background: #ffffff;
    }}
    table.metrics th, table.metrics td {{
      border: 1px solid #cfcfc8;
      padding: 7px 8px;
      text-align: right;
    }}
    table.metrics th:first-child, table.metrics td:first-child {{
      text-align: left;
    }}
    pre {{
      white-space: pre-wrap;
      line-height: 1.45;
      font-size: 13px;
    }}
    .missing {{
      color: #8a4b08;
      font-weight: 600;
    }}
  </style>
</head>
<body>
  <h1>{html.escape(config["experiment_name"])} MLP Classification Results</h1>
  <div class="panel">
    <h2>Numerical Metrics</h2>
    {metrics_html}
  </div>
  <div class="panel">
    <h2>Run Summary</h2>
    <pre>{html.escape(summary_text)}</pre>
  </div>
  {"".join(split_sections)}
  <h2>Model Interpretation</h2>
  {shap_note}
  <div class="grid">
    {image_block("Feature Importance Plot", relative_image(output_dir, "feature_importance_plot.png"))}
    {image_block("SHAP Summary Plot", relative_image(output_dir, "shap_summary_plot.png"))}
    {image_block("SHAP Dependence Plot", relative_image(output_dir, "shap_dependence_plot.png"))}
    {image_block("SHAP Heatmap Plot", relative_image(output_dir, "shap_heatmap_plot.png"))}
    {image_block("Training Loss Curve", relative_image(output_dir, "training_loss_curve.png"))}
  </div>
</body>
</html>
"""
    viewer_path = output_dir / "classification_results_viewer.html"
    viewer_path.write_text(html_text, encoding="utf-8")
    return viewer_path


def open_file(path):
    """Open a saved file or folder with the default Windows app."""
    try:
        if hasattr(os, "startfile"):
            os.startfile(str(path))
        else:
            webbrowser.open(Path(path).resolve().as_uri())
    except OSError as error:
        print(f"Could not automatically open {path}: {error}")


def open_classification_outputs(viewer_path, output_dir):
    """Open the popup viewer and results folder after the run finishes."""
    print("Opening plots/results viewer...")
    open_file(viewer_path)
    open_file(output_dir)
    print(f"Viewer: {viewer_path}")
    print(f"Results folder: {output_dir}")


def run_classification(config, args):
    """Run one complete experiment from input CSV to saved metrics/plots."""
    check_virtual_environment()
    backend_info = prepare_libraries(args.backend, install_missing=not args.no_install)

    output_dir = Path(config["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data, omitted_rows = load_experiment_dataframe(config)
    feature_columns = get_feature_columns(data)
    split_data = split_xy(data, feature_columns)

    max_display = len(feature_columns)
    if config.get("top_10_display", False) or len(feature_columns) > 25:
        max_display = min(10, len(feature_columns))

    model_bundle = train_model(split_data, len(feature_columns), backend_info)
    save_training_history(model_bundle, output_dir)

    metrics_tables = []
    for split_name in ["validation", "test"]:
        y_prob = predict_positive_probability(model_bundle, split_data[split_name]["x"])
        metrics_tables.append(
            evaluate_split(
                split_name,
                split_data[split_name]["y"],
                y_prob,
                split_data[split_name]["ids"],
                output_dir,
            )
        )

    all_metrics_df = pd.concat(metrics_tables, ignore_index=True)
    all_metrics_df.to_csv(output_dir / "validation_test_metrics.csv", index=False)

    permutation_importance(
        model_bundle,
        split_data["validation"]["x"],
        split_data["validation"]["y"],
        feature_columns,
        output_dir,
        max_display,
    )

    shap_skip_marker = output_dir / "shap_skipped.txt"
    if args.skip_shap:
        shap_skip_marker.write_text(
            "SHAP was skipped because --skip-shap was provided.\n",
            encoding="utf-8",
        )
    else:
        if shap_skip_marker.exists():
            shap_skip_marker.unlink()
        save_shap_plots(
            model_bundle,
            split_data,
            feature_columns,
            output_dir,
            max_display,
            install_missing=not args.no_install,
        )

    save_run_summary(config, data, omitted_rows, feature_columns, backend_info, output_dir)
    print_metrics_to_terminal(all_metrics_df)
    viewer_path = save_classification_viewer(
        config,
        output_dir,
        all_metrics_df,
        shap_was_skipped=args.skip_shap,
    )

    print(f"Finished {config['experiment_name']} MLP classification.")
    print(f"Backend used: {model_bundle['backend']}")
    print(f"Results saved in: {output_dir}")
    print("Key files:")
    print(f"  Metrics: {output_dir / 'validation_test_metrics.csv'}")
    print(f"  Summary: {output_dir / 'run_summary.txt'}")
    print(f"  Feature importance: {output_dir / 'feature_importance_plot.png'}")
    if not args.skip_shap:
        print(f"  SHAP summary: {output_dir / 'shap_summary_plot.png'}")
    print(f"  Results viewer: {viewer_path}")

    if not args.no_popups:
        open_classification_outputs(viewer_path, output_dir)


def add_common_arguments(parser):
    parser.add_argument(
        "--backend",
        choices=["auto", "keras", "sklearn"],
        default="auto",
        help="auto tries TensorFlow/Keras first, then falls back to sklearn MLP.",
    )
    parser.add_argument(
        "--no-install",
        action="store_true",
        help="Do not install missing libraries automatically.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Run training/evaluation but skip SHAP plots.",
    )
    parser.add_argument(
        "--no-popups",
        action="store_true",
        help="Save all outputs but do not open the results viewer or folder.",
    )
    return parser


def run_from_config(config):
    parser = argparse.ArgumentParser(description=f"Run MLP classification for {config['experiment_name']}.")
    add_common_arguments(parser)
    args = parser.parse_args()
    run_classification(config, args)
