from pathlib import Path
import os
import sys

# Fall back to the project's virtualenv packages when the global interpreter
# is used accidentally from VS Code or a fresh terminal.
project_site_packages = Path(__file__).resolve().parents[1] / "myenv" / "Lib" / "site-packages"
if project_site_packages.exists():
    project_site_packages_str = str(project_site_packages)
    if project_site_packages_str not in sys.path:
        sys.path.insert(0, project_site_packages_str)

import pandas as pd
matplotlib_cache_dir = Path(__file__).resolve().parent / ".matplotlib"
matplotlib_cache_dir.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))

import matplotlib.pyplot as plt

# path to the CSV file
# Default location: a DATA_SET folder placed next to the repository folder.
CT_FEATURES_PATH = Path(__file__).resolve().parents[2] / "DATA_SET" / "ct_features.csv"
df = pd.read_csv(CT_FEATURES_PATH)

# Displaying the first few rows
print(df.head())

FEATURE_GROUP_PREFIXES = {
    "GLCM": "CT_original_glcm_",
    "GLRLM": "CT_original_glrlm_",
    "GLSZM": "CT_original_glszm_",
    "NGTDM": "CT_original_ngtdm_",
}


def plot_histogram(column_name, title, color):
    feature_data = pd.to_numeric(df[column_name], errors="coerce").dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(feature_data, bins=20, color=color, edgecolor="black")
    plt.title(title)
    plt.xlabel(column_name)
    plt.ylabel("Number of Clients")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


def get_feature_group_columns(group_prefix):
    return [
        column_name
        for column_name in df.columns
        if str(column_name).startswith(group_prefix)
    ]


def plot_feature_group_heatmap(group_name, group_prefix):
    feature_columns = get_feature_group_columns(group_prefix)
    if len(feature_columns) < 2:
        print(f"Skipping {group_name}: fewer than 2 matching features were found.")
        return

    feature_data = (
        df[feature_columns]
        .apply(pd.to_numeric, errors="coerce")
        .dropna(axis=1, how="all")
    )
    if feature_data.shape[1] < 2:
        print(f"Skipping {group_name}: not enough numeric features were found.")
        return

    correlation_matrix = feature_data.corr()
    display_labels = [
        column_name.replace(group_prefix, "")
        for column_name in correlation_matrix.columns
    ]
    figure_size = (
        max(8, len(display_labels) * 0.45),
        max(6, len(display_labels) * 0.45),
    )

    plt.figure(figsize=figure_size)
    heatmap = plt.imshow(
        correlation_matrix,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    plt.colorbar(heatmap, label="Correlation")
    plt.title(f"Correlation Heatmap of {group_name} Features")
    plt.xticks(range(len(display_labels)), display_labels, rotation=90)
    plt.yticks(range(len(display_labels)), display_labels)
    plt.tight_layout()
    plt.show()


plot_histogram(
    "CT_original_firstorder_Mean",
    "Histogram of CT Original Firstorder Mean",
    "skyblue",
)

plot_histogram(
    "CT_original_firstorder_Entropy",
    "Histogram of CT Original Firstorder Entropy",
    "lightcoral",
)

plot_histogram(
    "CT_original_firstorder_Variance",
    "Histogram of CT Original Firstorder Variance",
    "mediumseagreen",
)

plot_histogram(
    "CT_original_shape_VoxelVolume",
    "Histogram of CT Original Shape VoxelVolume",
    "plum",
)

plot_histogram(
    "CT_original_shape_SurfaceArea",
    "Histogram of CT Original Shape SurfaceArea",
    "khaki",
)

plot_histogram(
    "CT_original_shape_Sphericity",
    "Histogram of CT Original Shape Sphericity",
    "lightsalmon",
)

for group_name, group_prefix in FEATURE_GROUP_PREFIXES.items():
    feature_columns = get_feature_group_columns(group_prefix)
    print(f"\n{group_name} features ({len(feature_columns)}):")
    for column_name in feature_columns:
        print(f" - {column_name}")

    plot_feature_group_heatmap(group_name, group_prefix)
