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

SCRIPT_DIR = Path(__file__).resolve().parent
# Default location: a DATA_SET folder placed next to the repository folder.
FALLBACK_CSV_PATH = SCRIPT_DIR.parents[1] / "DATA_SET" / "HECKTOR_2025_Training_Task_3.csv"
REQUIRED_COLUMNS = [
    "Age",
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "Treatment",
    "M-stage",
    "HPV Status",
]
MISSING_DATA_COLUMNS = [
    "Performance Status",
    "Alcohol Consumption",
    "Tobacco Consumption",
    "M-stage",
    "Treatment",
]
MISSING_DATA_DISPLAY_NAMES = {
    "Performance Status": "Performance Status",
    "Alcohol Consumption": "Alcohol",
    "Tobacco Consumption": "Tobacco",
    "M-stage": "M-stage",
    "Treatment": "Treatment",
}
MISSING_DATA_COMBINATIONS_TO_SHOW = [
    ("Performance Status", "Treatment"),
    ("Treatment", "M-stage"),
    ("Performance Status", "M-stage"),
]


def resolve_csv_path():
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()

    local_csv_files = sorted(
        SCRIPT_DIR.glob("*.csv"),
        key=lambda csv_path: csv_path.stat().st_mtime,
        reverse=True,
    )
    if local_csv_files:
        return local_csv_files[0]

    if FALLBACK_CSV_PATH.exists():
        return FALLBACK_CSV_PATH

    raise FileNotFoundError(
        "No CSV file was found. Add the corrected CSV next to this script or run "
        "the script with the CSV path as an argument."
    )


csv_path = resolve_csv_path()
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
if missing_columns:
    raise ValueError(
        "The CSV is missing required columns: " + ", ".join(missing_columns)
    )


def clean_category_series(series):
    return series.fillna("Unknown").astype(str).str.strip().replace("", "Unknown")


def map_binary_labels(series, label_prefix):
    return clean_category_series(series).replace(
        {
            "0": f"{label_prefix} 0",
            "0.0": f"{label_prefix} 0",
            "1": f"{label_prefix} 1",
            "1.0": f"{label_prefix} 1",
        }
    )


def build_missing_value_mask(dataframe, columns):
    selected_data = dataframe[columns]
    text_data = selected_data.astype("string").apply(
        lambda column: column.str.strip().str.lower()
    )
    text_missing_values = {"", "na", "n/a", "nan", "none", "null", "unknown"}

    return selected_data.isna() | text_data.isin(text_missing_values)


def format_missing_features(columns):
    return " + ".join(
        MISSING_DATA_DISPLAY_NAMES.get(column, column)
        for column in columns
    )


def add_required_missing_patterns(pattern_counts, required_patterns, columns):
    for display_pattern in required_patterns:
        unknown_columns = [
            column for column in display_pattern
            if column not in columns
        ]
        if unknown_columns:
            raise ValueError(
                "Unknown missing-data columns: " + ", ".join(unknown_columns)
            )

        count_key = tuple(
            column for column in columns
            if column in display_pattern
        )
        display_key = tuple(display_pattern)

        if count_key in pattern_counts and display_key != count_key:
            pattern_counts[display_key] = pattern_counts.pop(count_key)
        else:
            pattern_counts.setdefault(display_key, 0)


def build_missing_data_combination_summary(
    dataframe,
    columns,
    combine_isolated=True,
    required_patterns=None,
):
    missing_mask = build_missing_value_mask(dataframe, columns)
    missing_patterns = missing_mask.apply(
        lambda row: tuple(column for column in columns if row[column]),
        axis=1,
    )
    pattern_counts = missing_patterns.value_counts().to_dict()

    records = [
        {
            "Missing Pattern": "No missing",
            "Features Missing": "-",
            "Number of Patients": int(pattern_counts.pop((), 0)),
        }
    ]

    if combine_isolated:
        isolated_patterns = sorted(
            [pattern for pattern in pattern_counts if len(pattern) == 1],
            key=format_missing_features,
        )
        isolated_count = sum(pattern_counts.pop(pattern) for pattern in isolated_patterns)
        if isolated_count:
            isolated_features = " / ".join(
                format_missing_features(pattern)
                for pattern in isolated_patterns
            )
            records.append(
                {
                    "Missing Pattern": "1 missing",
                    "Features Missing": f"Any single feature ({isolated_features})",
                    "Number of Patients": int(isolated_count),
                }
            )

    if required_patterns:
        add_required_missing_patterns(pattern_counts, required_patterns, columns)

    sorted_patterns = sorted(
        pattern_counts.items(),
        key=lambda item: (len(item[0]), -item[1], format_missing_features(item[0])),
    )
    for pattern, count in sorted_patterns:
        records.append(
            {
                "Missing Pattern": f"{len(pattern)} missing",
                "Features Missing": format_missing_features(pattern),
                "Number of Patients": int(count),
            }
        )

    return pd.DataFrame(records)


def build_isolated_missing_summary(dataframe, columns):
    missing_mask = build_missing_value_mask(dataframe, columns)
    one_missing_rows = missing_mask.sum(axis=1).eq(1)
    isolated_counts = missing_mask.loc[one_missing_rows, columns].sum().astype(int)

    summary = pd.DataFrame(
        {
            "Feature Missing": [
                MISSING_DATA_DISPLAY_NAMES.get(column, column)
                for column in isolated_counts.index
            ],
            "Number of Patients": isolated_counts.values,
        }
    )

    return summary.sort_values(
        ["Number of Patients", "Feature Missing"],
        ascending=[False, True],
        ignore_index=True,
    )


def plot_dataframe_table(dataframe, title):
    if dataframe.empty:
        print(f"\n{title}: no rows to display.")
        return

    if len(dataframe.columns) == 3:
        column_widths = [0.24, 0.56, 0.20]
    elif len(dataframe.columns) == 2:
        column_widths = [0.70, 0.30]
    else:
        column_widths = None

    figure_height = max(2.4, 0.48 * (len(dataframe) + 1))
    figure, axis = plt.subplots(figsize=(12, figure_height))
    axis.axis("off")
    axis.set_title(title, loc="left", fontweight="bold", pad=12)

    table = axis.table(
        cellText=dataframe.astype(str).values,
        colLabels=dataframe.columns,
        cellLoc="left",
        colLoc="left",
        colWidths=column_widths,
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.35)

    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#d9d9d9")
        if row_index == 0:
            cell.set_facecolor("#3d3d3d")
            cell.set_text_props(weight="bold", color="white")
        else:
            cell.set_facecolor("#f7f7f7" if row_index % 2 == 0 else "white")
            if dataframe.columns[column_index] == "Number of Patients":
                cell.set_text_props(weight="bold")

    plt.tight_layout()
    plt.show()


def plot_age_by_hpv(age_series, hpv_series):
    age_data = pd.to_numeric(age_series, errors="coerce")
    hpv_data = map_binary_labels(hpv_series, "HPV")

    age_by_hpv = pd.DataFrame(
        {
            "Age": age_data,
            "HPV Status": hpv_data,
        }
    ).dropna(subset=["Age"])

    plot_groups = {
        "HPV 1": {
            "label": "HPV Positive",
            "color": "steelblue",
        },
        "HPV 0": {
            "label": "HPV Negative",
            "color": "mediumseagreen",
        },
    }
    age_by_hpv = age_by_hpv[age_by_hpv["HPV Status"].isin(plot_groups)]

    if age_by_hpv.empty:
        return

    bin_count = min(10, max(1, age_by_hpv["Age"].nunique()))

    plt.figure(figsize=(8, 5))
    for hpv_label, style in plot_groups.items():
        group_ages = age_by_hpv.loc[age_by_hpv["HPV Status"] == hpv_label, "Age"]
        if group_ages.empty:
            continue

        plt.hist(
            group_ages,
            bins=bin_count,
            alpha=0.6,
            label=style["label"],
            color=style["color"],
            edgecolor="black",
        )

    plt.title("Age Distribution by HPV Status")
    plt.xlabel("Age")
    plt.ylabel("Number of Clients")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="HPV Status")
    plt.tight_layout()
    plt.show()

# Displaying the first few rows
print(f"Loaded CSV file: {csv_path}")
print(df.head())

# Calculating missing data combinations and isolated missing data
missing_combination_summary = build_missing_data_combination_summary(
    df,
    MISSING_DATA_COLUMNS,
    combine_isolated=True,
    required_patterns=MISSING_DATA_COMBINATIONS_TO_SHOW,
)
isolated_missing_summary = build_isolated_missing_summary(
    df,
    MISSING_DATA_COLUMNS,
)

print("\nMissing Data Combinations (Full Summary)")
print(missing_combination_summary.to_string(index=False))

print("\nIsolated Missing Data")
print(isolated_missing_summary.to_string(index=False))

plot_dataframe_table(
    missing_combination_summary,
    "Missing Data Combinations (Full Summary)",
)
plot_dataframe_table(
    isolated_missing_summary,
    "Isolated Missing Data",
)

# Ploting the age distribution 
age_data = pd.to_numeric(df["Age"], errors="coerce").dropna()

plt.figure(figsize=(8, 5))
plt.hist(age_data, bins=10, color="skyblue", edgecolor="black")
plt.title("Age Distribution of Clients")
plt.xlabel("Age")
plt.ylabel("Number of Clients")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

# Plotting age by HPV status
plot_age_by_hpv(df["Age"], df["HPV Status"])

# Plotting the gender count
gender_counts = (
    clean_category_series(df["Gender"])
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8, 5))
plt.bar(gender_counts.index, gender_counts.values, color="lightcoral", edgecolor="black")
plt.title("Gender Count of Clients")
plt.xlabel("Gender")
plt.ylabel("Number of Clients")
plt.grid(axis="y", alpha=0.3)

for index, value in enumerate(gender_counts.values):
    plt.text(index, value, str(value), ha="center", va="bottom")

plt.tight_layout()
plt.show()


def plot_feature_by_hpv(feature_series, feature_prefix, title, xlabel):
    feature_data = map_binary_labels(feature_series, feature_prefix)
    hpv_data = map_binary_labels(df["HPV Status"], "HPV")

    feature_by_hpv = pd.crosstab(feature_data, hpv_data).sort_index()
    hpv_colors = {
        "HPV 0": "mediumseagreen",
        "HPV 1": "steelblue",
        "Unknown": "lightgray",
    }
    column_colors = [hpv_colors.get(column, "lightgray") for column in feature_by_hpv.columns]

    ax = feature_by_hpv.plot(
        kind="bar",
        figsize=(8, 5),
        color=column_colors,
        edgecolor="black",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Clients")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="HPV Status")

    for container in ax.containers:
        ax.bar_label(container, padding=3)

    plt.tight_layout()
    plt.show()


def plot_category_by_hpv(category_series, title, xlabel):
    category_data = clean_category_series(category_series)
    hpv_data = map_binary_labels(df["HPV Status"], "HPV")

    category_by_hpv = pd.crosstab(category_data, hpv_data).sort_index()
    hpv_colors = {
        "HPV 0": "mediumseagreen",
        "HPV 1": "steelblue",
        "Unknown": "lightgray",
    }
    column_colors = [hpv_colors.get(column, "lightgray") for column in category_by_hpv.columns]

    ax = category_by_hpv.plot(
        kind="bar",
        figsize=(8, 5),
        color=column_colors,
        edgecolor="black",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of Clients")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(title="HPV Status")

    for container in ax.containers:
        ax.bar_label(container, padding=3)

    plt.tight_layout()
    plt.show()


# Plotting HPV status by smoking
plot_feature_by_hpv(
    df["Tobacco Consumption"],
    "Smoking",
    "HPV Status by Smoking",
    "Smoking",
)


# Plotting HPV status by alcohol consumption
plot_feature_by_hpv(
    df["Alcohol Consumption"],
    "Alcohol",
    "HPV Status by Alcohol Consumption",
    "Alcohol Consumption",
)


# Plotting HPV status by M-stage
plot_category_by_hpv(
    df["M-stage"],
    "HPV Status by M-stage",
    "M-stage",
)


# Plotting the treatment distribution
treatment_data = (
    map_binary_labels(df["Treatment"], "Treatment")
)

gender_data = (
    map_binary_labels(df["Gender"], "Gender")
)

treatment_by_gender = pd.crosstab(treatment_data, gender_data).sort_index()

ax = treatment_by_gender.plot(
    kind="bar",
    figsize=(8, 5),
    color=["gold", "lightcoral", "skyblue"],
    edgecolor="black",
)
plt.title("Treatment Distribution by Gender")
plt.xlabel("Treatment")
plt.ylabel("Number of Clients")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Gender")

for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.tight_layout()
plt.show()


# Plotting M-stage in relation to treatment and gender
mstage_data = clean_category_series(df["M-stage"])
treatment_data = map_binary_labels(df["Treatment"], "Treatment")
gender_data = map_binary_labels(df["Gender"], "Gender")

mstage_by_treatment_gender = pd.crosstab(
    mstage_data,
    [treatment_data, gender_data],
)

treatment_order = [
    treatment_label
    for treatment_label in ["Treatment 0", "Treatment 1", "Unknown"]
    if treatment_label in treatment_data.unique()
]
gender_order = [
    gender_label
    for gender_label in ["Gender 0", "Gender 1", "Unknown"]
    if gender_label in gender_data.unique()
]
combo_order = [
    (treatment_label, gender_label)
    for treatment_label in treatment_order
    for gender_label in gender_order
    if (treatment_label, gender_label) in mstage_by_treatment_gender.columns
]

mstage_by_treatment_gender = mstage_by_treatment_gender.reindex(columns=combo_order)
mstage_by_treatment_gender.columns = [
    f"{treatment_label} | {gender_label}"
    for treatment_label, gender_label in mstage_by_treatment_gender.columns
]

combo_colors = {
    "Treatment 0 | Gender 0": "gold",
    "Treatment 0 | Gender 1": "khaki",
    "Treatment 0 | Unknown": "lightgray",
    "Treatment 1 | Gender 0": "lightcoral",
    "Treatment 1 | Gender 1": "skyblue",
    "Treatment 1 | Unknown": "silver",
    "Unknown | Gender 0": "sandybrown",
    "Unknown | Gender 1": "powderblue",
    "Unknown | Unknown": "gainsboro",
}
column_colors = [
    combo_colors.get(column, "lightgray")
    for column in mstage_by_treatment_gender.columns
]

ax = mstage_by_treatment_gender.plot(
    kind="bar",
    figsize=(10, 6),
    color=column_colors,
    edgecolor="black",
)
plt.title("M-stage by Treatment and Gender")
plt.xlabel("M-stage")
plt.ylabel("Number of Clients")
plt.xticks(rotation=0)
plt.grid(axis="y", alpha=0.3)
plt.legend(title="Treatment | Gender")

for container in ax.containers:
    ax.bar_label(container, padding=3)

plt.tight_layout()
plt.show()
