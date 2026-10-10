from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))

from mlp_binary_pipeline import PipelineConfig, run_mlp_classification


CLINICAL_CATEGORICAL_COLUMNS = [
    "Gender",
    "Tobacco Consumption",
    "Alcohol Consumption",
    "Performance Status",
    "Treatment",
    "Tobacco Consumption_missing",
    "Alcohol Consumption_missing",
    "Performance Status_missing",
    "M-stage_M0",
    "M-stage_M1",
    "M-stage_Mx",
    "M-stage_Unknown",
]


def main() -> None:
    config = PipelineConfig(
        model_name="Clinical-only",
        folder=CURRENT_DIR,
        input_csv="clinical_entry_preprocessed.csv",
        sampler="SMOTENC",
        categorical_columns=CLINICAL_CATEGORICAL_COLUMNS,
        excluded_columns=["CenterID"],
        scale_features=False,
        scale_feature_prefixes=[],
        compare_before_scaling=False,
        plot_top_n=None,
    )
    run_mlp_classification(config)


if __name__ == "__main__":
    main()
