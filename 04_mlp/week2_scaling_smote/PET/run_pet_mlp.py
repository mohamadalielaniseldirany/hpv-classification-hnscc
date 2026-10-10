from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))

from mlp_binary_pipeline import PipelineConfig, run_mlp_classification


def main() -> None:
    config = PipelineConfig(
        model_name="PET-only",
        folder=CURRENT_DIR,
        input_csv="pet_features.csv",
        sampler="SMOTE",
        categorical_columns=[],
        excluded_columns=["CenterID"],
        scale_features=True,
        scale_feature_prefixes=[],
        compare_before_scaling=True,
        plot_top_n=10,
    )
    run_mlp_classification(config)


if __name__ == "__main__":
    main()
