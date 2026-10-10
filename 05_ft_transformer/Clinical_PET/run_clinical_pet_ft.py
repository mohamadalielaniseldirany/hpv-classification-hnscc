from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR.parent))

from ft_transformer_pipeline import FTTransformerConfig, run_ft_transformer_classification


def main() -> None:
    config = FTTransformerConfig(
        experiment_name="Clinical + PET",
        folder=CURRENT_DIR,
        input_csv="clinical_pet_merged_preprocessed.csv",
        plot_top_n=10,
        shap_top_n=10,
    )
    run_ft_transformer_classification(config)


if __name__ == "__main__":
    main()
