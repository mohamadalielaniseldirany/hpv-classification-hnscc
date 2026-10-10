from pathlib import Path
import sys


# Combined clinical + PET MLP classification.
# This file already has PatientID, Split, HPV Status, and imputed numeric features.

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from mlp_classification_pipeline import run_from_config


FOLDER = Path(__file__).resolve().parent

CONFIG = {
    "experiment_name": "Clinical_PET_DLonly",
    "input_csv": FOLDER / "merged_clinical_pet_imputed.csv",
    "split_target_csv": None,
    "results_dir": FOLDER / "CLINICALPET_COMBINED_PLOTS" / "mlp_classification_results",
    "top_10_display": True,
}


if __name__ == "__main__":
    run_from_config(CONFIG)
