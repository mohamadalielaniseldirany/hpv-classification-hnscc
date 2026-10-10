from pathlib import Path
import sys


# Clinical-only MLP classification.
# This script uses the imputed clinical CSV and reuses the official split labels
# from the merged clinical+PET file. It does not create a new split.

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from mlp_classification_pipeline import run_from_config


FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent

CONFIG = {
    "experiment_name": "Clinical_DLonly",
    "input_csv": FOLDER / "HECKTOR_2025_Training_Task_3_imputed.csv",
    "split_target_csv": PROJECT / "Clinical_PET_DLonly" / "merged_clinical_pet_imputed.csv",
    "results_dir": FOLDER / "CLINICAL_ONLY_PLOTS" / "mlp_classification_results",
    "top_10_display": False,
}


if __name__ == "__main__":
    run_from_config(CONFIG)
