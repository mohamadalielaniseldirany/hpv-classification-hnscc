from pathlib import Path
import sys


# PET-only MLP classification.
# The PET feature file has no HPV Status or Split columns, so this script attaches
# those existing columns by PatientID from the merged clinical+PET file.

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from mlp_classification_pipeline import run_from_config


FOLDER = Path(__file__).resolve().parent
PROJECT = FOLDER.parent

CONFIG = {
    "experiment_name": "PET_DLonly",
    "input_csv": FOLDER / "pet_features_imputed.csv",
    "split_target_csv": PROJECT / "Clinical_PET_DLonly" / "merged_clinical_pet_imputed.csv",
    "results_dir": FOLDER / "PET_ONLY_PLOTS" / "mlp_classification_results",
    "top_10_display": True,
}


if __name__ == "__main__":
    run_from_config(CONFIG)
