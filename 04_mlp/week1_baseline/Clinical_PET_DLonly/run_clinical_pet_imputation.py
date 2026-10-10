import argparse
from pathlib import Path
import sys


# This file runs preprocessing only for the combined Clinical + PET base CSV.
# It does not modify train.csv, validation.csv, or test.csv.

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from preprocess_base_inputs import (
    add_popup_argument,
    open_preprocessing_outputs,
    preprocess_dataset,
    print_explanation_report,
    print_preprocessing_result,
)


CONFIG = {
    "name": "Clinical_PET_merged",
    "input_csv": Path(__file__).resolve().parent / "merged_clinical_pet.csv",
    "output_csv": Path(__file__).resolve().parent / "merged_clinical_pet_imputed.csv",
    "plot_dir": Path(__file__).resolve().parent / "CLINICALPET_COMBINED_PLOTS" / "imputation_preprocessing",
}


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess the combined Clinical + PET base input CSV."
    )
    add_popup_argument(parser)
    args = parser.parse_args()

    result = preprocess_dataset(CONFIG)
    print("Finished Clinical + PET preprocessing.\n")
    print_preprocessing_result(result)
    print_explanation_report(result)

    if not args.no_popups:
        open_preprocessing_outputs(result)


if __name__ == "__main__":
    main()
