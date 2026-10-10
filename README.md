# HPV Status Classification in Head & Neck Cancer (HNSCC)

AI-based prediction of **HPV status** in head and neck squamous cell carcinoma (HNSCC) patients from **clinical data** and **PET radiomic features**. Three models are compared — **LightGBM**, a **Multilayer Perceptron (MLP)** and an **FT-Transformer** — with a full evaluation of discrimination, calibration and interpretability (SHAP).

> Internship project — Bachelor in Computer Networks & Telecommunications Engineering, Lebanese University, Faculty of Technology.
> Hosted by **Together For Chehim** / CRPC (Committee for Collaborative Research and Projects), April – July 2026.

---

## Why it matters

HPV status is a key prognostic biomarker in head and neck cancer: HPV-positive tumours generally respond better to treatment. It is usually determined with laboratory tests on tissue samples, which are invasive, slow and not always available. This project investigates whether HPV status can be predicted **non-invasively** from routinely collected clinical variables and PET imaging features.

## Dataset

- **HECKTOR 2025 – Task 3** (clinical information + CT / PET radiomic features).
- After merging and removing patients without radiomic features: **514 patients**, strongly imbalanced (~89 % HPV-positive).
- Fixed stratified split, `random_state = 42`:

| Split | Patients | HPV-negative | HPV-positive |
|---|---|---|---|
| Train (70 %) | 359 | 40 | 319 |
| Validation (15 %) | 77 | 8 | 69 |
| Test (15 %) | 78 | 9 | 69 |

> ⚠️ **The data is not included in this repository.** HECKTOR is patient medical data distributed under its own licence. Request access from the official challenge page: https://hecktor25.grand-challenge.org/

## Pipeline

```
HECKTOR clinical + radiomics
        │
        ▼
01  Exploratory data analysis (distributions, missing values, radiomic feature groups)
        │
        ▼
02  Merge, clean, stratified 70/15/15 split
    · rule-based cleaning of Performance Status
    · missing-indicator columns
    · median (Age) / most-frequent / constant ("Unknown" M-stage) imputation — fitted on train only
    · Min-Max scaling of Age, one-hot encoding of M-stage
        │
        ▼
03  LightGBM           04  MLP (Keras)                  05  FT-Transformer (Keras)
    gradient boosting      week 1: baseline                 feature tokenizer + [CLS] token
                           week 2: StandardScaler           3 transformer blocks, 4 heads
                                   + SMOTE / SMOTENC        token dim 32, 60 epochs
        │
        ▼
Evaluation on validation & test (threshold 0.5)
ROC-AUC · PR-AUC · Accuracy · Balanced accuracy · Sensitivity · Specificity
Precision · F1 · Brier score · ECE + calibration curves · Confusion matrices
Permutation importance · SHAP (TreeExplainer / KernelExplainer)
```

Each model is trained on three feature sets: **Clinical only**, **PET only**, and **Clinical + PET**.

## Results

### Test ROC-AUC (78 unseen patients)

| Feature set | LightGBM | MLP (scaling + SMOTE) | FT-Transformer |
|---|---|---|---|
| Clinical only | **0.940** | 0.862 | 0.820 |
| PET only | 0.549 | 0.621 | 0.638 |
| Clinical + PET | 0.857 | 0.813 | **0.916** |

Best FT-Transformer (Clinical + PET), test set: ROC-AUC 0.916 · PR-AUC 0.988 · accuracy 0.936 · sensitivity 0.986 · specificity 0.556 · F1 0.965 · ECE 0.071.

### Key findings

- **Clinical data carries most of the signal.** M-stage, tobacco and alcohol consumption (and whether they are missing) are the dominant features across all models, according to both SHAP and permutation importance.
- **PET radiomics alone are weak**, but in the combined models PET shape and texture features (e.g. `Maximum2DDiameterSlice`, `Sphericity`, GLCM / GLSZM texture features) add complementary information.
- **Accuracy is misleading on imbalanced data.** The week 1 MLP and the unscaled models reached ~89 % accuracy while predicting *every* patient as HPV-positive (specificity 0, balanced accuracy 0.50). Balanced accuracy, specificity and calibration metrics were essential to detect this.
- **Feature scaling was decisive for neural networks.** Scaling only the PET columns raised the Clinical + PET MLP test ROC-AUC from 0.50 to 0.81 and the specificity from 0.00 to 0.56.

| MLP — test ROC-AUC | Week 1 baseline | Week 2 (scaling + SMOTE) |
|---|---|---|
| Clinical | 0.432 | 0.862 |
| PET | 0.519 | 0.621 |
| Clinical + PET | 0.471 | 0.813 |

- **Missingness bias.** SHAP shows that a *missing* M-stage (`M-stage_Unknown`) strongly pushes predictions towards HPV-negative. The model partly learns data-collection patterns rather than biology — an important limitation to report for any clinical use.
- **Specificity remains the main limitation** (few HPV-negative patients: 40 in training, 9 in test), so results should be interpreted with caution.

## Repository structure

```
hpv-classification-hnscc/
├── 01_data_analysis/        EDA scripts (clinical + CT radiomics) and figures
├── 02_preprocessing/        merge / clean / split script, preprocessing plan, missing-value reports
├── 03_lightgbm/             Clinical, PET and Clinical+PET LightGBM + results PDF + written analysis
├── 04_mlp/
│   ├── week1_baseline/      imputation + first MLP runs and their results
│   └── week2_scaling_smote/ shared pipeline, StandardScaler + SMOTE/SMOTENC, results and interpretation
├── 05_ft_transformer/       FT-Transformer pipeline, 3 experiments with interpretation, theory slides
├── docs/                    internship report and defence presentation (PDF)
├── requirements.txt
└── README.md
```

## How to run

1. Get access to HECKTOR 2025 Task 3 and place the CSV files in a `DATA_SET/` folder **next to** the repository folder:

   ```
   DATA_SET/
   ├── HECKTOR_2025_Training_Task_3.csv
   ├── ct_features.csv
   ├── pet_features.csv
   └── petct_features.csv
   hpv-classification-hnscc/
   ```

2. Create an environment and install the dependencies:

   ```bash
   python -m venv myenv
   # Windows: myenv\Scripts\activate    |    Linux/macOS: source myenv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the preprocessing to create the fixed split (`train.csv`, `validation.csv`, `test.csv`):

   ```bash
   python 02_preprocessing/preprocess_and_split_data.py
   ```

4. Copy the split files and the input feature file into the folder of the experiment you want to run (each model script reads its CSVs from its own folder), then run it, for example:

   ```bash
   python 03_lightgbm/clinical_lightgbm.py
   python 04_mlp/week2_scaling_smote/Clinical_PET/run_clinical_pet_mlp.py
   python 05_ft_transformer/Clinical_PET/run_clinical_pet_ft.py
   ```

   Plots, metrics tables and SHAP figures are written to a `plots/` folder next to each script. Set `POPUP_REPORT=0` to disable the interactive report window.

## Tech stack

Python · pandas · NumPy · scikit-learn · LightGBM · TensorFlow / Keras · imbalanced-learn (SMOTE, SMOTENC) · SHAP · Matplotlib · seaborn

## Acknowledgements

- Supervision and guidance: **Mr. Mohamad Hammoush** and **Dr. Aya Elhajj Chehade**
- Host organisation: **Together For Chehim** – CRPC
- Data: HECKTOR 2025 challenge organisers

## Author

**Mohamad Ali EL ANIS EL DIRANY** — Computer Science engineering student, UTBM (France)
[LinkedIn](https://www.linkedin.com/in/mohamad-ali-el-anis-el-dirany-b99b68417) · [GitHub](https://github.com/mohamadalielaniseldirany)
