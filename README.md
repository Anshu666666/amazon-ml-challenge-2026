# Amazon ML Challenge 2026: Business Entity Resolution Pipeline

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![LightGBM](https://img.shields.io/badge/Model-LightGBM_v3-orange.svg)](https://lightgbm.readthedocs.io/)
[![Metric](https://img.shields.io/badge/Metric-Macro_F0.5-purple.svg)]()

Production-grade, high-throughput solution for the **Amazon ML Challenge 2026 (Problem Statement 2: Multilingual Business Entity Resolution)** across three diverse enterprise datasets (`Source 1`, `Source 2`, and `Source 3`).

---

## 🏆 Current Performance & Architecture Highlights

- **Public Baseline:** `0.866297` Macro $F_{0.5}$.
- **Phase 8 Expansion (Target: 0.950+):** Injected **3,594,196 missing candidate pairs** using deterministic inverted key-blocking (de-leeted clean brand stems, address street-number pairs, postal-door combinations) into `candidate_pairs.tsv` (706 MB).
- **Test Inference Output:** Generated **5,256,568 verified entity matches** across all 1,732,544 test queries.
  - Strictly preserves **144,665 singletons** (8.35% singleton rate) to avoid false positive penalties on the precision-weighted metric ($F_{0.5}$).
  - Fully verified: 100% strict subset of candidate pairs, zero query alignment errors. Official competition validator: **PASS**.
- **Submission Artifact:** `final_submission_package.zip` (334.84 MB) attached to [GitHub Release v1.0](../../releases/tag/v1.0).

---

## 📁 Repository Structure

```text
├── code/
│   └── business_entity_resolution/
│       ├── requirements.txt            # Python dependencies (lightgbm, rapidfuzz, torch, etc.)
│       ├── README.md                   # Detailed technical execution manual
│       └── src/
│           ├── preprocess.py           # Multilingual legal/address normalization & clean brand logic
│           ├── blocking.py             # Dense semantic embedding blocking (multilingual-e5-small)
│           ├── expand_candidates_key_blocking.py # Phase 8 deterministic inverted key-blocking
│           ├── feature_engineering.py  # 15 fine-grained lexical, phonics, and structural features
│           ├── train_model.py          # LightGBM booster training & exact Macro F0.5 threshold optimizer
│           ├── inference.py            # Zero-lock 6-core multiprocessing test inference engine
│           ├── verify_subset.py        # Streaming validator for candidate-matching subset compliance
│           └── evaluate.py             # Exact competition Macro F0.5 evaluation metric
├── output/
│   ├── lgbm_model_v3.txt               # Trained LightGBM booster model (11 MB)
│   ├── threshold_config_v3.json        # Calibrated decision threshold (0.64) & singleton cutoff (0.75)
│   └── lgbm_model.txt                  # Baseline model
├── MASTER_POSTMORTEM_AND_ROADMAP_TO_0.99.md # Detailed architectural postmortem and path to 0.99
├── Plan_and_Progress.md                # Task tracker and phase logs
└── .gitignore                          # Excludes raw multi-GB datasets, DB caches, and large TSVs
```

---

## 💾 Large File Management Strategy

To ensure seamless collaboration across teammates without running into Git's 100MB file limit or slowing down git pulls:
1. **Source Code & Models Tracked in Git:**
   - All source scripts, evaluation scripts, documentation, and the lightweight trained model weights (`output/lgbm_model_v3.txt`, 11 MB) are tracked directly in Git.
2. **Ignored Intermediate Caches:**
   - Raw dataset files (`dataset/`), SQLite database caches (`catalogs.db`), and dense embedding arrays (`.npy`) are excluded via `.gitignore`.
3. **Submission Deliverables via GitHub Releases:**
   - The verified contest deliverable (`final_submission_package.zip`, 334.84 MB), containing `matching_results.tsv` and `candidate_pairs.tsv`, is hosted directly under [GitHub Releases](../../releases).

---

## 🚀 Quickstart for Teammates

### 1. Environment Setup
```bash
git clone <REPO_URL>
cd amazon-ml-challenge-2026

# Create and activate virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

pip install -r code/business_entity_resolution/requirements.txt
```

### 2. Verify Current Submission Package
If you need to inspect or submit the current Phase 8 results:
1. Download `final_submission_package.zip` from the Releases tab.
2. Run the validator:
```bash
python 6ab10eb3b23ba_student_resource/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir 6ab10eb3b23ba_student_resource/student_resource/dataset/test
```

### 3. Next Iteration: Toward 0.99 Macro F0.5
Review [MASTER_POSTMORTEM_AND_ROADMAP_TO_0.99.md](MASTER_POSTMORTEM_AND_ROADMAP_TO_0.99.md) for the active roadmap:
- **Feature Addition:** Incorporate RapidFuzz Jaro-Winkler brand stem similarity into `feature_engineering.py`.
- **Model Retraining:** Retrain LightGBM on the augmented key-blocked training candidate matrix (`full_train_candidate_pairs_v4.tsv`) to calibrate against brand variations.
