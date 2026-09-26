# Implementation Plan - Amazon ML Challenge: Business Entity Resolution

## 1. Current Status (Phases 1-3 Completed)
- **Environment**: CUDA 12.1 + RTX 3050 working seamlessly. Memory constraints (16GB RAM) actively managed using streaming architectures.
- **Data Normalization**: Complete. `preprocess.py` handles case, punctuation, and standardizes legal/address abbreviations.
- **Candidate Generation (Blocking)**: Complete. Semantic embedding pipeline using `intfloat/multilingual-e5-small`. We implemented a robust chunk-caching system that processes 10M+ rows efficiently without RAM OOM errors. Candidate limit bumped to `k=100` to maximize recall ceiling.

---

## 2. Phase 4: Feature Engineering (Completed)
**Goal**: Create highly discriminative numerical features for the candidate pairs identified in Phase 3 to solve the "Semantic Trap".
**Execution Steps**:
1. **Memory-Safe Extraction**: Load `val_candidate_pairs_sample.tsv`. Read the massive Source 2 and Source 3 files in chunks, extracting *only* the specific entities needed for our candidate pairs to keep RAM usage minimal.
2. **Advanced String Matching**: Compare `business_name` and `business_address` independently. Used `rapidfuzz` extensively.
3. **Lexical & Edge-Case Features**: Implemented `is_address_missing` and `exact_digits_match`.
4. **Ground Truth Labeling**: Cross-referenced candidate pairs against `val_gt_split.tsv`. Output to `val_features.csv`.

---

## 3. Phase 5: LightGBM Training & F0.5 Thresholding (Completed)
**Goal**: Train a gradient boosted tree to weigh the Phase 4 features and find the perfect probability cut-off.
**Execution Steps**:
1. **Model Initialization**: Trained a LightGBM Classifier (`binary` objective, `class_weight='balanced'`) on 80% of the query groups.
2. **Threshold Scan**: Scanned probability thresholds from `0.50` to `0.99`.
3. **F0.5 Optimization**: Achieved an outstanding **Validation F0.5 Score of 0.8927** with a strict threshold of **> 0.97**, resulting in near-perfect precision (93.74%).

---

## 4. Phase 6: Final Test Inference & Submission (In Progress)
**Goal**: Run the full pipeline on `test_source1.tsv` and produce the required submission format.
**Execution Steps**:
1. Run blocking search for the test queries against our pre-cached catalog embeddings. The script correctly identifies that `test_source2.tsv` and `test_source3.tsv` are entirely new databases, so it is generating new embeddings into a `test_cache` directory. *(Currently Running in Background)*
2. Generate tabular features for all test candidates.
3. Apply the trained LightGBM model and filter predictions strictly using the optimal threshold (`> 0.97`).
4. Format and save `submission.tsv` ensuring ALL original test queries are present.
