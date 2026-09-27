# Amazon ML Challenge: Business Entity Resolution
## Complete Plan & Progress Document

### 1. The Strategy (Two-Stage Pipeline)
Given the massive scale of the dataset (2.2M Source 1 entities and millions of candidates across Source 2 and Source 3) and the strict constraints (no internet lookups, <8B parameters, precision-focused F0.5 metric), we are utilizing a highly scalable, deterministic two-stage pipeline.

**Stage A: Candidate Generation (Blocking)**
*   **Goal:** Filter down millions of records to the top ~30 most likely matches for each Source 1 query, ensuring we don't miss true matches (High Recall).
*   **Methodology:** Semantic Embeddings using `Sentence-Transformers`. We encode standardized business names and addresses into dense vectors and use fast PyTorch Matrix Multiplication to find the most similar candidates.
*   **Optimization for 16GB RAM/4GB VRAM:** Processing data strictly in chunks (e.g. 50,000 at a time), executing matrix calculations on the GPU (RTX 3050), and keeping large catalog tensors neatly tucked into host CPU RAM, transferring batches to the GPU on demand.

**Stage B: Pairwise Matching Classifier**
*   **Goal:** Out of the 30 candidates, aggressively filter out false positives to predict the exact matches (High Precision).
*   **Methodology:** Gradient Boosted Trees (`LightGBM`). We calculate deterministic features (Jaccard similarity, Levenshtein distance, token overlap) between Source 1 and its candidate. LightGBM takes these features and predicts the probability of a match.
*   **Why LightGBM?** Extremely memory-efficient, blazingly fast to train on 1.7 million records, completely fits within the <8B parameter rule, and provides smooth probability distributions allowing us to surgically tune the threshold for the F0.5 score.

---

### 2. Progress Tracker

✅ **Phase 1: Project Setup & Validation Split**
- Created the codebase outside `student_resource` at `C:\Users\anshu\OneDrive\Desktop\amazon-ml\code\business_entity_resolution`.
- Generated a strictly stratified 20% validation split (`val_gt_split.tsv`) ensuring the distribution of countries and singleton businesses remains intact to prevent leaderboard overfitting.
- Wrote `evaluate.py` to match the exact F0.5 macro-averaging logic.

✅ **Phase 2: Data Normalization**
- Wrote `preprocess.py` to handle standardizations like stripping accents, converting "Corp" to "Corporation", "Ave" to "Avenue". This reduces variance and significantly boosts the model's ability to spot exact matches despite minor typos.

✅ **Phase 3: Candidate Generation (Blocking) - Completed**
- Switched from TF-IDF string matching to **Semantic Embeddings** to capture deeper contextual similarity.
- Re-wrote `blocking.py` to avoid memory (OOM) errors by implementing a robust chunk-based embedding generation, disk-caching the vector blocks, and utilizing PyTorch tensor dot products.
- Successfully encoded 10M+ businesses and extracted top 30 candidate pairs per query in a highly memory-optimized manner within the user's local constraint.

✅ **Phase 4: Feature Engineering - Completed**
- Built an ultra-fast, memory-bounded multiprocessing feature engineering pipeline (`feature_engineering.py`).
- Implemented 9 core features: RapidFuzz Levenshtein string distances, token sort ratio, partial ratio, token set ratio, address match ratios, numeric digit extraction, address missing indicators, and country code matching.
- Optimized with upfront negative downsampling, SQLite WAL-mode catalog indexing, zero-copy CSV line streaming, and bounded ProcessPoolExecutor concurrency (6 physical cores, max 8 in-flight chunks).
- Successfully processed all 2,206,821 Source 1 queries (2,207 chunks) in **457 seconds (~7.6 minutes)** without exceeding 7.6 GB RAM.
- Generated `output/full_train_features.csv` (1.43 GB, **17,658,741 total pairs**: 6,972,714 positives, 10,686,027 negatives).

✅ **Phase 5: LightGBM Training & Threshold Tuning - Completed**
- Built memory-safe streaming feature loader for all 17,658,741 feature rows in `train_model.py`.
- Split into 14,127,578 training pairs (80% queries) and 3,531,163 validation pairs (20% queries) strictly aligned with the stratified validation split `val_gt_split.tsv`.
- Trained 300 LightGBM gradient boosted decision trees across 6 physical cores in ~2 minutes with peak RAM under 2.2 GB.
- Binary validation logloss improved from 0.203 down to 0.0671.
- Validated probability thresholds across all 441,365 validation queries against the true competition macro-averaged F0.5 metric, correctly awarding 1.0 to singletons predicted empty.
- Discovered mathematically optimal decision threshold: **0.68**, achieving a validation **Macro F0.5 Score of 0.93278 (93.28%)** (compared to 0.8706 at the old 0.97 threshold).
- Saved booster to `output/lgbm_model.txt` and configuration to `output/threshold_config.json`.
- Key feature importances identified: `name_partial` (3,575) and `name_token_set` (3,460) are the top two drivers.

✅ **Phase 6: Final Inference & Packaging - Completed**
- Updated `preprocess.py` with French legal suffixes (`sarl`, `sas`, `sa`, `eurl`, `societe`, `etablissements`) and address road standardizations (`rue`, `chemin`, `impasse`, `allee`, `place`, `route`, `cedex`) to optimize for the 259K French entities in the test set.
- Built verified SQLite test catalog `output/test_catalog_temp.db` with 9,969,589 catalog records and 1,732,544 S1 records indexed on `entity_id`.
- Re-architected `inference.py` to stream test candidate pairs through 6 parallel worker processes with hot SQLite lookups, batch 9-feature RapidFuzz extraction, LightGBM inference, and strictly sequential output writing.
- Successfully completed full inference on all 1,732,544 test queries in 1,570s (~26 minutes) without exceeding 8.1 GB RAM.
- Predicted 8,497,332 matches across 1,700,277 queries, and preserved 32,267 true singletons (predicted empty).
- Validated output files using official competition validator `validate_submission.py`: **0 errors, 100% compliant**.
- Synchronized all files into `final_submission/` and compiled the final submission archive `Final_Submission.zip` (334.08 MB) containing all code, reproduction README, documentation, and final output TSVs.

---

### 3. The 0.98 Master Roadmap (Phase 7: Execution in Progress)
Following the diagnosis of public score `0.656` (caused by negative downsampling in validation and missed Indic/DBA/Domain patterns), we are executing the complete 0.98 end-to-end upgrade:

*   **Pillar 1: Indic Script Transliteration (`unidecode`)**
    - Unlocked 10.8% of the competition dataset (23.1% of Indian catalog names in Hindi, Telugu, Tamil, Bengali, Kannada, etc.) by converting native Indic characters into normalized Latin phonetics, boosting RapidFuzz similarity from 7% to 93%+.
*   **Pillar 2: Address Decomposition & Street Distractor Elimination**
    - Decomposing addresses into `street_num`, `street_name`, `city_state`, and `postal_code` to eliminate false merges between businesses on different streets in the same city.
*   **Pillar 3: DBA & Brand Matching Engine**
    - Extracted `is_dba_pattern` (1.0 if identical street number and street name in same country despite low name similarity) to capture trade name storefront matches.
*   **Pillar 4: Domain & Acronym Matching**
    - Implemented `name_compact_match` and `name_acronym_match` to link web domains (`warnersilverblueport.com`, `pcpartners.com`) directly to corporate names.
*   **Pillar 5: 100% Undownsampled Validation & Singleton Safeguard**
    - Split feature extraction into `full_train_features_v3.csv` (balanced with 8 negatives/positive) and `full_val_features_v3.csv` (100% undownsampled, top 30 candidates per query) to guarantee mathematically honest Macro F0.5 optimization.
    - Singleton safeguard: If max candidate probability < cutoff, output empty prediction to protect the 5.6% true singletons.

✅ **Phase 7: The 0.8663 Breakthrough (Completed & Submitted)**
- **Public Leaderboard Score:** `0.866297` (~0.8663), Local Validation: `0.89499`.
- Extracted 15 discriminative features across 43.25M training pairs and 13.25M undownsampled validation pairs.
- Trained 800-tree LightGBM model with exact Macro F0.5 threshold scanning (`optimal_threshold: 0.64`, `singleton_cutoff: 0.75`).
- Identified remaining mathematical ceiling: Candidate blocking recall capped at 91.15% (8.85% true match gap).

✅ **Phase 8: Inverted Key Blocking Candidate Expansion (Completed & Validated)**
- Built `expand_candidates_key_blocking.py` featuring multi-pathway blocking:
  - ISO Country Partitioning (`US`, `India`, `France`).
  - De-leeted, domain-stripped clean brand matching.
  - Decomposed street number + street name and postal code matching.
- Injected **3,594,196 high-recall candidate pairs** into `candidate_pairs.tsv` (706 MB) across all 1,732,544 test queries.
- Completed full test inference across 1,732,544 queries in 51.4 minutes:
  - Predicted **5,256,568 matches** across 1,587,879 entities.
  - Preserved **144,665 singletons** (8.35% empty predictions).
- Rigorously audited via `verify_subset.py`: **0 subset violations, 0 alignment errors across 1,732,544 rows**.
- Validated via official `validate_submission.py`: **PASS — no blocking issues found**.
- Compiled final contest package: `final_submission_package.zip` (334.84 MB).

---

### 4. Phase 9: Roadmap to 0.950 – 0.990+ & Scalable Multi-Pathway Candidate Expansion (Active)

#### Methodology Checkpoint & Empirical Findings:
1. **Empirical Evaluation of Dual-Threshold Step 1 (Completed):**
   - **Hypothesis:** Lowering probability threshold to $P \ge 0.50$ when clean brands match and address is missing rescues false negatives and boosts score.
   - **Evaluation on Full 441,365 Validation Entities (`test_dual_threshold_eval.py`):**
     - Baseline Macro $F_{0.5}$: `0.89285` ($T=0.64$, singleton cutoff $0.75$).
     - Step 1 Dual Threshold ($P \ge 0.50$): dropped to `0.89107` (-0.00178 regression).
     - Rescued candidates achieved only **48.37% precision** (8,380 false positives vs 7,851 true positives).
   - **Mathematical Proof:** Under $F_{0.5} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$, precision is weighted $4\times$ heavier than recall ($\frac{\partial F_{0.5}}{\partial P} / \frac{\partial F_{0.5}}{\partial R} = 4$). Any rescued candidate group must achieve $\ge 80.0\%$ precision; naive threshold lowering on the old model degrades score.

2. **Root Cause Analysis of Recall Bottleneck (Completed):**
   - Dense E5 semantic embeddings alone captured 1,368,042 / 1,526,765 true links = **89.60% recall** (158,723 true links missed).
   - Phase 8 key blocking raised recall to **92.99%** (capped theoretical ceiling $< 0.93$).
   - Granular inspection of 50+ missed pairs (`inspect_unrecovered_pairs.py`) revealed:
     - 40%+ of Indian and 15%+ of US addresses put landmarks/cities before street numbers (e.g., `Sector 57, Noida, C-66`, `Alliance, OH, 71 Oxford Street`), which prefix regex `^\s*(\d+)` completely missed.
     - Multi-pathway blocking: (1) Clean brand exact match, (2) Rare brand words ($\ge 4$ chars, vowel-collapsed, frequency-filtered), (3) Unanchored address tokens `(country, number, street_word)`, and (4) Postal keys.
     - Empirical test on 27,788 missed pairs (`test_super_recovery.py`): recovered 20,516 pairs (**73.83% recovery rate**).
     - **Projected candidate recall jumps to 97.28%**!

3. **Memory Safety & Hardware Utilization Architecture (Active):**
   - In multi-process candidate expansion on Windows (`spawn`), passing 5.3M dictionary entries via IPC queue duplicates memory across workers, risking `MemoryError` on 16GB RAM.
   - Designed bounded single-pass streaming architecture ($< 2.5\text{ GB}$ peak RAM), zero memory duplication, periodic per-core CPU and available RAM logging (`flush=True`).
   - Built and validated `test_demo_expansion_robust.py` on 50,000 queries and 200,000 catalog rows with complete RAM stability (8.66 GB RAM free).
   - **Candidate Expansion v4 Successfully Completed:**
     - Scanned all 9,969,589 catalog rows in 584.8s (9.7 minutes).
     - Added **15,249,142 extra high-recall candidates** across 1,684,832 queries (97.2% query coverage).
     - Generated and promoted `output/candidate_pairs.tsv` (849.3 MB, exactly 1,732,544 queries).
   - **6-Core Multiprocessing Smoke Test Passed (`test_inference_smoke.py`):**
     - Processed 3,000 queries across 6 cores in 15.5s (0.26 min) with zero errors and 8.14 GB available RAM.
   - **Full 6-Core Test Inference Completed Successfully:**
     - Processed all 1,732,544 test queries across 6 cores in 3,711.1s (61.85 minutes) at 467 queries/sec.
     - Total matches predicted: **5,243,327** across 1,590,066 queries.
     - Preserved singletons (predicted empty): **142,478** (8.22%).
     - Available RAM remained stable at **~7.50 GB free** throughout the run.
   - **Strict Subset Verification (erify_subset.py):**
     - **100% PERFECT: 0 violations, 0 alignment errors across all 1,732,544 queries**.
   - **Official Competition Validator (alidate_submission.py):**
     - **PASS: All 1,732,544 rows compliant with zero blocking issues**.
   - **Final Contest Package Generated:**
     - Successfully built inal_submission_package.zip (394.50 MB) containing all code, reproduction README, documentation, and final output TSVs.
