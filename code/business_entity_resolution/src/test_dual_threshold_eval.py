import os
import gc
import time
import numpy as np
import pandas as pd

def compute_f05_single(pred_set, true_set):
    if len(true_set) == 0:
        return 1.0 if len(pred_set) == 0 else 0.0
    if len(pred_set) == 0:
        return 0.0
    tp = len(pred_set.intersection(true_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def evaluate_predictions(pred_dict, gt_map, total_entities):
    score_sum = 0.0
    for s1_id, true_set in gt_map.items():
        preds = pred_dict.get(s1_id, set())
        score_sum += compute_f05_single(preds, true_set)
    return score_sum / total_entities

print("1. Loading validation ground truth...", flush=True)
val_gt_path = 'output/val_gt_split.tsv'
gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)
total_val_entities = len(gt_df)
gt_map = {}
for _, row in gt_df.iterrows():
    sid = row['source1_entity_id']
    m = row.get('matched_entity_ids', '')
    if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
        gt_map[sid] = set(x.strip() for x in str(m).split(',') if x.strip())
    else:
        gt_map[sid] = set()

print(f"Total validation entities: {total_val_entities:,}", flush=True)

print("2. Loading validation probabilities...", flush=True)
val_probs = np.load('output/val_probs_lgb_v3.npy')

val_csv_path = 'output/full_val_features_v3.csv'
print(f"3. Streaming through {val_csv_path} and extracting features for candidate pairs...", flush=True)

cols_needed = ['source1_entity_id', 'candidate_entity_id', 'name_ratio', 'name_compact_match', 'is_addr_missing', 'street_name_sim']
chunk_size = 500000
idx = 0

candidates_by_s1 = {}
t0 = time.time()

for chunk in pd.read_csv(val_csv_path, chunksize=chunk_size, usecols=cols_needed, dtype={'source1_entity_id': str, 'candidate_entity_id': str, 'name_ratio': np.float32, 'name_compact_match': np.float32, 'is_addr_missing': np.float32, 'street_name_sim': np.float32}):
    n = len(chunk)
    chunk_p = val_probs[idx:idx+n]
    mask = chunk_p >= 0.40
    if np.any(mask):
        sub_s1 = chunk['source1_entity_id'].to_numpy()[mask]
        sub_c = chunk['candidate_entity_id'].to_numpy()[mask]
        sub_p = chunk_p[mask]
        sub_nr = chunk['name_ratio'].to_numpy()[mask]
        sub_nc = chunk['name_compact_match'].to_numpy()[mask]
        sub_iam = chunk['is_addr_missing'].to_numpy()[mask]
        sub_sns = chunk['street_name_sim'].to_numpy()[mask]
        
        for s1, c, p, nr, nc, iam, sns in zip(sub_s1, sub_c, sub_p, sub_nr, sub_nc, sub_iam, sub_sns):
            if s1 not in candidates_by_s1:
                candidates_by_s1[s1] = []
            candidates_by_s1[s1].append((c, float(p), float(nr), float(nc), float(iam), float(sns)))
    idx += n

print(f"Loaded {len(candidates_by_s1):,} S1 queries with candidates in {time.time()-t0:.1f}s", flush=True)

# Baseline Evaluation: T=0.64, cutoff=0.75
baseline_preds = {}
for s1, cands in candidates_by_s1.items():
    max_p = max(c[1] for c in cands)
    if max_p >= 0.75:
        p_set = set(c[0] for c in cands if c[1] >= 0.64)
        if p_set:
            baseline_preds[s1] = p_set

base_score = evaluate_predictions(baseline_preds, gt_map, total_val_entities)
print(f"\n=======================================================", flush=True)
print(f"Baseline Validation Macro F0.5 (T=0.64, cutoff=0.75): {base_score:.5f}", flush=True)
print(f"=======================================================\n", flush=True)

# Now evaluate Roadmap Step 1:
# Rule A: EXACT Roadmap proposal:
# if not is_match and prob >= 0.50:
#     if (name_compact_match == 1.0 or name_ratio >= 0.92) and (is_addr_missing == 1.0 or street_name_sim == 0.5):
#         is_match = True

for min_rescue_p in [0.50, 0.55, 0.58, 0.60]:
    step1_preds = {}
    rescued_total = 0
    rescued_tp = 0
    rescued_fp = 0
    
    for s1, cands in candidates_by_s1.items():
        max_p = max(c[1] for c in cands)
        # Does singleton guard apply? Let's test with max_p >= 0.75 first:
        if max_p >= 0.75:
            matched = set()
            for c, p, nr, nc, iam, sns in cands:
                is_m = (p >= 0.64)
                if not is_m and p >= min_rescue_p:
                    if (nc == 1.0 or nr >= 0.92) and (iam == 1.0 or sns == 0.5):
                        is_m = True
                        rescued_total += 1
                        if c in gt_map.get(s1, set()):
                            rescued_tp += 1
                        else:
                            rescued_fp += 1
                if is_m:
                    matched.add(c)
            if matched:
                step1_preds[s1] = matched
                
    s1_score = evaluate_predictions(step1_preds, gt_map, total_val_entities)
    diff = s1_score - base_score
    print(f"Roadmap Step 1 (rescue_p >= {min_rescue_p:.2f}, cutoff=0.75): Macro F0.5 = {s1_score:.5f} (Diff: {diff:+.5f}) | Rescued: {rescued_total:,} (TP: {rescued_tp:,}, FP: {rescued_fp:,}, Precision: {rescued_tp/(rescued_total+1e-9)*100:.2f}%)", flush=True)

# What if singleton cutoff was bypassed for rescued queries?
print("\n--- Testing if singleton cutoff is ALSO relaxed for Step 1 ---", flush=True)
for min_rescue_p in [0.50, 0.55, 0.60]:
    step1_preds_no_guard = {}
    rescued_total = 0
    rescued_tp = 0
    rescued_fp = 0
    
    for s1, cands in candidates_by_s1.items():
        matched = set()
        has_anchor = max(c[1] for c in cands) >= 0.75
        for c, p, nr, nc, iam, sns in cands:
            is_m = (p >= 0.64 and has_anchor)
            if not is_m and p >= min_rescue_p:
                if (nc == 1.0 or nr >= 0.92) and (iam == 1.0 or sns == 0.5):
                    is_m = True
                    rescued_total += 1
                    if c in gt_map.get(s1, set()):
                        rescued_tp += 1
                    else:
                        rescued_fp += 1
            if is_m:
                matched.add(c)
        if matched:
            step1_preds_no_guard[s1] = matched
            
    score_no_guard = evaluate_predictions(step1_preds_no_guard, gt_map, total_val_entities)
    diff = score_no_guard - base_score
    print(f"Step 1 (rescue_p >= {min_rescue_p:.2f}, NO cutoff): Macro F0.5 = {score_no_guard:.5f} (Diff: {diff:+.5f}) | Rescued: {rescued_total:,} (TP: {rescued_tp:,}, FP: {rescued_fp:,}, Precision: {rescued_tp/(rescued_total+1e-9)*100:.2f}%)", flush=True)
