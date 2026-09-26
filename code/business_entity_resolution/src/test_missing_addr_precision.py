import pandas as pd
import numpy as np

print("Loading validation ground truth...", flush=True)
val_gt_path = 'output/val_gt_split.tsv'
gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)
gt_map = {}
for _, row in gt_df.iterrows():
    sid = row['source1_entity_id']
    m = row.get('matched_entity_ids', '')
    if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
        gt_map[sid] = set(x.strip() for x in str(m).split(',') if x.strip())

val_probs = np.load('output/val_probs_lgb_v3.npy')

print("Analyzing pairs with missing address in full_val_features_v3.csv...", flush=True)
chunk_size = 1000000
idx = 0

stats = {
    'total_missing_addr': 0,
    'missing_addr_p_50_64': 0,
    'missing_addr_p_50_64_exact_name': 0,
    'exact_name_tp': 0,
    'exact_name_fp': 0,
    'high_token_sort_tp': 0,
    'high_token_sort_fp': 0
}

for chunk in pd.read_csv('output/full_val_features_v3.csv', chunksize=chunk_size, usecols=['source1_entity_id', 'candidate_entity_id', 'name_ratio', 'name_token_sort', 'name_compact_match', 'is_addr_missing']):
    n = len(chunk)
    cp = val_probs[idx:idx+n]
    
    # Missing address
    iam = chunk['is_addr_missing'].to_numpy() == 1.0
    nr = chunk['name_ratio'].to_numpy()
    nts = chunk['name_token_sort'].to_numpy()
    ncm = chunk['name_compact_match'].to_numpy()
    s1s = chunk['source1_entity_id'].to_numpy()
    cands = chunk['candidate_entity_id'].to_numpy()
    
    mask = iam & (cp >= 0.50) & (cp < 0.64)
    stats['missing_addr_p_50_64'] += int(np.sum(mask))
    
    for s1, c, p_val, r, ts, cm in zip(s1s[mask], cands[mask], cp[mask], nr[mask], nts[mask], ncm[mask]):
        is_true = c in gt_map.get(s1, set())
        if r >= 0.98 or (cm == 1.0 and ts >= 0.95):
            stats['missing_addr_p_50_64_exact_name'] += 1
            if is_true:
                stats['exact_name_tp'] += 1
            else:
                stats['exact_name_fp'] += 1
                
    idx += n

print("\n--- RESULTS ON MISSING ADDRESS PAIRS WITH P in [0.50, 0.64) ---", flush=True)
print(f"Total pairs in [0.50, 0.64) with missing address: {stats['missing_addr_p_50_64']:,}", flush=True)
print(f"Pairs with near-exact name (ratio >= 0.98 or compact+token_sort >= 0.95): {stats['missing_addr_p_50_64_exact_name']:,}", flush=True)
tp = stats['exact_name_tp']
fp = stats['exact_name_fp']
total = tp + fp
print(f"  True Positives:  {tp:,}", flush=True)
print(f"  False Positives: {fp:,}", flush=True)
print(f"  Precision:       {tp/(total+1e-9)*100:.2f}%", flush=True)
