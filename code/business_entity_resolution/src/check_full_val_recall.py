import pandas as pd
import numpy as np

print("Loading validation ground truth...", flush=True)
gt_df = pd.read_csv('output/val_gt_split.tsv', sep='\t', dtype=str)
gt_map = {}
total_true_links = 0
for _, row in gt_df.iterrows():
    sid = row['source1_entity_id']
    m = row.get('matched_entity_ids', '')
    if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
        s = set(x.strip() for x in str(m).split(',') if x.strip())
        gt_map[sid] = s
        total_true_links += len(s)
    else:
        gt_map[sid] = set()

print(f"Total S1 validation entities: {len(gt_df):,}", flush=True)
print(f"Entities with true matches:   {len([s for s in gt_map.values() if s]):,}", flush=True)
print(f"True singletons:              {len([s for s in gt_map.values() if not s]):,}", flush=True)
print(f"Total true matching links:    {total_true_links:,}", flush=True)

print("\nStreaming through full_val_features_v3.csv to calculate E5 candidate recall...", flush=True)
found_links = 0
chunk_size = 1000000

for chunk in pd.read_csv('output/full_val_features_v3.csv', chunksize=chunk_size, usecols=['source1_entity_id', 'candidate_entity_id'], dtype=str):
    s1s = chunk['source1_entity_id'].to_numpy()
    cands = chunk['candidate_entity_id'].to_numpy()
    for s1, c in zip(s1s, cands):
        if c in gt_map.get(s1, set()):
            found_links += 1

print(f"True links captured by E5 candidates: {found_links:,} / {total_true_links:,} ({found_links/total_true_links*100:.2f}%)", flush=True)
print(f"True links MISSED by E5 candidates:   {total_true_links - found_links:,} ({(total_true_links - found_links)/total_true_links*100:.2f}%)", flush=True)
