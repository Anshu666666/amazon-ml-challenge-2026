import os
import sys
import sqlite3
import pandas as pd
import time
from collections import defaultdict
import unidecode
import re

print("1. Loading validation ground truth...", flush=True)
val_gt_path = 'output/val_gt_split.tsv'
gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)
val_gt = {}
total_true_links = 0
for _, row in gt_df.iterrows():
    sid = row['source1_entity_id']
    m = row.get('matched_entity_ids', '')
    if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
        s = set(x.strip() for x in str(m).split(',') if x.strip())
        val_gt[sid] = s
        total_true_links += len(s)

print(f"Validation S1 entities with true matches: {len(val_gt):,} | Total true links: {total_true_links:,}", flush=True)

print("2. Loading existing E5 candidate pairs for validation...", flush=True)
e5_cands = defaultdict(set)
chunk_size = 1000000
for chunk in pd.read_csv('output/full_val_features_v3.csv', chunksize=chunk_size, usecols=['source1_entity_id', 'candidate_entity_id'], dtype=str):
    s1s = chunk['source1_entity_id'].to_numpy()
    cands = chunk['candidate_entity_id'].to_numpy()
    for s1, c in zip(s1s, cands):
        e5_cands[s1].add(c)

print(f"E5 candidate map built for {len(e5_cands):,} S1 entities.", flush=True)

# Count how many are missed
missed_by_s1 = {}
total_missed = 0
for s1, true_set in val_gt.items():
    diff = true_set - e5_cands[s1]
    if diff:
        missed_by_s1[s1] = diff
        total_missed += len(diff)

print(f"Total true links missed by E5: {total_missed:,} across {len(missed_by_s1):,} S1 queries.", flush=True)

# 3. Test Key Blocking logic on these missed queries
sys.path.append('code/business_entity_resolution/src')
from expand_candidates_key_blocking import get_clean_brand, get_addr_keys

print("\n3. Testing Key Blocking Recovery against missed pairs...", flush=True)
db_path = 'output/train_catalog_temp.db'
conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
cur = conn.cursor()

# Sample 20,000 missed S1 queries for fast, statistically rigorous benchmark
sample_s1 = list(missed_by_s1.keys())[:20000]
sample_missed_count = sum(len(missed_by_s1[s]) for s in sample_s1)
print(f"Testing on sample of {len(sample_s1):,} S1 queries ({sample_missed_count:,} missed true links)...", flush=True)

# Build S1 keys for sample
s1_brand_map = defaultdict(list)
s1_street_map = defaultdict(list)
s1_pc_map = defaultdict(list)

# Fetch S1 records in batches
batch_size = 1000
for i in range(0, len(sample_s1), batch_size):
    batch = sample_s1[i:i+batch_size]
    placeholders = ','.join(['?']*len(batch))
    cur.execute(f"SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id IN ({placeholders})", batch)
    for eid, name, addr, country in cur.fetchall():
        cntry = country or ''
        b = get_clean_brand(name)
        if b and len(b) >= 4:
            s1_brand_map[(cntry, b)].append(eid)
        sk, pk = get_addr_keys(addr)
        if sk:
            s1_street_map[(cntry, sk[0], sk[1])].append(eid)
        if pk:
            s1_pc_map[(cntry, pk[0], pk[1])].append(eid)

print(f"S1 sample keys built: {len(s1_brand_map):,} brands, {len(s1_street_map):,} streets, {len(s1_pc_map):,} postal keys.", flush=True)

# Check all missed candidate targets
recovered_count = 0
recovered_by_brand = 0
recovered_by_street = 0
recovered_by_pc = 0

for s1 in sample_s1:
    needed_cands = missed_by_s1[s1]
    placeholders = ','.join(['?']*len(needed_cands))
    cur.execute(f"SELECT entity_id, business_name, business_address, country FROM catalog WHERE entity_id IN ({placeholders})", list(needed_cands))
    for cid, c_name, c_addr, c_country in cur.fetchall():
        cntry = c_country or ''
        cb = get_clean_brand(c_name)
        csk, cpk = get_addr_keys(c_addr)
        
        hit_brand = (cb and len(cb) >= 4 and (cntry, cb) in s1_brand_map and s1 in s1_brand_map[(cntry, cb)])
        hit_street = (csk and (cntry, csk[0], csk[1]) in s1_street_map and s1 in s1_street_map[(cntry, csk[0], csk[1])])
        hit_pc = (cpk and (cntry, cpk[0], cpk[1]) in s1_pc_map and s1 in s1_pc_map[(cntry, cpk[0], cpk[1])])
        
        if hit_brand: recovered_by_brand += 1
        if hit_street: recovered_by_street += 1
        if hit_pc: recovered_by_pc += 1
        
        if hit_brand or hit_street or hit_pc:
            recovered_count += 1

print("\n" + "="*60, flush=True)
print(f"RECOVERY RESULTS ON MISSED TRUE MATCHES ({sample_missed_count:,} pairs):", flush=True)
print(f"  Recovered by Clean Brand:   {recovered_by_brand:,} ({recovered_by_brand/sample_missed_count*100:.2f}%)", flush=True)
print(f"  Recovered by Street Key:    {recovered_by_street:,} ({recovered_by_street/sample_missed_count*100:.2f}%)", flush=True)
print(f"  Recovered by Postal Key:    {recovered_by_pc:,} ({recovered_by_pc/sample_missed_count*100:.2f}%)", flush=True)
print(f"  Total Unique Recovered:     {recovered_count:,} / {sample_missed_count:,} ({recovered_count/sample_missed_count*100:.2f}%)", flush=True)
print(f"  Total Candidate Recall (E5 + Key Blocking): {89.60 + 10.40 * (recovered_count/sample_missed_count):.2f}%", flush=True)
print("="*60, flush=True)
conn.close()
