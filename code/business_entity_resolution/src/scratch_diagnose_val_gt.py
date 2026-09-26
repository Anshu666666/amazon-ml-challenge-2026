import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import defaultdict
import unidecode
import re

sys.path.insert(0, r"c:\Users\anshu\OneDrive\Desktop\amazon-ml\code\business_entity_resolution\src")
from inference import (
    fast_15_features, normalize_text, get_compact_signature,
    get_acronym, decompose_address, extract_digits, FEATURE_COLS
)
from expand_candidates_key_blocking import get_clean_brand, get_addr_keys

out_dir = r"C:\Users\anshu\OneDrive\Desktop\amazon-ml\output"
db_path = os.path.join(out_dir, "train_catalog_temp.db")
model_path = os.path.join(out_dir, "lgbm_model_v3.txt")
val_gt_path = os.path.join(out_dir, "val_gt_split.tsv")

bst = lgb.Booster(model_file=model_path)
abs_db = os.path.abspath(db_path).replace('\\', '/')
conn = sqlite3.connect(f"file:{abs_db}?mode=ro&immutable=1", uri=True)
cur = conn.cursor()

# 1. Load validation ground truth
print("Loading validation ground truth...")
gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)
val_gt = {}
for _, row in gt_df.iterrows():
    sid = row['source1_entity_id']
    m = row.get('matched_entity_ids', '')
    if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
        val_gt[sid] = set(x.strip() for x in str(m).split(',') if x.strip())
    else:
        val_gt[sid] = set()

# 2. Load existing E5 validation candidates (sample)
val_cands_path = os.path.join(out_dir, "val_candidate_pairs_sample.tsv")
e5_cands = defaultdict(set)
if os.path.exists(val_cands_path):
    with open(val_cands_path, 'r', encoding='utf-8') as f:
        f.readline()
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1 and parts[1]:
                e5_cands[parts[0]] = set(parts[1].split(','))

# Let's take 5,000 validation S1 entities
val_s1_sample = list(val_gt.keys())[:10000]

# Check how many ground truth matches were missed by E5
missed_gt_pairs = []
for sid in val_s1_sample:
    true_matches = val_gt[sid]
    missed = true_matches - e5_cands[sid]
    for cid in missed:
        missed_gt_pairs.append((sid, cid))

print(f"Total True Matches in 10,000 S1 sample: {sum(len(val_gt[s]) for s in val_s1_sample):,}")
print(f"True Matches MISSED by E5:            {len(missed_gt_pairs):,}")

# 3. Check if key blocking generates these missed pairs
cur.execute("CREATE TEMP TABLE IF NOT EXISTS sample_s1 (eid TEXT PRIMARY KEY)")
cur.execute("DELETE FROM sample_s1")
cur.executemany("INSERT INTO sample_s1 VALUES (?)", [(s,) for s in val_s1_sample])

# Fetch S1 records
cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id IN (SELECT eid FROM sample_s1)")
s1_recs = {r[0]: r for r in cur.fetchall()}

# Build S1 key maps
s1_brand_map = defaultdict(list)
s1_street_map = defaultdict(list)
s1_pc_map = defaultdict(list)
for eid, (eid_val, name, addr, country) in s1_recs.items():
    b = get_clean_brand(name)
    cntry = country or ''
    if b and len(b) >= 4:
        s1_brand_map[(cntry, b)].append(eid)
    sk, pk = get_addr_keys(addr)
    if sk:
        s1_street_map[(cntry, sk[0], sk[1])].append(eid)
    if pk:
        s1_pc_map[(cntry, pk[0], pk[1])].append(eid)

# Check which missed GT pairs are recovered by key blocking
recovered_by_key = []
for sid, cid in missed_gt_pairs:
    cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog WHERE entity_id = ?", (cid,))
    c_row = cur.fetchone()
    if not c_row:
        continue
    _, c_name, c_addr, c_country = c_row
    cntry = c_country or ''
    b = get_clean_brand(c_name)
    sk, pk = get_addr_keys(c_addr)
    
    hit = False
    if b and len(b) >= 4 and (cntry, b) in s1_brand_map and sid in s1_brand_map[(cntry, b)]:
        hit = True
    elif sk and (cntry, sk[0], sk[1]) in s1_street_map and sid in s1_street_map[(cntry, sk[0], sk[1])]:
        hit = True
    elif pk and (cntry, pk[0], pk[1]) in s1_pc_map and sid in s1_pc_map[(cntry, pk[0], pk[1])]:
        hit = True
        
    if hit:
        recovered_by_key.append((sid, cid, s1_recs[sid], c_row))

print(f"Missed GT pairs RECOVERED by key blocking: {len(recovered_by_key):,} / {len(missed_gt_pairs):,} ({len(recovered_by_key)/len(missed_gt_pairs)*100:.1f}%)")

# 4. Now evaluate these recovered true matches with LightGBM!
print("\n--- What does LightGBM predict for these TRUE MATCHES? ---")
recovered_features = []
for sid, cid, s1_row, c_row in recovered_by_key:
    _, name, addr, country = s1_row
    n_name = normalize_text(name)
    c_comp = get_compact_signature(name)
    c_acro = get_acronym(name)
    c_addr, s_num, s_name, c_cs, c_pc = decompose_address(addr)
    d_str = extract_digits(f"{n_name} {addr}")
    s1_rec = (n_name, c_comp, c_acro, c_addr, s_num, s_name, c_cs, d_str, country or "")
    
    _, c_name, c_addr_raw, c_country = c_row
    cn_name = normalize_text(c_name)
    cc_comp = get_compact_signature(c_name)
    cc_acro = get_acronym(c_name)
    cc_addr, cs_num, cs_name, cc_cs, cc_pc = decompose_address(c_addr_raw)
    cd_str = extract_digits(f"{cn_name} {c_addr_raw}")
    c_rec = (cn_name, cc_comp, cc_acro, cc_addr, cs_num, cs_name, cc_cs, cd_str, c_country or "")
    
    feats = fast_15_features(s1_rec, c_rec, cid)
    recovered_features.append(feats)

X_rec = np.array(recovered_features, dtype=np.float32)
probs_rec = bst.predict(X_rec)

print(f"Count of Recovered True Matches: {len(probs_rec)}")
print(f"Predicted >= 0.64 (Won by Model):  {(probs_rec >= 0.64).sum()} ({(probs_rec >= 0.64).mean()*100:.1f}%)")
print(f"Predicted in [0.50, 0.64) (Missed): {((probs_rec >= 0.50) & (probs_rec < 0.64)).sum()} ({((probs_rec >= 0.50) & (probs_rec < 0.64)).mean()*100:.1f}%)")
print(f"Predicted in [0.40, 0.50) (Missed): {((probs_rec >= 0.40) & (probs_rec < 0.50)).sum()} ({((probs_rec >= 0.40) & (probs_rec < 0.50)).mean()*100:.1f}%)")
print(f"Predicted < 0.40 (Rejected):        {(probs_rec < 0.40).sum()} ({(probs_rec < 0.40).mean()*100:.1f}%)")

# Let's inspect some of the missed ones
print("\n--- Examples of Recovered True Matches with Prob < 0.64 ---")
for i in range(len(probs_rec)):
    if probs_rec[i] < 0.64:
        sid, cid, s1_row, c_row = recovered_by_key[i]
        print(f"P={probs_rec[i]:.4f} | S1: '{s1_row[1]}' | Cand: '{c_row[1]}'")
        print(f"           | S1 Addr: '{s1_row[2]}' | Cand Addr: '{c_row[2]}'")
        print(f"           | name_ratio={X_rec[i, 0]:.2f}, compact={X_rec[i, 4]:.1f}, addr_sort={X_rec[i, 10]:.2f}, street_sim={X_rec[i, 8]:.2f}")
        print()
        if i >= 10: break
