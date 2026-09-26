import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import defaultdict
import re

out_dir = r"C:\Users\anshu\OneDrive\Desktop\amazon-ml\output"
db_path = os.path.join(out_dir, "train_catalog_temp.db")
val_gt_path = os.path.join(out_dir, "val_gt_split.tsv")
val_cands_path = os.path.join(out_dir, "val_candidate_pairs_sample.tsv")

# Connect to DB
abs_db = os.path.abspath(db_path).replace('\\', '/')
conn = sqlite3.connect(f"file:{abs_db}?mode=ro&immutable=1", uri=True)
cur = conn.cursor()

# 1. Check address missingness in catalogs
cur.execute("SELECT count(*), sum(case when business_address is null or business_address = 'None' or trim(business_address) = '' then 1 else 0 end) FROM catalog")
total_cat, missing_cat = cur.fetchone()
print(f"Catalog (S2 + S3): {total_cat:,} total rows | {missing_cat:,} missing address ({missing_cat/total_cat*100:.2f}%)")

cur.execute("SELECT count(*), sum(case when business_address is null or business_address = 'None' or trim(business_address) = '' then 1 else 0 end) FROM s1_catalog")
total_s1, missing_s1 = cur.fetchone()
print(f"S1 Catalog:        {total_s1:,} total rows | {missing_s1:,} missing address ({missing_s1/total_s1*100:.2f}%)")

# 2. Check in test catalog
test_db_path = os.path.join(out_dir, "test_catalog_temp.db")
abs_test_db = os.path.abspath(test_db_path).replace('\\', '/')
conn_test = sqlite3.connect(f"file:{abs_test_db}?mode=ro&immutable=1", uri=True)
cur_test = conn_test.cursor()
cur_test.execute("SELECT count(*), sum(case when business_address is null or business_address = 'None' or trim(business_address) = '' then 1 else 0 end) FROM catalog")
t_cat, m_cat = cur_test.fetchone()
cur_test.execute("SELECT count(*), sum(case when business_address is null or business_address = 'None' or trim(business_address) = '' then 1 else 0 end) FROM s1_catalog")
t_s1, m_s1 = cur_test.fetchone()
print(f"Test Catalog (S2 + S3): {t_cat:,} total rows | {m_cat:,} missing address ({m_cat/t_cat*100:.2f}%)")
print(f"Test S1:                {t_s1:,} total rows | {m_s1:,} missing address ({m_s1/t_s1*100:.2f}%)")

# 3. Analyze what was missed by key blocking
# Look at 100 missed GT pairs
sys.path.insert(0, r"c:\Users\anshu\OneDrive\Desktop\amazon-ml\code\business_entity_resolution\src")
from expand_candidates_key_blocking import get_clean_brand, get_addr_keys

gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)
val_gt = {}
for _, row in gt_df.iterrows():
    sid = row['source1_entity_id']
    m = row.get('matched_entity_ids', '')
    if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
        val_gt[sid] = set(x.strip() for x in str(m).split(',') if x.strip())

e5_cands = defaultdict(set)
with open(val_cands_path, 'r', encoding='utf-8') as f:
    f.readline()
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) > 1 and parts[1]:
            e5_cands[parts[0]] = set(parts[1].split(','))

sample_s1 = list(val_gt.keys())[:3000]
missed_by_both = []
for sid in sample_s1:
    cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id = ?", (sid,))
    s1_row = cur.fetchone()
    if not s1_row: continue
    s1_b = get_clean_brand(s1_row[1])
    s1_sk, s1_pk = get_addr_keys(s1_row[2])
    
    for cid in val_gt[sid] - e5_cands[sid]:
        cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog WHERE entity_id = ?", (cid,))
        c_row = cur.fetchone()
        if not c_row: continue
        c_b = get_clean_brand(c_row[1])
        c_sk, c_pk = get_addr_keys(c_row[2])
        
        # Check why key blocking failed
        brand_match = (s1_b and c_b and s1_b == c_b and len(s1_b) >= 4)
        street_match = (s1_sk and c_sk and s1_sk == c_sk)
        pc_match = (s1_pk and c_pk and s1_pk == c_pk)
        
        if not (brand_match or street_match or pc_match):
            missed_by_both.append((s1_row, c_row))
            if len(missed_by_both) >= 15:
                break
    if len(missed_by_both) >= 15:
        break

print(f"\n--- Deep Sample of True Pairs Missed by BOTH E5 and Key-Blocking ---")
for s1_r, c_r in missed_by_both[:10]:
    print(f"S1:   Name: '{s1_r[1]}' | Addr: '{s1_r[2]}' | Country: '{s1_r[3]}'")
    print(f"Cand: Name: '{c_r[1]}' | Addr: '{c_r[2]}' | Country: '{c_r[3]}'")
    print(f"      Clean Brands: S1='{get_clean_brand(s1_r[1])}' vs Cand='{get_clean_brand(c_r[1])}'")
    s1_sk, s1_pk = get_addr_keys(s1_r[2])
    c_sk, c_pk = get_addr_keys(c_r[2])
    print(f"      Street Keys:  S1={s1_sk} vs Cand={c_sk}")
    print()
