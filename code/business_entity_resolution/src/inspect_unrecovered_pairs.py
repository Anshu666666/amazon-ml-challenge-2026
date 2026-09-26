import os
import sys
import sqlite3
import pandas as pd
from collections import defaultdict
import unidecode
import re
from rapidfuzz import fuzz

sys.path.append('code/business_entity_resolution/src')
from expand_candidates_key_blocking import get_clean_brand, get_addr_keys

val_gt_path = 'output/val_gt_split.tsv'
gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)
val_gt = {}
for _, row in gt_df.iterrows():
    sid = row['source1_entity_id']
    m = row.get('matched_entity_ids', '')
    if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
        val_gt[sid] = set(x.strip() for x in str(m).split(',') if x.strip())

e5_cands = defaultdict(set)
chunk_size = 1000000
for chunk in pd.read_csv('output/full_val_features_v3.csv', chunksize=chunk_size, usecols=['source1_entity_id', 'candidate_entity_id'], dtype=str):
    s1s = chunk['source1_entity_id'].to_numpy()
    cands = chunk['candidate_entity_id'].to_numpy()
    for s1, c in zip(s1s, cands):
        e5_cands[s1].add(c)

db_path = 'output/train_catalog_temp.db'
conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
cur = conn.cursor()

sample_s1 = list(val_gt.keys())[:10000]

unrecovered = []
for s1 in sample_s1:
    true_set = val_gt[s1]
    missed = true_set - e5_cands[s1]
    if not missed:
        continue
    
    cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id = ?", (s1,))
    s1_row = cur.fetchone()
    if not s1_row: continue
    s1_b = get_clean_brand(s1_row[1])
    s1_sk, s1_pk = get_addr_keys(s1_row[2])
    cntry = s1_row[3] or ''
    
    for cid in missed:
        cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog WHERE entity_id = ?", (cid,))
        c_row = cur.fetchone()
        if not c_row: continue
        c_b = get_clean_brand(c_row[1])
        c_sk, c_pk = get_addr_keys(c_row[2])
        
        hit_brand = (s1_b and c_b and s1_b == c_b and len(s1_b) >= 4)
        hit_street = (s1_sk and c_sk and s1_sk == c_sk)
        hit_pc = (s1_pk and c_pk and s1_pk == c_pk)
        
        if not (hit_brand or hit_street or hit_pc):
            unrecovered.append((s1_row, c_row))
            if len(unrecovered) >= 50:
                break
    if len(unrecovered) >= 50:
        break

print(f"Sampled {len(unrecovered)} pairs missed by BOTH E5 and Current Key Blocking.\n")

# Classify why they were missed
categories = defaultdict(int)
for s1_r, c_r in unrecovered:
    s1_name, s1_addr, s1_c = s1_r[1], s1_r[2], s1_r[3]
    c_name, c_addr, c_c = c_r[1], c_r[2], c_r[3]
    
    # 1. Non-ascii in either name?
    has_non_ascii = (not s1_name.isascii()) or (not c_name.isascii())
    
    # 2. Token overlap of clean words?
    w1 = set(re.findall(r'[a-z0-9]{3,}', unidecode.unidecode(s1_name).lower()))
    w2 = set(re.findall(r'[a-z0-9]{3,}', unidecode.unidecode(c_name).lower()))
    shared_name_words = w1 & w2
    
    # 3. Numeric tokens in address?
    d1 = set(re.findall(r'\b\d+\b', str(s1_addr)))
    d2 = set(re.findall(r'\b\d+\b', str(c_addr)))
    shared_digits = d1 & d2
    
    # 4. Address word overlap?
    aw1 = set(re.findall(r'[a-z]{4,}', str(s1_addr).lower()))
    aw2 = set(re.findall(r'[a-z]{4,}', str(c_addr).lower()))
    shared_addr_words = aw1 & aw2
    
    cat = []
    if has_non_ascii:
        cat.append("Non-ASCII/Indic")
    if shared_name_words:
        cat.append(f"NameWordOverlap:{shared_name_words}")
    if shared_digits and shared_addr_words:
        cat.append(f"AddrDigits+Word:{shared_digits}&{len(shared_addr_words)}w")
    elif shared_digits:
        cat.append(f"AddrDigitsOnly:{shared_digits}")
    elif shared_addr_words:
        cat.append(f"AddrWordOnly:{len(shared_addr_words)}w")
        
    print(f"[{'/'.join(cat) or 'Unknown'}]")
    print(f"  S1: '{s1_name}' | '{s1_addr}'")
    print(f"  C : '{c_name}' | '{c_addr}'")
    print()

conn.close()
