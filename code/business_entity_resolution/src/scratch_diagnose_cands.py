import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import lightgbm as lgb

sys.path.insert(0, r"c:\Users\anshu\OneDrive\Desktop\amazon-ml\code\business_entity_resolution\src")
from inference import (
    fast_15_features, normalize_text, get_compact_signature,
    get_acronym, decompose_address, extract_digits, FEATURE_COLS
)

out_dir = r"C:\Users\anshu\OneDrive\Desktop\amazon-ml\output"
db_path = os.path.join(out_dir, "test_catalog_temp.db")
model_path = os.path.join(out_dir, "lgbm_model_v3.txt")

# Load model
bst = lgb.Booster(model_file=model_path)

# Connect to DB
abs_db = os.path.abspath(db_path).replace('\\', '/')
conn = sqlite3.connect(f"file:{abs_db}?mode=ro&immutable=1", uri=True)
cur = conn.cursor()

# Find queries that had extra candidates in candidate_pairs.tsv vs backup
old_cands = {}
with open(os.path.join(out_dir, "candidate_pairs_v3_backup.tsv"), 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10000: break
        parts = line.strip().split('\t')
        old_cands[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 and parts[1] else set()

new_cands = {}
with open(os.path.join(out_dir, "candidate_pairs.tsv"), 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10000: break
        parts = line.strip().split('\t')
        new_cands[parts[0]] = set(parts[1].split(',')) if len(parts) > 1 and parts[1] else set()

# Find pairs added by key blocking
added_pairs = []
for sid in old_cands:
    diff = new_cands.get(sid, set()) - old_cands[sid]
    for cid in diff:
        added_pairs.append((sid, cid))
        if len(added_pairs) >= 1000:
            break
    if len(added_pairs) >= 1000:
        break

print(f"Sampled {len(added_pairs)} candidate pairs added by key blocking.")

# Evaluate them with LightGBM
probs = []
feature_rows = []
for sid, cid in added_pairs:
    cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id = ?", (sid,))
    s1_row = cur.fetchone()
    cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog WHERE entity_id = ?", (cid,))
    c_row = cur.fetchone()
    
    if s1_row and c_row:
        # Pre-process s1
        _, name, addr, country = s1_row
        n_name = normalize_text(name)
        c_comp = get_compact_signature(name)
        c_acro = get_acronym(name)
        c_addr, s_num, s_name, c_cs, c_pc = decompose_address(addr)
        d_str = extract_digits(f"{n_name} {addr}")
        s1_rec = (n_name, c_comp, c_acro, c_addr, s_num, s_name, c_cs, d_str, country or "")
        
        # Pre-process candidate
        _, c_name, c_addr_raw, c_country = c_row
        cn_name = normalize_text(c_name)
        cc_comp = get_compact_signature(c_name)
        cc_acro = get_acronym(c_name)
        cc_addr, cs_num, cs_name, cc_cs, cc_pc = decompose_address(c_addr_raw)
        cd_str = extract_digits(f"{cn_name} {c_addr_raw}")
        c_rec = (cn_name, cc_comp, cc_acro, cc_addr, cs_num, cs_name, cc_cs, cd_str, c_country or "")
        
        feats = fast_15_features(s1_rec, c_rec, cid)
        feature_rows.append(feats)

X = np.array(feature_rows, dtype=np.float32)
probs = bst.predict(X)

print("\n--- Probability Distribution of Key-Blocking Added Candidates ---")
print(f"Min Prob:    {probs.min():.4f}")
print(f"25% Prob:    {np.percentile(probs, 25):.4f}")
print(f"Median Prob: {np.median(probs):.4f}")
print(f"75% Prob:    {np.percentile(probs, 75):.4f}")
print(f"90% Prob:    {np.percentile(probs, 90):.4f}")
print(f"Max Prob:    {probs.max():.4f}")
print(f"Fraction >= 0.64 (Passed threshold): {(probs >= 0.64).mean()*100:.2f}%")
print(f"Fraction in [0.50, 0.64):             {((probs >= 0.50) & (probs < 0.64)).mean()*100:.2f}%")
print(f"Fraction in [0.40, 0.50):             {((probs >= 0.40) & (probs < 0.50)).mean()*100:.2f}%")
print(f"Fraction < 0.40:                      {(probs < 0.40).mean()*100:.2f}%")

# Let's inspect feature means for these pairs
df_feats = pd.DataFrame(X, columns=FEATURE_COLS)
df_feats['prob'] = probs

print("\n--- Top Features Importance in Model ---")
imp = bst.feature_importance(importance_type='gain')
for feat, score in sorted(zip(FEATURE_COLS, imp), key=lambda x: -x[1])[:8]:
    print(f"  {feat:<20}: {score:>10.1f}")

print("\n--- Inspection of High Probability Matches (Prob >= 0.64) ---")
print(f"Count: {(probs >= 0.64).sum()}")
if (probs >= 0.64).sum() > 0:
    print(df_feats[df_feats['prob'] >= 0.64][['name_ratio', 'name_token_sort', 'name_compact_match', 'street_name_sim', 'prob']].head(5))

print("\n--- Inspection of Near Misses (0.50 <= Prob < 0.64) ---")
print(f"Count: {((probs >= 0.50) & (probs < 0.64)).sum()}")
if ((probs >= 0.50) & (probs < 0.64)).sum() > 0:
    print(df_feats[(df_feats['prob'] >= 0.50) & (df_feats['prob'] < 0.64)][['name_ratio', 'name_token_sort', 'name_compact_match', 'street_name_sim', 'prob']].head(5))

print("\n--- Inspection of Low Probability (Prob < 0.40) ---")
print(f"Count: {(probs < 0.40).sum()}")
if (probs < 0.40).sum() > 0:
    print(df_feats[df_feats['prob'] < 0.40][['name_ratio', 'name_token_sort', 'name_compact_match', 'street_name_sim', 'prob']].head(5))
