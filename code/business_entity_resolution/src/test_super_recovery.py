import os
import sys
import sqlite3
import pandas as pd
from collections import defaultdict
import unidecode
import re
from rapidfuzz import fuzz

sys.path.append('code/business_entity_resolution/src')
from expand_candidates_key_blocking import get_clean_brand, legal_pat, domain_pat

# Common words to exclude from rare brand words
COMMON_WORDS = set([
    'enterprises', 'services', 'solutions', 'company', 'group', 'india', 'international',
    'technologies', 'trading', 'consulting', 'industries', 'ventures', 'associates', 'partners',
    'management', 'holdings', 'global', 'agency', 'center', 'national', 'corporation', 'limited',
    'private', 'public', 'society', 'store', 'market', 'hotel', 'restaurant', 'cafe', 'hospital',
    'clinic', 'school', 'academy', 'trust', 'foundation', 'club', 'care', 'logistics', 'transport',
    'construction', 'builders', 'properties', 'realty', 'infra', 'infrastructure', 'software',
    'systems', 'digital', 'tech', 'smart', 'super', 'first', 'prime', 'golden', 'royal', 'apex',
    'shree', 'om', 'sai', 'ram', 'sri', 'guru', 'bharat', 'hindustan', 'delhi', 'mumbai',
    'america', 'american', 'united', 'city', 'state', 'north', 'south', 'east', 'west', 'new'
])

def extract_rare_brand_words(name):
    if not name or name == 'None':
        return []
    if not name.isascii():
        name = unidecode.unidecode(name)
    name = name.lower()
    name = domain_pat.sub('', name)
    name = legal_pat.sub(' ', name)
    # Collapse double vowels for transliterated words
    name = re.sub(r'([aeiou])\1+', r'\1', name)
    words = re.findall(r'[a-z]{4,}', name)
    return [w for w in words if w not in COMMON_WORDS]

# Unanchored address key extractor
addr_num_re = re.compile(r'(?:#|no\.?|plot|door|flat|unit|apt|suite|sector|h\.?no\.?|b-?|c-?|e-?|f-?|hs-?|sy\.?no\.?)?[-/#\s]*\b(\d{1,5}[a-z]?)\b', re.I)
street_word_re = re.compile(r'[a-z]{4,}', re.I)

COMMON_ADDR_WORDS = set([
    'street', 'road', 'avenue', 'lane', 'drive', 'court', 'circle', 'boulevard', 'highway',
    'floor', 'ground', 'first', 'second', 'third', 'block', 'sector', 'nagar', 'colony',
    'enclave', 'phase', 'cross', 'main', 'east', 'west', 'north', 'south', 'near', 'opposite',
    'behind', 'beside', 'above', 'below', 'plot', 'door', 'flat', 'house', 'building', 'tower',
    'complex', 'plaza', 'bazaar', 'market', 'city', 'state', 'india', 'pradesh', 'uttar',
    'maharashtra', 'karnataka', 'tamil', 'nadu', 'delhi', 'mumbai', 'bangalore', 'chennai',
    'hyderabad', 'kolkata', 'texas', 'california', 'florida', 'york', 'ohio', 'none'
])

def extract_unanchored_addr_keys(addr):
    if not addr or addr == 'None':
        return []
    addr_clean = unidecode.unidecode(str(addr)).lower()
    # Find all numbers
    numbers = set(re.findall(r'\b\d{1,5}[a-z]?\b', addr_clean))
    # Filter out zip codes (5-6 digits)
    numbers = [n for n in numbers if len(n) <= 4 or (len(n) == 5 and n.startswith('0'))]
    if not numbers:
        return []
    # Find distinctive street words
    words = [w for w in street_word_re.findall(addr_clean) if w not in COMMON_ADDR_WORDS]
    if not words:
        return []
    keys = []
    for n in numbers[:2]:
        for w in words[:3]:
            keys.append((n, w))
    return keys

print("1. Loading validation ground truth...", flush=True)
val_gt_path = 'output/val_gt_split.tsv'
gt_df = pd.read_csv(val_gt_path, sep="\t", dtype=str)
val_gt = {}
for _, row in gt_df.iterrows():
    sid = row['source1_entity_id']
    m = row.get('matched_entity_ids', '')
    if pd.notna(m) and str(m).strip() and str(m).strip() != 'nan':
        val_gt[sid] = set(x.strip() for x in str(m).split(',') if x.strip())

print("2. Loading existing E5 candidate pairs...", flush=True)
e5_cands = defaultdict(set)
for chunk in pd.read_csv('output/full_val_features_v3.csv', chunksize=1000000, usecols=['source1_entity_id', 'candidate_entity_id'], dtype=str):
    for s1, c in zip(chunk['source1_entity_id'].to_numpy(), chunk['candidate_entity_id'].to_numpy()):
        e5_cands[s1].add(c)

missed_by_s1 = {}
for s1, true_set in val_gt.items():
    diff = true_set - e5_cands[s1]
    if diff:
        missed_by_s1[s1] = diff

db_path = 'output/train_catalog_temp.db'
conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
cur = conn.cursor()

sample_s1 = list(missed_by_s1.keys())[:20000]
sample_missed_count = sum(len(missed_by_s1[s]) for s in sample_s1)
print(f"Testing on sample of {len(sample_s1):,} S1 queries ({sample_missed_count:,} missed true links)...", flush=True)

# Build Enhanced S1 keys
s1_clean_brand = defaultdict(list)
s1_rare_words = defaultdict(list)
s1_unanchored_addr = defaultdict(list)

batch_size = 1000
for i in range(0, len(sample_s1), batch_size):
    batch = sample_s1[i:i+batch_size]
    placeholders = ','.join(['?']*len(batch))
    cur.execute(f"SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE entity_id IN ({placeholders})", batch)
    for eid, name, addr, country in cur.fetchall():
        cntry = country or ''
        # Clean brand
        b = get_clean_brand(name)
        if b and len(b) >= 4:
            s1_clean_brand[(cntry, b)].append(eid)
        # Rare brand words
        for rw in extract_rare_brand_words(name):
            s1_rare_words[(cntry, rw)].append(eid)
        # Unanchored address keys
        for num, sword in extract_unanchored_addr_keys(addr):
            s1_unanchored_addr[(cntry, num, sword)].append(eid)

print(f"Enhanced S1 keys built: {len(s1_clean_brand):,} brands, {len(s1_rare_words):,} rare words, {len(s1_unanchored_addr):,} unanchored addr keys.", flush=True)

# Test Recovery
rec_brand = 0
rec_rare = 0
rec_addr = 0
rec_total = 0

for s1 in sample_s1:
    needed_cands = missed_by_s1[s1]
    placeholders = ','.join(['?']*len(needed_cands))
    cur.execute(f"SELECT entity_id, business_name, business_address, country FROM catalog WHERE entity_id IN ({placeholders})", list(needed_cands))
    for cid, c_name, c_addr, c_country in cur.fetchall():
        cntry = c_country or ''
        cb = get_clean_brand(c_name)
        c_rare = extract_rare_brand_words(c_name)
        c_addr_keys = extract_unanchored_addr_keys(c_addr)
        
        hit_b = (cb and len(cb) >= 4 and (cntry, cb) in s1_clean_brand and s1 in s1_clean_brand[(cntry, cb)])
        hit_r = any((cntry, rw) in s1_rare_words and s1 in s1_rare_words[(cntry, rw)] for rw in c_rare)
        hit_a = any((cntry, num, sword) in s1_unanchored_addr and s1 in s1_unanchored_addr[(cntry, num, sword)] for num, sword in c_addr_keys)
        
        if hit_b: rec_brand += 1
        if hit_r: rec_rare += 1
        if hit_a: rec_addr += 1
        if hit_b or hit_r or hit_a:
            rec_total += 1

print("\n" + "="*65, flush=True)
print(f"ENHANCED MULTI-PATHWAY KEY RECOVERY ON MISSED TRUE MATCHES ({sample_missed_count:,} pairs):", flush=True)
print(f"  Old Clean Brand Match:           {rec_brand:,} ({rec_brand/sample_missed_count*100:.2f}%)", flush=True)
print(f"  + Rare Brand Words:              {rec_rare:,} ({rec_rare/sample_missed_count*100:.2f}%)", flush=True)
print(f"  + Unanchored Address (Num+Word): {rec_addr:,} ({rec_addr/sample_missed_count*100:.2f}%)", flush=True)
print(f"  TOTAL UNIQUE TRUE MATCHES RECOVERED: {rec_total:,} / {sample_missed_count:,} ({rec_total/sample_missed_count*100:.2f}%)", flush=True)
print(f"  >>> PROJECTED TOTAL CANDIDATE RECALL: {89.60 + 10.40 * (rec_total/sample_missed_count):.2f}% <<<", flush=True)
print("="*65, flush=True)
conn.close()
