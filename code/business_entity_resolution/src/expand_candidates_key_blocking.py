import os
import sys
import time
import psutil
import sqlite3
import unidecode
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

def get_sys_util_string():
    per_cpu = psutil.cpu_percent(percpu=True)
    mem = psutil.virtual_memory()
    cpu_str = " ".join([f"C{i}:{int(c)}%" for i, c in enumerate(per_cpu)])
    ram_str = f"RAM Avail: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}% used)"
    return f"[{cpu_str}] | {ram_str}"

def log_memory(label=""):
    print(f"[{label}] {get_sys_util_string()}", flush=True)

legal_pat = re.compile(
    r'\b(pvt|pv\.t|private|ltd|lt\.d|limited|corp|corporation|inc|incorporated|llc|sarl|sas|sa|eurl|enterprises|services|solutions|company|co|group|praaivett|praivet|limittedd|kampani|kampanii)\b',
    re.I
)
domain_pat = re.compile(r'\.(com|org|net|in|fr|co|biz|info)\b', re.I)

COMMON_BRAND_WORDS = set([
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

COMMON_ADDR_WORDS = set([
    'street', 'road', 'avenue', 'lane', 'drive', 'court', 'circle', 'boulevard', 'highway',
    'floor', 'ground', 'first', 'second', 'third', 'block', 'sector', 'nagar', 'colony',
    'enclave', 'phase', 'cross', 'main', 'east', 'west', 'north', 'south', 'near', 'opposite',
    'behind', 'beside', 'above', 'below', 'plot', 'door', 'flat', 'house', 'building', 'tower',
    'complex', 'plaza', 'bazaar', 'market', 'city', 'state', 'india', 'pradesh', 'uttar',
    'maharashtra', 'karnataka', 'tamil', 'nadu', 'delhi', 'mumbai', 'bangalore', 'chennai',
    'hyderabad', 'kolkata', 'texas', 'california', 'florida', 'york', 'ohio', 'none'
])

def get_clean_brand(name):
    if not name or name == 'None':
        return ''
    if not name.isascii():
        name = unidecode.unidecode(name)
    name = name.lower()
    name = domain_pat.sub('', name)
    name = name.replace('0', 'o').replace('5', 's').replace('1', 'i').replace('3', 'e')
    name = legal_pat.sub(' ', name)
    name = re.sub(r'([aeiou])\1+', r'\1', name)
    clean = re.sub(r'[^a-z0-9]', '', name)
    return clean

def extract_rare_brand_words(name):
    if not name or name == 'None':
        return []
    if not name.isascii():
        name = unidecode.unidecode(name)
    name = name.lower()
    name = domain_pat.sub('', name)
    name = legal_pat.sub(' ', name)
    name = re.sub(r'([aeiou])\1+', r'\1', name)
    words = re.findall(r'[a-z]{4,}', name)
    return [w for w in words if w not in COMMON_BRAND_WORDS]

street_word_re = re.compile(r'[a-z]{4,}', re.I)

def extract_unanchored_addr_keys(addr):
    if not addr or addr == 'None':
        return []
    addr_clean = unidecode.unidecode(str(addr)).lower()
    numbers = set(re.findall(r'\b\d{1,5}[a-z]?\b', addr_clean))
    numbers = [n for n in numbers if len(n) <= 4 or (len(n) == 5 and n.startswith('0'))]
    if not numbers:
        return []
    words = [w for w in street_word_re.findall(addr_clean) if w not in COMMON_ADDR_WORDS]
    if not words:
        return []
    keys = []
    for n in numbers[:2]:
        for w in words[:2]:
            keys.append((n, w))
    return keys

pc_re = re.compile(r'\b\d{5,6}\b')
num_re = re.compile(r'^\s*(?:#|no\.?|plot|door)?\s*([0-9]+[a-z]?)', re.I)

def get_addr_keys(addr):
    if not addr or addr == 'None':
        return None, None
    pc_m = pc_re.search(addr)
    pc = pc_m.group(0) if pc_m else None
    
    idx = addr.find(',')
    first_part = addr[:idx] if idx != -1 else addr
    num_m = num_re.search(first_part)
    num = num_m.group(1).lower() if num_m else None
    if num_m:
        rem = first_part[num_m.end():].strip(' -#/,')
    else:
        rem = first_part.strip()
    street = re.sub(r'[^a-z0-9 ]', '', rem.lower()).strip()
    
    street_key = (num, street) if (num and len(street) >= 3) else None
    pc_key = (pc, num) if (pc and num) else None
    return street_key, pc_key


def worker_scan_catalog_slice(args):
    """
    Worker process: Scans a designated rowid slice of the SQLite catalog table.
    """
    worker_id, db_path, start_rowid, end_rowid, key_bundles, max_extra_per_query, max_keys_freq = args
    brand_map, rare_map, unanchored_addr_map, street_map, pc_map = key_bundles
    
    abs_db = os.path.abspath(db_path).replace('\\', '/')
    conn = sqlite3.connect(f"file:{abs_db}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    
    cur.execute(
        "SELECT entity_id, business_name, business_address, country FROM catalog WHERE rowid BETWEEN ? AND ?",
        (start_rowid, end_rowid)
    )
    
    local_hits = defaultdict(list)
    scanned = 0
    total_to_scan = end_rowid - start_rowid + 1
    t0 = time.time()
    batch_size = 50000
    
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        scanned += len(rows)
        for eid, name, addr, country in rows:
            cntry = country or ''
            
            # 1. Clean brand
            b = get_clean_brand(name)
            if b and len(b) >= 4:
                k = (cntry, b)
                if k in brand_map:
                    targets = brand_map[k]
                    if len(targets) <= max_keys_freq:
                        for sid in targets:
                            if len(local_hits[sid]) < max_extra_per_query:
                                local_hits[sid].append(eid)
                                
            # 2. Rare brand words
            for rw in extract_rare_brand_words(name):
                k = (cntry, rw)
                if k in rare_map:
                    targets = rare_map[k]
                    if len(targets) <= max_keys_freq:
                        for sid in targets:
                            if len(local_hits[sid]) < max_extra_per_query:
                                local_hits[sid].append(eid)
                                
            # 3. Unanchored address
            for num, sword in extract_unanchored_addr_keys(addr):
                k = (cntry, num, sword)
                if k in unanchored_addr_map:
                    targets = unanchored_addr_map[k]
                    if len(targets) <= max_keys_freq:
                        for sid in targets:
                            if len(local_hits[sid]) < max_extra_per_query:
                                local_hits[sid].append(eid)
                                
            # 4. Street and postal keys
            sk, pk = get_addr_keys(addr)
            if sk:
                k = (cntry, sk[0], sk[1])
                if k in street_map:
                    targets = street_map[k]
                    if len(targets) <= max_keys_freq:
                        for sid in targets:
                            if len(local_hits[sid]) < max_extra_per_query:
                                local_hits[sid].append(eid)
            if pk:
                k = (cntry, pk[0], pk[1])
                if k in pc_map:
                    targets = pc_map[k]
                    if len(targets) <= max_keys_freq:
                        for sid in targets:
                            if len(local_hits[sid]) < max_extra_per_query:
                                local_hits[sid].append(eid)
                                
        if scanned % 200000 == 0 or scanned == total_to_scan:
            elapsed = time.time() - t0
            util = get_sys_util_string()
            print(f"  [Worker {worker_id}] Scanned {scanned:,}/{total_to_scan:,} ({scanned/total_to_scan*100:.1f}%) in {elapsed:.1f}s | Hits: {len(local_hits):,} queries | {util}", flush=True)
            
    conn.close()
    return local_hits


def expand_candidates(input_cands_path, output_cands_path, db_path, max_extra_per_query=15, max_keys_freq=40):
    start_time = time.time()
    num_cores = min(6, os.cpu_count())
    log_memory("Expansion Start")
    print(f"\n{'='*75}", flush=True)
    print(f"High-Recall 6-Core Candidate Expansion: {input_cands_path} -> {output_cands_path}", flush=True)
    print(f"Catalog DB: {db_path} | Workers: {num_cores} | Max Extra Cands/Query: {max_extra_per_query} | Key Freq Cap: {max_keys_freq}", flush=True)
    print(f"{'='*75}\n", flush=True)
    
    abs_db = os.path.abspath(db_path).replace('\\', '/')
    conn = sqlite3.connect(f"file:{abs_db}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    
    # 1. Load S1 query records from SQLite and index keys
    print("Step 1/4: Loading S1 queries and building multi-pathway blocking keys...", flush=True)
    t0 = time.time()
    cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog")
    
    s1_brand_map = defaultdict(list)
    s1_rare_map = defaultdict(list)
    s1_unanchored_addr_map = defaultdict(list)
    s1_street_map = defaultdict(list)
    s1_pc_map = defaultdict(list)
    total_s1 = 0
    
    batch_size = 250000
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        for eid, name, addr, country in rows:
            total_s1 += 1
            cntry = country or ''
            
            # Pathway 1: Exact clean brand
            b = get_clean_brand(name)
            if b and len(b) >= 4:
                s1_brand_map[(cntry, b)].append(eid)
                
            # Pathway 2: Rare brand words
            for rw in extract_rare_brand_words(name):
                s1_rare_map[(cntry, rw)].append(eid)
                
            # Pathway 3: Unanchored address keys
            for num, sword in extract_unanchored_addr_keys(addr):
                s1_unanchored_addr_map[(cntry, num, sword)].append(eid)
                
            # Pathway 4: Standard street and postal keys
            sk, pk = get_addr_keys(addr)
            if sk:
                s1_street_map[(cntry, sk[0], sk[1])].append(eid)
            if pk:
                s1_pc_map[(cntry, pk[0], pk[1])].append(eid)
                
    print(f"  Indexed {total_s1:,} S1 queries in {time.time()-t0:.2f}s.", flush=True)
    print(f"  Keys: {len(s1_brand_map):,} brands | {len(s1_rare_map):,} rare words | {len(s1_unanchored_addr_map):,} unanchored addrs | {len(s1_street_map):,} streets | {len(s1_pc_map):,} postal keys.", flush=True)
    log_memory("After S1 Keys")
    
    # 2. Stream through catalog table and match with multi-pathway blocking
    print(f"\nStep 2/4: Streaming through catalog table to retrieve key-matching candidates...", flush=True)
    t0 = time.time()
    cur.execute("SELECT min(rowid), max(rowid), count(*) FROM catalog")
    min_rowid, max_rowid, total_cat = cur.fetchone()
    
    cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog")
    
    total_merged = defaultdict(list)
    cat_scanned = 0
    matched_hits = 0
    batch_size = 100000
    
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        cat_scanned += len(rows)
        for eid, name, addr, country in rows:
            cntry = country or ''
            
            # Pathway 1: Clean brand
            b = get_clean_brand(name)
            if b and len(b) >= 4:
                k = (cntry, b)
                if k in s1_brand_map and len(s1_brand_map[k]) <= max_keys_freq:
                    for sid in s1_brand_map[k]:
                        if len(total_merged[sid]) < max_extra_per_query:
                            total_merged[sid].append(eid)
                            matched_hits += 1
                            
            # Pathway 2: Rare brand words
            for rw in extract_rare_brand_words(name):
                k = (cntry, rw)
                if k in s1_rare_map and len(s1_rare_map[k]) <= max_keys_freq:
                    for sid in s1_rare_map[k]:
                        if len(total_merged[sid]) < max_extra_per_query:
                            total_merged[sid].append(eid)
                            matched_hits += 1
                            
            # Pathway 3: Unanchored address
            for num, sword in extract_unanchored_addr_keys(addr):
                k = (cntry, num, sword)
                if k in s1_unanchored_addr_map and len(s1_unanchored_addr_map[k]) <= max_keys_freq:
                    for sid in s1_unanchored_addr_map[k]:
                        if len(total_merged[sid]) < max_extra_per_query:
                            total_merged[sid].append(eid)
                            matched_hits += 1
                            
            # Pathway 4: Street and postal keys
            sk, pk = get_addr_keys(addr)
            if sk:
                k = (cntry, sk[0], sk[1])
                if k in s1_street_map and len(s1_street_map[k]) <= max_keys_freq:
                    for sid in s1_street_map[k]:
                        if len(total_merged[sid]) < max_extra_per_query:
                            total_merged[sid].append(eid)
                            matched_hits += 1
            if pk:
                k = (cntry, pk[0], pk[1])
                if k in s1_pc_map and len(s1_pc_map[k]) <= max_keys_freq:
                    for sid in s1_pc_map[k]:
                        if len(total_merged[sid]) < max_extra_per_query:
                            total_merged[sid].append(eid)
                            matched_hits += 1
                            
        if cat_scanned % 250000 == 0 or cat_scanned == total_cat:
            elapsed = time.time() - t0
            util = get_sys_util_string()
            rate = cat_scanned / max(0.01, elapsed)
            print(f"  Scanned {cat_scanned:,}/{total_cat:,} ({cat_scanned/total_cat*100:.1f}%) in {elapsed:.1f}s ({rate:,.0f} rows/s) | S1 Hits: {len(total_merged):,} queries | {util}", flush=True)
            
    conn.close()
    elapsed_scan = time.time() - t0
    print(f"  Catalog scan finished in {elapsed_scan:.2f}s ({cat_scanned:,} rows)!", flush=True)
    print(f"  Queries with extra candidates: {len(total_merged):,} / {total_s1:,} ({len(total_merged)/total_s1*100:.1f}%)", flush=True)
    print(f"  Total candidate links generated: {matched_hits:,}", flush=True)
    
    # Clean up key maps to release memory
    del s1_brand_map, s1_rare_map, s1_unanchored_addr_map, s1_street_map, s1_pc_map
    import gc
    gc.collect()
    log_memory("After Catalog Merge")
    
    # 4. Stream through input candidate pairs and merge
    print(f"\nStep 3/4: Merging existing candidates with multi-pathway candidates into {output_cands_path}...", flush=True)
    t0 = time.time()
    
    total_written = 0
    total_added_cands = 0
    
    with open(input_cands_path, 'r', encoding='utf-8') as f_in, \
         open(output_cands_path, 'w', encoding='utf-8', buffering=4*1024*1024) as f_out:
         
        header = f_in.readline()
        f_out.write(header)
        
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            idx = line.find('\t')
            if idx == -1:
                continue
            s1_id = line[:idx]
            cands_str = line[idx+1:]
            
            existing = [c.strip() for c in cands_str.split(',') if c.strip()]
            existing_set = set(existing)
            
            extra = total_merged.get(s1_id, [])
            for cand in extra:
                if cand not in existing_set:
                    existing.append(cand)
                    existing_set.add(cand)
                    total_added_cands += 1
                    
            f_out.write(f"{s1_id}\t{','.join(existing)}\n")
            total_written += 1
            if total_written % 500000 == 0:
                util = get_sys_util_string()
                print(f"  Merged {total_written:,} / {total_s1:,} queries ({total_written/total_s1*100:.1f}%) | {util}", flush=True)
                
    elapsed = time.time() - start_time
    print(f"\n{'='*75}", flush=True)
    print(f"High-Recall Candidate Expansion Complete in {elapsed:.1f}s ({elapsed/60:.1f}m)!", flush=True)
    print(f"  Total Queries Processed: {total_written:,}", flush=True)
    print(f"  Total Extra Candidates Added: {total_added_cands:,} (avg {total_added_cands/max(1, total_written):.2f} / query)", flush=True)
    print(f"  Output File: {output_cands_path} ({os.path.getsize(output_cands_path)/(1024**2):.1f} MB)", flush=True)
    print(f"{'='*75}\n", flush=True)
    log_memory("End")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Expand candidate pairs using 6-Core High-Recall Multi-Pathway Blocking.")
    parser.add_argument("--mode", choices=["test", "train", "sample"], default="test", help="Target mode: test, train, or sample")
    args = parser.parse_args()
    
    base_dir = r"C:\Users\anshu\OneDrive\Desktop\amazon-ml"
    
    if args.mode == "test":
        in_path = os.path.join(base_dir, "output", "candidate_pairs_v3_backup.tsv")
        out_path = os.path.join(base_dir, "output", "candidate_pairs_v4.tsv")
        db_path = os.path.join(base_dir, "output", "test_catalog_temp.db")
        expand_candidates(in_path, out_path, db_path, max_extra_per_query=15, max_keys_freq=40)
    elif args.mode == "train":
        in_path = os.path.join(base_dir, "output", "full_train_candidate_pairs.tsv")
        out_path = os.path.join(base_dir, "output", "full_train_candidate_pairs_v4.tsv")
        db_path = os.path.join(base_dir, "output", "train_catalog_temp.db")
        expand_candidates(in_path, out_path, db_path, max_extra_per_query=15, max_keys_freq=40)
    elif args.mode == "sample":
        in_path = os.path.join(base_dir, "output", "val_candidate_pairs_sample.tsv")
        out_path = os.path.join(base_dir, "output", "val_candidate_pairs_sample_v4.tsv")
        db_path = os.path.join(base_dir, "output", "train_catalog_temp.db")
        expand_candidates(in_path, out_path, db_path, max_extra_per_query=15, max_keys_freq=40)
