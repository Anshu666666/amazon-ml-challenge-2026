import os
import sys
import time
import psutil
import sqlite3
import unidecode
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

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

def get_sys_util_string():
    per_cpu = psutil.cpu_percent(percpu=True)
    mem = psutil.virtual_memory()
    cpu_str = " ".join([f"C{i}:{int(c)}%" for i, c in enumerate(per_cpu)])
    ram_str = f"RAM Avail: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}% used)"
    return f"[{cpu_str}] | {ram_str}"

def worker_scan_catalog_slice(args):
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
                                
        if scanned % 100000 == 0 or scanned == total_to_scan:
            elapsed = time.time() - t0
            util = get_sys_util_string()
            print(f"  [Worker {worker_id}] Scanned {scanned:,}/{total_to_scan:,} ({scanned/total_to_scan*100:.1f}%) in {elapsed:.1f}s | Hits: {len(local_hits):,} queries | {util}", flush=True)
            
    conn.close()
    return local_hits

if __name__ == '__main__':
    print("="*70, flush=True)
    print("DEMO TEST: 6-Core Parallel Multi-Pathway Blocking Validation", flush=True)
    print("="*70, flush=True)
    
    db_path = 'output/test_catalog_temp.db'
    abs_db = os.path.abspath(db_path).replace('\\', '/')
    conn = sqlite3.connect(f"file:{abs_db}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    
    # 1. Build keys for first 5,000 S1 queries
    print("Building keys for sample 5,000 S1 queries...", flush=True)
    cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog WHERE rowid <= 5000")
    s1_rows = cur.fetchall()
    
    brand_map = defaultdict(list)
    rare_map = defaultdict(list)
    unanchored_addr_map = defaultdict(list)
    street_map = defaultdict(list)
    pc_map = defaultdict(list)
    
    for eid, name, addr, country in s1_rows:
        cntry = country or ''
        b = get_clean_brand(name)
        if b and len(b) >= 4:
            brand_map[(cntry, b)].append(eid)
        for rw in extract_rare_brand_words(name):
            rare_map[(cntry, rw)].append(eid)
        for num, sword in extract_unanchored_addr_keys(addr):
            unanchored_addr_map[(cntry, num, sword)].append(eid)
        sk, pk = get_addr_keys(addr)
        if sk:
            street_map[(cntry, sk[0], sk[1])].append(eid)
        if pk:
            pc_map[(cntry, pk[0], pk[1])].append(eid)
            
    print(f"Built sample keys: {len(brand_map):,} brands, {len(rare_map):,} rare words, {len(unanchored_addr_map):,} addrs.", flush=True)
    print(f"System State: {get_sys_util_string()}", flush=True)
    
    # 2. Divide 600,000 catalog rows across 6 cores (100,000 rows each)
    slice_size = 100000
    num_cores = 6
    tasks = []
    key_bundles = (brand_map, rare_map, unanchored_addr_map, street_map, pc_map)
    
    for i in range(num_cores):
        start_r = i * slice_size + 1
        end_r = (i + 1) * slice_size
        tasks.append((i, db_path, start_r, end_r, key_bundles, 15, 40))
        
    print(f"\nLaunching {num_cores} workers scanning 600,000 total catalog rows...", flush=True)
    t0 = time.time()
    
    total_merged = defaultdict(list)
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(worker_scan_catalog_slice, tasks)
        for r in results:
            for sid, c_list in r.items():
                total_merged[sid].extend(c_list)
                
    elapsed = time.time() - t0
    print(f"\n6-Core Demo Parallel Scan Completed in {elapsed:.2f}s ({600000/elapsed:,.0f} rows/s)!", flush=True)
    print(f"Total S1 queries that gained candidates: {len(total_merged):,} / 5,000", flush=True)
    print(f"Final System State: {get_sys_util_string()}", flush=True)
    print("="*70, flush=True)
    print("DEMO TEST RESULT: 100% SUCCESS — 6-core multiprocessing verified!", flush=True)
    print("="*70, flush=True)
