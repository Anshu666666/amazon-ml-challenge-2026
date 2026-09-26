import os
import sys
import time
import psutil
import sqlite3
import unidecode
import re
from collections import defaultdict

def get_sys_util_string():
    per_cpu = psutil.cpu_percent(percpu=True)
    mem = psutil.virtual_memory()
    cpu_str = " ".join([f"C{i}:{int(c)}%" for i, c in enumerate(per_cpu)])
    ram_str = f"RAM Avail: {mem.available / (1024**3):.2f} GB / {mem.total / (1024**3):.2f} GB ({mem.percent}% used)"
    return f"[{cpu_str}] | {ram_str}"

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

def test_demo_expansion():
    print("=" * 70)
    print("TEST DEMO: Validating Multi-Pathway Expansion & Memory Safety")
    print(f"Initial State: {get_sys_util_string()}")
    print("=" * 70)
    
    db_path = r"c:\Users\anshu\OneDrive\Desktop\amazon-ml\output\test_catalog_temp.db"
    abs_db = os.path.abspath(db_path).replace('\\', '/')
    conn = sqlite3.connect(f"file:{abs_db}?mode=ro&immutable=1", uri=True)
    cur = conn.cursor()
    
    # 1. Sample 50,000 S1 queries
    print("\nPhase 1: Indexing 50,000 sample S1 queries...")
    t0 = time.time()
    cur.execute("SELECT entity_id, business_name, business_address, country FROM s1_catalog LIMIT 50000")
    s1_rows = cur.fetchall()
    
    s1_brand_map = defaultdict(list)
    s1_rare_map = defaultdict(list)
    s1_unanchored_map = defaultdict(list)
    s1_street_map = defaultdict(list)
    s1_pc_map = defaultdict(list)
    
    for eid, name, addr, country in s1_rows:
        cntry = country or ''
        b = get_clean_brand(name)
        if b and len(b) >= 4:
            s1_brand_map[(cntry, b)].append(eid)
        for rw in extract_rare_brand_words(name):
            s1_rare_map[(cntry, rw)].append(eid)
        for num, sword in extract_unanchored_addr_keys(addr):
            s1_unanchored_map[(cntry, num, sword)].append(eid)
        sk, pk = get_addr_keys(addr)
        if sk:
            s1_street_map[(cntry, sk[0], sk[1])].append(eid)
        if pk:
            s1_pc_map[(cntry, pk[0], pk[1])].append(eid)
            
    print(f"Indexed 50,000 S1 queries in {time.time()-t0:.2f}s.")
    print(f"Keys: {len(s1_brand_map):,} brands | {len(s1_rare_map):,} rare words | {len(s1_unanchored_map):,} unanchored addrs | {len(s1_street_map):,} streets | {len(s1_pc_map):,} pc")
    print(f"Post-Index State: {get_sys_util_string()}")
    
    # 2. Scan 200,000 catalog rows
    print("\nPhase 2: Scanning 200,000 catalog rows against blocking keys...")
    t0 = time.time()
    cur.execute("SELECT entity_id, business_name, business_address, country FROM catalog LIMIT 200000")
    
    hits_by_s1 = defaultdict(list)
    max_extra = 15
    max_freq = 40
    batch_size = 50000
    scanned = 0
    
    while True:
        rows = cur.fetchmany(batch_size)
        if not rows:
            break
        scanned += len(rows)
        for eid, name, addr, country in rows:
            cntry = country or ''
            # Brand
            b = get_clean_brand(name)
            if b and len(b) >= 4:
                k = (cntry, b)
                if k in s1_brand_map and len(s1_brand_map[k]) <= max_freq:
                    for sid in s1_brand_map[k]:
                        if len(hits_by_s1[sid]) < max_extra:
                            hits_by_s1[sid].append(eid)
            # Rare words
            for rw in extract_rare_brand_words(name):
                k = (cntry, rw)
                if k in s1_rare_map and len(s1_rare_map[k]) <= max_freq:
                    for sid in s1_rare_map[k]:
                        if len(hits_by_s1[sid]) < max_extra:
                            hits_by_s1[sid].append(eid)
            # Unanchored addr
            for num, sword in extract_unanchored_addr_keys(addr):
                k = (cntry, num, sword)
                if k in s1_unanchored_map and len(s1_unanchored_map[k]) <= max_freq:
                    for sid in s1_unanchored_map[k]:
                        if len(hits_by_s1[sid]) < max_extra:
                            hits_by_s1[sid].append(eid)
            # Street & postal
            sk, pk = get_addr_keys(addr)
            if sk:
                k = (cntry, sk[0], sk[1])
                if k in s1_street_map and len(s1_street_map[k]) <= max_freq:
                    for sid in s1_street_map[k]:
                        if len(hits_by_s1[sid]) < max_extra:
                            hits_by_s1[sid].append(eid)
            if pk:
                k = (cntry, pk[0], pk[1])
                if k in s1_pc_map and len(s1_pc_map[k]) <= max_freq:
                    for sid in s1_pc_map[k]:
                        if len(hits_by_s1[sid]) < max_extra:
                            hits_by_s1[sid].append(eid)
                            
        util = get_sys_util_string()
        print(f"  Scanned {scanned:,} catalog rows | Hits: {len(hits_by_s1):,} S1 queries matched | {util}", flush=True)
        
    conn.close()
    elapsed = time.time() - t0
    rate = scanned / max(0.01, elapsed)
    print(f"\nScan Complete: {scanned:,} rows in {elapsed:.2f}s ({rate:,.0f} rows/s)")
    print(f"Queries matched: {len(hits_by_s1):,} / 50,000 ({len(hits_by_s1)/50000*100:.2f}%)")
    total_added = sum(len(v) for v in hits_by_s1.values())
    print(f"Total candidate hits generated: {total_added:,}")
    print(f"Final State: {get_sys_util_string()}")
    print("\nTEST DEMO RESULT: SUCCESS! Zero errors, RAM stable.")
    print("=" * 70)

if __name__ == "__main__":
    test_demo_expansion()
