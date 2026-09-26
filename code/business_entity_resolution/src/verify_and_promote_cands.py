import os
import shutil

v4_path = r"output/candidate_pairs_v4.tsv"
target_path = r"output/candidate_pairs.tsv"

print("Verifying candidate_pairs_v4.tsv...")
count = 0
with open(v4_path, 'r', encoding='utf-8') as f:
    header = f.readline().strip()
    assert header == "source1_entity_id\tcandidate_entity_ids", f"Bad header: {header}"
    for line in f:
        count += 1
        if count <= 3:
            parts = line.strip().split('\t')
            num_cands = len(parts[1].split(',')) if len(parts) > 1 and parts[1] else 0
            print(f"Sample {count}: {parts[0]} -> {num_cands} candidates")
            
print(f"Total data lines: {count:,}")
assert count == 1732544, f"Expected 1,732,544 lines, got {count}"

print(f"Promoting {v4_path} to {target_path}...")
if os.path.exists(target_path):
    os.remove(target_path)
shutil.copyfile(v4_path, target_path)
print(f"Done! {target_path} size: {os.path.getsize(target_path)/(1024**2):.1f} MB")
