import os
import shutil
import zipfile

base_dir = r"C:\Users\anshu\OneDrive\Desktop\amazon-ml"
pkg_dir = os.path.join(base_dir, "final_submission_package")
zip_out = os.path.join(base_dir, "final_submission_package.zip")

# 1. Clean previous build directory
if os.path.exists(pkg_dir):
    shutil.rmtree(pkg_dir)

os.makedirs(os.path.join(pkg_dir, "output"), exist_ok=True)
os.makedirs(os.path.join(pkg_dir, "code", "business_entity_resolution", "src"), exist_ok=True)

# 2. Copy output deliverables
print("Copying output TSV files...")
shutil.copy2(
    os.path.join(base_dir, "output", "matching_results.tsv"),
    os.path.join(pkg_dir, "output", "matching_results.tsv")
)
shutil.copy2(
    os.path.join(base_dir, "output", "candidate_pairs.tsv"),
    os.path.join(pkg_dir, "output", "candidate_pairs.tsv")
)

# 3. Copy code deliverables
print("Copying code & documentation files...")
shutil.copy2(
    os.path.join(base_dir, "code", "business_entity_resolution", "README.md"),
    os.path.join(pkg_dir, "code", "business_entity_resolution", "README.md")
)
shutil.copy2(
    os.path.join(base_dir, "code", "business_entity_resolution", "requirements.txt"),
    os.path.join(pkg_dir, "code", "business_entity_resolution", "requirements.txt")
)

src_dir = os.path.join(base_dir, "code", "business_entity_resolution", "src")
dest_src = os.path.join(pkg_dir, "code", "business_entity_resolution", "src")
for f in os.listdir(src_dir):
    if f.endswith(".py"):
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dest_src, f))

# 4. Copy Documentation_template.md
shutil.copy2(
    os.path.join(base_dir, "6ab10eb3b23ba_student_resource", "student_resource", "Documentation_template.md"),
    os.path.join(pkg_dir, "Documentation_template.md")
)

# 5. Create zip archive
print("Compressing final zip archive...")
if os.path.exists(zip_out):
    os.remove(zip_out)

with zipfile.ZipFile(zip_out, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
    for root, dirs, files in os.walk(pkg_dir):
        for file in files:
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, pkg_dir)
            zipf.write(full_path, rel_path)

size_mb = os.path.getsize(zip_out) / (1024 * 1024)
print(f"Successfully created final submission package: {zip_out} ({size_mb:.2f} MB)")
